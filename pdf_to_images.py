import os
import time
import json
import base64
import re
import hashlib
from mimetypes import guess_type
import fitz # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import models
from utils_gpt_vision import gpt_vision_call

ROTATION_DETECTION_MODEL = models.GPT_5_2

# Conservative max path to avoid hitting Windows MAX_PATH (260). Leave headroom for filenames.
MAX_WINDOWS_PATH = 240


def _as_extended_path(path: str) -> str:
    """Return Windows extended-length path to avoid MAX_PATH issues."""
    if os.name != "nt":
        return path
    if path.startswith("\\\\?\\") or path.startswith("//?/"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path.lstrip("\\")
    return "\\\\?\\" + path


def _safe_subfolder_name(filename: str, max_len: int = 80) -> str:
    """
    Build a Windows-friendly folder name for a PDF so paths stay under MAX_PATH.
    - Sanitizes to alnum/underscore/dot/dash
    - Truncates and appends a short hash if still too long
    Deterministic: same filename -> same folder name.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename).strip("_")
    if len(safe) <= max_len:
        return safe

    root, ext = os.path.splitext(safe)
    digest = hashlib.md5(safe.encode("utf-8")).hexdigest()[:8]
    # leave room for underscore + hash + extension
    reserve = len(ext) + len(digest) + 1  # "_" between root and hash
    truncated_root = root[: max(1, max_len - reserve)]
    return f"{truncated_root}_{digest}{ext}"


def detect_rotation(image_path):
    """
    Detects if a PDF page needs to be rotated to be read correctly.
    Vision good at knowing when rotation is needed, but not good at knowing the degrees to fix.
    """

    detection_prompt = """
You are tasked as a document orientation specialist. Your primary responsibility is to analyze and correct the orientation of various documents, including pages with text, tables, handwriting, and images, ensuring they are in a standard reading orientation.

You will be provided with the exact same exact page four times, each with a different rotation. The task is to identify the only image that reflects the normal reading orientation out of the four images.

The normal reading orientation would ensure text and tables are read from left to right and top to bottom.
Conflicting Cues: In situations where different elements (like text and images) suggest conflicting orientations, give priority to the orientation suggested by text and tables.

Your output should select which image is correctly oriented by responding with a JSON array in the following format:
Ensure the JSON is valid. No Markdown, no explanation.
[
    {
        "correct_image": <1, 2, 3, or 4>
    }
]
"""

    # Take in the image_path
    # Generate the 4 orientations
    # Pass main prompt & all of them into messages using the code from the blog
    # Get the output from vision to select which of the 4 images to use

    # Generate the 4 orientations and save them
    img = Image.open(image_path)
    rotations = [0, 90, 180, 270]
    rotated_image_paths = []
    base, ext = os.path.splitext(image_path)
    for idx, angle in enumerate(rotations):
        rotated_img = img.rotate(-angle, expand=True)
        rotated_path = f"{base}_rot{angle}{ext}"
        rotated_img.save(rotated_path)
        rotated_image_paths.append(rotated_path)

    # Encodes image to format for Prompt
    def encode_image(image_path):
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream' # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    # Build messages for GPT Vision in the requested format
    messages = [
        {"role": "user", "content": detection_prompt},
        *[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Below is Image. No.{i + 1}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encode_image(rotated_image_paths[i])
                        }
                    }
                ]
            }
            for i in range(4)
        ]
    ]

    for attempt in range(1, 6):
        response, input_tokens, output_tokens = gpt_vision_call(None,
                                                                None,
                                                                ROTATION_DETECTION_MODEL['model_name'],
                                                                ROTATION_DETECTION_MODEL['api_version'],
                                                                messages)

        try:
            data = json.loads(response)
            if not isinstance(data, list) or not data:
                raise ValueError("Response is not a non-empty JSON array")
            correct_image = data[0].get("correct_image")
            if correct_image not in [1, 2, 3, 4]:
                raise ValueError("correct_image is not 1, 2, 3, or 4")
            # Overwrite the original image_path with the selected image
            selected_path = rotated_image_paths[correct_image - 1]
            Image.open(selected_path).save(image_path)
            # Delete all rotated images
            for path in rotated_image_paths:
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Warning: Could not delete {path}: {e}")
            # Return degrees rotated (0 if original form)
            return [0, 90, 180, 270][correct_image - 1]
        except Exception as e:
            print(f"(attempt): Error parsing GPT Vision response: {e}")
            if attempt < 5:
                time.sleep(2)
            else:
                print("Max retries reached. Keeping original image.")
                for path in rotated_image_paths:
                    if path != image_path:
                        try:
                            os.remove(path)
                        except Exception as e:
                            print(f"Warning: Could not delete {path}: {e}")
                return 0  # Default to 0 degrees if failed, since original is kept


def pdf_to_images(pdf_path, output_dir, bar_height=160, dpi=450):
    """
    Converts each page of a PDF to an image, adds a black bar with page number, and saves to output_dir.
    Returns a list of saved image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_abs = os.path.abspath(pdf_path)
    pdf_path_for_open = _as_extended_path(pdf_abs)
    doc = fitz.open(pdf_path_for_open)
    image_paths = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        image_path = os.path.join(output_dir, f"page_{i+1}.jpg")
        pix.save(image_path)

        # Open image and correct orientation if needed
        img = Image.open(image_path)
        rotation_value = detect_rotation(image_path)
        if rotation_value and rotation_value in [90, 180, 270]:
            print(f"    GPT Vision Rotated: {pdf_path} page {i+1} by {rotation_value} degrees")

        # Re open with the updated image
        img = Image.open(image_path)

        # Add black bar with page number
        new_img = Image.new("RGB", (img.width, img.height + bar_height), "black")
        new_img.paste(img, (0, bar_height))
        draw = ImageDraw.Draw(new_img)
        try:
            font = ImageFont.truetype("arialbd.ttf", 96)
        except:
            font = ImageFont.load_default()
        text = f"Page {i+1}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (img.width - text_width) // 2
        text_y = (bar_height - text_height) // 2
        draw.text((text_x, text_y), text, fill="white", font=font)

        new_img.save(image_path)
        image_paths.append(image_path)

    return image_paths


def turn_pdf_to_images(companies, input_root, output_root, is_file_loss_run_dict):

    max_image_filename = "image-page-999.jpg"  # reserve space for largest expected page filename

    for company in companies:
        company_dir = os.path.join(input_root, company)
        if not os.path.isdir(company_dir):
            continue

        # Base output path for the company, created once
        company_output_dir = os.path.join(output_root, company)
        os.makedirs(company_output_dir, exist_ok=True)

        for filename in os.listdir(company_dir):
            if filename.lower().endswith(".pdf"):
                # If the file was not detected as a loss run, don't make images
                key = (company, filename)
                info = is_file_loss_run_dict.get(key)
                # If there is an entry and it says NOT a loss run -> skip
                if info is not None and not bool(info.get("is_loss_run")):
                    continue

                pdf_path = os.path.join(company_dir, filename)

                # Calculate a safe subfolder length so the eventual image path stays under MAX_WINDOWS_PATH
                base_path_len = len(os.path.abspath(company_output_dir))
                # account for path separator and longest image filename
                available_len = MAX_WINDOWS_PATH - base_path_len - len(os.path.sep) - len(max_image_filename)
                # ensure we still have room for a hashed folder name if needed
                available_len = max(16, available_len)

                subfolder_name = _safe_subfolder_name(filename, max_len=available_len)
                output_dir = os.path.join(company_output_dir, subfolder_name)

                # Fallback: if somehow still too long, collapse to a short hash
                if len(os.path.abspath(output_dir)) > MAX_WINDOWS_PATH:
                    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()[:16]
                    subfolder_name = f"file_{digest}"
                    output_dir = os.path.join(company_output_dir, subfolder_name)

                os.makedirs(output_dir, exist_ok=True)

                # Propagate detection metadata to the sanitized folder key so downstream steps match
                if info is not None:
                    is_file_loss_run_dict.setdefault((company, subfolder_name), info)

                # NEW: if images already exist for this file, skip
                already_has_images = any(
                    fname.lower().startswith("image-page-")
                    and fname.lower().endswith(".jpg")
                    for fname in os.listdir(output_dir)
                )
                if already_has_images:
                    continue

                image_paths = pdf_to_images(pdf_path, output_dir)
                for idx, img_path in enumerate(image_paths, 1):
                    new_name = os.path.join(output_dir, f"image-page-{idx}.jpg")
                    if os.path.exists(new_name):
                        os.remove(new_name)
                    os.rename(img_path, new_name)

if __name__ == "__main__":
    pdf_to_images(["Jason Douglas Holdings LLC"])
