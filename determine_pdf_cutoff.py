import os
import json
import pandas as pd
import time
import tiktoken
from prompts import CHUNK_DETECTION_PROMPT
from utils_gpt import gpt_call
import models
from metrics import record_stage

CHUNK_DETECTION_MODEL = models.GPT_5_2

_chunk_token_usage = {} # key: (company, pdf_name) -> {"input": int, "output": int}
_chunk_time_usage = {}  # key: (company, pdf_name) -> float seconds

def _to_long_path(path: str) -> str:
    """
    Ensure Windows paths are prefixed with \\?\\ to bypass MAX_PATH (260) limits.
    Keeps non-Windows paths untouched.
    """
    if os.name != "nt":
        return path
        
    abs_path = os.path.abspath(path)
    if abs_path.startswith("\\\\?\\"):
        return abs_path
    return "\\\\?\\" + abs_path


def _log_chunk_decision(company: str, pdf_name: str, pages_to_join: list[list[int]]):
    """
    Print per-PDF chunking decision and LLM cost / time.
    """
    key = (company, pdf_name)
    elapsed = _chunk_time_usage.get(key, 0.0)
    usage = _chunk_token_usage.get(key, {"input": 0, "output": 0})
    
    print(f"        Company: {company}")
    print(f"        PDF Chunking: {pdf_name}")
    print(f"        Page groups: {pages_to_join}")
    print(f"                 ⏱  Chunking Time: {elapsed:.2f} seconds")
    
    if usage["input"] or usage["output"]:
        cost = models.compute_cost(
            usage["input"],
            usage["output"],
            CHUNK_DETECTION_MODEL,
        )
    else:
        cost = 0.0
    print(f"                 💰 Chunking Cost: ${cost:.6f}")
    
    record_stage(
        company,
        pdf_name,
        stage="chunk_detection",
        time_seconds=elapsed,
        input_tokens=usage["input"],
        output_tokens=usage["output"],
        cost=cost,
    )


def image_token_calc(width: int, height: int, detail: str = "high") -> int:
    """
    Function to calculate how many tokens an image will use for GPT-4.1 VLM
    """
    if detail == "low":
        return 85
    
    # Scale down to fit within a 2048 x 2048 square if necessary
    if width > 2048 or height > 2048:
        max_size = 2048
        aspect_ratio = width / height
        if aspect_ratio > 1:
            width = max_size
            height = int(max_size / aspect_ratio)
        else:
            height = max_size
            width = int(max_size * aspect_ratio)
    
    # Resize such that the shortest side is 768px if the original dimensions exceed 768px
    min_size = 768
    aspect_ratio = width / height
    if width > min_size and height > min_size:
        if aspect_ratio > 1:
            height = min_size
            width = int(min_size * aspect_ratio)
        else:
            width = min_size
            height = int(min_size / aspect_ratio)
    
    tiles_width = -(-width // 512)  # Ceiling division
    tiles_height = -(-height // 512)
    return 85 + 170 * (tiles_width * tiles_height)


def text_token_calc(text: str, model_name: str = "gpt-4.1") -> int:
    """Function to calculate how many tokens a text will use for a given model."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        print(f"Warning: Model '{model_name}' not found. Using 'cl100k_base' encoding as a fallback.")
    
    num_tokens = len(encoding.encode(text))
    return num_tokens


def _load_page_texts(pdf_subfolder: str) -> dict:
    """Load OCR text mapping from page_response.json: {page_num:int -> text:str}"""
    page_texts = {}
    page_response_path = os.path.join(pdf_subfolder, "page_response.json")
    if not os.path.exists(_to_long_path(page_response_path)):
        return page_texts
    
    with open(_to_long_path(page_response_path), "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for k, v in data.items():
        try:
            page_num = int(k)
        except ValueError:
            continue
        if v is None or (isinstance(v, float) and pd.isna(v)):
            v = ""
        page_texts[page_num] = str(v)
    return page_texts


def _llm_choose_chunk_end_page(
    *,
    candidate_end_pages: list[int],
    page_texts: dict,
    end_page: int,
    usage_key: tuple | None = None,   # (company, pdf_name)
) -> int:
    """
    Ask LLM to select the best chunk end page among candidates.
    Provide OCR context for end_page and up to 4 pages before it (or fewer).
    """
    if not candidate_end_pages:
        raise ValueError("candidate_end_pages cannot be empty")
    
    # Provide only the pages the prompt needs: end page + up to 4 pages before
    start_context_page = max(1, end_page - 4)
    context_pages = {
        str(p): page_texts.get(p, "")
        for p in range(start_context_page, end_page + 1)
    }
    
    prompt = CHUNK_DETECTION_PROMPT.format(
        candidate_end_pages=candidate_end_pages,
        pages_ocr_json=json.dumps(context_pages, ensure_ascii=False),
    )
    
    raw, input_tokens, output_tokens = gpt_call(prompt,
                                                CHUNK_DETECTION_MODEL['model_name'],
                                                CHUNK_DETECTION_MODEL['api_version'])
    
    # Track token usage if key provided
    if usage_key is not None:
        usage = _chunk_token_usage.setdefault(usage_key, {"input": 0, "output": 0})
        usage["input"] += input_tokens or 0
        usage["output"] += output_tokens or 0
    
    try:
        parsed = json.loads(raw)
        chosen = int(parsed["chunk_end_page"])
    except Exception:
        # On any parse/format failure, fall back conservatively to the latest page (max tokens utilization)
        return max(candidate_end_pages)
    
    if chosen not in candidate_end_pages:
        return max(candidate_end_pages)
    
    return chosen


def find_context_limit_page(
    pdf_subfolder, start_page=1, prompt_tokens=2500, token_limit=35000, max_images=50
):
    """
    Find the maximum page number that can be processed without exceeding token_limit or max_images.
    """
    def get_page_num(fname):
        try:
            return int(fname.split('-')[-1].split('.')[0])
        except Exception:
            return 0
    
    all_image_paths = [
        os.path.join(pdf_subfolder, fname)
        for fname in sorted(
            [f for f in os.listdir(pdf_subfolder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=get_page_num
        )
    ]
    
    if not all_image_paths:
        print(f"No images found in {pdf_subfolder}")
        return None, 0
    
    image_paths = [path for path in all_image_paths if get_page_num(os.path.basename(path)) >= start_page]
    
    if not image_paths:
        print(f"No images found from page {start_page} onwards in {pdf_subfolder}")
        return None, 0
    
    page_texts = _load_page_texts(pdf_subfolder)
    
    total_tokens = prompt_tokens
    last_valid_page = None
    images_processed = 0
    
    for image_path in image_paths:
        page_num = get_page_num(os.path.basename(image_path))
        
        images_processed += 1
        if images_processed > max_images:
            break
            
        from PIL import Image
        with Image.open(_to_long_path(image_path)) as img:
            width, height = img.size
            image_tokens = image_token_calc(width, height, detail="high")
            
        text = page_texts.get(page_num, "")
        if text is None or (isinstance(text, float) and pd.isna(text)):
            text = ""
        else:
            text = str(text)
        text_tokens = text_token_calc(text) if text else 0
        
        page_total_tokens = image_tokens + text_tokens
        if total_tokens + page_total_tokens > token_limit:
            break
            
        total_tokens += page_total_tokens
        last_valid_page = page_num
        
    return last_valid_page, total_tokens


def find_context_limit_page_refined(
    pdf_subfolder: str,
    start_page: int = 1,
    prompt_tokens: int = 2500,
    token_limit: int = 30000,
    max_images: int = 50,
    candidate_backoff: int = 4,
    usage_key: tuple | None = None, # (company, pdf_name)
):
    """
    1) Compute the max end page by token/image budget (existing logic).
    2) Ask LLM to choose the BEST chunk boundary among nearby candidate pages,
       using OCR context for [end_page-4 .. end_page].
    """
    max_page, total_tokens = find_context_limit_page(
        pdf_subfolder,
        start_page=start_page,
        prompt_tokens=prompt_tokens,
        token_limit=token_limit,
        max_images=max_images,
    )
    
    if max_page is None:
        return None, total_tokens
        
    def get_page_num(fname):
        try:
            return int(fname.split("-")[-1].split(".")[0])
        except Exception:
            return 0
            
    all_image_pages = [
        get_page_num(f)
        for f in os.listdir(pdf_subfolder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    last_doc_page = max(all_image_pages) if all_image_pages else None
    
    if last_doc_page is not None and max_page >= last_doc_page:
        return max_page, total_tokens
        
    # Can't choose an end page earlier than start_page
    low = max(start_page, max_page - candidate_backoff)
    candidate_end_pages = list(range(low, max_page + 1))
    
    # If we only have one candidate, no need to call LLM
    if len(candidate_end_pages) == 1:
        return max_page, total_tokens
        
    page_texts = _load_page_texts(pdf_subfolder)
    chosen_end_page = _llm_choose_chunk_end_page(
        candidate_end_pages=candidate_end_pages,
        page_texts=page_texts,
        end_page=max_page,
        usage_key=usage_key,
    )
    
    return chosen_end_page, total_tokens


def determine_page_cutoffs(output_data_dir, company_folders, is_file_loss_run_dict):
    """
    Detect if any PDF pages need to be split based on token/image limits and store which pages need to be joined.
    """
    pages_to_join_dict = {}
    
    for company_folder in company_folders:
        company_path = os.path.join(output_data_dir, company_folder)
        pdf_checked = False
        
        if os.path.isdir(company_path):
            for subfolder in os.listdir(company_path):
                if subfolder.lower() in ["csvs", "excel_sheets"]:
                    continue
                
                key = (company_folder, subfolder)
                if key in is_file_loss_run_dict:
                    info = is_file_loss_run_dict[key]
                    if info.get("is_loss_run") is False:
                        continue
                        
                subfolder_path = os.path.join(company_path, subfolder)
                
                if not os.path.isdir(subfolder_path):
                    continue
                    
                image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if not image_files:
                    continue
                    
                total_pages = len(image_files)
                pdf_checked = True
                
                pages_to_join = []
                current_start_page = 1
                
                # initialize time and token usage for this PDF
                usage_key = (company_folder, subfolder)
                _chunk_token_usage.setdefault(usage_key, {"input": 0, "output": 0})
                start_time = time.time()
                
                while current_start_page <= total_pages:
                    max_page, total_tokens = find_context_limit_page_refined(
                        subfolder_path,
                        start_page=current_start_page,
                        usage_key=usage_key,
                    )
                    
                    if max_page is None:
                        print(f"        ⚠ Warning: Could not process any pages starting from page {current_start_page}")
                        break
                        
                    page_group = list(range(current_start_page, max_page + 1))
                    pages_to_join.append(page_group)
                    
                    if max_page >= total_pages:
                        break
                        
                    current_start_page = max_page + 1 # next chunk begins after chosen LLM cut
                    
                # record elapsed time (0 if loop never ran)
                _chunk_time_usage[usage_key] = time.time() - start_time
                
                if pages_to_join:
                    pages_to_join_dict[(company_folder, subfolder)] = pages_to_join
                    _log_chunk_decision(company_folder, subfolder, pages_to_join)
                    
    return pages_to_join_dict
