import os
import requests
import time
import random
import json
import base64
from mimetypes import guess_type
from dotenv import load_dotenv
import models

load_dotenv()

# Conditionally set CA certificate bundle for Liberty Mutual proxy if the file exists
cert_path = r'C:\Users\n1700803\OneDrive - Liberty Mutual\Documents\Groundspeed-Replacement_IU\cert_certificate\cacert.pem'
if os.path.exists(cert_path):
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path

def exponential_backoff(base_delay=2, max_delay=60, factor=2, jitter=True):
    delay = base_delay
    while True:
        yield delay
        if jitter:
            delay = min(max_delay, delay * factor) * (0.5 + random.random() / 2)
        else:
            delay = min(max_delay, delay * factor)

def gpt_vision_call(
    prompt,
    folder_with_images_path,
    model_name,
    api_version,
    messages=None,
    temperature=0,
):
    model_cfg = models.MODEL_REGISTRY.get(model_name, {})
    endpoint_var = model_cfg.get("endpoint_env")
    key_var = model_cfg.get("key_env")
    
    endpoint = os.getenv(endpoint_var) if endpoint_var else None
    api_key = os.getenv(key_var) if key_var else None
    
    if not endpoint:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not api_key:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
    if not endpoint:
        print(f"[ERROR] Endpoint for model '{model_name}' not found. Check environment ({endpoint_var} or AZURE_OPENAI_ENDPOINT).")
        return "", 0, 0
    if not api_key:
        print(f"[ERROR] API key for model '{model_name}' not found. Check environment ({key_var} or AZURE_OPENAI_API_KEY).")
        return "", 0, 0

    endpoint = endpoint.rstrip("/")
    url = f"{endpoint}/openai/deployments/{model_name}/chat/completions?api-version={api_version}"

    if messages:
        data = {
            "messages": messages,
        }
    else:
        def get_page_num(fname):
            try:
                return int(os.path.basename(fname).split('-')[-1].split('.')[0])
            except Exception:
                return 0
                
        folder_with_images_path = sorted(folder_with_images_path, key=get_page_num)
        
        def encode_image(image_path):
            mime_type, _ = guess_type(image_path)
            if mime_type is None:
                mime_type = 'application/octet-stream'
                
            with open(image_path, "rb") as image_file:
                base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
                
            return f"data:{mime_type};base64,{base64_encoded_data}"
            
        image_contents = []
        for i, image_path in enumerate(folder_with_images_path):
            image_contents.append({"type": "text", "text": f"Below is Page No. {i + 1}"})
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": encode_image(image_path),
                    "detail": "high"
                }
            })
            
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful underwriter tasked with extracting information from insurance documents."},
                {"role": "user", "content": [{"type": "text", "text": prompt}, *image_contents]}
            ]
        }
        
    if "gpt-5" not in (model_name or ""):
        data['temperature'] = temperature
        
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    retries = exponential_backoff()
    for attempt in range(5):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            resp_json = response.json() or {}
            
            usage = resp_json.get("usage", {}) or {}
            input_tokens = usage.get("prompt_tokens", 0) or 0
            output_tokens = usage.get("completion_tokens", 0) or 0
            
            output = (
                resp_json.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            
            return output, input_tokens, output_tokens
            
        except Exception as e:
            delay = next(retries)
            print(f"Vision API call failed with error: {e}. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
            
    print(f"Vision API call failed after 5 attempts")
    return "", 0, 0
