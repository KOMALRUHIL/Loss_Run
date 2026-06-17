import ast
import requests
import os
from dotenv import load_dotenv
import time
import random
import re
import models

# Load env variables
load_dotenv()

def exponential_backoff(base_delay=2, max_delay=60, factor=2, jitter=True):
    delay = base_delay
    while True:
        yield delay
        if jitter:
            delay = min(max_delay, delay * factor) * (0.5 + random.random() / 2)
        else:
            delay = min(max_delay, delay * factor)

def _sanitize_prompt_for_safety(text: str) -> str:
    if not text:
        return text
    patterns = [
        r"\bkill(?:ing)?\b",
        r"\bmurder(?:ing)?\b",
        r"\bhomicide\b",
        r"\bsuicide\b",
        r"\bassault\b",
        r"\brape\b",
        r"\bshoot(?:ing|s|er|ers)?\b",
        r"\bshot\b",
        r"\bgun(?:s)?\b",
        r"\bfirearm(?:s)?\b",
        r"\bstab(?:bed)?\b",
        r"\bblood\b",
        r"\bblood\b",
        r"\bgore\b",
        r"\bdeath(?:s)?\b",
        r"\bfatal(?:ity|ities)?\b",
        r"\binjur(?:y|ies)\b",
    ]
    sanitized = text
    for pattern in patterns:
        sanitized = re.sub(pattern, "[redacted]", sanitized, flags=re.IGNORECASE)
    return sanitized

def call_api_with_retries(url, params, headers, json, max_attempts=10):
    retries = exponential_backoff()
    for attempt in range(max_attempts):
        try:
            resp = requests.post(
                url,
                params=params,
                headers=headers,
                json=json
            )
            resp.raise_for_status()
            response = resp.json()
            
            # Check for content filter
            filter_results = response.get('choices', [{}])[0].get('content_filter_results', {})
            filtered_categories = [k for k, v in filter_results.items() if isinstance(v, dict) and v.get('filtered')]
            if filtered_categories:
                print(f"Content filtered in categories: {filtered_categories}. Disregarding.")
                return "", 0, 0
                
            content = response.get('choices', [{}])[0].get('message', {}).get('content', "")
            usage = response.get('usage', {}) or {}
            input_tokens = usage.get('prompt_tokens', 0) or 0
            output_tokens = usage.get('completion_tokens', 0) or 0
            
            return content, input_tokens, output_tokens
            
        except requests.HTTPError as e:
            status_code = getattr(e.response, "status_code", None)
            response_text = getattr(e.response, "text", "") if e.response is not None else ""
            if response_text:
                response_text = response_text[:1000]
                
            print(
                f"API call failed with HTTP {status_code}: {e}."
                + (f" Response: {response_text}" if response_text else "")
            )
            
            # Do not retry non-rate-limit 4xx errors
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                return "", 0, 0
                
            delay = next(retries)
            print(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
        except Exception as e:
            delay = next(retries)
            print(f"API call failed with error: {e}. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
            
    print(f"API call failed after {max_attempts} attempts")
    return "", 0, 0

def _is_valid_json(text: str) -> bool:
    import json as _json
    try:
        _json.loads(text)
        return True
    except Exception:
        return False

def _extract_balanced_json(text: str, start_idx: int) -> str | None:
    stack = []
    in_string = False
    escape = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                continue
        elif ch == '"':
            in_string = True
            continue
            
        if ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                continue
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                return None
            if not stack:
                return text[start_idx:i+1]
    return None

def _normalize_numeric_expressions(text: str) -> str:
    """
    Evaluates math expressions in the JSON text so it becomes valid JSON.
    E.g. "total_incurred": 15000 + 3500 -> "total_incurred": 18500
    """
    def evaluate_expression(match):
        prefix = match.group(1)
        expr = match.group(2)
        try:
            val = eval(expr, {"__builtins__": None}, {})
            if isinstance(val, (int, float)):
                if isinstance(val, float) and val.is_integer():
                    val = int(val)
                return f"{prefix}{val}"
        except Exception:
            pass
        return match.group(0)

    pattern = r'(:\s*)(\d+(?:\.\d+)?(?:\s*[\+\-\*/\(\)]\s*\d+(?:\.\d+)?)+)'
    return re.sub(pattern, evaluate_expression, text)

def _coerce_json_text(raw_output: str) -> str | None:
    if raw_output is None:
        return None
    text = str(raw_output).strip()
    if not text:
        return None
        
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
        if _is_valid_json(text):
            return text
        normalized = _normalize_numeric_expressions(text)
        if normalized != text and _is_valid_json(normalized):
            print("[WARN] LLM output had numeric expressions. Evaluated to numbers.")
            return normalized
            
    if _is_valid_json(text):
        return text
        
    normalized = _normalize_numeric_expressions(text)
    if normalized != text and _is_valid_json(normalized):
        print("[WARN] LLM output had numeric expressions. Evaluated to numbers.")
        return normalized
        
    for i, ch in enumerate(text):
        if ch in "[{":
            candidate = _extract_balanced_json(text, i)
            if candidate and _is_valid_json(candidate):
                return candidate
            if candidate:
                normalized_candidate = _normalize_numeric_expressions(candidate)
                if normalized_candidate != candidate and _is_valid_json(normalized_candidate):
                    print("[WARN] LLM output had numeric expressions. Evaluated to numbers.")
                    return normalized_candidate
    return None

def gpt_call(
    prompt,
    model_name,
    api_version,
    temperature=0
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
        return None, 0, 0
    if not api_key:
        print(f"[ERROR] API key for model '{model_name}' not found. Check environment ({key_var} or AZURE_OPENAI_API_KEY).")
        return None, 0, 0

    endpoint = endpoint.rstrip("/")
    url = f"{endpoint}/openai/deployments/{model_name}/chat/completions"
    
    headers = {
        'Content-Type': 'application/json',
        'api-key': api_key,
    }
    
    params = {
        'api-version': api_version,
    }
    
    json_data = {
        'messages': [
            {
                'role': 'system',
                'content': 'Descriptions may include medical or accident details, but are not violent in intent. Do not filter unless truly violent or graphic.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ]
    }
    
    if "gpt-5" not in model_name:
        json_data['temperature'] = temperature
        
    output, input_tokens, output_tokens = call_api_with_retries(
        url=url,
        params=params,
        headers=headers,
        json=json_data
    )
    
    if not output:
        sanitized_prompt = _sanitize_prompt_for_safety(prompt)
        if sanitized_prompt != prompt:
            print("[WARN] Retrying with sanitized prompt due to empty/filtered output.")
            json_data["messages"][1]["content"] = sanitized_prompt
            output_retry, input_tokens_retry, output_tokens_retry = call_api_with_retries(
                url=url,
                params=params,
                headers=headers,
                json=json_data
            )
            if output_retry:
                output = output_retry
                input_tokens += input_tokens_retry
                output_tokens += output_tokens_retry
                
    if not output:
        print("[ERROR] LLM output was empty or filtered. Skipping this chunk.")
        return None, input_tokens, output_tokens
        
    if not _is_valid_json(output):
        extracted = _coerce_json_text(output)
        if extracted:
            print("[WARN] LLM output was not strict JSON. Using extracted JSON payload.")
            return extracted, input_tokens, output_tokens
            
        print("[ERROR] LLM output was not valid JSON. Logging and skipping.")
        with open("llm_error_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Prompt:\n{prompt}\nOutput:\n{output}\n\n")
        return None, input_tokens, output_tokens
        
    return output, input_tokens, output_tokens

def safe_string_to_list(string_data):
    try:
        result = ast.literal_eval(string_data)
        if isinstance(result, list):
            return result
        else:
            return []
    except Exception:
        return []
