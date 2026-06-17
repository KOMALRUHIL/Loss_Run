import os
import re
import json
import time
import tiktoken
from utils_excel_csv_to_json_advanced import get_excel_csv_chunks_advanced as get_excel_csv_chunks
from prompts import LOSS_RUN_DETECTION_PROMPT
from utils_gpt import gpt_call
import models
from metrics import record_stage

LOSS_RUN_DETECTION_MODEL = models.GPT_5_2

# Guardrails to prevent oversized prompts for detection
MAX_DETECTION_PROMPT_TOKENS = 20000
MAX_DETECTION_ITEM_TOKENS = 1500

import warnings
warnings.filterwarnings(
    "ignore",
    message="Data Validation extension is not supported and will be removed",
    category=UserWarning,
)


def _is_loss_run_filename(filename: str) -> bool:
    """
    Heuristic detection based only on the filename (no path, no extension).
    
    Rules:
    - Case-insensitive.
    - 'lr' / 'lrs' etc. must be whole tokens, not substrings
      inside other words, e.g. "color" should NOT match.
    """
    name_no_ext = os.path.splitext(filename)[0].lower()
    
    # Normalize common separators to spaces
    normalized = re.sub(r"[_\-\.\+]+", " ", name_no_ext)
    
    # 1) Simple substring match for the multi-word or concatenated forms
    phrase_like_keywords = [
        "loss run",
        "loss runs",
        "lossrun",
        "lossruns",
        "lrun",
        "loss summary",
        "loss summaries",
    ]
    if any(kw in normalized for kw in phrase_like_keywords):
        return True
    
    # 2) Token-based match for short forms that must be whole tokens
    tokens = re.split(r"[^a-z0-9]+", normalized)
    tokens = [t for t in tokens if t]
    
    token_keywords = {"lr", "lrs"}
    if any(t in token_keywords for t in tokens):
        return True
    
    return False


def _score_text_by_keywords(text: str) -> int:
    """
    Very simple scoring: for each keyword, if present in the lowercased text,
    add 1 to the score. You can later add weights if you want.
    """
    keywords = [
        # Core loss-run document identifiers
        "loss run",
        "loss runs",
        "loss history",
        "loss summary",
        "loss schedule",
        "schedule of losses",
        "loss information",
        "loss experience",
        "loss development",
        "loss detail",
        "loss detail report",
        "loss report",
        "losses valued as of",
        "loss run valued as of",
        "valuation date",
        "valuation as of",
        "valued as of",
        "claims experience",
        "claims summary",
        "summary of losses",
        "loss summary by year",
        "loss summary by policy year",
        "loss summary by location",
        "summary of claims",
        "claims by year",
        "claims by policy year",
        "no claims",
        "no losses",
        
        # Claim table / field labels
        "claim number",
        "claim no",
        "claim #",
        "claim id",
        "claim reference",
        "claim ref",
        "date of loss",
        "loss date",
        "reported date",
        "report date",
        "date reported",
        "status",
        "open date",
        "close date",
        "closed date",
        "report date",
        "claimant",
        "claimant name",
        "description of loss",
        "loss description",
        "claim description",
        "accident description",
        "injury description",
        "location of loss",
        "occurrence id",
        "policy number",
        "policy no",
        "policy #",
        
        # Loss / financial metric labels typical of loss runs
        "total incurred",
        "net incurred",
        "incurred loss",
        "gross incurred",
        "incurred amount",
        "loss incurred",
        "total paid",
        "paid loss",
        "loss paid",
        "paid to date",
        "paid-to-date",
        "paid amount",
        "outstanding reserve",
        "case reserve",
        "loss reserve",
        "outstanding loss",
        "remaining reserve",
        "reserve amount",
        "total claim cost",
        "unpaid loss",
        "unpaid amount",
        "incurred alae",
        "alae incurred",
        "allocated loss adjustment expense",
        "allocated loss adj expense",
        "paid alae",
        "alae paid",
        "total alae",
        "expense incurred",
        "expense paid",
        "total expense",
        "expense reserve",
        "total recoveries",
        "recoveries",
        "subrogation recoveries",
        "subrogation",
        "salvage",
        "deductible",
        "loss amount",
        "claim amount",
        "large loss",
        "major loss",
        
        # Aggregation / count indicators within loss runs
        "number of claims",
        "claim count",
        "claims count",
        "total number of claims",
        "incurred by year",
        "paid by year",
        "total incurred by year",
        "total paid by year",
    ]
    if not text:
        return 0
    text = str(text)
    text_l = text.lower()
    score = 0
    for kw in keywords:
        if kw in text_l:
            score += 1
    return score


def _get_top_ocr_pages_for_detection(page_text_map: dict, top_n: int = 5):
    """
    Given page_text_map = {"1": "page 1 text", "2": "page 2 text", ...},
    return a list of (page_number_str, text) for the top_n most likely
    loss-run pages by keyword hits.
    
    If no pages have any hits, return the first top_n pages in page-number order.
    """
    # Score each page
    page_scores = []
    for page_str, text in page_text_map.items():
        score = _score_text_by_keywords(text)
        page_scores.append((page_str, text, score))
    
    # Any hits at all?
    max_score = max((s for _, _, s in page_scores), default=0)
    if max_score > 0:
        # Sort by score desc, then by page number asc
        page_scores.sort(key=lambda x: (-x[2], int(x[0]) if x[0].isdigit() else 999999))
        selected = page_scores[:top_n]
    else:
        # No hits -> default to first N pages by page number
        page_scores.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 999999)
        selected = page_scores[:top_n]
    
    # Return list of (page_number_str, page_text)
    return [(p, t) for p, t, _ in selected]


def _truncate_chunks_for_detection(chunks: list, max_words_per_sheet: int = 500) -> list:
    """
    Take every chunk of excel/csv and truncates its content to the first
    'max_words_per_sheet' words. No keyword scoring / ranking.
    
    Important: we preserve headers (column names) if present so keyword
    detection can still fire even when a sheet has only headers and no rows.
    """
    if not chunks:
        return []
    
    truncated_chunks = []
    for ch in chunks:
        # Capture headers if provided, or infer from dict-of-rows content
        headers: List[str] = []
        if isinstance(ch.get("headers"), list):
            headers = ch.get("headers") or []
        else:
            content_obj = ch.get("content")
            if isinstance(content_obj, dict):
                # If this is a dict of rows -> dict of columns
                row_dicts = [v for v in content_obj.values() if isinstance(v, dict)]
                header_set = set()
                for rd in row_dicts:
                    header_set.update(str(k) for k in rd.keys())
                headers = sorted(header_set)
        
        # Build truncated content string
        content = (str(ch.get("content")) or "").strip()
        words = content.split()
        if len(words) > max_words_per_sheet:
            words = words[:max_words_per_sheet]
        truncated_content = " ".join(words)
        
        ch_copy = dict(ch)
        ch_copy["content"] = truncated_content
        if headers:
            ch_copy["headers"] = headers
        truncated_chunks.append(ch_copy)
    
    return truncated_chunks


def _estimate_tokens(text: str, model_name: str | None = None) -> int:
    if not text:
        return 0
    try:
        encoding = tiktoken.encoding_for_model(model_name or "gpt-4.0")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _truncate_text_to_tokens(text: str, max_tokens: int, model_name: str | None = None) -> str:
    if not text:
        return ""
    try:
        encoding = tiktoken.encoding_for_model(model_name or "gpt-4.0")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def _prepare_detection_payload(
    input_payload,
    max_prompt_tokens: int = MAX_DETECTION_PROMPT_TOKENS,
    max_tokens_per_item: int = MAX_DETECTION_ITEM_TOKENS,
) -> str:
    def truncate_payload(payload, per_item_tokens: int):
        if isinstance(payload, dict):
            trimmed = {}
            for k, v in payload.items():
                if isinstance(v, str):
                    trimmed[k] = _truncate_text_to_tokens(v, per_item_tokens, LOSS_RUN_DETECTION_MODEL["model_name"])
                else:
                    trimmed[k] = v
            return trimmed
        if isinstance(payload, list):
            trimmed_list = []
            for item in payload:
                if isinstance(item, dict):
                    item_copy = dict(item)
                    if isinstance(item_copy.get("content"), str):
                        item_copy["content"] = _truncate_text_to_tokens(
                            item_copy["content"],
                            per_item_tokens,
                            LOSS_RUN_DETECTION_MODEL["model_name"],
                        )
                    trimmed_list.append(item_copy)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    page = item[0]
                    text = item[1]
                    if isinstance(text, str):
                        text = _truncate_text_to_tokens(text, per_item_tokens, LOSS_RUN_DETECTION_MODEL["model_name"])
                    if isinstance(item, tuple):
                        trimmed_list.append((page, text))
                    else:
                        trimmed_list.append([page, text])
                else:
                    trimmed_list.append(item)
            return trimmed_list
        return payload
    
    def prompt_tokens_for(payload):
        input_json = json.dumps(payload, ensure_ascii=False)
        prompt = LOSS_RUN_DETECTION_PROMPT.format(input=input_json)
        return _estimate_tokens(prompt, LOSS_RUN_DETECTION_MODEL["model_name"]), input_json
    
    payload = truncate_payload(input_payload, max_tokens_per_item)
    tokens, input_json_str = prompt_tokens_for(payload)
    
    if tokens <= max_prompt_tokens:
        return payload, input_json_str
    
    # Reduce list size if needed
    if isinstance(payload, list):
        trimmed = payload
        while len(trimmed) > 1 and tokens > max_prompt_tokens:
            trimmed = trimmed[:-1]
            tokens, input_json_str = prompt_tokens_for(trimmed)
        payload = trimmed
    
    # Further reduce per-item tokens if still too large
    per_item = max_tokens_per_item
    while tokens > max_prompt_tokens and per_item > 200:
        per_item = int(per_item * 0.7)
        payload = truncate_payload(payload, per_item)
        tokens, input_json_str = prompt_tokens_for(payload)
    
    if tokens > max_prompt_tokens:
        if isinstance(payload, list) and payload:
            payload = truncate_payload([payload[0]], 200)
            tokens, input_json_str = prompt_tokens_for(payload)
    
    if tokens > max_prompt_tokens:
        print("        ⚠ Detection payload still exceeds token budget after truncation.")
    else:
        print("        ⚠ Detection payload truncated to fit token budget.")
    
    return payload, input_json_str


_detection_token_usage = {}   # key: (company, filename) -> {"input": int, "output": int}
_detection_time_usage = {}    # key: (company, filename) -> float seconds


def _log_detection_decision(company: str, filename: str, info: dict):
    """
    Print per-file detection decision and LLM cost in a readable format.
    """
    is_loss_run = bool(info.get("is_loss_run"))
    elapsed = _detection_time_usage.get((company, filename), 0.0)
    
    emoji = "✅" if is_loss_run else "❌"
    print(f"        Company: {company}")
    print(f"        {emoji} Loss Run Detection: {filename}")
    print(f"        Filtering Time: {elapsed:.2f} seconds")
    usage = _detection_token_usage.get((company, filename), {"input": 0, "output": 0})
    
    if usage["input"] or usage["output"]:
        cost = models.compute_cost(
            usage["input"],
            usage["output"],
            LOSS_RUN_DETECTION_MODEL,
        )
    else:
        cost = 0.0
    print(f"                 💰 Filtering Cost: ${cost:.6f}")
    
    record_stage(
        company,
        filename,
        stage="loss_run_detection",
        time_seconds=elapsed,
        input_tokens=usage["input"],
        output_tokens=usage["output"],
        cost=cost,
    )


def _call_llm_detection(input_payload: dict, max_retries: int = 5, account_name: str = None, file_name: str = None) -> bool:
    """
    Calls the loss-run detection LLM with retry + strict JSON parsing.
    
    input_payload: dict or list that will be JSON-serialized and inserted into LOSS_RUN_DETECTION_PROMPT
                    via the {input} placeholder.
    Returns:
        bool: True if LLM decides it is a loss run, else False.
    """
    # The detection prompt expects the Input section to be JSON text,
    # which is substituted into {input}.
    _, input_json_str = _prepare_detection_payload(input_payload)
    
    for attempt in range(1, max_retries + 1):
        try:
            prompt = LOSS_RUN_DETECTION_PROMPT.format(input=input_json_str)
            
            # gpt_call is assumed to return the raw model text response
            raw_output, input_tokens, output_tokens = gpt_call(
                prompt,
                LOSS_RUN_DETECTION_MODEL["model_name"],
                LOSS_RUN_DETECTION_MODEL["api_version"],
            )
            
            if raw_output is None:
                raise ValueError("LLM output was empty or invalid JSON")
            
            # --- track token usage for this file (for cost reporting) ---
            if account_name is not None and file_name is not None:
                key = (account_name, file_name)
                usage = _detection_token_usage.setdefault(key, {"input": 0, "output": 0})
                usage["input"] += input_tokens or 0
                usage["output"] += output_tokens or 0
            
            # Response must be a single JSON object per the prompt spec
            parsed = json.loads(raw_output)
            
            if not isinstance(parsed, dict):
                raise ValueError("LLM response is not a JSON object")
            
            if "is_loss_run" not in parsed or "reason" not in parsed:
                raise ValueError("LLM response missing required keys")
            
            parsed["is_loss_run"] = bool(parsed["is_loss_run"])
            
            return parsed["is_loss_run"]
            
        except Exception as e:
            print(f"        ⚠ LLM detection attempt {attempt} failed: {e}")
            if attempt == max_retries:
                return False


def detect_loss_runs(output_data_dir, company_folders):
    """
    Store the results
    """
    loss_run_detection_dict = {}
    
    # Loop through each company
    for company_folder in company_folders:
        company_path = os.path.join(output_data_dir, company_folder)
        if not os.path.isdir(company_path):
            continue
        
        # Walk through the subfolders inside the company folder
        # Subfolder structure:
        # Can be PDF name (folder includes the .pdf extension): inside will be the images of each page & the page_response.json
        # Can be "csvs": inside will be a copy of the input csv files
        # Can be "excel_sheets": inside will be a copy of the input Excel files
        for subfolder in os.listdir(company_path):
            subfolder_path = os.path.join(company_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            
            # --- CSV / Excel processing using utils_excel_csv_to_json ---
            if subfolder in ["csvs", "excel_sheets"]:
                for root, dirs, files in os.walk(subfolder_path):
                    # Loop through each csv and excel file
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_lower = file.lower()
                        is_csv = file_lower.endswith(".csv")
                        is_excel = file_lower.endswith((".xls", ".xlsx", ".xlsm", ".xlsb"))
                        
                        if not (is_csv or is_excel):
                            continue
                        
                        # Key for this file
                        dict_key = (company_folder, file)
                        
                        # File-name based check
                        file_name_check = _is_loss_run_filename(file)
                        
                        if file_name_check:
                            loss_run_detection_dict[dict_key] = {
                                "is_loss_run": bool(file_name_check),
                                "method": "Filename"
                            }
                            _log_detection_decision(company_folder, file, loss_run_detection_dict[dict_key])
                            continue
                        
                        # If file name doesn't hit, use the text from the excel/csv
                        try:
                            # Force sheet-level chunking to avoid empty block-level results
                            chunks = get_excel_csv_chunks(file_path, hybrid_approach=False)
                        except Exception as e:
                            print(f"        ❌ Failed to build chunks for {file_path}: {e}")
                            loss_run_detection_dict[dict_key] = {
                                "is_loss_run": False,
                                "method": None
                            }
                            _log_detection_decision(company_folder, file, loss_run_detection_dict[dict_key])
                            continue
                        
                        # Fallback: if no chunks produced, try any precomputed *_chunks.json alongside
                        if isinstance(chunks, list) and len(chunks) == 0:
                            alt_jsons = [
                                os.path.join(os.path.dirname(file_path), f"{os.path.splitext(file)[0]}_chunks.json"),
                                os.path.join(os.path.dirname(os.path.dirname(file_path)), f"{os.path.splitext(file)[0]}_chunks.json"),
                            ]
                            for alt_json in alt_jsons:
                                if os.path.isfile(alt_json):
                                    try:
                                        with open(alt_json, "r", encoding="utf-8") as f:
                                            chunks_from_json = json.load(f)
                                        if isinstance(chunks_from_json, list) and chunks_from_json:
                                            chunks = chunks_from_json
                                            break
                                    except Exception:
                                        pass
                        
                        # Use all sheets (1 for csv), but truncate it
                        top_chunks = _truncate_chunks_for_detection(chunks)
                        # --- Always check headers and free text for keywords ---
                        keyword_score = 0
                        for ch in top_chunks:
                            content = ch.get("content", "")
                            # Try to extract headers if present
                            headers = []
                            if isinstance(ch.get("headers"), list):
                                headers = ch["headers"]
                            elif isinstance(ch.get("content"), dict):
                                headers = list(ch["content"].keys())
                            # Compose a string with headers and content
                            header_str = " ".join(str(h) for h in headers)
                            combined_text = f"{header_str} {content}".strip()
                            score = _score_text_by_keywords(combined_text)
                            keyword_score += score
                        if keyword_score > 0:
                            loss_run_detection_dict[dict_key] = {
                                "is_loss_run": True,
                                "method": "Keyword"
                            }
                            _log_detection_decision(company_folder, file, loss_run_detection_dict[dict_key])
                            continue

                        # Pass the top chunks into LLM call for True/False of if loss run detected
                        start_time = time.time()
                        llm_check = _call_llm_detection(
                            top_chunks,
                            account_name=company_folder,
                            file_name=file,
                        )
                        _detection_time_usage[(company_folder, file)] = time.time() - start_time
                        if llm_check:
                            loss_run_detection_dict[dict_key] = {
                                "is_loss_run": bool(llm_check),
                                "method": "GPT"
                            }
                            _log_detection_decision(company_folder, file, loss_run_detection_dict[dict_key])
                            continue
                        
                        # If neither checks are passed -> deem as not a loss run
                        loss_run_detection_dict[dict_key] = {
                            "is_loss_run": False,
                            "method": None
                        }
                        _log_detection_decision(company_folder, file, loss_run_detection_dict[dict_key])
            
            # --- PDF (& DOCX) Processing ---
            # Each DOCX file should have been converted to a PDF, so it will be handled the same way
            else:
                if subfolder.lower().endswith(".pdf"):
                    # Key for this file
                    dict_key = (company_folder, subfolder)
                    
                    # File-name based check
                    file_name_check = _is_loss_run_filename(subfolder)
                    
                    if file_name_check:
                        loss_run_detection_dict[dict_key] = {
                            "is_loss_run": bool(file_name_check),
                            "method": "Filename"
                        }
                        _log_detection_decision(company_folder, subfolder, loss_run_detection_dict[dict_key])
                        continue
                    
                    # Get the OCR text for the PDF
                    # Single JSON per PDF with page_number -> text
                    json_path = os.path.join(subfolder_path, "page_response.json")
                    if not os.path.isfile(json_path):
                        print(f"        ❌ page_response.json not found in {subfolder_path}")
                        loss_run_detection_dict[dict_key] = {
                            "is_loss_run": False,
                            "method": None
                        }
                        _log_detection_decision(company_folder, subfolder, loss_run_detection_dict[dict_key])
                        continue
                    
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            page_text_map = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"        ❌ Failed to load JSON in {json_path}: {e}")
                        loss_run_detection_dict[dict_key] = {
                            "is_loss_run": False,
                            "method": None
                        }
                        _log_detection_decision(company_folder, subfolder, loss_run_detection_dict[dict_key])
                        continue
                    
                    # Select top pages based on loss run related keyword hits
                    # Reason we use the scoring is because in a PDF it is much more likely for a
                    # loss run to be hidden in a long PDF or all submission docs to be in 1 long PDF,
                    # so using just the first 5 pages would hurt recall
                    top_pages = _get_top_ocr_pages_for_detection(page_text_map, top_n=5)
                    
                    # Pass the top pages into LLM call for True/False of if loss run detected
                    start_time = time.time()
                    llm_check = _call_llm_detection(
                        top_pages,
                        account_name=company_folder,
                        file_name=subfolder,
                    )
                    _detection_time_usage[(company_folder, subfolder)] = time.time() - start_time
                    if llm_check:
                        loss_run_detection_dict[dict_key] = {
                            "is_loss_run": bool(llm_check),
                            "method": "GPT"
                        }
                        _log_detection_decision(company_folder, subfolder, loss_run_detection_dict[dict_key])
                        continue
                    
                    # If no checks are passed -> deem as not a loss run
                    loss_run_detection_dict[dict_key] = {
                        "is_loss_run": False,
                        "method": None
                    }
                    _log_detection_decision(company_folder, subfolder, loss_run_detection_dict[dict_key])
            
    return loss_run_detection_dict

if __name__ == "__main__":
    
    GROUND_TRUTH = {
        ('AAA_DOCX_TEST', 'TEST_LOSS.pdf'): {'is_loss_run': True, 'method': 'groundtruth'},
        ('Acosta Tractors', 'LRUN 2020-2021 IM Liberty val 051525.pdf'): {'is_loss_run': True, 'method': 'groundtruth'},
        ('Acosta Tractors', 'LRUN 2020-2023 Auto AL Liberty val 05152...'): {'is_loss_run': True, 'method': 'groundtruth'},
    }
    
    output_data_dir = "output_data"
    
    # Build list of company folders from immediate subdirectories of output_data
    # company_folders = [
    #     name
    #     for name in os.listdir(output_data_dir)
    #     if os.path.isdir(os.path.join(output_data_dir, name))
    # ]
    # TEMP: only run on a single company subfolder
    company_folders = ["AAA_TEST_NO_LOSS_RUNS"]
    
    # Run the detection
    loss_run_detection_dict = detect_loss_runs(output_data_dir, company_folders)
    
    # Summarize counts of True / False
    true_count = sum(
        1 for v in loss_run_detection_dict.values()
        if v.get("is_loss_run") is True
    )
    false_count = sum(
        1 for v in loss_run_detection_dict.values()
        if v.get("is_loss_run") is False
    )
    
    print(f"Total entries: {len(loss_run_detection_dict)}")
    print(f"is_loss_run=True: {true_count}")
    print(f"is_loss_run=False: {false_count}")
    
    # --- Count detected loss runs by method ---
    method_counts = {}
    for info in loss_run_detection_dict.values():
        if info.get("is_loss_run") is True:
            method = info.get("method") or "unknown"
            method_counts[method] = method_counts.get(method, 0) + 1
            
    print("\nDetected loss runs by method:")
    for method, count in sorted(method_counts.items()):
        print(f"  {method}: {count}")
        
    # --- Evaluation vs GROUND_TRUTH ---
    # Ground truth rule:
    # - If key is in GROUND_TRUTH -> actual is_loss_run = True
    # - If key is NOT in GROUND_TRUTH -> actual is_loss_run = False
    def gt_label(key):
        return key in GROUND_TRUTH
        
    tp = fp = tn = fn = 0
    
    for key, pred_info in loss_run_detection_dict.items():
        pred = bool(pred_info.get("is_loss_run"))
        actual = gt_label(key)
        
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and not actual:
            tn += 1
        elif not pred and actual:
            fn += 1
            
    # Extra recall checks: ensure every ground-truth file is predicted and True
    missing_gt = []
    false_gt = []
    
    for gt_key in GROUND_TRUTH.keys():
        if gt_key not in loss_run_detection_dict:
            missing_gt.append(gt_key)
        else:
            if not bool(loss_run_detection_dict[gt_key].get("is_loss_run")):
                false_gt.append(gt_key)
                
    # For a strict recall definition on ground-truth positives only:
    strict_tp = len(GROUND_TRUTH) - len(missing_gt) - len(false_gt)
    strict_fn = len(missing_gt) + len(false_gt)
    strict_recall = strict_tp / (strict_tp + strict_fn) if (strict_tp + strict_fn) else 0.0
    
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    
    print("\nEvaluation vs GROUND_TRUTH:")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {strict_recall:.3f}")
    
    # --- Detailed mistakes printout (FP and FN) ---
    print("\nMistakes (pred != actual):")
    for key, pred_info in loss_run_detection_dict.items():
        pred = bool(pred_info.get("is_loss_run"))
        actual = gt_label(key) # True if in GROUND_TRUTH, else False
        if pred != actual:
            company, filename = key
            print(f"Company: {company} | File: {filename} | Pred: {pred} | Actual: {actual}")

