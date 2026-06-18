import os
import re
import json
import difflib
import pandas as pd
import models
from utils_gpt import gpt_call, _coerce_json_text

def normalize_name(name: str) -> str:
    """
    Cleans name for comparison: lowercases, removes punctuation,
    and strips extra spaces. Used in fallback logic.
    """
    if pd.isna(name) or not name:
        return ""
    cleaned = str(name).lower()
    cleaned = re.sub(r'[^a-z0-9\s]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def names_match(n1: str, n2: str) -> bool:
    """
    Checks if two normalized names are highly similar or substring matches.
    """
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    if n1 in n2 or n2 in n1:
        return True
    similarity = difflib.SequenceMatcher(None, n1, n2).ratio()
    return similarity > 0.85

def programmatic_fallback_roll_numbers(df: pd.DataFrame) -> list:
    """
    Programmatic fallback grouping: Groups by name & date (showing matched formula),
    or defaults.
    """
    print("[WARN] Using programmatic fallback for claims roll-up.")
    groups = []  # list of dicts: {'accident_date': ..., 'accident_state': ..., 'names': set(), 'roll_no': ...}
    roll_numbers = []

    for idx, row in df.iterrows():
        raw_date = row.get("accident_date") if "accident_date" in df.columns else None
        raw_state = row.get("accident_state") if "accident_state" in df.columns else None
        raw_driver = row.get("driver") if "driver" in df.columns else None
        raw_claimant = row.get("claimant") if "claimant" in df.columns else None

        acc_date = str(raw_date).strip() if pd.notna(raw_date) else None
        acc_state = str(raw_state).strip().upper() if pd.notna(raw_state) else None
        
        driver_val = str(raw_driver).strip() if pd.notna(raw_driver) else None
        claimant_val = str(raw_claimant).strip() if pd.notna(raw_claimant) else None

        norm_driver = normalize_name(driver_val) if driver_val else ""
        norm_claimant = normalize_name(claimant_val) if claimant_val else ""

        matched_group = None
        for g in groups:
            date_match = (g['accident_date'] == acc_date)
            state_match = (g['accident_state'] == acc_state)
            
            name_match = False
            if g['names']:
                for existing_name in g['names']:
                    if norm_driver and names_match(norm_driver, existing_name):
                        name_match = True
                        break
                    if norm_claimant and names_match(norm_claimant, existing_name):
                        name_match = True
                        break
            else:
                if not norm_driver and not norm_claimant:
                    name_match = True

            if date_match and state_match and name_match:
                matched_group = g
                break

        if matched_group is not None:
            if norm_driver:
                matched_group['names'].add(norm_driver)
            if norm_claimant:
                matched_group['names'].add(norm_claimant)
            roll_no = matched_group['roll_no']
        else:
            names_set = set()
            if norm_driver:
                names_set.add(norm_driver)
            if norm_claimant:
                names_set.add(norm_claimant)

            # Build formula roll_no based on name, date, state
            name_slug = "unknown_person"
            if driver_val:
                name_slug = normalize_name(driver_val).replace(" ", "_")
            elif claimant_val:
                name_slug = normalize_name(claimant_val).replace(" ", "_")
                
            date_slug = acc_date if acc_date else "unknown_date"
            state_slug = acc_state if acc_state else "unknown_state"
            
            roll_no = f"{name_slug}&{date_slug}&{state_slug}"
            groups.append({
                'accident_date': acc_date,
                'accident_state': acc_state,
                'names': names_set,
                'roll_no': roll_no
            })

        roll_numbers.append(roll_no)

    return roll_numbers

def get_llm_roll_numbers(df: pd.DataFrame) -> list:
    """
    Sends claims to GPT-5.2 to group them and generate unique formula-based roll numbers.
    """
    records_to_send = []
    for idx, row in df.iterrows():
        record = {"row_index": idx}
        for col in df.columns:
            val = row[col]
            if pd.notna(val):
                record[col] = str(val)
        records_to_send.append(record)

    json_input_str = json.dumps(records_to_send, indent=2)

    prompt = f"""You are an insurance data assistant. You are given a list of extracted claims.
Your task is to identify which claims belong to the same customer/driver and represent the same accident/event, and group them together by assigning them the same unique `roll_no` identifier.

CRITICAL RULES:
1. Do NOT assume any fixed set of identifying fields. Instead, dynamically determine which attributes uniquely identify a group of claims representing the same customer/incident. This can include:
   - Claimant or driver names (allow for spelling variations, typos, middle initials, name order)
   - Dates (accident date)
   - States (accident state, garage state, coverage state)
   - Policy number suffixes (or entire policy numbers if they align)
   - Specific details from descriptions or line of business
2. If two or more rows represent the same customer/incident, they MUST be assigned the EXACT same `roll_no`. If a row represents a unique incident, assign it a unique `roll_no`.
3. Construct the `roll_no` by combining the matched values/attributes to show the matching formula. Do NOT use generic IDs like ROLL_001. Instead, create a string showing the formula of fields you matched on.
   - Example 1 (matched on name, date, and state): "suman&2024-03-12&il"
   - Example 2 (matched on name and policy suffix): "john_smith&pol_1234"
   - Example 3 (matched on claimant and description keyword): "jane_doe&house_fire"
   - Make the roll_no strings completely lowercased, replacing spaces with underscores.
4. You must map EVERY single row index in the input.

INPUT DATA (JSON list of claims with their 'row_index'):
{json_input_str}

Your response must be ONLY a valid JSON dictionary mapping the string representation of each 'row_index' to its assigned 'roll_no'. Do not include any explanation or markdown formatting other than JSON.
Example format:
{{
  "0": "suman&2024-03-12&il",
  "1": "suman&2024-03-12&il",
  "2": "john_smith&pol_1234"
}}
"""

    model_name = models.GPT_5_2["model_name"]
    api_version = models.GPT_5_2["api_version"]

    print(f"Calling LLM ({model_name}) to group claims using dynamic matching formulas...")
    try:
        content, in_tokens, out_tokens = gpt_call(
            prompt=prompt,
            model_name=model_name,
            api_version=api_version,
            temperature=0
        )
        
        if not content:
            print("[ERROR] LLM returned empty response.")
            return programmatic_fallback_roll_numbers(df)
            
        json_str = _coerce_json_text(content)
        if not json_str:
            print("[ERROR] LLM output could not be parsed as JSON.")
            return programmatic_fallback_roll_numbers(df)
            
        mapping = json.loads(json_str)
        
        # Verify we got mapping for all row indices
        roll_numbers = []
        for idx in range(len(df)):
            roll_no = mapping.get(str(idx))
            if not roll_no:
                print(f"[WARN] Row index {idx} was not mapped by LLM. Falling back.")
                return programmatic_fallback_roll_numbers(df)
            roll_numbers.append(str(roll_no))
            
        return roll_numbers
        
    except Exception as e:
        print(f"[ERROR] Exception during LLM roll-up matching: {e}")
        return programmatic_fallback_roll_numbers(df)

def aggregate_claims(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups by roll_no and aggregates all claims.
    """
    numeric_cols = [
        "total_incurred",
        "total_paid",
        "incurred_alae",
        "paid_alae",
        "total_recoveries"
    ]
    
    agg_funcs = {}
    for col in df.columns:
        if col == 'roll_no':
            continue
        
        if col in numeric_cols:
            agg_funcs[col] = 'sum'
        else:
            def _join_unique(series, col_name=col):
                vals = []
                for val in series:
                    if pd.notna(val):
                        s = str(val).strip()
                        if s != "" and s.lower() != "nan":
                            vals.append(s)
                
                unique_vals = []
                for v in vals:
                    if v not in unique_vals:
                        unique_vals.append(v)
                
                if not unique_vals:
                    return ""
                if len(unique_vals) == 1:
                    return unique_vals[0]
                return "?".join(unique_vals)
                
            agg_funcs[col] = _join_unique

    aggregated = df.groupby('roll_no', as_index=False).agg(agg_funcs)
    
    # Put roll_no first
    cols = ['roll_no'] + [col for col in aggregated.columns if col != 'roll_no']
    aggregated = aggregated[cols]
    
    return aggregated

def run_rollup_post_processing(input_path: str, output_detailed_path: str, output_rollup_path: str):
    """
    Loads, processes, and writes the two output CSV files.
    """
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file {input_path} not found.")
        return

    print(f"Reading claims from: {input_path}")
    df = pd.read_csv(input_path)

    if df.empty:
        print("[INFO] Claims file is empty. Skipping rollup.")
        df.insert(0, 'roll_no', [])
        df.to_csv(output_detailed_path, index=False)
        df.to_csv(output_rollup_path, index=False)
        return

    print("Running LLM rollup logic...")
    roll_numbers = get_llm_roll_numbers(df)
    
    detailed_df = df.copy()
    detailed_df.insert(0, 'roll_no', roll_numbers)
    
    rollup_df = aggregate_claims(detailed_df)

    # Write output files
    print(f"Saving detailed claims with roll_no to: {output_detailed_path}")
    detailed_df.to_csv(output_detailed_path, index=False)

    print(f"Saving rolled up claims to: {output_rollup_path}")
    rollup_df.to_csv(output_rollup_path, index=False)
    print("Rollup step completed successfully!")

if __name__ == "__main__":
    # Standard output paths
    input_file = "extraction_output/sample_claims.csv"
    output_detailed = "extraction_output/sample_claims.csv"
    output_rollup = "extraction_output/sample_Rolled_up_claims.csv"
    
    run_rollup_post_processing(input_file, output_detailed, output_rollup)
