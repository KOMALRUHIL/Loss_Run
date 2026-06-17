import pandas as pd
import os
from typing import Dict, List, Any
import json
from prompts import DUPLICATE_DETECTION_PROMPT
import models
from utils_gpt import gpt_call

# Columns to consider when detecting duplicates across files
DUPLICATE_KEY_COLUMNS: List[str] = [
    "claim_id",
    "loss_date",
    "claim_description",
]

DUPLICATE_DETECTION_MODEL = models.GPT_5_2

def _candidate_columns(df: pd.DataFrame) -> List[str]:
    """
    Columns that can be used to detect potential duplicates.
    
    RULES:
    - Start from a fixed list (DUPLICATE_KEY_COLUMNS).
    - Keep only those that are actually present in the DataFrame.
    - Drop columns that are all-null or have only a single unique non-null value.
    - We DO NOT require matches to be across different file_path values anymore.
    """
    cols: List[str] = []

    for col in DUPLICATE_KEY_COLUMNS:
        if col not in df.columns:
            continue
        
        s = df[col]
        
        # Skip all-null columns
        if s.dropna().empty:
            continue
        
        # Skip columns where everything (non-null) is the same
        if s.nunique(dropna=True) <= 1:
            continue
        
        cols.append(col)
    
    return cols


def build_potential_duplicates_table_any_column(
    claims_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a DataFrame containing all rows that are potential duplicates,
    based on having the same value in ANY selected duplicate key column
    (see DUPLICATE_KEY_COLUMNS), and coming from different file_path values.
    
    Adds a stable '_row_id' column so we can refer back to the original rows.
    Also adds '_dup_group_id' so rows that must be processed together by the LLM
    are kept in the same batch.
    """
    df = claims_df.copy()
    if "_row_id" not in df.columns:
        df["_row_id"] = range(len(df))
    
    candidate_cols = _candidate_columns(df)
    if not candidate_cols:
        return pd.DataFrame(columns=df.columns)
    
    # Optional: normalize candidate columns for comparison
    for col in candidate_cols:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
            )
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Coerce to numeric where possible, keep NaN for invalid
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if "file_path" not in df.columns:
        # Can't enforce cross-file duplicates without file_path
        return pd.DataFrame(columns=df.columns)
    
    duplicate_row_ids: set[int] = set()
    dup_groups: list[pd.Index] = []
    
    # For each candidate column independently:
    for col in candidate_cols:
        # work only with rows where this column is non-null
        col_sub = df.dropna(subset=[col])
        if col_sub.empty:
            continue

        # Group by this single column
        for _, group in col_sub.groupby(col, dropna=False, sort=False):
            distinct_paths = group["file_path"].dropna().nunique()
            if distinct_paths < 2:
                # All rows for this value come from the same file; skip
                continue
            
            # Same value in this column, from at least 2 different files
            duplicate_row_ids.update(group["_row_id"].tolist())
            # Remember this whole group as something the LLM should see together
            dup_groups.append(group.index)
    
    if not duplicate_row_ids:
        return pd.DataFrame(columns=df.columns)
    
    dup_df = df[df["_row_id"].isin(duplicate_row_ids)].copy()

    # Build a stable _dup_group_id across all groupings.
    # A row can appear in multiple groups; we assign it the smallest group id.
    dup_df["_dup_group_id"] = -1
    next_group_id = 0
    for idx in dup_groups:
        # Only consider rows that actually ended up in dup_df
        idx_in_dup = dup_df.index.intersection(idx)
        if idx_in_dup.empty:
            continue
        
        # Assign group id where not yet assigned or keep the minimum
        mask_unassigned = dup_df.loc[idx_in_dup, "_dup_group_id"] == -1
        dup_df.loc[idx_in_dup[mask_unassigned], "_dup_group_id"] = next_group_id
        next_group_id += 1

    # Any row that is in dup_df but didn't get a group (defensive) becomes its own group
    ungrouped_mask = dup_df["_dup_group_id"] == -1
    if ungrouped_mask.any():
        dup_df.loc[ungrouped_mask, "_dup_group_id"] = range(
            next_group_id, next_group_id + ungrouped_mask.sum()
        )
    
    sort_cols = [
        c
        for c in [
            "AccountName",
            "_dup_group_id",
            "file_path",
            "claim_id",
            "loss_date",
            "net_incurred",
        ]
        if c in dup_df.columns
    ]
    if sort_cols:
        dup_df = dup_df.sort_values(by=sort_cols + ["_dup_group_id"], kind="stable")
    
    return dup_df


def _duplicates_table_to_gpt_payload(duplicates_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert the duplicates DataFrame into a JSON-serializable payload.
    
    We keep rows as list-of-objects keyed by column.
    '_row_id' is the key that GPT should use to say what to do with each row.
    """
    records = duplicates_df.to_dict(orient="records")
    return {
        "schema": {
            "description": "Potentially duplicated claim records extracted from multiple files.",
            "primary_key": "_row_id",
        },
        "rows": records,
    }


def build_gpt_prompt_for_duplicates_table(duplicates_df: pd.DataFrame) -> str:
    """
    Build a single prompt containing the entire duplicates table,
    asking GPT to decide which rows are true duplicates that should be dropped.
    """
    payload = _duplicates_table_to_gpt_payload(duplicates_df)
    table_json = json.dumps(payload, default=str)
    return DUPLICATE_DETECTION_PROMPT.format(table_json=table_json)


def call_gpt_for_duplicates_table(duplicates_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build the prompt, send to gpt_call, and parse the JSON response.
    
    Expected response shape:
    {
        "rows_to_drop": [<int>, ...]
    }
    
    This function now handles batching: we send at most 25 rows to GPT per call, and union the results. Rows with the same '_dup_group_id' are always kept in the same batch. If a single group is larger than MAX_ROWS_PER_CALL, it is sent as one oversized batch rather than being split.
    """
    if duplicates_df.empty:
        return {"rows_to_drop": []}
    
    MAX_ROWS_PER_CALL = 25
    all_rows_to_drop: set[int] = set()
    
    df = duplicates_df.copy()

    # If no grouping column is present, fall back to simple slicing
    if "_dup_group_id" not in df.columns:
        n = len(df)
        for start in range(0, n, MAX_ROWS_PER_CALL):
            batch = df.iloc[start : start + MAX_ROWS_PER_CALL]
            prompt = build_gpt_prompt_for_duplicates_table(batch)
            raw_response, input_tokens, output_tokens = gpt_call(prompt,
                                                                 DUPLICATE_DETECTION_MODEL['model_name'],
                                                                 DUPLICATE_DETECTION_MODEL['api_version'])
            try:
                result = json.loads(raw_response)
                rows = result.get("rows_to_drop", [])
                if not isinstance(rows, list):
                    continue
                for r in rows:
                    try:
                        all_rows_to_drop.add(int(r))
                    except Exception:
                        continue
            except Exception:
                continue
        return {"rows_to_drop": sorted(all_rows_to_drop)}

    # --- Group-aware batching using _dup_group_id ---
    df = df.sort_values(by=["_dup_group_id", "_row_id"], kind="stable")

    batches: List[pd.DataFrame] = []
    current_group_ids: List[int] = []
    current_batch_rows = 0

    for group_id, group_df in df.groupby("_dup_group_id", sort=False):
        group_size = len(group_df)

        if group_size > MAX_ROWS_PER_CALL:
            # Oversized group: flush any current batch, then send this group alone
            if current_group_ids:
                batch_df = df[df["_dup_group_id"].isin(current_group_ids)]
                batches.append(batch_df)
                current_group_ids = []
                current_batch_rows = 0

            batches.append(group_df)
            continue

        # Normal case: pack groups without splitting them
        if current_batch_rows > 0 and current_batch_rows + group_size > MAX_ROWS_PER_CALL:
            batch_df = df[df["_dup_group_id"].isin(current_group_ids)]
            batches.append(batch_df)
            current_group_ids = []
            current_batch_rows = 0

        current_group_ids.append(group_id)
        current_batch_rows += group_size

    # Flush last batch
    if current_group_ids:
        batch_df = df[df["_dup_group_id"].isin(current_group_ids)]
        batches.append(batch_df)

    # --- Call GPT per batch ---
    for batch in batches:
        payload_df = batch.drop(columns=["_dup_group_id"], errors="ignore")
        prompt = build_gpt_prompt_for_duplicates_table(payload_df)
        raw_response, input_tokens, output_tokens = gpt_call(prompt,
                                                             DUPLICATE_DETECTION_MODEL['model_name'],
                                                             DUPLICATE_DETECTION_MODEL['api_version'])
        try:
            result = json.loads(raw_response)
            rows = result.get("rows_to_drop", [])
            if not isinstance(rows, list):
                continue
            for r in rows:
                try:
                    all_rows_to_drop.add(int(r))
                except Exception:
                    continue
        except Exception:
            continue

    rows_to_drop_sorted = sorted(all_rows_to_drop)

    return {"rows_to_drop": rows_to_drop_sorted}


def apply_gpt_duplicate_resolution(
    claims_df: pd.DataFrame,
    gpt_result: Dict[str, Any],
) -> pd.DataFrame:
    """
    Apply GPT's duplicate resolution instructions to the original claims_df.
    
    Assumes claims_df has an '_row_id' column that matches the IDs used in gpt_result.
    
    GPT is only allowed to specify which existing rows to drop. No consolidation or
    value editing is performed here.
    """
    if "_row_id" not in claims_df.columns:
        raise ValueError("claims_df must contain '_row_id' to apply GPT resolution.")
    
    df = claims_df.copy()
    
    rows_to_drop = gpt_result.get("rows_to_drop", []) or []
    if rows_to_drop:
        df = df[~df["_row_id"].isin(rows_to_drop)]
    
    # Drop the helper column before returning
    df = df.drop(columns=["_row_id"], errors="ignore")
    
    return df


def deduplicate_claims_with_gpt_any_column(
    claims_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    High-level helper:
    1. Build potential duplicates table (any column same in different files).
    2. Call GPT once with that table.
    3. Apply GPT's decisions to the original claims_df.
    """
    # Ensure we have _row_id to track rows
    if "_row_id" not in claims_df.columns:
        claims_df = claims_df.copy()
        claims_df["_row_id"] = range(len(claims_df))
    
    duplicates_df = build_potential_duplicates_table_any_column(claims_df)
    
    if duplicates_df.empty:
        # Nothing to do
        return claims_df
    
    gpt_result = call_gpt_for_duplicates_table(duplicates_df)
    resolved_df = apply_gpt_duplicate_resolution(claims_df, gpt_result)
    
    return resolved_df


def detect_duplicate_claims_across_files(claims_df):
    """
    Useful because some submission files will repeat claims in different files - especially large losses
    The amount of information in each file about a claim typically differs so we can't easily programmatically detect these instances
    So we don't want to over-represent their claims and get a worse picture of their previous risk
    """
    # If completely empty, nothing to deduplicate
    if claims_df.empty:
        print("\n        0 duplicates detected\n")
        return claims_df
    
    deduped_parts = []
    for acct, group in claims_df.groupby("AccountName", dropna=False):
        deduped = deduplicate_claims_with_gpt_any_column(group)
        deduped_parts.append(deduped)
    deduped_df = pd.concat(deduped_parts, ignore_index=True)
    
    # Ensure helper column is not in the final output
    deduped_df = deduped_df.drop(columns=["_row_id"], errors="ignore")
    
    print(f"        {claims_df.shape[0] - deduped_df.shape[0]} duplicates detected\n")
    return deduped_df
