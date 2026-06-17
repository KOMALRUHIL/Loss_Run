import pandas as pd
import re
import string
import warnings
from extract import CLAIM_LEVEL_VALID_JSON_FIELDS, NO_CLAIMS_VALID_JSON_FIELDS

US_STATES_MAP = {
    # basic mapping; extend as needed
    "AL": "AL", "ALABAMA": "AL",
    "AK": "AK", "ALASKA": "AK",
    "AZ": "AZ", "ARIZONA": "AZ",
    "AR": "AR", "ARKANSAS": "AR",
    "CA": "CA", "CALIFORNIA": "CA",
    "CO": "CO", "COLORADO": "CO",
    "CT": "CT", "CONNECTICUT": "CT",
    "DE": "DE", "DELAWARE": "DE",
    "FL": "FL", "FLORIDA": "FL",
    "GA": "GA", "GEORGIA": "GA",
    "HI": "HI", "HAWAII": "HI",
    "ID": "ID", "IDAHO": "ID",
    "IL": "IL", "ILLINOIS": "IL",
    "IN": "IN", "INDIANA": "IN",
    "IA": "IA", "IOWA": "IA",
    "KS": "KS", "KANSAS": "KS",
    "KY": "KY", "KENTUCKY": "KY",
    "LA": "LA", "LOUISIANA": "LA",
    "ME": "ME", "MAINE": "ME",
    "MD": "MD", "MARYLAND": "MD",
    "MA": "MA", "MASSACHUSETTS": "MA",
    "MI": "MI", "MICHIGAN": "MI",
    "MN": "MN", "MINNESOTA": "MN",
    "MS": "MS", "MISSISSIPPI": "MS",
    "MO": "MO", "MISSOURI": "MO",
    "MT": "MT", "MONTANA": "MT",
    "NE": "NE", "NEBRASKA": "NE",
    "NV": "NV", "NEVADA": "NV",
    "NH": "NH", "NEW HAMPSHIRE": "NH",
    "NJ": "NJ", "NEW JERSEY": "NJ",
    "NM": "NM", "NEW MEXICO": "NM",
    "NY": "NY", "NEW YORK": "NY",
    "NC": "NC", "NORTH CAROLINA": "NC",
    "ND": "ND", "NORTH DAKOTA": "ND",
    "OH": "OH", "OHIO": "OH",
    "OK": "OK", "OKLAHOMA": "OK",
    "OR": "OR", "OREGON": "OR",
    "PA": "PA", "PENNSYLVANIA": "PA",
    "RI": "RI", "RHODE ISLAND": "RI",
    "SC": "SC", "SOUTH CAROLINA": "SC",
    "SD": "SD", "SOUTH DAKOTA": "SD",
    "TN": "TN", "TENNESSEE": "IN",
    "TX": "TX", "TEXAS": "TX",
    "UT": "UT", "UTAH": "UT",
    "VT": "VT", "VERMONT": "VT",
    "VA": "VA", "VIRGINIA": "VA",
    "WA": "WA", "WASHINGTON": "WA",
    "WV": "WV", "WEST VIRGINIA": "WV",
    "WI": "WI", "WISCONSIN": "WI",
    "WY": "WY", "WYOMING": "WY"
}

def _clean_date_str(s: pd.Series) -> pd.Series:
    """Convert anything parseable to MM/DD/YYYY as string; else NaN."""
    def _parse(v):
        if pd.isna(v):
            return pd.NA
        try:
            dt = pd.to_datetime(str(v), errors="raise")
            return dt.strftime("%m/%d/%Y")
        except Exception:
            return pd.NA
    return s.apply(_parse)

def _clean_lob(s: pd.Series) -> pd.Series:
    allowed = {
        "auto": "auto",
        "property": "property",
        "general liability": "general liability",
        "workers compensation": "workers compensation",
        "unknown": "unknown",
    }
    def _norm(v):
        if pd.isna(v):
            return pd.NA
        txt = str(v).strip().lower()
        return allowed.get(txt, pd.NA)
    return s.apply(_norm)

def _to_str_strip(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()

def _clean_id_str(s: pd.Series) -> pd.Series:
    """Policy/claim numbers as string; drop trailing .0 from floats."""
    def _conv(v):
        if pd.isna(v):
            return pd.NA
        txt = str(v).strip()
        # handle float-like "1234.0"
        if re.fullmatch(r"^\d+\.0", txt):
            return txt[:-2]
        return txt
    return s.apply(_conv).astype("string")

def _to_float_with_default(s: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    out = out.fillna(default)
    return out.astype(float)

def _clean_status(s: pd.Series) -> pd.Series:
    mapping = {"o": "O", "open": "O",
               "c": "C", "closed": "C",
               "r": "R", "reopen": "R", "re-open": "R", "reopened": "R"}
    def _norm(v):
        if pd.isna(v):
            return pd.NA
        txt = str(v).strip().lower()
        return mapping.get(txt, pd.NA)
    return s.apply(_norm)

def _clean_state_col(s: pd.Series) -> pd.Series:
    def _norm(v):
        if pd.isna(v):
            return pd.NA
        txt = str(v).strip().upper()
        txt = re.sub(r"[^A-Z ]", "", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return US_STATES_MAP.get(txt, pd.NA)
    return s.apply(_norm)

def _count_changes(before: pd.Series, after: pd.Series) -> int:
    """Count number of rows where value actually changed (ignoring NaN==NaN)."""
    b = before.reset_index(drop=True)
    a = after.reset_index(drop=True)
    # equal if both NaN
    both_nan = b.isna() & a.isna()
    # equal if exactly same (and not both NaN already counted)
    same = (b == a) | both_nan
    return (~same).sum()

def _print_if_changes(changed: int, msg: str) -> None:
    if changed > 0:
        print(msg)

def process_claims_df(claims_df: pd.DataFrame) -> pd.DataFrame:
    df = claims_df.copy()
    # Log all claim numbers before cleaning
    claim_numbers_before = set()
    for col in ["claim_number", "claim_id"]:
        if col in df.columns:
            claim_numbers_before.update(df[col].dropna().astype(str).str.strip())

    # --- identify rows that are "empty" other than AccountName / file_path ---
    # Only drop rows if both claim_id and claim_number are missing or empty
    def has_id(row):
        for col in ["claim_id", "claim_number"]:
            if col in row and str(row[col]).strip() not in ["", "nan", "None"]:
                return True
        return False
    mask_has_id = df.apply(has_id, axis=1)
    df_empty = df[~mask_has_id].copy()   # rows with no ID at all
    df_keep = df[mask_has_id].copy()    # keep rows with at least one ID
    df = df_keep

    # --- Dates: "MM/DD/YYYY" as string ---
    date_cols = [
        "policy_effective_date",
        "policy_expiration_date",
        "evaluation_date",
        "accident_date",
        "report_date",
    ]
    for col in date_cols:
        if col in df.columns:
            before = df[col].copy()
            df[col] = _clean_date_str(df[col]).astype("string")
            changed = _count_changes(before, df[col])
            _print_if_changes(changed, f"        Date cleaning: changed {changed} rows in '{col}'")

    # --- line_of_business normalization ---
    if "line_of_business" in df.columns:
        before = df["line_of_business"].copy()
        df["line_of_business"] = _clean_lob(df["line_of_business"]).astype("string")
        changed = _count_changes(before, df["line_of_business"])
        _print_if_changes(changed, f"        LOB cleaning: changed {changed} rows in 'line_of_business'")

    # --- prior_carrier as string ---
    if "prior_carrier" in df.columns:
        before = df["prior_carrier"].copy()
        df["prior_carrier"] = _to_str_strip(df["prior_carrier"])
        changed = _count_changes(before, df["prior_carrier"])
        _print_if_changes(changed, f"        Prior carrier cleaning: changed {changed} rows in 'prior_carrier'")

    # --- policy_number and claim_number as strings with .0 removed ---
    if "policy_number" in df.columns:
        before = df["policy_number"].copy()
        df["policy_number"] = _clean_id_str(df["policy_number"])
        changed = _count_changes(before, df["policy_number"])
        _print_if_changes(changed, f"        ID cleaning: changed {changed} rows in 'policy_number'")

    if "claim_id" in df.columns:
        before = df["claim_id"].copy()
        df["claim_id"] = _clean_id_str(df["claim_id"])
        changed = _count_changes(before, df["claim_id"])
        _print_if_changes(changed, f"        ID cleaning: changed {changed} rows in 'claim_id'")

        if "policy_number" in df.columns:
            eq_mask = df["claim_id"].notna() & df["policy_number"].notna() & (df["claim_id"] == df["policy_number"])
            if eq_mask.any():
                df.loc[eq_mask, "claim_id"] = pd.NA
                _print_if_changes(int(eq_mask.sum()), f"        Claim ID sanity: nullified {eq_mask.sum()} rows where claim_id == policy_number")

    if "claim_number" in df.columns:
        before = df["claim_number"].copy()
        df["claim_number"] = _clean_id_str(df["claim_number"])
        changed = _count_changes(before, df["claim_number"])
        _print_if_changes(changed, f"        ID cleaning: changed {changed} rows in 'claim_number'")

    # --- subline as string; for auto, only "AL" or "APD" ---
    if "subline" in df.columns:
        before_all = df["subline"].copy()
        df["subline"] = _to_str_strip(df["subline"])
        after_strip = df["subline"].copy()
        changed_strip = _count_changes(before_all.astype("string"), after_strip)
        _print_if_changes(changed_strip, f"        Subline cleaning (strip): changed {changed_strip} rows in 'subline'")

        if "line_of_business" in df.columns:
            is_auto = df["line_of_business"].eq("auto")

            def _auto_subline(v):
                if pd.isna(v):
                    return pd.NA
                txt = str(v).strip().upper()
                if txt in ("AL", "AUTO LIABILITY"):
                    return "AL"
                if txt in ("APD", "AUTO PHYSICAL DAMAGE", "PD"):
                    return "APD"
                return pd.NA # anything else becomes NaN

            before_auto = df.loc[is_auto, "subline"].copy()
            df.loc[is_auto, "subline"] = df.loc[is_auto, "subline"].apply(_auto_subline).astype("string")
            after_auto = df.loc[is_auto, "subline"]
            changed_auto = _count_changes(before_auto, after_auto)
            _print_if_changes(changed_auto, f"              Subline cleaning (auto-only): changed {changed_auto} rows in 'subline' for auto LOB")

    # --- Financials as float with default 0.0 ---
    float_cols = [
        "total_incurred",
        "total_paid",
        "incurred_alae",
        "paid_alae",
        "total_recoveries",
    ]
    for col in float_cols:
        if col in df.columns:
            before = df[col].copy()
            df[col] = _to_float_with_default(df[col], 0.0)
            # compare as strings to avoid float precision noise
            changed = _count_changes(before.astype("string"), df[col].astype("string"))
            _print_if_changes(changed, f"        Financial cleaning: changed {changed} rows in '{col}'")

    # --- status: "O", "C", or "R" ---
    if "status" in df.columns:
        before = df["status"].copy()
        df["status"] = _clean_status(df["status"]).astype("string")
        changed = _count_changes(before.astype("string"), df["status"])
        _print_if_changes(changed, f"        Status cleaning: changed {changed} rows in 'status'")

    # --- claimant, driver, description as string ---
    for col in ["claimant", "driver", "description"]:
        if col in df.columns:
            before = df[col].copy()
            df[col] = _to_str_strip(df[col])
            changed = _count_changes(before.astype("string"), df[col])
            _print_if_changes(changed, f"        Text cleaning: changed {changed} rows in '{col}'")

    # --- state columns: 2-letter codes ---
    for col in ["accident_state", "garage_state", "coverage_state"]:
        if col in df.columns:
            before = df[col].copy()
            df[col] = _clean_state_col(df[col]).astype("string")
            changed = _count_changes(before.astype("string"), df[col])
            _print_if_changes(changed, f"        State cleaning: changed {changed} rows in '{col}'")

    # --- garage_state and driver NaN if not auto ---
    if "line_of_business" in df.columns:
        not_auto = ~df["line_of_business"].eq("auto")

        if "garage_state" in df.columns:
            before = df["garage_state"].copy()
            df.loc[not_auto, "garage_state"] = pd.NA
            changed = _count_changes(before.astype("string"), df["garage_state"])
            _print_if_changes(changed, f"              Non-auto adjustment: changed {changed} rows in 'garage_state'")

        if "driver" in df.columns:
            before = df["driver"].copy()
            df.loc[not_auto, "driver"] = pd.NA
            changed = _count_changes(before.astype("string"), df["driver"])
            _print_if_changes(changed, f"              Non-auto adjustment: changed {changed} rows in 'driver'")

    # --- coverage_state NaN if not general liability ---
    if "line_of_business" in df.columns and "coverage_state" in df.columns:
        not_gl = ~df["line_of_business"].eq("general liability")
        before = df["coverage_state"].copy()
        df.loc[not_gl, "coverage_state"] = pd.NA
        changed = _count_changes(before.astype("string"), df["coverage_state"])
        _print_if_changes(changed, f"              Non-GL adjustment: changed {changed} rows in 'coverage_state'")

    # Log all claim numbers after cleaning (exclude rows with no IDs)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
        )
        def _normalize_for_diff(val):
            if pd.isna(val):
                return None
            s = str(val).strip()
            if re.fullmatch(r"^\d+\.0+", s):
                s = s.split(".")[0]
            s = s.replace(" ", "")
            return re.sub(r"[^A-Za-z0-9]", "", s).lower() or None

        claim_numbers_after = set()
        for col in ["claim_number", "claim_id"]:
            if col in df.columns:
                claim_numbers_after.update(df[col].dropna().astype(str).str.strip())

        norm_before = {_normalize_for_diff(v) for v in claim_numbers_before if _normalize_for_diff(v)}
        norm_after = {_normalize_for_diff(v) for v in claim_numbers_after if _normalize_for_diff(v)}
        missing_norm = norm_before - norm_after
        if missing_norm:
            raw_lookup = {}
            for raw in claim_numbers_before:
                n = _normalize_for_diff(raw)
                if n:
                    raw_lookup.setdefault(n, set()).add(str(raw).strip())
            missing_raw = sorted({item for n in missing_norm for item in raw_lookup.get(n, (n,))})
            print(f"[CLEANING] Dropped claim numbers during cleaning: {missing_raw}")

    return df

def process_claim_free_policies_df(no_claims_df):
    return no_claims_df
    # Convert None to NaN for all columns
    no_claims_df = no_claims_df.where(pd.notnull(no_claims_df), None).replace({None: pd.NA})

    # # Drop rows where all columns except 'AccountName' and 'file_path' are NaN (no policy in the given input)
    # before = len(no_claims_df)
    # no_claims_df = no_claims_df.dropna(
    #     subset=[col for col in no_claims_df.columns if col not in ['AccountName', 'file_path']],
    #     how='all'
    # )
    # print(f"            \n            Dropped {before - len(no_claims_df)} rows with all NaN except AccountName/file_path")

    # # If empty, return an empty DataFrame with expected columns
    # if no_claims_df.empty:
    #     expected_cols = ['AccountName'] + [col for col in NO_CLAIMS_VALID_JSON_FIELDS] + ['file_path']
    #     return pd.DataFrame(columns=expected_cols)

    # # Drop rows where policy_effective_year is NaN
    # before = len(no_claims_df)
    # no_claims_df = no_claims_df.dropna(subset=['policy_effective_year'], how='any')
    # print(f"            Dropped {before - len(no_claims_df)} rows with missing policy_effective_year")

    # # Drop duplicates
    # before = len(no_claims_df)
    # no_claims_df = no_claims_df.drop_duplicates()
    # print(f"            Dropped {before - len(no_claims_df)} duplicate rows (using all columns)")

    # # Filter allowed lines of business
    # allowed_lobs = ["property", "auto", "general liability", "workers compensation", "umbrella"]
    # if 'line_of_business' in no_claims_df.columns:
    #     mask = no_claims_df['line_of_business'].str.strip().str.lower().isin(allowed_lobs)
    #     no_claims_df = no_claims_df[mask]

    # # If empty, return an empty DataFrame with expected columns
    # if no_claims_df.empty:
    #     expected_cols = ['AccountName'] + [col for col in NO_CLAIMS_VALID_JSON_FIELDS] + ['file_path']
    #     return pd.DataFrame(columns=expected_cols)

    # # Ensure dates are in "YYYY-MM-DD" format, else set to NaN
    # def validate_date(date_str):
    #     try:
    #         dt = pd.to_datetime(date_str, format='%Y-%m-%d', errors='raise')
    #         return dt.strftime('%Y-%m-%d')
    #     except Exception:
    #         return pd.NA

    # # List all date columns to validate
    # date_columns = [
    #     'evaluation_date'
    # ]
    # for col in date_columns:
    #     if col in no_claims_df.columns:
    #         no_claims_df[col] = no_claims_df[col].apply(validate_date)

    # # Final column order
    # final_order = ['AccountName'] + [col for col in NO_CLAIMS_VALID_JSON_FIELDS if col in no_claims_df.columns] + ['file_path']
    # final_order = [col for col in final_order if col in no_claims_df.columns]
    # no_claims_df = no_claims_df[final_order]

    # # If empty, return an empty DataFrame with expected columns
    # if no_claims_df.empty:
    #     expected_cols = ['AccountName'] + [col for col in NO_CLAIMS_VALID_JSON_FIELDS] + ['file_path']
    #     return pd.DataFrame(columns=expected_cols)

    return no_claims_df
