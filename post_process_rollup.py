import os
import re
import difflib
import pandas as pd

def normalize_name(name: str) -> str:
    """
    Cleans name for comparison: lowercases, removes punctuation,
    and strips extra spaces.
    """
    if pd.isna(name) or not name:
        return ""
    cleaned = str(name).lower()
    cleaned = re.sub(r'[^a-z0-9\s]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def clean_for_slug(name: str) -> str:
    """
    Cleans name and replaces spaces with underscores for roll_no formatting.
    """
    norm = normalize_name(name)
    return norm.replace(' ', '_')

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
    # Sequence similarity threshold of 0.85
    similarity = difflib.SequenceMatcher(None, n1, n2).ratio()
    return similarity > 0.85

def assign_roll_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a roll_no to each claim in the DataFrame.
    """
    df = df.copy()
    groups = []  # list of dicts: {'accident_date': ..., 'accident_state': ..., 'names': set(), 'roll_no': ...}
    roll_numbers = []

    for idx, row in df.iterrows():
        # Retrieve fields, default to None if missing
        raw_date = row.get("accident_date") if "accident_date" in df.columns else None
        raw_state = row.get("accident_state") if "accident_state" in df.columns else None
        raw_driver = row.get("driver") if "driver" in df.columns else None
        raw_claimant = row.get("claimant") if "claimant" in df.columns else None

        acc_date = str(raw_date).strip() if pd.notna(raw_date) else None
        acc_state = str(raw_state).strip().upper() if pd.notna(raw_state) else None
        
        driver_val = str(raw_driver).strip() if pd.notna(raw_driver) else None
        claimant_val = str(raw_claimant).strip() if pd.notna(raw_claimant) else None

        # Normalize names for fuzzy comparison
        norm_driver = normalize_name(driver_val) if driver_val else ""
        norm_claimant = normalize_name(claimant_val) if claimant_val else ""

        # Find matching group
        matched_group = None
        for g in groups:
            # Match date
            date_match = (g['accident_date'] == acc_date)
            # Match state
            state_match = (g['accident_state'] == acc_state)
            
            # Match names
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
                # Both have no names
                if not norm_driver and not norm_claimant:
                    name_match = True

            if date_match and state_match and name_match:
                matched_group = g
                break

        if matched_group is not None:
            # Update names in matched group
            if norm_driver:
                matched_group['names'].add(norm_driver)
            if norm_claimant:
                matched_group['names'].add(norm_claimant)
            roll_no = matched_group['roll_no']
        else:
            # Create a new group
            names_set = set()
            if norm_driver:
                names_set.add(norm_driver)
            if norm_claimant:
                names_set.add(norm_claimant)

            # Determine human-readable name for roll_no
            roll_name = "unknown_person"
            if driver_val:
                roll_name = clean_for_slug(driver_val)
            elif claimant_val:
                roll_name = clean_for_slug(claimant_val)

            # Format dates/states
            date_slug = acc_date if acc_date else "unknown_date"
            state_slug = acc_state if acc_state else "unknown_state"

            roll_no = f"{date_slug}&{state_slug}*{roll_name}"
            
            groups.append({
                'accident_date': acc_date,
                'accident_state': acc_state,
                'names': names_set,
                'roll_no': roll_no
            })

        roll_numbers.append(roll_no)

    # Insert roll_no as the first column
    df.insert(0, 'roll_no', roll_numbers)
    return df

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
        
        # If numeric, sum
        if col in numeric_cols:
            agg_funcs[col] = 'sum'
        else:
            # Custom function to join unique non-null values with '?'
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

    # Group by roll_no and apply aggregation
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

    print("Running rollup logic...")
    detailed_df = assign_roll_numbers(df)
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
