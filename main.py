import os
import shutil
from docx_to_pdf import convert_docx_to_pdf
from detect_loss_runs import detect_loss_runs
from pdf_to_images import turn_pdf_to_images
from determine_pdf_cutoff import determine_page_cutoffs
from extract import extract_claims
from clean_extraction_output import process_claims_df
from detect_duplicates import detect_duplicate_claims_across_files
from metrics import save_metrics_json

def prepare_excel_csv_files(input_data_dir, output_data_dir, company_names):
    """
    Finds Excel and CSV files in the input_data folder and copies them
    to the output_data folder under excel_sheets/ and csvs/ subfolders
    as expected by the downstream detection and extraction steps.
    """
    for company in company_names:
        company_input_dir = os.path.join(input_data_dir, company)
        if not os.path.isdir(company_input_dir):
            continue
            
        company_output_dir = os.path.join(output_data_dir, company)
        excel_out_dir = os.path.join(company_output_dir, "excel_sheets")
        csv_out_dir = os.path.join(company_output_dir, "csvs")
        
        for root, _, files in os.walk(company_input_dir):
            for file in files:
                file_lower = file.lower()
                src_path = os.path.join(root, file)
                
                if file_lower.endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
                    os.makedirs(excel_out_dir, exist_ok=True)
                    shutil.copy2(src_path, os.path.join(excel_out_dir, file))
                    print(f"Copied Excel: {file} to output_data")
                elif file_lower.endswith(".csv"):
                    os.makedirs(csv_out_dir, exist_ok=True)
                    shutil.copy2(src_path, os.path.join(csv_out_dir, file))
                    print(f"Copied CSV: {file} to output_data")

def main():
    input_data_dir = "input_data"
    output_data_dir = "output_data"
    extraction_data_dir = "extraction_output"
    
    # Configure the folder name under input_data and the output prefix
    company_names = ["sample"]
    file_path_prefix = "sample_"
    
    # 1. Convert Word (.docx) documents to PDF
    print("\nConverting DOCX to PDF....\n")
    convert_docx_to_pdf(input_data_dir, company_names)
    
    # 2. Copy raw Excel/CSV files to output folders
    print("\nCopying Excel and CSV files to output_data....\n")
    prepare_excel_csv_files(input_data_dir, output_data_dir, company_names)
    
    # 3. Detect which documents are loss runs
    print("\nDetecting which input files are loss runs....\n")
    is_file_loss_run_dict = detect_loss_runs(output_data_dir, company_names)
    
    # 4. Convert PDF pages to image sheets (handles rotation check)
    print("\nCreating images of each PDF page...")
    turn_pdf_to_images(company_names, input_data_dir, output_data_dir, is_file_loss_run_dict)
    
    # 5. Calculate page cutoff boundaries
    print("\nDetermining PDF page groupings based on context limits...")
    pages_to_join_dict = determine_page_cutoffs(output_data_dir, company_names, is_file_loss_run_dict)
    
    # 6. Extract structured claims JSON
    print("\nBeginning Loss Run Extraction....\n")
    claims_df = extract_claims(output_data_dir, company_names, pages_to_join_dict, is_file_loss_run_dict)
    
    # 7. Clean and normalize the output DataFrame
    print("\n Cleaning Claims Dataframe and Implementing Logic")
    claims_df = process_claims_df(claims_df)
    
    # 8. Filter duplicate claims across files (optional)
    # print("\n Detecting Claims Reported Across Multiple Files")
    # claims_df = detect_duplicate_claims_across_files(claims_df)
    
    # 9. Save CSV output
    os.makedirs(extraction_data_dir, exist_ok=True)
    claims_df.to_csv(f"{extraction_data_dir}/{file_path_prefix}claims.csv", index=False)
    print(f"\nFinal claims CSV saved to: {extraction_data_dir}/{file_path_prefix}claims.csv")
    
    # 9.5 Run rollup post-processing
    print("\nRunning claims rollup post-processing...")
    try:
        from post_process_rollup import run_rollup_post_processing
        run_rollup_post_processing(
            input_path=f"{extraction_data_dir}/{file_path_prefix}claims.csv",
            output_detailed_path=f"{extraction_data_dir}/{file_path_prefix}claims.csv",
            output_rollup_path=f"{extraction_data_dir}/{file_path_prefix}Rolled_up_claims.csv"
        )
    except Exception as e:
        print(f"[ERROR] Failed to run rollup post-processing: {e}")
    
    # 10. Save metrics JSON
    metrics_output_path = f"{extraction_data_dir}/{file_path_prefix}metrics.json"
    save_metrics_json(metrics_output_path)
    print(f"Metrics saved to: {metrics_output_path}")

if __name__ == "__main__":
    main()
