import os
from docx2pdf import convert

def convert_docx_to_pdf(input_data_path, company_names):
    """
    Recursively convert all .docx files in input_data_path (and subfolders) to PDF.
    The PDF will be saved in the same folder as the .docx file.
    """
    for root, _, files in os.walk(input_data_path):
        # root looks like: input_data_path/company_name[/sub/dirs...]
        rel_path = os.path.relpath(root, input_data_path)
        if rel_path == ".":
            # This is the base folder itself; skip files directly under input_data_path
            # (or remove this `continue` if you *do* want to process them)
            continue
        
        # Get the first directory under input_data_path (the company name)
        company_folder = rel_path.split(os.path.sep)[0]
        if company_folder not in company_names:
            # Skip anything not under an allowed company_name
            continue
        
        for file in files:
            if file.lower().endswith('.docx'):
                docx_path = os.path.join(root, file)
                pdf_path = os.path.splitext(docx_path)[0] + ".pdf"
                if os.path.exists(pdf_path):
                    print(f"PDF already exists, not coverting: {docx_path}")
                    continue
                try:
                    convert(docx_path, pdf_path)
                    print(f"Converted: {docx_path} -> {pdf_path}")
                except Exception as e:
                    print(f"Failed to convert {docx_path}: {e}")
