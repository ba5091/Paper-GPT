from src.pdf_processor import extract_text_from_pdf

pdf_path = "data/raw_pdfs/sample.pdf"

text = extract_text_from_pdf(pdf_path)

print(text[:1000])