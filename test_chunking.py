from src.pdf_processor import extract_text_from_pdf
from src.chunking import chunk_text

pdf_path = "data/raw_pdfs/sample.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

print("TOTAL CHUNKS:", len(chunks))

print("\nFIRST CHUNK:\n")

print(chunks[0])