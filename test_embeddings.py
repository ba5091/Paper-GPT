from src.pdf_processor import extract_text_from_pdf
from src.chunking import chunk_text
from src.embeddings import create_embeddings

pdf_path = "data/raw_pdfs/sample.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

embeddings = create_embeddings(chunks)

print("TOTAL EMBEDDINGS:", len(embeddings))

print("\nEMBEDDING DIMENSION:")

print(len(embeddings[0]))

print("\nFIRST EMBEDDING VECTOR:\n")

print(embeddings[0])