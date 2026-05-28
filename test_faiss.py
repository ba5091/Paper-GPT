from src.pdf_processor import extract_text_from_pdf
from src.chunking import chunk_text
from src.embeddings import create_embeddings
from src.vector_store import build_faiss_index

pdf_path = "data/raw_pdfs/sample.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

embeddings = create_embeddings(chunks)

index = build_faiss_index(embeddings)

print("FAISS INDEX CREATED SUCCESSFULLY")

print("TOTAL VECTORS STORED:")

print(index.ntotal)