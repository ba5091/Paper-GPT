from src.pdf_processor import extract_text_from_pdf
from src.chunking import chunk_text
from src.embeddings import create_embeddings, model
from src.vector_store import build_faiss_index
from src.retriever import retrieve_chunks
from src.llm_engine import generate_answer

pdf_path = "data/raw_pdfs/sample.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

embeddings = create_embeddings(chunks)

index = build_faiss_index(embeddings)

query = "What methodology was used?"

query_embedding = model.encode([query])[0]

retrieved_chunks = retrieve_chunks(
    query_embedding,
    index,
    chunks
)

context = "\n".join(retrieved_chunks)

answer = generate_answer(
    context,
    query
)

print("\nFINAL ANSWER:\n")

print(answer)