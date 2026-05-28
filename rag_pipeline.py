from src.pdf_processor import extract_text_from_pdf
from src.chunking import chunk_text
from src.embeddings import create_embeddings, model
from src.vector_store import build_faiss_index
from src.retriever import retrieve_chunks
from src.llm_engine import generate_answer

def run_rag(pdf_path, question):

    text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)

    index = build_faiss_index(embeddings)

    query_embedding = model.encode([question])[0]

    retrieved_chunks = retrieve_chunks(
        query_embedding,
        index,
        chunks
    )

    context = "\n".join(retrieved_chunks)

    answer = generate_answer(
        context,
        question
    )

    return answer, retrieved_chunks