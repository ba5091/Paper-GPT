from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(chunked_data):

    texts = [chunk["text"] for chunk in chunked_data]

    embeddings = model.encode(texts)

    return embeddings