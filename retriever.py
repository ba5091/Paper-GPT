import numpy as np

def retrieve_chunks(query_embedding, index, chunked_data, top_k=3):

    distances, indices = index.search(
        np.array([query_embedding]),
        top_k
    )

    retrieved = [chunked_data[i] for i in indices[0]]

    return retrieved