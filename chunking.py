from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunked_data = []

    for page in pages:

        chunks = splitter.split_text(page["text"])

        for chunk in chunks:

            chunked_data.append({
                "page": page["page"],
                "text": chunk
            })

    return chunked_data