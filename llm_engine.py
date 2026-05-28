from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)

def generate_answer(context, question):

    prompt = f"""
    Answer only using the context below.

    Context:
    {context}

    Question:
    {question}

    If answer is not found,
    say:
    Information not available in document.
    """

    result = generator(
        prompt,
        max_new_tokens=200
    )

    return result[0]['generated_text']