import streamlit as st
import os

from src.pdf_processor import extract_text_from_pdf
from src.chunking import chunk_text
from src.embeddings import create_embeddings, model
from src.vector_store import build_faiss_index
from src.retriever import retrieve_chunks
from src.llm_engine import generate_answer

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="ResearchMind AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

/* -------------------------------------------------
   GLOBAL
------------------------------------------------- */

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: white !important;
    color: black !important;
}

/* Main App */

.stApp {
    background-color: white !important;
}

/* -------------------------------------------------
   SIDEBAR
------------------------------------------------- */

section[data-testid="stSidebar"] {
    background-color: white !important;
    border-right: 2px solid #f0f0f0;
}

/* Sidebar Text */

section[data-testid="stSidebar"] * {
    color: black !important;
}

/* -------------------------------------------------
   MAIN TITLE
------------------------------------------------- */

.main-title {
    text-align: center;
    color: #E50914;
    font-size: 70px;
    font-weight: 800;
    margin-top: -20px;
}

.subtitle {
    text-align: center;
    color: #555555;
    font-size: 24px;
    margin-bottom: 30px;
}

/* -------------------------------------------------
   HEADINGS
------------------------------------------------- */

h1, h2, h3, h4 {
    color: #E50914 !important;
}

/* -------------------------------------------------
   TEXT INPUT
------------------------------------------------- */

.stTextInput input {
    background-color: white !important;
    color: black !important;
    border: 2px solid #E50914 !important;
    border-radius: 12px !important;
    padding: 12px !important;
    font-size: 16px !important;
}

/* Placeholder */

.stTextInput input::placeholder {
    color: #666666 !important;
}

/* -------------------------------------------------
   FILE UPLOADER
------------------------------------------------- */

[data-testid="stFileUploader"] {
    background-color: #1E1E2F !important;
    border: 2px dashed #E50914 !important;
    border-radius: 16px !important;
    padding: 25px !important;
}

/* Upload Box Text */

[data-testid="stFileUploader"] * {
    color: white !important;
}

/* Upload Browse Button */

[data-testid="stFileUploader"] section button {
    background-color: #E50914 !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
}

[data-testid="stFileUploader"] section button:hover {
    background-color: #b20710 !important;
    color: white !important;
}

/* -------------------------------------------------
   BUTTONS
------------------------------------------------- */

.stButton button {
    background-color: #E50914 !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    height: 50px !important;
    font-size: 18px !important;
    font-weight: bold !important;
}

.stButton button:hover {
    background-color: #b20710 !important;
    color: white !important;
}

/* -------------------------------------------------
   METRIC CARDS
------------------------------------------------- */

[data-testid="metric-container"] {
    background-color: white !important;
    border: 2px solid #f0f0f0 !important;
    padding: 20px !important;
    border-radius: 15px !important;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08) !important;
}

/* Metric Text */

[data-testid="metric-container"] * {
    color: black !important;
}

/* -------------------------------------------------
   ANSWER BOX
------------------------------------------------- */

.answer-box {
    background-color: white;
    color: black;
    border-left: 6px solid #E50914;
    padding: 25px;
    border-radius: 15px;
    font-size: 18px;
    line-height: 1.8;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
}

/* -------------------------------------------------
   CHUNK BOX
------------------------------------------------- */

.chunk-box {
    background-color: white;
    color: black;
    border-left: 5px solid #E50914;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    line-height: 1.7;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
}

/* -------------------------------------------------
   CHAT BOX
------------------------------------------------- */

.chat-box {
    background-color: white;
    color: black;
    border-left: 5px solid #E50914;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    line-height: 1.7;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
}

/* -------------------------------------------------
   EXPANDERS
------------------------------------------------- */

.streamlit-expanderHeader {
    background-color: white !important;
    color: black !important;
    border-radius: 10px !important;
}

/* -------------------------------------------------
   FOOTER
------------------------------------------------- */

.footer {
    text-align: center;
    color: #666666;
    padding: 20px;
    font-size: 15px;
}

/* -------------------------------------------------
   HR
------------------------------------------------- */

hr {
    border: 1px solid #f0f0f0;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.markdown(
        "<h1 style='color:#E50914;'>🧠 ResearchMind AI</h1>",
        unsafe_allow_html=True
    )

    st.markdown("""
    ### AI-Powered Research Assistant
    """)

    st.markdown("---")

    st.markdown("## 🚀 Features")

    st.markdown("""
    - Multi PDF Upload  
    - Research Paper QA  
    - Semantic Search  
    - RAG Architecture  
    - FAISS Vector Search  
    - LLM-Powered Answers  
    - Research Summary  
    - Literature Review Generation  
    - Source Citations  
    - Chat History  
    """)

    st.markdown("---")

    st.markdown("## ⚙️ Tech Stack")

    st.markdown("""
    - LangChain  
    - Sentence Transformers  
    - FAISS  
    - FLAN-T5  
    - Streamlit  
    """)

    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.success("✅ System Ready")

# ---------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------

st.markdown(
    "<div class='main-title'>ResearchMind AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI-Powered Research Paper Assistant using RAG</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------
# MULTI PDF UPLOAD
# ---------------------------------------------------

st.markdown("## 📄 Upload Research Papers")

uploaded_files = st.file_uploader(
    "",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------------------------------------------
# QUESTION INPUT
# ---------------------------------------------------

st.markdown("## ❓ Ask a Question")

question = st.text_input(
    "",
    placeholder="Ask anything about your research papers..."
)

# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------

if uploaded_files:

    all_chunked_data = []

    with st.spinner("📚 Processing PDFs..."):

        for uploaded_file in uploaded_files:

            save_path = os.path.join(
                "data/raw_pdfs",
                uploaded_file.name
            )

            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())

            # Extract Pages
            pages = extract_text_from_pdf(save_path)

            # Chunking
            chunked_data = chunk_text(pages)

            # Add Source File Name
            for chunk in chunked_data:
                chunk["source"] = uploaded_file.name

            all_chunked_data.extend(chunked_data)

    st.success("✅ PDFs Processed Successfully!")

    if question:

        with st.spinner("🧠 Generating AI Response..."):

            # Embeddings
            embeddings = create_embeddings(all_chunked_data)

            # FAISS Index
            index = build_faiss_index(embeddings)

            # Query Embedding
            query_embedding = model.encode([question])[0]

            # Retrieval
            retrieved_chunks = retrieve_chunks(
                query_embedding,
                index,
                all_chunked_data
            )

            # Context
            context = "\n".join([
                chunk["text"]
                for chunk in retrieved_chunks
            ])

            # Generate Answer
            answer = generate_answer(
                context,
                question
            )

            # Save Chat History
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer
            })

        # ---------------------------------------------------
        # METRICS
        # ---------------------------------------------------

        st.markdown("## 📊 System Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "📚 Total Chunks",
            len(all_chunked_data)
        )

        col2.metric(
            "🔍 Retrieved Chunks",
            len(retrieved_chunks)
        )

        col3.metric(
            "🧠 Embedding Size",
            len(embeddings[0])
        )

        st.markdown("---")

        # ---------------------------------------------------
        # AI ANSWER
        # ---------------------------------------------------

        st.markdown("## ✨ AI Generated Answer")

        st.markdown(
            f"""
            <div class="answer-box">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # ---------------------------------------------------
        # RESEARCH SUMMARY
        # ---------------------------------------------------

        if st.button("📄 Generate Research Summary"):

            summary_prompt = """
            Summarize the uploaded research papers.
            """

            summary = generate_answer(
                context,
                summary_prompt
            )

            st.markdown("## 📑 Research Summary")

            st.markdown(
                f"""
                <div class="answer-box">
                {summary}
                </div>
                """,
                unsafe_allow_html=True
            )

        # ---------------------------------------------------
        # LITERATURE REVIEW
        # ---------------------------------------------------

        if st.button("📚 Generate Literature Review"):

            literature_prompt = """
            Generate a literature review using the uploaded papers.
            """

            literature_review = generate_answer(
                context,
                literature_prompt
            )

            st.markdown("## 📘 Literature Review")

            st.markdown(
                f"""
                <div class="answer-box">
                {literature_review}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ---------------------------------------------------
        # RETRIEVED CONTEXT
        # ---------------------------------------------------

        st.markdown("## 📖 Retrieved Context")

        for i, chunk in enumerate(retrieved_chunks):

            with st.expander(
                f"Chunk {i+1} | Page {chunk['page']} | Source: {chunk['source']}"
            ):

                st.markdown(
                    f"""
                    <div class="chunk-box">
                    {chunk['text']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # ---------------------------------------------------
        # CHAT HISTORY
        # ---------------------------------------------------

        st.markdown("## 💬 Chat History")

        for chat in reversed(st.session_state.chat_history):

            st.markdown(
                f"""
                <div class="chat-box">
                <b>Question:</b><br>
                {chat['question']}
                <br><br>
                <b>Answer:</b><br>
                {chat['answer']}
                </div>
                """,
                unsafe_allow_html=True
            )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div class='footer'>
    Built with ❤️ using Streamlit, LangChain, FAISS, and Transformers
    </div>
    """,
    unsafe_allow_html=True
)