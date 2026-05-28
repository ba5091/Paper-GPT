# 🧠 PaperGPT – RAG-Based Research Paper Assistant

PaperGPT is an AI-powered research assistant that enables users to interact with research papers using Retrieval-Augmented Generation (RAG), semantic search, vector databases, and Large Language Models (LLMs).

The system allows users to upload multiple research papers in PDF format and ask contextual questions about them. It retrieves the most relevant information using FAISS vector similarity search and generates intelligent responses using FLAN-T5.

---

# 🚀 Features

✅ Multi-PDF Upload Support
✅ Research Paper Question Answering
✅ Semantic Search using Embeddings
✅ Retrieval-Augmented Generation (RAG)
✅ FAISS Vector Database Integration
✅ AI-Powered Answer Generation
✅ Research Summary Generation
✅ Literature Review Generation
✅ Source Citations with Page Numbers
✅ Conversational Chat History
✅ Modern Streamlit UI

---

# 🏗️ System Architecture

```text
PDF Upload
     ↓
Text Extraction
     ↓
Chunking
     ↓
Embeddings Generation
     ↓
FAISS Vector Storage
     ↓
Semantic Retrieval
     ↓
LLM Answer Generation
     ↓
Final AI Response
```

---

# 🧠 Technologies Used

## Frontend

* Streamlit

## Backend

* Python

## AI / NLP

* Sentence Transformers
* FLAN-T5
* LangChain
* HuggingFace Transformers

## Vector Database

* FAISS

## PDF Processing

* PyMuPDF (fitz)

---

# 📂 Project Structure

```text
PaperGPT/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── raw_pdfs/
│   └── processed/
│
├── src/
│   ├── pdf_processor.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── llm_engine.py
│
├── test_pdf.py
├── test_chunking.py
├── test_embeddings.py
├── test_faiss.py
├── test_retrieval.py
└── test_llm.py
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/papergpt-ai.git
cd papergpt-ai
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📖 How It Works

## Step 1 — Upload PDFs

Users upload one or more research papers.

## Step 2 — Text Extraction

The system extracts text from PDFs using PyMuPDF.

## Step 3 — Chunking

Large documents are divided into smaller overlapping chunks.

## Step 4 — Embedding Generation

Chunks are converted into semantic vectors using Sentence Transformers.

## Step 5 — Vector Storage

Embeddings are stored inside FAISS for efficient retrieval.

## Step 6 — Semantic Retrieval

User queries are converted into embeddings and matched against stored vectors.

## Step 7 — LLM Response Generation

Retrieved chunks are passed to FLAN-T5 to generate contextual answers.

---

# 📸 Key Functionalities

## 🔍 Semantic Research Search

Finds relevant information based on meaning rather than keywords.

## 📑 Research Summary Generation

Automatically summarizes uploaded papers.

## 📚 Literature Review Generation

Generates AI-assisted literature reviews from uploaded research documents.

## 💬 Conversational QA

Maintains chat history for interactive research conversations.

## 📖 Source Citations

Displays retrieved chunks along with page references.

---

# 🎯 Use Cases

* Academic Research Assistance
* Literature Review Automation
* Research Paper Analysis
* AI-Powered Knowledge Retrieval
* Educational AI Tools
* Enterprise Document Intelligence

---

# 🔥 Future Improvements

* OpenAI/Groq API Integration
* PDF Highlighting
* Research Citation Export
* Cloud Deployment
* Docker Support
* Authentication System
* Streaming LLM Responses
* Pinecone/ChromaDB Integration

---

# 📊 Tech Highlights

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Embeddings
* FAISS Similarity Search
* Conversational AI
* Multi-Document Intelligence
* LLM Orchestration

---

# 👨‍💻 Author

Aryaneel Bhaduri
B.Tech CSE – Data Science
SRM Institute of Science and Technology

---

# ⭐ Acknowledgements

* HuggingFace
* LangChain
* Streamlit
* FAISS
* Sentence Transformers
* PyMuPDF
