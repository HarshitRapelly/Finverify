# ⚖️ FinVerify AI - Hallucination-Resistant Financial RAG

FinVerify AI is a production-grade, hackathon-ready Retrieval-Augmented Generation (RAG) system specifically designed for financial documents (10-K, 10-Q, quarterly earnings reports, audit memos).

Traditional RAG systems frequently hallucinate metrics, confuse fiscal quarters, or extrapolate beyond the text. FinVerify AI solves this with **deterministic provenance tracking**, **local FAISS vector similarity thresholds**, **lexical entity guardrails**, and **strict zero-hallucination fallback enforcement**.

---

## 🚀 Key Features

1. **Multi-PDF Ingestion with Provenance Tracking**:
   - Extracts page-by-page text using `pypdf`.
   - Immutable metadata preserved for every chunk: `document_name`, `page_number`, and `chunk_id`.
   - Recursive chunking: ~700 characters with 120 character overlap respecting paragraph and sentence boundaries.

2. **100% Local Embeddings & FAISS Vector Store**:
   - Embeddings generated on CPU with `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions).
   - Fast Inner Product (`IndexFlatIP`) cosine similarity search in `faiss-cpu`.
   - Top-5 most relevant chunks retrieved per query.

3. **Multi-Layered Anti-Hallucination Guardrail**:
   - **Semantic Similarity Threshold**: Filters out queries without sufficient semantic proximity.
   - **Lexical Overlap Verification**: Cross-references financial entities, ratios, and terms.
   - **Strict Fallback Guarantee**: If evidence is insufficient, responds **exactly**:
     ```text
     "I don't know based on the provided documents."
     ```
   - **Post-Generation Hallucination Interception**: Detects if an LLM is uncertain or speculating and substitutes the exact fallback.

4. **Multi-Model Support (Gemini & Groq)**:
   - Built-in support for **Google Gemini** (`gemini-2.5-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`) via `google-genai`.
   - Built-in support for **Groq** (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) via `groq`.
   - Zero hardcoded keys: dynamically read from `.env` or input through the sidebar.
   - Grounded extractive offline mode works even without API keys!

5. **Hackathon Demo Polish**:
   - Real-time confidence badge: `🟢 High`, `🟡 Medium`, or `🔴 Insufficient Evidence`.
   - Detailed citations for every answer with document name, page badge, chunk ID, and similarity percentage.
   - Expandable retrieved evidence drawer with full chunk texts.
   - One-click sample 10-Q filing loader and clickable test prompt chips.

---

## 📁 Project Architecture

```text
finverify/
├── app.py              # Streamlit web application & UI
├── ingest.py           # PDF extraction, metadata tracking & recursive chunker
├── rag.py              # Local MiniLM embeddings, FAISS indexing & LLM orchestration
├── guardrail.py        # Evidence sufficiency & hallucination prevention engine
├── requirements.txt    # Dependencies
├── .env.example        # Environment variable template
├── .env                # Local secrets (never committed)
└── test_pipeline.py    # Automated test script for answerable & unanswerable queries
```

---

## 🛠️ Quick Start

### 1. Activate Environment & Install Dependencies
```bash
# In the finverify directory
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
Copy `.env.example` to `.env` and insert your API key(s):
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy...
# or
GROQ_API_KEY=gsk_...
```
*(You can also paste your API key directly into the Streamlit sidebar during the demo!)*

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing
Run the verification test suite:
```bash
python test_pipeline.py
```
This tests:
1. PDF ingestion and page/chunk preservation.
2. Embedding and FAISS indexing.
3. **Answerable query**: verifies citations, confidence score, and grounded response.
4. **Unanswerable query**: verifies detection of insufficient evidence and exact output of `"I don't know based on the provided documents."`.
