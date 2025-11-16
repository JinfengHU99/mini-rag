# ![Mini RAG](https://img.shields.io/badge/Mini--RAG-Retrieval--Augmented%20Generation-blue)

A **minimal Retrieval-Augmented Generation (RAG) pipeline** using FastAPI, OpenAI Chat API, FAISS, and Sentence Transformers.  
This project allows you to query documents using LLMs while retrieving the most relevant context chunks.

---

## 🔹 Project Structure
- **Mini-RAG/**  
  - `app.py` — FastAPI web app  
  - `embedder.py` — Convert text chunks to vector embeddings  
  - `retriever.py` — FAISS-based vector retrieval  
  - `textsplitter.py` — Split long documents into smaller chunks  
  - `rag_chain.py` — RAG pipeline: build prompts & query OpenAI LLM  
  - `requirements.txt` — Python dependencies  
  - `README.md` — Project description  
  - `data/`  
    - `sample.txt` — Example document(s)  

---


## Pipeline Flow

**Step-by-step flow of the Mini RAG system:**

```text
Document(s)
    ↓
TextSplitter
    ↓
Chunks
    ↓
Embedder
    ↓
Embeddings
    ↓
Retriever
    ↓
Top-K Chunks
    ↓
RAG Pipeline
    ↓
LLM Answer
    ↓
API Response
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mini-rag.git
cd mini-rag
```
2. Create a virtual environment (recommended):
```bash
3. python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Set your OpenAI API key:
```bash
# Linux / Mac
export OPENAI_API_KEY="your_api_key"
# Windows
setx OPENAI_API_KEY "your_api_key"
```
##Usage

1. Run the FastAPI server:
```bash
uvicorn app:app --reload
```
2. Send a POST request to the /query endpoint.

**Request Example:**
```json
{
  "question": "What is Mini RAG?"
}
```

**Response Example:**
```json
{
  "answer": "Mini RAG is a minimal Retrieval-Augmented Generation pipeline...",
  "context": ["This is your example document text."]
}
```
3. Access interactive API docs at:
```arduino
http://127.0.0.1:8000/docs
```
---

##How It Works

**1. Text Splitting** — `textsplitter.py` splits long documents into chunks.  
**2. Embedding** — `embedder.py`generates vector embeddings for each chunk.  
**3. Retrieval** — `retriever.py` builds a FAISS index and retrieves top-k relevant chunks.  
**4. RAG Pipeline**  — `rag_chain.py` builds prompts combining retrieved chunks with the user query and calls OpenAI Chat API.  
**5. API** — `app.py` exposes a FastAPI endpoint for querying the system.

---

## Example Python Usage

```python
import requests

url = "http://127.0.0.1:8000/query"
data = {"question": "What is RAG?"}
response = requests.post(url, json=data)
print(response.json())
```
---

##Dependencies
- `fastapi` — Web framework
- `uvicorn` — ASGI server
- `pydantic` — Data validation
- `sentence-transformers` — Embedding model
- `faiss-cpu` — Vector search
- `openai` — LLM API

---

##Notes / Tips
- Replace `texts` in `app.py` with documents from the `data/` folder.   
- Adjust `chunk_size` and `chunk_overlap` in `textsplitter.py` for your document length.   
- Adjust `k` in `retriever.query` for the number of retrieved chunks.   
- Use `temperature=0.0` in `rag_chain.py` to reduce hallucinations from the LLM.