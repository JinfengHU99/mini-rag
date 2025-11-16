from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embedder import Embedder
from retriever import Retriever
from textsplitter import SimpleTextSplitter
from rag_chain import RAGPipeline

# Initialize FastAPI app
app = FastAPI(title="Mini RAG API")

# Initialize components for RAG pipeline
splitter = SimpleTextSplitter(chunk_size=500, chunk_overlap=50)  # Split documents into chunks
embedder = Embedder()  # Embedding model

# Example documents (replace with file reading if needed)
texts = ["This is your example document text."]
chunks = splitter.split_text(" ".join(texts))  # Split into chunks
embs = embedder.encode(chunks)  # Encode chunks into embeddings
retriever = Retriever(embs, chunks)  # Build vector index
rag = RAGPipeline()  # RAG pipeline instance


# Define request model
class Query(BaseModel):
    question: str  # User's question


# Define POST API endpoint
@app.post("/query")
def query_rag(q: Query):
    """
    API endpoint to query the Mini RAG system.

    Args:
        q: Query object containing user's question

    Returns:
        JSON with 'answer' from LLM and 'context' retrieved from documents
    """
    try:
        # Encode user's question
        q_emb = embedder.encode(q.question)[0]

        # Retrieve top-3 relevant chunks
        context = retriever.query(q_emb, k=3)

        # Generate answer using RAG pipeline
        answer = rag.answer(q.question, context)

        # Return response
        return {"answer": answer, "context": context}
    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))
