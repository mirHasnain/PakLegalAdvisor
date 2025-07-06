import os
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PakLegalAdvisor API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[str]

# Global variables
faiss_index = None
legal_documents = []
embeddings_model = None

# Configure Gemini API
genai.configure(api_key="AIzaSyApfAsLZ5_G41N2gg_W3Uar5tlrPOsRjoM")
gemini_model = genai.GenerativeModel(model_name="models/gemma-3n-e4b-it")

def load_faiss_index(index_path: str):
    """Load FAISS index, texts, and metadatas from disk"""
    global faiss_index, legal_documents, texts, metadatas

    try:
        # Load FAISS index
        faiss_index = faiss.read_index(index_path)
        logger.info(f"✅ FAISS index loaded from: {index_path}")

        # Load associated text and metadata
        text_path = index_path.replace(".idx", "_texts.pkl")
        meta_path = index_path.replace(".idx", "_metadatas.pkl")

        with open(text_path, 'rb') as f:
            texts = pickle.load(f)

        with open(meta_path, 'rb') as f:
            metadatas = pickle.load(f)

        legal_documents = texts  # for compatibility if code elsewhere uses legal_documents
        logger.info(f"✅ Loaded {len(texts)} texts and {len(metadatas)} metadata records.")

    except Exception as e:
        logger.error(f"❌ Failed to load FAISS data: {e}")
        raise

def get_embeddings(text: str) -> np.ndarray:
    """Generate embeddings for the query text"""
    try:
        # Initialize the model (fixed the typo: modeeel -> embedding_model)
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Return numpy array, not list
        return embedding_model.encode(text)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def search_similar_documents(query: str, k: int = 5) -> List[str]:
    """Search for similar documents using FAISS"""
    try:
        # Get query embeddings
        query_embedding = get_embeddings(query)
        query_vector = np.array([query_embedding])
        
        # Search in FAISS index
        scores, indices = faiss_index.search(query_vector, k)
        
        # Retrieve relevant documents
        retrieved_chunks = []
        for idx in indices[0]:
            if idx < len(legal_documents):
                retrieved_chunks.append(legal_documents[idx])
        
        return retrieved_chunks
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []

def generate_response(user_query: str, retrieved_chunks: List[str]) -> str:
    """Generate response using Gemini model with RAG"""
    try:
        # Combine retrieved chunks
        context = "\n".join(retrieved_chunks)
        
        # Create RAG prompt
        rag_prompt = f"""You are PakLegalAdvisor, an AI assistant specialized in Pakistani law and legal procedures. 
        Provide accurate, helpful legal guidance based on the provided context. Always mention that this is general guidance 
        and users should consult with qualified legal professionals for specific legal matters.

Context: "{context}"

Question: "{user_query}"

Answer: Please provide a comprehensive answer based on the legal context provided above strictly. Include relevant legal provisions, 
procedures, and practical guidance while emphasizing the importance of professional legal consultation. (keep it concise and guide user step by step, and if its citical issue help him mentaly as well like condemn)"""

        # Generate response using gemini_model (fixed variable name conflict)
        response = gemini_model.generate_content(rag_prompt)
        
        return response.text
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I apologize, but I encountered an error while generating a response. Please try again."

@app.on_event("startup")
async def startup_event():
    """Load FAISS index on startup"""
    index_path = r"D:\DS projects\PAKLEGALADVISOR\faiss_index.idx"
    load_faiss_index(index_path)

@app.get("/")
async def root():
    return {"message": "PakLegalAdvisor API is running"}

@app.post("/query", response_model=QueryResponse)
async def query_legal_advice(request: QueryRequest):
    """Process legal query and return advice"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        print(f"////////////{request.question}//////////////")
        # Search for relevant documents
        retrieved_chunks = search_similar_documents(request.question)
        
        if not retrieved_chunks:
            return QueryResponse(
                answer="I couldn't find relevant legal information for your query. Please try rephrasing your question or consult with a legal professional.",
                retrieved_chunks=[]
            )
        
        # Generate response
        answer = generate_response(request.question, retrieved_chunks)
        
        return QueryResponse(
            answer=answer,
            retrieved_chunks=retrieved_chunks
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "index_loaded": faiss_index is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)