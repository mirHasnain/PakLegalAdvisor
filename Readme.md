# 🏛️ PakLegalAdvisor – AI-Powered Legal Chatbot for Pakistani Law

## ⚖️ Overview

**PakLegalAdvisor** is an AI chatbot that provides general guidance on **Pakistani legal matters** using a powerful **Retrieval-Augmented Generation (RAG)** architecture.

This system combines:
- **FAISS** for fast semantic search,
- **SentenceTransformers** (`all-MiniLM-L6-v2`) for query/document embeddings,
- **Google Gemma (via Generative AI API)** for generating legally relevant responses,
- A **FastAPI** backend for handling RAG and model logic,
- A clean **Flask frontend** to simulate an intelligent chat experience.

---

## 🧠 How It Works

1. User types a legal question into the **Flask web interface**.
2. The question is sent to the **FastAPI server**, where:
   - It's converted into embeddings.
   - FAISS performs a similarity search on preprocessed legal document chunks.
   - A custom prompt is built and passed to **Gemma (Google Generative AI)**.
3. The AI response is returned and displayed with markdown styling (e.g., headings, bold).

---

## 🚀 Tech Stack

- **Frontend**: Flask (HTML, CSS, JS)
- **Backend**: FastAPI
- **Embedding Model**: `sentence-transformers` (all-MiniLM-L6-v2)
- **Search**: FAISS (`faiss_index.idx`)
- **LLM**: Google Gemma (via `google-generativeai`)
- **Others**: Pickle, NumPy, Logging, Uvicorn, CORS

---

## How to run?
1. get your google API key, put in main.py
2. install all requirements
3. run main.py and app.py on different terminals.
