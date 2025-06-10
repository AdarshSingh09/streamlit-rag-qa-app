# Streamlit RAG QA App

A lightweight Retrieval-Augmented Generation (RAG) Question-Answering app built with Streamlit. It uses Gemini Flash 2.0 as the LLM, Recurssive Text Splitter from LangChain for document chunking, FAISS for vector storage, and sentence-transformers embeddings for semantic search.

## 🚀 Features

- 📄 Upload documents (PDFs or text)
- 🔍 Semantic chunking via LangChain's Recursive Text Splitter
- 🔗 Embeddings with `sentence-transformers/all-mpnet-base-v2`
- ⚡ Retrieval with FAISS
- 🧠 Answer generation using Gemini Flash 2.0
- 📝 Source attribution: See which document and chunk contributed to the answer
- 🖥️ Intuitive Streamlit UI

## 🧰 Tech Stack

- **Frontend:** Streamlit
- **LLM:** Gemini Flash 2.0 (via API)
- **Embeddings:** Hugging Face (`all-mpnet-base-v2`)
- **Chunking:** LangChain's RecursiveTextSplitter
- **Vector DB:** FAISS
- **RAG Framework:** LangChain
