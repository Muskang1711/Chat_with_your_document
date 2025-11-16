# 📚 RAG Document Chat System

A production-ready **Retrieval-Augmented Generation (RAG)** system for document Q&A, built with FastAPI, ChromaDB, and Gemini LLM. Features query-based chunking, TOON format for token efficiency, and a beautiful Streamlit UI.

## 🌟 Features

- **Query-Based Chunking**: Smart document chunking that identifies what questions each chunk can answer
- **TOON Format**: Token-efficient format for LLM communication (30-60% fewer tokens)
- **ChromaDB Integration**: Persistent vector storage with metadata filtering
- **Modular Architecture**: Clean separation of concerns with independent modules
- **FastAPI REST API**: Production-ready API with comprehensive endpoints
- **Streamlit UI**: Interactive web interface for document Q&A
- **Docker Support**: Fully containerized with multi-stage builds
- **LLM Flexibility**: Support for Gemini, OpenAI, and HuggingFace models

## 📋 Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional)
- Gemini API Key (get from [Google AI Studio](https://makersuite.google.com/app/apikey))

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-document-chat.git
cd rag-document-chat