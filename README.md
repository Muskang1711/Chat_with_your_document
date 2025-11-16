# **README.md**

```markdown
# 📚 RAG Document Chat System

A production-ready Retrieval-Augmented Generation (RAG) system for intelligent document Q&A. Built with FastAPI, FAISS vector store, HuggingFace models, and designed specifically for the "Attention Is All You Need" paper.

## 🌟 Features

- **Query-Based Chunking**: Smart document segmentation based on semantic content
- **FAISS Vector Store**: Fast and efficient similarity search
- **HuggingFace Integration**: Free inference API for text generation
- **TOON Format Support**: Token-efficient data serialization for LLM prompts
- **FastAPI Backend**: Production-ready REST API
- **Streamlit UI**: Interactive web interface (Bonus)
- **Docker Support**: Fully containerized deployment

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Running the FastAPI Server](#running-the-fastapi-server)
- [Docker Deployment](#docker-deployment)
- [Running the Streamlit UI](#running-the-streamlit-ui-bonus)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+** (Python 3.10, 3.11, or 3.12 recommended)
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **Docker & Docker Compose** (optional, for containerized deployment)
- **CUDA** (optional, for GPU acceleration)

## 📁 Project Structure

```
rag-document-chat/
│
├── src/
│   ├── core/
│   │   ├── config.py              # Configuration management
│   │   └── models.py              # Pydantic data models
│   ├── utils/
│   │   └── toon_handler.py        # TOON format handler
│   ├── ingestion/
│   │   ├── loader.py              # PDF document loader
│   │   └── chunker.py             # Query-based chunking
│   ├── embeddings/
│   │   └── encoder.py             # Embedding generation
│   ├── storage/
│   │   └── vector_store.py        # FAISS vector database
│   ├── retrieval/
│   │   └── retriever.py           # Semantic search
│   ├── generation/
│   │   ├── augmenter.py           # Context augmentation
│   │   └── generator.py           # LLM integration
│   └── api/
│       └── endpoints.py           # FastAPI routes
│
├── config/
│   └── settings.toml              # Configuration file
│
├── data/
│   ├── transformer_paper.pdf      # Source document
│   └── vector_store/              # FAISS index storage
│
├── logs/                          # Application logs
│
├── main.py                        # FastAPI entry point
├── app_streamlit.py               # Streamlit UI (bonus)
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose setup
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 🚀 Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/rag-document-chat.git
cd rag-document-chat
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Installation time:** ~2-5 minutes (depending on your internet speed)

### Step 4: Download the Transformer Paper

Place the "Attention Is All You Need" paper in the `data/` folder:

```bash
# Create data directory
mkdir -p data

# Download the paper
wget https://arxiv.org/pdf/1706.03762.pdf -O data/transformer_paper.pdf

# Or use curl
curl -o data/transformer_paper.pdf https://arxiv.org/pdf/1706.03762.pdf
```

### Step 5: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file
nano .env  # or use your preferred editor
```

**.env File:**
```env
# HuggingFace API Token (Optional but recommended)
# Get from: https://huggingface.co/settings/tokens
HUGGINGFACE_API_KEY=your_huggingface_token_here

# Environment
ENVIRONMENT=development
DEBUG=true
```

**Note:** HuggingFace API works without a token but has rate limits. For better performance, create a free account and get a token.

### Step 6: Verify Installation

```bash
python -c "import fastapi, faiss, sentence_transformers; print('✓ All dependencies installed successfully!')"
```

---

## 🖥️ Running the FastAPI Server

### Method 1: Using Python directly

```bash
python main.py
```

### Method 2: Using Uvicorn

```bash
uvicorn src.api.endpoints:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
============================================================
INITIALIZING RAG SYSTEM
============================================================

[1/5] Loading embedding model...
✓ Embedding model loaded: sentence-transformers/all-MiniLM-L6-v2

[2/5] Initializing vector store...
✓ Vector store initialized. Current chunks: 0

[3/5] Loading and processing PDF...
  • Loaded 15 pages
  • Created 87 chunks
  • Generated embeddings for 87 chunks
  ✓ Added to vector store. Total: 87

[4/5] Initializing retriever...
✓ Retriever ready

[5/5] Initializing LLM generator...
✓ Generator ready (huggingface)

============================================================
RAG SYSTEM READY - 87 chunks indexed
============================================================

INFO:     Uvicorn running on http://0.0.0.0:8000
```

**First-time startup:** ~30-60 seconds (downloads models and processes PDF)  
**Subsequent startups:** ~5-10 seconds (loads from cache)

### Access the API

- **API Base URL:** http://localhost:8000
- **Interactive Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health
- **Statistics:** http://localhost:8000/stats

### Test the API

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Transformer architecture?"}'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What is self-attention?"}
)

print(response.json())
```

**Expected Response:**
```json
{
  "question": "What is self-attention?",
  "answer": "Self-attention is a mechanism that allows the model to weigh the importance of different parts of the input sequence when processing each element...",
  "context": "[Page 3] An attention function can be described as mapping a query and a set of key-value pairs to an output..."
}
```

---

## 🐳 Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 1.29+

### Step 1: Build Docker Image

```bash
docker build -t rag-document-chat .
```

**Build time:** ~5-10 minutes (first time)

### Step 2: Run with Docker

**Option A: Docker Run (Simple)**
```bash
docker run -d \
  --name rag-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  rag-document-chat
```

**Option B: Docker Compose (Recommended)**
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Step 3: Verify Docker Deployment

```bash
# Check if container is running
docker ps

# Test the API
curl http://localhost:8000/health
```

### Docker Commands Reference

```bash
# View logs
docker logs -f rag-api

# Restart container
docker restart rag-api

# Stop container
docker stop rag-api

# Remove container
docker rm rag-api

# Rebuild image
docker-compose up -d --build
```

### Docker Compose Configuration

The `docker-compose.yml` includes:
- FastAPI service on port 8000
- Volume mounts for data persistence
- Environment variable configuration
- Automatic restart policy

---

## 🎨 Running the Streamlit UI (Bonus)

### Step 1: Install Streamlit (if not already installed)

```bash
pip install streamlit
```

### Step 2: Ensure FastAPI is Running

The Streamlit UI connects to the FastAPI backend, so make sure it's running first:

```bash
# In terminal 1
python main.py
```

### Step 3: Run Streamlit App

```bash
# In terminal 2
streamlit run app_streamlit.py
```

**Alternative with custom settings:**
```bash
streamlit run app_streamlit.py --server.port 8501 --server.address 0.0.0.0
```

### Step 4: Access the UI

Open your browser and navigate to:
- **Local:** http://localhost:8501
- **Network:** http://YOUR_IP:8501

### Streamlit Features

- ✅ Interactive question input
- ✅ Real-time answer generation
- ✅ Source context display
- ✅ System statistics
- ✅ Sample questions
- ✅ Processing metrics

### Using the Streamlit UI

1. **Enter your question** in the text area
2. **Click "Get Answer"** or press Enter
3. **View the answer** and source context
4. **Check metrics** (processing time, tokens used)
5. **Try sample questions** for quick testing

---

## 📖 API Documentation

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "chunks": 87
}
```

#### 2. Get Statistics
```http
GET /stats
```

**Response:**
```json
{
  "total_chunks": 87,
  "dimension": 384,
  "persist_directory": "data/vector_store",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "storage_type": "FAISS"
}
```

#### 3. Ask Question (Main Endpoint)
```http
POST /ask
Content-Type: application/json

{
  "question": "Your question here"
}
```

**Response:**
```json
{
  "question": "What is the Transformer architecture?",
  "answer": "Generated answer from the LLM",
  "context": "Relevant text chunks from the document"
}
```

### Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🧪 Testing

### Quick API Test

```bash
# Health check
curl http://localhost:8000/health

# Statistics
curl http://localhost:8000/stats

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is multi-head attention?"}'
```

### Test Script

Create a file `test_api.py`:

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Test questions
questions = [
    "What is the Transformer architecture?",
    "How does self-attention work?",
    "What are the advantages of the Transformer model?",
    "What datasets were used for evaluation?",
    "What are the performance results?"
]

print("Testing RAG API...\n")

for i, question in enumerate(questions, 1):
    print(f"{i}. Question: {question}")
    
    response = requests.post(
        f"{BASE_URL}/ask",
        json={"question": question}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Answer: {data['answer'][:100]}...")
        print(f"   Status: ✓ Success\n")
    else:
        print(f"   Status: ✗ Error {response.status_code}\n")
```

Run:
```bash
python test_api.py
```

---

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. **Module Not Found Errors**

```bash
# Error: ModuleNotFoundError: No module named 'fastapi'

# Solution: Reinstall dependencies
pip install -r requirements.txt
```

#### 2. **PDF File Not Found**

```bash
# Error: FileNotFoundError: PDF not found: data/transformer_paper.pdf

# Solution: Download the paper
wget https://arxiv.org/pdf/1706.03762.pdf -O data/transformer_paper.pdf
```

#### 3. **Port Already in Use**

```bash
# Error: [Errno 48] Address already in use

# Solution: Use a different port
uvicorn src.api.endpoints:app --port 8001

# Or kill the process using port 8000
# Linux/Mac:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

#### 4. **CUDA/GPU Issues**

```bash
# If you don't have GPU, embeddings will use CPU (slower but works)

# To force CPU usage, edit src/embeddings/encoder.py:
# Change: self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
# To: self.device = 'cpu'
```

#### 5. **HuggingFace API Rate Limits**

```bash
# Error: 429 Too Many Requests

# Solution: Add HuggingFace API token to .env
HUGGINGFACE_API_KEY=your_token_here

# Get token from: https://huggingface.co/settings/tokens
```

#### 6. **Empty Search Results**

```bash
# If you get "No relevant context found"

# Solution 1: Delete vector store and restart
rm -rf data/vector_store/*
python main.py

# Solution 2: Lower the score threshold in config/settings.toml
[retrieval]
score_threshold = 0.1  # Lower value = more results
```

#### 7. **Slow Startup**

```bash
# First startup is slow due to:
# - Model downloads (~300MB)
# - PDF processing
# - Embedding generation

# Subsequent startups are fast (~5-10 seconds)

# To speed up: Use Docker with cached models
```

#### 8. **Docker Build Fails**

```bash
# Error: failed to solve with frontend dockerfile.v0

# Solution: Update Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io

# Or use --no-cache flag
docker build --no-cache -t rag-document-chat .
```

---

## 📊 Performance Metrics

- **Startup Time:** 5-60 seconds (depending on cache)
- **Query Response Time:** 1-5 seconds
- **Vector Search:** <100ms
- **LLM Generation:** 1-3 seconds
- **Embedding Generation:** ~100 chunks/second (CPU)
- **Memory Usage:** ~2-4GB
- **Disk Space:** ~1GB (models + data)

---

## 🛠️ Configuration

Edit `config/settings.toml` to customize:

```toml
[document]
pdf_path = "data/transformer_paper.pdf"
chunk_size = 500                    # Adjust chunk size
chunk_overlap = 50                  # Overlap between chunks

[embeddings]
model_name = "sentence-transformers/all-MiniLM-L6-v2"
dimension = 384
batch_size = 32

[llm]
provider = "huggingface"
model_name = "google/flan-t5-base"  # Change model here
temperature = 0.7
max_tokens = 200

[retrieval]
top_k = 3                           # Number of chunks to retrieve
score_threshold = 0.1               # Minimum similarity score

[api]
host = "0.0.0.0"
port = 8000
reload = true
```

---

## 📝 Development

### Running in Development Mode

```bash
# With auto-reload
uvicorn src.api.endpoints:app --reload

# With debug logging
export DEBUG=true
python main.py
```

### Adding New Features

1. Create new module in `src/`
2. Update imports in relevant files
3. Test locally
4. Update `requirements.txt` if needed
5. Rebuild Docker image

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **FAISS:** Facebook AI Similarity Search
- **HuggingFace:** Transformers and Inference API
- **FastAPI:** Modern web framework
- **Sentence Transformers:** Efficient embeddings

---

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check troubleshooting section above
- Review API documentation at http://localhost:8000/docs

---

## 🎯 Quick Start Summary

```bash
# 1. Setup
git clone <repo-url>
cd rag-document-chat
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download PDF
wget https://arxiv.org/pdf/1706.03762.pdf -O data/transformer_paper.pdf

# 3. Configure
cp .env.example .env
# Add your HUGGINGFACE_API_KEY (optional)

# 4. Run
python main.py

# 5. Test
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Transformer?"}'

# 6. UI (Optional)
streamlit run app_streamlit.py
```

---

**Built with ❤️ for efficient document Q&A**
```

Save this as `README.md` in your project root. Professional, complete, aur easy to follow! 🚀