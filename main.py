import os
# Fix ChromaDB-FastAPI conflict
os.environ["CHROMA_SERVER_IMPL"] = "chromadb.server.fastapi.FastAPI"

import uvicorn
from src.core.config import get_settings
from src.api.endpoints import app

def main():
    """Run the FastAPI application"""
    settings = get_settings()
    
    print("Starting RAG Document Chat API...")
    print(f"API URL: http://{settings.api.host}:{settings.api.port}")
    print(f"Docs URL: http://{settings.api.host}:{settings.api.port}/docs")
    
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload
    )

if __name__ == "__main__":
    main()