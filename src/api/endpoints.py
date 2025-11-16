import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

from src.core.config import get_settings
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import QueryBasedChunker
from src.embeddings.encoder import EmbeddingEncoder
from src.storage.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Settings
settings = get_settings()

# FastAPI app
app = FastAPI(
    title="RAG Document Chat API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
vector_store = None
encoder = None
retriever = None
generator = None


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    context: str


@app.on_event("startup")
async def startup_event():
    global vector_store, encoder, retriever, generator
    
    print("\n" + "="*60)
    print("INITIALIZING RAG SYSTEM")
    print("="*60)
    
    try:
        # 1. Encoder
        print("\n[1/5] Loading embedding model...")
        encoder = EmbeddingEncoder(settings)
        print(f"✓ Embedding model loaded: {settings.embeddings.model_name}")
        
        # 2. Vector Store
        print("\n[2/5] Initializing vector store...")
        vector_store = VectorStore(settings)
        print(f"✓ Vector store initialized. Current chunks: {vector_store.count()}")
        
        # 3. Load PDF if empty
        if vector_store.count() == 0:
            print("\n[3/5] Loading and processing PDF...")
            pdf_path = settings.document.pdf_path
            
            # Load
            loader = DocumentLoader(settings)
            doc = loader.load_pdf(pdf_path)
            print(f"  • Loaded {doc.total_pages} pages")
            
            # Chunk
            chunker = QueryBasedChunker(settings)
            chunks = chunker.chunk_document(doc)
            print(f"  • Created {len(chunks)} chunks")
            
            # Embed
            chunks = encoder.encode_chunks(chunks, show_progress=True)
            print(f"  • Generated embeddings for {len(chunks)} chunks")
            
            # Store
            vector_store.add_chunks(chunks)
            print(f"  ✓ Added to vector store. Total: {vector_store.count()}")
        else:
            print("\n[3/5] PDF already loaded, skipping...")
        
        # 4. Retriever
        print("\n[4/5] Initializing retriever...")
        retriever = Retriever(vector_store, encoder, settings)
        print(f"✓ Retriever ready")
        
        # 5. Generator
        print("\n[5/5] Initializing LLM generator...")
        generator = Generator(settings)
        print(f"✓ Generator ready ({settings.llm.provider})")
        
        print("\n" + "="*60)
        print(f"RAG SYSTEM READY - {vector_store.count()} chunks indexed")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ STARTUP FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        raise


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "RAG Document Chat API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    if not all([vector_store, encoder, retriever, generator]):
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "status": "healthy",
        "chunks": vector_store.count() if vector_store else 0
    }


@app.get("/stats")
async def get_stats():
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    stats = vector_store.get_stats()
    return stats


@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    if not all([retriever, generator]):
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        question = request.question
        logger.info(f"Question: {question}")
        
        # Retrieve
        search_results = retriever.retrieve(question, top_k=3)
        
        if not search_results:
            return QuestionResponse(
                question=question,
                answer="I couldn't find relevant information in the document to answer this question.",
                context="No relevant context found in the vector store."
            )
        
        # Generate answer
        result = generator.generate_answer(question, search_results, use_toon=True)
        
        # Format context
        context_texts = []
        for sr in search_results:
            context_texts.append(f"[Page {sr.chunk.page}] {sr.chunk.content}")
        
        combined_context = "\n\n".join(context_texts)
        
        return QuestionResponse(
            question=question,
            answer=result["answer"],
            context=combined_context
        )
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))