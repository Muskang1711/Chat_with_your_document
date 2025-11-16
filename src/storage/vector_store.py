import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss

from src.core.config import get_settings
from src.core.models import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.dimension = self.settings.embeddings.dimension
        self.persist_dir = Path(self.settings.vector_store.persist_directory)
        
        self.index = None
        self.chunks = []
        self.id_to_index = {}
        
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()
    
    def _initialize(self):
        index_file = self.persist_dir / "faiss.index"
        chunks_file = self.persist_dir / "chunks.pkl"
        
        if index_file.exists() and chunks_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(chunks_file, 'rb') as f:
                    data = pickle.load(f)
                    self.chunks = data['chunks']
                    self.id_to_index = data['id_to_index']
                logger.info(f"Loaded {len(self.chunks)} chunks from disk")
            except Exception as e:
                logger.warning(f"Could not load existing data: {e}")
                self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("Created new FAISS index")
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return
        
        valid_chunks = [c for c in chunks if c.embedding is not None]
        if not valid_chunks:
            logger.warning("No valid chunks with embeddings")
            return
        
        embeddings = np.array([c.embedding for c in valid_chunks]).astype('float32')
        faiss.normalize_L2(embeddings)
        
        start_idx = len(self.chunks)
        self.index.add(embeddings)
        
        for i, chunk in enumerate(valid_chunks):
            idx = start_idx + i
            self.chunks.append(chunk)
            self.id_to_index[chunk.chunk_id] = idx
        
        self._save()
        logger.info(f"Added {len(valid_chunks)} chunks. Total: {len(self.chunks)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks:
            logger.warning("No chunks in store")
            return []
        
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def search_by_text(self, query_text: str, encoder, top_k: int = 5):
        query_embedding = encoder.encode_query(query_text)
        return self.search(query_embedding, top_k)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        idx = self.id_to_index.get(chunk_id)
        if idx is not None and 0 <= idx < len(self.chunks):
            return self.chunks[idx]
        return None
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        return self.chunks.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_chunks': len(self.chunks),
            'dimension': self.dimension,
            'persist_directory': str(self.persist_dir),
            'embedding_model': self.settings.embeddings.model_name,
            'storage_type': 'FAISS'
        }
    
    def _save(self):
        try:
            faiss.write_index(self.index, str(self.persist_dir / "faiss.index"))
            with open(self.persist_dir / "chunks.pkl", 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'id_to_index': self.id_to_index
                }, f)
            logger.debug("FAISS index saved")
        except Exception as e:
            logger.error(f"Error saving: {e}")
    
    def persist(self):
        self._save()
        logger.info("Data persisted")
    
    def load(self):
        logger.info(f"Loaded {len(self.chunks)} chunks")
    
    def clear(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.id_to_index = {}
        self._save()
        logger.info("Store cleared")
    
    def count(self):
        return len(self.chunks)


def create_vector_store(chunks: List[DocumentChunk], settings=None) -> VectorStore:
    store = VectorStore(settings)
    store.add_chunks(chunks)
    return store


def load_existing_store(settings=None) -> VectorStore:
    return VectorStore(settings)