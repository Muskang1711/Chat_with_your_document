import logging
from typing import List, Optional

from src.core.config import get_settings
from src.core.models import VectorSearchResult
from src.storage.vector_store import VectorStore
from src.embeddings.encoder import EmbeddingEncoder

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, vector_store: VectorStore, encoder: EmbeddingEncoder, settings=None):
        self.vector_store = vector_store
        self.encoder = encoder
        self.settings = settings or get_settings()
        self.top_k = self.settings.retrieval.top_k
        self.score_threshold = self.settings.retrieval.score_threshold
    
    def retrieve(self, query: str, top_k: Optional[int] = None, score_threshold: Optional[float] = None) -> List[VectorSearchResult]:
        top_k = top_k or self.top_k
        score_threshold = score_threshold or self.score_threshold
        
        logger.info(f"Retrieving chunks for query: '{query[:50]}...'")
        logger.debug(f"Parameters: top_k={top_k}, threshold={score_threshold}")
        
        # Encode query
        query_embedding = self.encoder.encode_query(query)
        logger.debug(f"Query embedding shape: {query_embedding.shape}")
        
        # Search with lower threshold for debugging
        raw_results = self.vector_store.search(query_embedding, top_k=top_k)
        
        logger.info(f"Raw results from vector store: {len(raw_results)}")
        
        # Log scores
        if raw_results:
            for i, (chunk, score) in enumerate(raw_results[:3]):
                logger.info(f"  Result {i+1}: score={score:.4f}, page={chunk.page}")
        
        # Filter by threshold (use lower threshold for testing)
        filtered_results = [
            (chunk, score) 
            for chunk, score in raw_results 
            if score >= 0.1  # Lower threshold for testing
        ]
        
        logger.info(f"After threshold filter: {len(filtered_results)} results")
        
        # Create VectorSearchResult objects
        search_results = []
        for rank, (chunk, score) in enumerate(filtered_results[:top_k], start=1):
            result = VectorSearchResult(
                chunk=chunk,
                score=score,
                rank=rank
            )
            search_results.append(result)
        
        logger.info(f"Retrieved {len(search_results)} chunks")
        return search_results