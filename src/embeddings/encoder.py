import logging
from typing import List, Optional, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.core.config import get_settings
from src.core.models import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """
    Handles embedding generation for text chunks
    Uses sentence-transformers for efficient encoding
    """
    
    def __init__(self, settings=None):
        """Initialize embedding encoder"""
        self.settings = settings or get_settings()
        self.model_name = self.settings.embeddings.model_name
        self.dimension = self.settings.embeddings.dimension
        self.batch_size = self.settings.embeddings.batch_size
        
        self.model = None
        self.device = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load embedding model"""
        try:
            # Detect device (GPU/CPU)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            
            # Load model
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Verify embedding dimension
            test_embedding = self.model.encode("test", show_progress_bar=False)
            actual_dim = len(test_embedding)
            
            if actual_dim != self.dimension:
                logger.warning(
                    f"Model dimension ({actual_dim}) differs from config ({self.dimension}). "
                    f"Updating config to match model."
                )
                self.dimension = actual_dim
            
            logger.info(
                f"Model loaded successfully. Embedding dimension: {self.dimension}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode single text into embedding vector
        
        Args:
            text: Input text
            normalize: Whether to normalize embedding (recommended for cosine similarity)
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise
    
    def encode_batch(self, texts: List[str], 
                    normalize: bool = True,
                    show_progress: bool = True) -> np.ndarray:
        """
        Encode multiple texts in batches (efficient)
        
        Args:
            texts: List of input texts
            normalize: Whether to normalize embeddings
            show_progress: Show progress bar
            
        Returns:
            Array of embeddings (shape: [num_texts, embedding_dim])
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            logger.info(f"Encoded {len(texts)} texts into embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
            raise
    
    def encode_chunks(self, chunks: List[DocumentChunk],
                     show_progress: bool = True) -> List[DocumentChunk]:
        """
        Encode document chunks and add embeddings to them
        
        Args:
            chunks: List of DocumentChunk objects
            show_progress: Show progress bar
            
        Returns:
            Chunks with embeddings added
        """
        logger.info(f"Encoding {len(chunks)} document chunks...")
        
        # Extract text content
        texts = [chunk.content for chunk in chunks]
        
        # Encode in batches
        embeddings = self.encode_batch(
            texts,
            normalize=True,
            show_progress=show_progress
        )
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()
        
        logger.info(f"Successfully encoded all chunks")
        return chunks
    
    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode search query
        
        Args:
            query: Search query text
            normalize: Whether to normalize
            
        Returns:
            Query embedding vector
        """
        return self.encode_text(query, normalize=normalize)
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0 to 1)
        """
        # Convert to numpy if needed
        if isinstance(embedding1, list):
            embedding1 = np.array(embedding1)
        if isinstance(embedding2, list):
            embedding2 = np.array(embedding2)
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        return float(similarity)
    
    def compute_similarities(self, query_embedding: np.ndarray,
                           chunk_embeddings: List[np.ndarray]) -> List[float]:
        """
        Compute similarities between query and multiple chunks
        
        Args:
            query_embedding: Query vector
            chunk_embeddings: List of chunk vectors
            
        Returns:
            List of similarity scores
        """
        # Convert to numpy array
        chunk_embeddings = np.array(chunk_embeddings)
        
        # Batch similarity computation
        similarities = np.dot(chunk_embeddings, query_embedding)
        
        return similarities.tolist()
    
    def get_model_info(self) -> dict:
        """Get information about loaded model"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.dimension,
            'device': self.device,
            'batch_size': self.batch_size,
            'max_sequence_length': self.model.max_seq_length
        }


# Utility functions
def encode_text(text: str, settings=None) -> np.ndarray:
    """Quick utility to encode single text"""
    encoder = EmbeddingEncoder(settings)
    return encoder.encode_text(text)


def encode_chunks(chunks: List[DocumentChunk], settings=None) -> List[DocumentChunk]:
    """Quick utility to encode chunks"""
    encoder = EmbeddingEncoder(settings)
    return encoder.encode_chunks(chunks)