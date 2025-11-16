from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from src.utils.toon_handler import TOONHandler


class DocumentChunk(BaseModel):
    """Model for document chunks with query-based metadata"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk text content")
    page: int = Field(..., description="Page number in document")
    section: Optional[str] = Field(None, description="Document section")
    token_count: int = Field(..., description="Number of tokens")
    questions: List[str] = Field(default_factory=list, 
                                 description="Questions this chunk answers")
    embedding: Optional[List[float]] = Field(None, exclude=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_toon(self) -> str:
        """Convert chunk to TOON format"""
        data = self.dict(exclude={'embedding'})
        return TOONHandler.json_to_toon(data)
    
    @classmethod
    def from_toon(cls, toon_str: str) -> "DocumentChunk":
        """Create chunk from TOON string"""
        data = TOONHandler.toon_to_json(toon_str)
        return cls(**data)


class QueryRequest(BaseModel):
    """API request model for queries"""
    question: str = Field(..., min_length=1, max_length=500)
    max_chunks: Optional[int] = Field(3, ge=1, le=10)
    include_metadata: Optional[bool] = Field(False)
    
    @validator('question')
    def clean_question(cls, v):
        return v.strip()


class ContextChunk(BaseModel):
    """Context chunk with relevance score"""
    chunk_id: str
    content: str
    page: int
    section: Optional[str]
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    questions: List[str] = Field(default_factory=list)
    
    def to_toon(self) -> str:
        """Convert to TOON format"""
        return TOONHandler.json_to_toon(self.dict())


class RAGResponse(BaseModel):
    """Response model for RAG queries"""
    question: str
    answer: str
    contexts: List[ContextChunk]
    total_tokens: Optional[int] = None
    processing_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def to_toon(self) -> str:
        """
        Convert entire response to TOON format
        Optimized for token efficiency
        """
        response_dict = {
            'question': self.question,
            'answer': self.answer,
            'context_count': len(self.contexts),
            'sources': [
                {
                    'chunk_id': ctx.chunk_id,
                    'page': ctx.page,
                    'section': ctx.section or '',
                    'score': round(ctx.relevance_score, 3)
                }
                for ctx in self.contexts
            ]
        }
        
        if self.total_tokens:
            response_dict['tokens'] = self.total_tokens
        if self.processing_time:
            response_dict['time_ms'] = round(self.processing_time * 1000, 2)
        
        return TOONHandler.json_to_toon(response_dict)
    
    def to_full_toon(self) -> str:
        """
        Full TOON with complete context content
        For detailed analysis
        """
        response_dict = {
            'question': self.question,
            'answer': self.answer,
            'contexts': [ctx.dict() for ctx in self.contexts],
            'metadata': {
                'tokens': self.total_tokens,
                'time_ms': round(self.processing_time * 1000, 2) if self.processing_time else None,
                'timestamp': self.timestamp.isoformat()
            }
        }
        return TOONHandler.json_to_toon(response_dict)


class ChunkingStrategy(BaseModel):
    """Configuration for chunking strategy"""
    type: str = Field("query_based", pattern="^(query_based|recursive|semantic)$")
    chunk_size: int = Field(500, gt=0)
    chunk_overlap: int = Field(50, ge=0)
    questions_per_chunk: int = Field(3, ge=1, le=5)
    

class VectorSearchResult(BaseModel):
    """Result from vector similarity search"""
    chunk: DocumentChunk
    score: float = Field(..., ge=0.0, le=1.0)
    rank: int = Field(..., ge=1)
    
    def to_context_chunk(self) -> ContextChunk:
        """Convert to context chunk for response"""
        return ContextChunk(
            chunk_id=self.chunk.chunk_id,
            content=self.chunk.content,
            page=self.chunk.page,
            section=self.chunk.section,
            relevance_score=self.score,
            questions=self.chunk.questions
        )


class DocumentMetadata(BaseModel):
    """Metadata for processed document"""
    filename: str
    total_pages: int
    total_chunks: int
    processing_time: float
    chunking_strategy: str
    average_chunk_size: float
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def to_toon(self) -> str:
        """Convert metadata to TOON"""
        return TOONHandler.json_to_toon(self.dict())