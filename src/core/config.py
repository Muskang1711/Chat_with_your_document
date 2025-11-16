"""
Configuration Management Module
Handles all configuration loading and validation
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import toml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class DocumentConfig(BaseModel):
    """Document processing configuration"""
    pdf_path: str = Field(default="data/transformer_paper.pdf")
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    chunking_strategy: str = Field(default="query_based")
    
    @validator('chunking_strategy')
    def validate_strategy(cls, v):
        allowed = ["query_based", "recursive", "semantic"]
        if v not in allowed:
            raise ValueError(f"Strategy must be one of {allowed}")
        return v


class EmbeddingConfig(BaseModel):
    """Embedding model configuration"""
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    dimension: int = Field(default=384)
    batch_size: int = Field(default=32, gt=0)


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""
    type: str = Field(default="chromadb")
    persist_directory: str = Field(default="data/chromadb")
    collection_name: str = Field(default="document_chunks")


class LLMConfig(BaseModel):
    """LLM configuration"""
    provider: str = Field(default="gemini")
    model_name: str = Field(default="gemini-pro")
    temperature: float = Field(default=0.7, ge=0, le=1)
    max_tokens: int = Field(default=500, gt=0)


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    top_k: int = Field(default=3, gt=0)
    score_threshold: float = Field(default=0.7, ge=0, le=1)


class APIConfig(BaseModel):
    """API server configuration"""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, gt=0)
    reload: bool = Field(default=True)


class Settings(BaseSettings):
    """Main settings class combining all configurations"""
    
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    # Environment variables
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    
    # API Keys - Handle them separately
    gemini_api_key: Optional[str] = Field(default=None, env='GEMINI_API_KEY')
    openai_api_key: Optional[str] = Field(default=None, env='OPENAI_API_KEY')
    huggingface_api_key: Optional[str] = Field(default=None, env='HUGGINGFACE_API_KEY')
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Allow extra fields
    
    @classmethod
    def load_from_toml(cls, config_path: str = "config/settings.toml") -> "Settings":
        """Load configuration from TOML file"""
        config_file = Path(config_path)
        
        # Start with defaults
        settings_dict = {}
        
        if config_file.exists():
            try:
                config_data = toml.load(config_file)
                
                # Map TOML sections to Settings fields
                if "document" in config_data:
                    settings_dict["document"] = DocumentConfig(**config_data["document"])
                if "embeddings" in config_data:
                    settings_dict["embeddings"] = EmbeddingConfig(**config_data["embeddings"])
                if "vector_store" in config_data:
                    settings_dict["vector_store"] = VectorStoreConfig(**config_data["vector_store"])
                if "llm" in config_data:
                    settings_dict["llm"] = LLMConfig(**config_data["llm"])
                if "retrieval" in config_data:
                    settings_dict["retrieval"] = RetrievalConfig(**config_data["retrieval"])
                if "api" in config_data:
                    settings_dict["api"] = APIConfig(**config_data["api"])
                    
            except Exception as e:
                print(f"Error loading TOML config: {e}")
        
        # Load environment variables (this will override TOML settings)
        # Create Settings instance which will also load from .env file
        settings = cls(**settings_dict)
        
        # Get API keys from environment
        settings.gemini_api_key = os.getenv("GEMINI_API_KEY", settings.gemini_api_key)
        settings.openai_api_key = os.getenv("OPENAI_API_KEY", settings.openai_api_key)
        settings.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY", settings.huggingface_api_key)
        
        return settings
    
    def validate_paths(self) -> bool:
        """Validate that required paths exist"""
        pdf_path = Path(self.document.pdf_path)
        if not pdf_path.exists():
            print(f"Warning: PDF file {pdf_path} not found")
            return False
        
        # Create vector store directory if it doesn't exist
        vector_dir = Path(self.vector_store.persist_directory)
        vector_dir.mkdir(parents=True, exist_ok=True)
        
        return True
    
    def get_llm_api_key(self) -> Optional[str]:
        """Get API key for LLM provider"""
        if self.llm.provider == "gemini":
            return self.gemini_api_key
        elif self.llm.provider == "openai":
            return self.openai_api_key
        elif self.llm.provider == "huggingface":
            return self.huggingface_api_key
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "document": self.document.dict(),
            "embeddings": self.embeddings.dict(),
            "vector_store": self.vector_store.dict(),
            "llm": self.llm.dict(),
            "retrieval": self.retrieval.dict(),
            "api": self.api.dict(),
            "environment": self.environment,
            "debug": self.debug
        }


# Singleton instance
settings = Settings.load_from_toml()


# Utility function for easy access
def get_settings() -> Settings:
    """Get singleton settings instance"""
    return settings