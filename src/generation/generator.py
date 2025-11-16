import logging
from typing import List, Dict, Any
import time

from src.core.config import get_settings
from src.core.models import VectorSearchResult
from src.generation.augmenter import ContextAugmenter

logger = logging.getLogger(__name__)


class HuggingFaceLLM:
    """HuggingFace Free Inference API"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.model_name = self.settings.llm.model_name
        self.temperature = self.settings.llm.temperature
        self.max_tokens = self.settings.llm.max_tokens
        
        # Import HuggingFace
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                device=-1  # CPU
            )
            logger.info(f"HuggingFace model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            max_length = kwargs.get('max_tokens', self.max_tokens)
            
            # Generate
            result = self.pipeline(
                prompt,
                max_length=max_length,
                temperature=kwargs.get('temperature', self.temperature),
                do_sample=True,
                num_return_sequences=1
            )
            
            return result[0]['generated_text'].strip()
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "I apologize, but I encountered an error generating the response."
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'provider': 'huggingface',
            'model': self.model_name,
            'device': 'cpu'
        }


class Generator:
    """Main Generator class"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.augmenter = ContextAugmenter()
        
        # Initialize HuggingFace LLM
        self.llm = HuggingFaceLLM(settings)
        logger.info("Generator initialized with HuggingFace")
    
    def generate_answer(self, query: str, search_results: List[VectorSearchResult], use_toon: bool = True) -> Dict[str, Any]:
        start_time = time.time()
        
        # Create prompt
        prompt_format = "toon" if use_toon else "text"
        prompt = self.augmenter.augment_query(query, search_results, format=prompt_format)
        
        prompt_tokens = len(prompt.split())
        logger.info(f"Generated prompt with ~{prompt_tokens} tokens")
        
        # Generate answer
        try:
            answer = self.llm.generate(prompt)
            answer_tokens = len(answer.split())
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = "I apologize, but I encountered an error generating the response."
            answer_tokens = 0
        
        # Format context
        contexts = self.augmenter.format_response_context(search_results)
        
        processing_time = time.time() - start_time
        
        return {
            'answer': answer,
            'contexts': contexts,
            'metadata': {
                'prompt_tokens': prompt_tokens,
                'answer_tokens': answer_tokens,
                'total_tokens': prompt_tokens + answer_tokens,
                'processing_time': processing_time,
                'num_contexts': len(contexts),
                'llm_provider': 'huggingface',
                'model': self.settings.llm.model_name,
                'use_toon': use_toon
            }
        }
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'llm_info': self.llm.get_info(),
            'augmenter': 'ContextAugmenter',
            'toon_support': True
        }


def create_generator(settings=None) -> Generator:
    return Generator(settings)