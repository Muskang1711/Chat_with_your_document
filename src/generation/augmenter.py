import logging
from typing import List, Dict, Any
from src.core.models import VectorSearchResult
from src.utils.toon_handler import TOONHandler

logger = logging.getLogger(__name__)


class ContextAugmenter:
    """
    Augments user queries with retrieved context
    Creates efficient prompts using TOON format
    """
    
    def __init__(self):
        """Initialize augmenter"""
        self.toon_handler = TOONHandler()
    
    def augment_query(self, query: str, 
                     search_results: List[VectorSearchResult],
                     format: str = "toon") -> str:
        """
        Create augmented prompt from query and context
        
        Args:
            query: User question
            search_results: Retrieved chunks
            format: Output format ("toon" or "text")
            
        Returns:
            Formatted prompt string
        """
        if not search_results:
            logger.warning("No search results to augment")
            return self._create_fallback_prompt(query)
        
        if format == "toon":
            return self._create_toon_prompt(query, search_results)
        else:
            return self._create_text_prompt(query, search_results)
    
    def _create_toon_prompt(self, query: str,
                           results: List[VectorSearchResult]) -> str:
        """
        Create TOON-formatted prompt (token efficient)
        """
        # Prepare context data
        contexts = []
        for result in results:
            context = {
                'id': result.chunk.chunk_id,
                'page': result.chunk.page,
                'section': result.chunk.section or 'N/A',
                'score': round(result.score, 3),
                'content': result.chunk.content[:500]  # Limit length
            }
            contexts.append(context)
        
        # Create prompt structure
        prompt_data = {
            'instruction': 'Answer the question using ONLY the provided context',
            'question': query,
            'context_sources': contexts
        }
        
        # Convert to TOON
        toon_context = self.toon_handler.json_to_toon(prompt_data)
        
        # Final prompt
        prompt = f"""You are a helpful AI assistant. Answer based on the context below.

{toon_context}

Requirements:
- Use ONLY information from the context
- Be concise and accurate
- If context doesn't contain the answer, say "I cannot answer based on the provided context"
- Cite the page number when possible

Answer:"""
        
        return prompt
    
    def _create_text_prompt(self, query: str,
                           results: List[VectorSearchResult]) -> str:
        """
        Create traditional text prompt
        """
        # Combine contexts
        context_parts = []
        for i, result in enumerate(results, 1):
            context_text = f"""
[Context {i}] (Page {result.chunk.page}, Relevance: {result.score:.2f})
Section: {result.chunk.section or 'N/A'}
---
{result.chunk.content}
---
"""
            context_parts.append(context_text)
        
        combined_context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{combined_context}

Question: {query}

Instructions:
- Answer using ONLY the information in the context above
- Be precise and cite page numbers when relevant
- If the context doesn't contain enough information, say so clearly

Answer:"""
        
        return prompt
    
    def _create_fallback_prompt(self, query: str) -> str:
        """Create prompt when no context available"""
        return f"""Question: {query}

I apologize, but I couldn't find relevant context in the document to answer this question.
Please try rephrasing your question or ask about a different topic covered in the document.
"""
    
    def create_system_message(self) -> str:
        """Create system message for chat-based models"""
        return """You are an AI assistant specialized in answering questions about technical documents.
Your role is to provide accurate, concise answers based solely on the provided context.
Always cite page numbers and sections when available.
If information is not in the context, clearly state that you cannot answer."""
    
    def format_response_context(self, search_results: List[VectorSearchResult]) -> List[Dict]:
        """
        Format search results for API response
        
        Args:
            search_results: Retrieved chunks
            
        Returns:
            List of formatted context dictionaries
        """
        formatted = []
        for result in search_results:
            ctx = {
                'chunk_id': result.chunk.chunk_id,
                'content': result.chunk.content,
                'page': result.chunk.page,
                'section': result.chunk.section,
                'relevance_score': round(result.score, 4),
                'rank': result.rank
            }
            formatted.append(ctx)
        
        return formatted