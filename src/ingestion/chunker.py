import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib

from src.core.config import get_settings
from src.core.models import DocumentChunk
from src.ingestion.loader import LoadedDocument

logger = logging.getLogger(__name__)


@dataclass
class ChunkCandidate:
    """Temporary chunk during processing"""
    text: str
    start_page: int
    end_page: int
    section: Optional[str]
    potential_questions: List[str]


class QueryBasedChunker:
    """
    Advanced chunking that creates chunks based on semantic coherence
    and potential questions they can answer
    """
    
    def __init__(self, settings=None):
        """Initialize chunker with settings"""
        self.settings = settings or get_settings()
        self.chunk_size = self.settings.document.chunk_size
        self.chunk_overlap = self.settings.document.chunk_overlap
        
        # Question patterns for different content types
        self.question_patterns = {
            'definition': r'(?:what is|define|definition of|meaning of)',
            'explanation': r'(?:how does|explain|why|reason)',
            'comparison': r'(?:difference between|compare|versus|vs)',
            'process': r'(?:steps to|how to|procedure|process)',
            'example': r'(?:example|instance|case study)',
            'formula': r'(?:equation|formula|calculation)',
            'architecture': r'(?:architecture|structure|design|model)',
            'performance': r'(?:result|performance|accuracy|benchmark)',
        }
    
    def chunk_document(self, loaded_doc: LoadedDocument) -> List[DocumentChunk]:
        """
        Main chunking method - creates query-aware chunks
        
        Args:
            loaded_doc: Loaded document from DocumentLoader
            
        Returns:
            List of DocumentChunk objects with metadata
        """
        logger.info(f"Starting query-based chunking for {loaded_doc.filename}")
        
        # Step 1: Identify document sections
        sections = self._identify_sections(loaded_doc)
        
        # Step 2: Create semantic chunks
        chunk_candidates = []
        for section in sections:
            section_chunks = self._chunk_section(section, loaded_doc)
            chunk_candidates.extend(section_chunks)
        
        # Step 3: Generate questions for each chunk
        chunks_with_questions = self._generate_questions(chunk_candidates)
        
        # Step 4: Create final DocumentChunk objects
        final_chunks = self._finalize_chunks(chunks_with_questions, loaded_doc)
        
        logger.info(
            f"Created {len(final_chunks)} chunks from {loaded_doc.total_pages} pages"
        )
        
        return final_chunks
    
    def _identify_sections(self, loaded_doc: LoadedDocument) -> List[Dict]:
        """
        Identify logical sections in document
        Uses headers, spacing, and content patterns
        """
        sections = []
        current_section = {
            'title': 'Introduction',
            'start_page': 1,
            'end_page': 1,
            'content': []
        }
        
        for page in loaded_doc.pages:
            lines = page.text.split('\n')
            
            for line in lines:
                # Detect section headers (all caps, short lines, etc.)
                if self._is_section_header(line):
                    # Save previous section
                    if current_section['content']:
                        current_section['end_page'] = page.page_number
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        'title': line.strip(),
                        'start_page': page.page_number,
                        'end_page': page.page_number,
                        'content': []
                    }
                else:
                    current_section['content'].append(line)
                    current_section['end_page'] = page.page_number
        
        # Add last section
        if current_section['content']:
            sections.append(current_section)
        
        logger.info(f"Identified {len(sections)} sections in document")
        return sections
    
    def _is_section_header(self, line: str) -> bool:
        """Detect if line is a section header"""
        line = line.strip()
        
        if not line:
            return False
        
        # Common header patterns
        patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 Introduction"
            r'^[A-Z][A-Z\s]{3,}$',  # All caps short line
            r'^Abstract$',
            r'^Introduction$',
            r'^Conclusion$',
            r'^References$',
            r'^Appendix',
        ]
        
        return any(re.match(pattern, line) for pattern in patterns)
    
    def _chunk_section(self, section: Dict, 
                       loaded_doc: LoadedDocument) -> List[ChunkCandidate]:
        """
        Chunk a section into semantic units
        """
        content = '\n'.join(section['content'])
        chunks = []
        
        # Split into sentences
        sentences = self._split_into_sentences(content)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # Check if adding sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(ChunkCandidate(
                    text=chunk_text,
                    start_page=section['start_page'],
                    end_page=section['end_page'],
                    section=section['title'],
                    potential_questions=[]
                ))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, 
                    self.chunk_overlap
                )
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(ChunkCandidate(
                text=chunk_text,
                start_page=section['start_page'],
                end_page=section['end_page'],
                section=section['title'],
                potential_questions=[]
            ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences intelligently"""
        # Handle common abbreviations
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Fig|vs|etc|et al)\.',
                     r'\1<DOT>', text)
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _get_overlap_sentences(self, sentences: List[str], 
                               overlap_words: int) -> List[str]:
        """Get last N words worth of sentences for overlap"""
        overlap = []
        word_count = 0
        
        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= overlap_words:
                overlap.insert(0, sentence)
                word_count += sentence_words
            else:
                break
        
        return overlap
    
    def _generate_questions(self, 
                          candidates: List[ChunkCandidate]) -> List[ChunkCandidate]:
        """
        Generate potential questions each chunk can answer
        This is the KEY feature of query-based chunking
        """
        for candidate in candidates:
            questions = []
            text_lower = candidate.text.lower()
            
            # Pattern-based question generation
            if re.search(r'\b(is|are|was|were)\s+(?:a|an|the)?\s*\w+', text_lower):
                # Definition patterns
                questions.append(
                    f"What is mentioned in the {candidate.section} section?"
                )
            
            if re.search(r'\b(model|architecture|framework|system)\b', text_lower):
                questions.append(
                    f"What architecture/model is described in {candidate.section}?"
                )
            
            if re.search(r'\b(achieve|result|performance|accuracy)\b', text_lower):
                questions.append(
                    f"What results/performance are reported in {candidate.section}?"
                )
            
            if re.search(r'\b(equation|formula|calculate)\b', text_lower):
                questions.append(
                    f"What formulas/equations are presented in {candidate.section}?"
                )
            
            if re.search(r'\b(compare|comparison|versus|vs|better|worse)\b', text_lower):
                questions.append(
                    f"What comparisons are made in {candidate.section}?"
                )
            
            # Extract key terms for question generation
            key_terms = self._extract_key_terms(candidate.text)
            for term in key_terms[:2]:  # Top 2 terms
                questions.append(f"What is {term}?")
                questions.append(f"How does {term} work?")
            
            # Limit questions per chunk
            candidate.potential_questions = questions[:5]
        
        return candidates
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract important terms from text"""
        # Find capitalized phrases (likely important terms)
        terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Find technical terms (words with special chars or all caps)
        technical = re.findall(r'\b[A-Z]{2,}\b', text)
        
        all_terms = terms + technical
        
        # Count occurrences and return most frequent
        term_counts = {}
        for term in all_terms:
            term_counts[term] = term_counts.get(term, 0) + 1
        
        # Sort by frequency
        sorted_terms = sorted(term_counts.items(), 
                            key=lambda x: x[1], 
                            reverse=True)
        
        return [term for term, _ in sorted_terms[:5]]
    
    def _finalize_chunks(self, candidates: List[ChunkCandidate],
                        loaded_doc: LoadedDocument) -> List[DocumentChunk]:
        """Convert candidates to final DocumentChunk objects"""
        final_chunks = []
        
        for idx, candidate in enumerate(candidates):
            # Generate unique chunk ID
            chunk_id = self._generate_chunk_id(
                loaded_doc.filename, 
                idx, 
                candidate.text
            )
            
            # Count tokens (simple word-based)
            token_count = len(candidate.text.split())
            
            # Create DocumentChunk
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                content=candidate.text,
                page=candidate.start_page,
                section=candidate.section,
                token_count=token_count,
                questions=candidate.potential_questions,
                metadata={
                    'end_page': candidate.end_page,
                    'chunk_index': idx,
                    'source_document': loaded_doc.filename,
                    'chunking_strategy': 'query_based'
                }
            )
            
            final_chunks.append(chunk)
        
        return final_chunks
    
    def _generate_chunk_id(self, filename: str, index: int, text: str) -> str:
        """Generate unique chunk ID"""
        # Use hash of content for uniqueness
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        clean_filename = re.sub(r'[^a-zA-Z0-9]', '_', filename)
        return f"{clean_filename}_chunk_{index:04d}_{content_hash}"


# Utility function
def chunk_document(loaded_doc: LoadedDocument, settings=None) -> List[DocumentChunk]:
    """
    Quick utility to chunk a loaded document
    
    Args:
        loaded_doc: LoadedDocument object
        settings: Optional settings
        
    Returns:
        List of DocumentChunk objects
    """
    chunker = QueryBasedChunker(settings)
    return chunker.chunk_document(loaded_doc)