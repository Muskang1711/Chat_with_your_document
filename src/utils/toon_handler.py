import json
import re
from typing import Any, Dict, List, Union, Optional
from dataclasses import dataclass, asdict


@dataclass
class ChunkTOON:
    """TOON representation of document chunks"""
    chunk_id: str
    content: str
    page: int
    section: str
    token_count: int
    questions: List[str]  # Questions this chunk answers


class TOONHandler:
    """
    Handler for converting between JSON and TOON formats
    Optimized for RAG system data structures
    """
    
    @staticmethod
    def json_to_toon(data: Union[Dict, List]) -> str:
        """
        Convert JSON to TOON format
        Optimized for uniform arrays and nested structures
        """
        if isinstance(data, list):
            return TOONHandler._list_to_toon(data)
        elif isinstance(data, dict):
            return TOONHandler._dict_to_toon(data)
        else:
            return str(data)
    
    @staticmethod
    def _dict_to_toon(data: Dict, indent: int = 0) -> str:
        """Convert dictionary to TOON format with indentation"""
        lines = []
        indent_str = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                # Uniform array of objects - use tabular format
                fields = list(value[0].keys())
                header = f"{key}[{len(value)}]{{{','.join(fields)}}}:"
                lines.append(f"{indent_str}{header}")
                
                for item in value:
                    row_values = []
                    for field in fields:
                        val = item.get(field, '')
                        # Escape commas and quotes if needed
                        if isinstance(val, str) and (',' in val or '"' in val):
                            val = f'"{val}"'
                        row_values.append(str(val))
                    lines.append(f"{indent_str}  {','.join(row_values)}")
            
            elif isinstance(value, dict):
                # Nested dictionary
                lines.append(f"{indent_str}{key}:")
                lines.append(TOONHandler._dict_to_toon(value, indent + 1))
            
            elif isinstance(value, list):
                # Simple list
                lines.append(f"{indent_str}{key}[{len(value)}]:")
                for item in value:
                    lines.append(f"{indent_str}  {item}")
            
            else:
                # Primitive value
                lines.append(f"{indent_str}{key}: {value}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def _list_to_toon(data: List, indent: int = 0) -> str:
        """Convert list to TOON format"""
        if not data:
            return "[]"
        
        indent_str = "  " * indent
        
        # Check if it's a uniform array of objects
        if all(isinstance(item, dict) for item in data):
            fields = list(data[0].keys())
            lines = [f"[{len(data)}]{{{','.join(fields)}}}:"]
            
            for item in data:
                row_values = []
                for field in fields:
                    val = item.get(field, '')
                    if isinstance(val, str) and (',' in val or '"' in val):
                        val = f'"{val}"'
                    row_values.append(str(val))
                lines.append(f"{indent_str}  {','.join(row_values)}")
            
            return '\n'.join(lines)
        else:
            # Simple list
            lines = [f"[{len(data)}]:"]
            for item in data:
                lines.append(f"{indent_str}  {item}")
            return '\n'.join(lines)
    
    @staticmethod
    def toon_to_json(toon_str: str) -> Union[Dict, List]:
        """
        Parse TOON format back to JSON
        """
        lines = toon_str.strip().split('\n')
        return TOONHandler._parse_toon_lines(lines, 0)[0]
    
    @staticmethod
    def _parse_toon_lines(lines: List[str], start_idx: int = 0) -> tuple:
        """Parse TOON lines recursively"""
        result = {}
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            
            # Skip empty lines
            if not stripped:
                i += 1
                continue
            
            # Parse array header: name[count]{fields}:
            array_match = re.match(r'(\w+)\[(\d+)\]\{([^}]+)\}:', stripped)
            if array_match:
                name, count, fields = array_match.groups()
                count = int(count)
                fields = [f.strip() for f in fields.split(',')]
                
                items = []
                i += 1
                for _ in range(count):
                    if i < len(lines):
                        row = lines[i].strip()
                        values = TOONHandler._parse_csv_row(row)
                        item = dict(zip(fields, values))
                        items.append(item)
                        i += 1
                
                result[name] = items
                continue
            
            # Parse simple key-value
            if ':' in stripped:
                key, value = stripped.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if value:
                    result[key] = value
                else:
                    # Nested structure
                    i += 1
                    nested_result = {}
                    while i < len(lines):
                        next_line = lines[i]
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= indent:
                            break
                        # Parse nested content
                        i += 1
                    result[key] = nested_result
            
            i += 1
        
        return result, i
    
    @staticmethod
    def _parse_csv_row(row: str) -> List[str]:
        """Parse CSV-style row handling quoted values"""
        values = []
        current = []
        in_quotes = False
        
        for char in row:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                values.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        
        if current:
            values.append(''.join(current).strip())
        
        return values
    
    @staticmethod
    def chunks_to_toon(chunks: List[Dict]) -> str:
        """
        Convert document chunks to TOON format
        Optimized for RAG system
        """
        toon_chunks = []
        
        for chunk in chunks:
            toon_chunk = ChunkTOON(
                chunk_id=chunk.get('id', ''),
                content=chunk.get('content', ''),
                page=chunk.get('page', 0),
                section=chunk.get('section', ''),
                token_count=chunk.get('token_count', 0),
                questions=chunk.get('questions', [])
            )
            toon_chunks.append(asdict(toon_chunk))
        
        return TOONHandler.json_to_toon({'chunks': toon_chunks})
    
    @staticmethod
    def format_rag_response_toon(question: str, answer: str, 
                                 context: List[Dict]) -> str:
        """
        Format RAG response in TOON format
        Token-efficient for LLM consumption
        """
        response = {
            'question': question,
            'answer': answer,
            'context_count': len(context),
            'sources': []
        }
        
        for ctx in context:
            source = {
                'chunk_id': ctx.get('id', ''),
                'page': ctx.get('page', 0),
                'section': ctx.get('section', ''),
                'relevance_score': ctx.get('score', 0.0)
            }
            response['sources'].append(source)
        
        return TOONHandler.json_to_toon(response)
    
    @staticmethod
    def create_prompt_toon(query: str, contexts: List[Dict]) -> str:
        """
        Create LLM prompt in TOON format
        Minimizes tokens while maintaining structure
        """
        prompt_data = {
            'instruction': 'Answer based on provided context',
            'query': query,
            'contexts': []
        }
        
        for ctx in contexts:
            prompt_data['contexts'].append({
                'id': ctx.get('id', ''),
                'text': ctx.get('content', ''),
                'relevance': ctx.get('score', 0.0)
            })
        
        return TOONHandler.json_to_toon(prompt_data)


# Utility functions for quick access
def to_toon(data: Union[Dict, List]) -> str:
    """Quick conversion to TOON"""
    return TOONHandler.json_to_toon(data)


def from_toon(toon_str: str) -> Union[Dict, List]:
    """Quick conversion from TOON"""
    return TOONHandler.toon_to_json(toon_str)


def create_chunk_toon(chunk_id: str, content: str, page: int, 
                     section: str, questions: List[str]) -> str:
    """Create a single chunk in TOON format"""
    chunk = ChunkTOON(
        chunk_id=chunk_id,
        content=content,
        page=page,
        section=section,
        token_count=len(content.split()),
        questions=questions
    )
    return TOONHandler.json_to_toon(asdict(chunk))