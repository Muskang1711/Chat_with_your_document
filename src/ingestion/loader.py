import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import PyPDF2

from src.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentPage:
    page_number: int
    text: str
    metadata: Dict[str, Any]


@dataclass
class LoadedDocument:
    filename: str
    total_pages: int
    pages: List[DocumentPage]
    metadata: Dict[str, Any]
    raw_content: str


class DocumentLoader:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
    
    def load_pdf(self, pdf_path: str) -> LoadedDocument:
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Loading PDF: {pdf_path}")
        
        pages = []
        with open(pdf_file, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                pages.append(DocumentPage(
                    page_number=i,
                    text=text,
                    metadata={}
                ))
        
        raw_content = "\n\n".join([p.text for p in pages])
        
        logger.info(f"Loaded {len(pages)} pages from {pdf_file.name}")
        
        return LoadedDocument(
            filename=pdf_file.name,
            total_pages=len(pages),
            pages=pages,
            metadata={'source': str(pdf_file)},
            raw_content=raw_content
        )


def load_document(pdf_path: str, settings=None) -> LoadedDocument:
    loader = DocumentLoader(settings)
    return loader.load_pdf(pdf_path)