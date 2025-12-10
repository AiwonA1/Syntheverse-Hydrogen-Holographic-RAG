"""
LangChain-based PDF Processor
Uses LangChain's document loaders and text splitters for robust PDF processing.
Based on LangChain (121K+ stars on GitHub) - the most popular RAG framework.
"""

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Optional
from pathlib import Path
import os
import json


class LangChainPDFProcessor:
    """
    Processes PDFs using LangChain's document loaders and text splitters.
    Provides better text extraction and chunking than basic PDF parsers.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 use_recursive_splitter: bool = True):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            use_recursive_splitter: Use RecursiveCharacterTextSplitter (better for code/docs)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # LangChain's RecursiveCharacterTextSplitter is better for structured documents
        if use_recursive_splitter:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        else:
            self.text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
    
    def load_pdf(self, pdf_path: str, use_unstructured: bool = False) -> List[Document]:
        """
        Load PDF using LangChain document loader.
        
        Args:
            pdf_path: Path to PDF file
            use_unstructured: Use UnstructuredPDFLoader (requires unstructured library)
        
        Returns:
            List of LangChain Document objects
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Try PyPDFLoader first (simpler, no extra dependencies)
        # UnstructuredPDFLoader is more powerful but requires additional setup
        try:
            if use_unstructured:
                loader = UnstructuredPDFLoader(pdf_path)
            else:
                loader = PyPDFLoader(pdf_path)
            
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error loading PDF with LangChain: {e}")
            # Fallback to basic loader
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                return documents
            except Exception as e2:
                raise Exception(f"Failed to load PDF: {e2}")
    
    def load_pdf_from_url(self, url: str, temp_dir: Path = None) -> List[Document]:
        """
        Load PDF directly from URL using LangChain.
        
        Args:
            url: URL to PDF file
            temp_dir: Temporary directory to save PDF (optional)
        
        Returns:
            List of LangChain Document objects
        """
        import requests
        import tempfile
        
        # Download to temporary file
        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir())
        
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"temp_{hash(url)}.pdf"
        
        # Download PDF
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        try:
            # Load with LangChain
            documents = self.load_pdf(str(temp_path))
            return documents
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
    
    def chunk_documents(self, documents: List[Document], metadata: Dict = None) -> List[Document]:
        """
        Split documents into chunks using LangChain text splitter.
        
        Args:
            documents: List of LangChain Document objects
            metadata: Additional metadata to add to all chunks
        
        Returns:
            List of chunked Document objects
        """
        # Add metadata to all documents
        if metadata:
            for doc in documents:
                if doc.metadata:
                    doc.metadata.update(metadata)
                else:
                    doc.metadata = metadata.copy()
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        return chunks
    
    def process_pdf(self, pdf_path: str, metadata: Dict = None) -> List[Dict]:
        """
        Process PDF: load, chunk, and return as list of dictionaries.
        
        Args:
            pdf_path: Path to PDF file
            metadata: Metadata to attach to chunks
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        print(f"Processing PDF with LangChain: {os.path.basename(pdf_path)}")
        
        # Load PDF
        documents = self.load_pdf(pdf_path)
        
        if not documents:
            print(f"Warning: No content extracted from {pdf_path}")
            return []
        
        # Add PDF filename to metadata
        if metadata is None:
            metadata = {}
        metadata['pdf_filename'] = os.path.basename(pdf_path)
        metadata['source'] = pdf_path
        
        # Chunk documents
        chunks = self.chunk_documents(documents, metadata)
        
        # Convert to dictionary format
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dicts.append({
                'text': chunk.page_content,
                'metadata': chunk.metadata,
                'chunk_index': i
            })
        
        print(f"Extracted {len(chunk_dicts)} chunks from {os.path.basename(pdf_path)}")
        return chunk_dicts
    
    def process_pdf_from_url(self, url: str, metadata: Dict = None, save_path: Path = None) -> List[Dict]:
        """
        Process PDF from URL: download, load, chunk, and return chunks.
        
        Args:
            url: URL to PDF file
            metadata: Metadata to attach to chunks
            save_path: Optional path to save downloaded PDF
        
        Returns:
            List of chunk dictionaries
        """
        import requests
        import tempfile
        
        # Download PDF
        if save_path:
            pdf_path = save_path
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = Path(tempfile.gettempdir())
            pdf_path = temp_dir / f"temp_{hash(url)}.pdf"
        
        if not pdf_path.exists():
            print(f"Downloading PDF from URL...")
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        try:
            # Process PDF
            return self.process_pdf(str(pdf_path), metadata)
        finally:
            # Clean up temp file if we created it
            if save_path is None and pdf_path.exists():
                pdf_path.unlink()


if __name__ == "__main__":
    # Test processor
    processor = LangChainPDFProcessor(chunk_size=1000, chunk_overlap=200)
    
    # Example: process a PDF
    # chunks = processor.process_pdf("path/to/file.pdf", metadata={"source": "test"})
    # print(f"Created {len(chunks)} chunks")
