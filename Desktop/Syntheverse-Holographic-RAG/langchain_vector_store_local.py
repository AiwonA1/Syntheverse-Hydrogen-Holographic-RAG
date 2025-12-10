"""
LangChain-based Vector Store with Local Embeddings (Free!)
Uses LangChain's ChromaDB integration with sentence-transformers for free local embeddings.
Based on LangChain (121K+ stars on GitHub) - the most popular RAG framework.
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Dict, Optional
from pathlib import Path
import os
import hashlib
import json


class LangChainVectorStoreLocal:
    """
    Vector store using LangChain's ChromaDB integration with free local embeddings.
    Uses sentence-transformers - no API calls, no costs, no quota limits!
    """
    
    def __init__(self,
                 collection_name: str = "syntheverse_rag",
                 persist_directory: str = "./data/chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize LangChain vector store with local embeddings.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: HuggingFace embedding model name
                - "all-MiniLM-L6-v2" (default, fast, 384 dims)
                - "all-mpnet-base-v2" (slower, better quality, 768 dims)
                - "sentence-transformers/all-MiniLM-L6-v2" (explicit)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        print(f"Loading local embedding model: {embedding_model}")
        print("  (First time will download ~80MB, then cached locally)")
        
        # Initialize HuggingFace embeddings (free, local, no API needed!)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
            encode_kwargs={'normalize_embeddings': True}  # Better for cosine similarity
        )
        
        print(f"  ✓ Model loaded successfully")
        
        # Initialize or load ChromaDB vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        print(f"LangChain vector store initialized (FREE local embeddings): {collection_name}")
        print(f"Persist directory: {persist_directory}")
    
    def add_documents(self, documents: List[Document], batch_size: int = 100) -> List[str]:
        """
        Add LangChain Document objects to vector store.
        
        Args:
            documents: List of LangChain Document objects
            batch_size: Batch size for adding documents
        
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Add documents in batches
        all_ids = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = self.vectorstore.add_documents(batch)
            all_ids.extend(ids)
            print(f"  Added batch {i//batch_size + 1} ({len(batch)} documents)")
        
        # Persist to disk
        self.vectorstore.persist()
        
        return all_ids
    
    def add_chunks(self, chunks: List[Dict], batch_size: int = 100) -> List[str]:
        """
        Add chunk dictionaries to vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' keys
            batch_size: Batch size for adding chunks
        
        Returns:
            List of document IDs
        """
        # Convert chunk dicts to LangChain Documents
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk['text'],
                metadata=chunk.get('metadata', {})
            )
            documents.append(doc)
        
        return self.add_documents(documents, batch_size=batch_size)
    
    def similarity_search(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
        
        Returns:
            List of similar Document objects
        """
        if filter:
            return self.vectorstore.similarity_search(query, k=k, filter=filter)
        else:
            return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of (Document, score) tuples
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        collection = self.vectorstore._collection
        count = collection.count()
        
        # Get unique PDFs from metadata
        unique_pdfs = set()
        if count > 0:
            results = collection.get()
            if results and 'metadatas' in results:
                for metadata in results['metadatas']:
                    if metadata and 'pdf_filename' in metadata:
                        unique_pdfs.add(metadata['pdf_filename'])
        
        return {
            'total_chunks': count,
            'unique_pdfs': len(unique_pdfs),
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory,
            'embedding_model': self.embedding_model
        }
    
    def get_processed_pdfs(self) -> set:
        """Get set of PDF filenames that have been processed."""
        collection = self.vectorstore._collection
        unique_pdfs = set()
        
        if collection.count() > 0:
            results = collection.get()
            if results and 'metadatas' in results:
                for metadata in results['metadatas']:
                    if metadata and 'pdf_filename' in metadata:
                        unique_pdfs.add(metadata['pdf_filename'])
        
        return unique_pdfs
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.vectorstore.delete_collection()
        print(f"Deleted collection: {self.collection_name}")

