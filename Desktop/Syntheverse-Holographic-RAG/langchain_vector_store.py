"""
LangChain-based Vector Store
Uses LangChain's ChromaDB integration for vector storage and retrieval.
Based on LangChain (121K+ stars on GitHub) - the most popular RAG framework.
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import List, Dict, Optional
from pathlib import Path
import os
import hashlib
import json


class LangChainVectorStore:
    """
    Vector store using LangChain's ChromaDB integration.
    Provides seamless integration with LangChain's RAG pipeline.
    """
    
    def __init__(self,
                 collection_name: str = "syntheverse_rag",
                 persist_directory: str = "./data/chroma_db",
                 embedding_model: str = "text-embedding-3-small",
                 openai_api_key: Optional[str] = None):
        """
        Initialize LangChain vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: OpenAI embedding model to use
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # Initialize OpenAI embeddings
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=api_key
        )
        
        # Initialize or load ChromaDB vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        print(f"LangChain vector store initialized: {collection_name}")
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
            'persist_directory': self.persist_directory
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


if __name__ == "__main__":
    # Test vector store
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in .env file")
    else:
        vector_store = LangChainVectorStore()
        stats = vector_store.get_stats()
        print(f"Vector store stats: {stats}")
