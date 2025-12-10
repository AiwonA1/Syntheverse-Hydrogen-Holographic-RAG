"""
LangChain-based RAG Pipeline with FREE Local Embeddings
Complete pipeline using sentence-transformers - no API costs, no quota limits!
Based on LangChain (121K+ stars on GitHub) - the most popular RAG framework.
"""

import os
import json
from typing import List, Dict, Optional
from pathlib import Path
from langchain_zenodo_scraper import LangChainZenodoScraper
from langchain_pdf_processor import LangChainPDFProcessor
from langchain_vector_store_local import LangChainVectorStoreLocal
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


class LangChainRAGPipelineLocal:
    """
    Complete RAG pipeline using LangChain with FREE local embeddings.
    Uses sentence-transformers for embeddings (no API needed!).
    Optionally uses Gemini for LLM generation.
    """
    
    def __init__(self,
                 collection_name: str = "syntheverse_rag",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 data_dir: str = "./data",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 use_gemini: bool = True,
                 llm_model: str = "gemini-pro"):
        """
        Initialize RAG pipeline with free local embeddings.
        
        Args:
            collection_name: Name for vector store collection
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            data_dir: Base directory for data storage
            embedding_model: HuggingFace embedding model (free, local)
            use_gemini: Whether to use Gemini for LLM (requires API key)
            llm_model: Google Gemini LLM model (if use_gemini=True)
        """
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data_dir = Path(data_dir)
        self.embedding_model = embedding_model
        self.use_gemini = use_gemini
        self.llm_model = llm_model
        
        # Setup directories
        self.pdfs_dir = self.data_dir / "pdfs"
        self.vector_db_dir = self.data_dir / "chroma_db"
        self.metadata_dir = self.data_dir / "metadata"
        self.parsed_dir = self.data_dir / "parsed"
        
        for dir_path in [self.pdfs_dir, self.vector_db_dir, self.metadata_dir, self.parsed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.scraper = LangChainZenodoScraper()
        self.processor = LangChainPDFProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Vector store will be initialized when needed
        self.vector_store = None
        self.qa_chain = None
    
    def initialize_vector_store(self):
        """Initialize vector store with free local embeddings."""
        if self.vector_store is None:
            self.vector_store = LangChainVectorStoreLocal(
                collection_name=self.collection_name,
                persist_directory=str(self.vector_db_dir),
                embedding_model=self.embedding_model
            )
        return self.vector_store
    
    def scrape_zenodo_repositories(self, zenodo_urls: List[str]) -> List[Dict]:
        """Scrape Zenodo repositories for PDF files."""
        print("\n" + "=" * 80)
        print("Step 1: Scraping Zenodo Repositories")
        print("=" * 80)
        
        results = self.scraper.scrape_multiple_repositories(zenodo_urls)
        
        # Save scrape results
        metadata_path = self.metadata_dir / "zenodo_scrape_results.json"
        with open(metadata_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nScrape results saved to: {metadata_path}")
        
        total_pdfs = sum(len(r['pdf_files']) for r in results)
        print(f"Total PDFs found: {total_pdfs}")
        
        return results
    
    def process_pdfs(self, scrape_results: List[Dict], skip_existing: bool = True) -> int:
        """
        Download and process PDFs from scrape results.
        
        Args:
            scrape_results: Results from scrape_zenodo_repositories
            skip_existing: Skip PDFs that are already processed
        
        Returns:
            Total number of chunks created
        """
        print("\n" + "=" * 80)
        print("Step 2: Processing PDFs with LangChain")
        print("=" * 80)
        
        # Initialize vector store
        vector_store = self.initialize_vector_store()
        
        # Get already processed PDFs
        processed_pdfs = set()
        if skip_existing:
            processed_pdfs = vector_store.get_processed_pdfs()
            if processed_pdfs:
                print(f"Found {len(processed_pdfs)} already processed PDF(s), will skip duplicates")
        
        total_chunks = 0
        processed_urls = set()
        
        for repo_result in scrape_results:
            if 'error' in repo_result.get('metadata', {}):
                print(f"\n⚠ Skipping repository with error: {repo_result['metadata'].get('url', 'Unknown')}")
                continue
            
            repo_metadata = repo_result['metadata']
            pdf_files = repo_result['pdf_files']
            
            print(f"\nProcessing repository: {repo_metadata.get('title', 'Unknown')}")
            print(f"  PDFs in repository: {len(pdf_files)}")
            
            for pdf_file in pdf_files:
                pdf_url = pdf_file['url']
                pdf_filename = pdf_file['filename']
                
                # Skip if already processed
                if pdf_filename in processed_pdfs:
                    print(f"  ⊘ Skipping already processed: {pdf_filename}")
                    continue
                
                # Skip duplicate URLs
                if pdf_url in processed_urls:
                    print(f"  ⊘ Skipping duplicate URL: {pdf_filename}")
                    continue
                
                processed_urls.add(pdf_url)
                
                try:
                    # Download PDF
                    pdf_path = self.pdfs_dir / pdf_filename
                    if not pdf_path.exists():
                        self.scraper.download_pdf(pdf_url, pdf_path)
                    
                    # Process PDF with LangChain
                    chunks = self.processor.process_pdf(
                        str(pdf_path),
                        metadata={
                            'source_repository': repo_metadata.get('title', 'Unknown'),
                            'record_id': repo_metadata.get('record_id', ''),
                            'doi': repo_metadata.get('doi', ''),
                            'publication_date': repo_metadata.get('publication_date', ''),
                            'creators': ', '.join(repo_metadata.get('creators', [])),
                            'pdf_url': pdf_url
                        }
                    )
                    
                    if chunks:
                        # Add to vector store (using FREE local embeddings!)
                        vector_store.add_chunks(chunks, batch_size=100)
                        total_chunks += len(chunks)
                        processed_pdfs.add(pdf_filename)
                        print(f"  ✓ Added {len(chunks)} chunks from {pdf_filename}")
                    else:
                        print(f"  ⚠ No chunks extracted from {pdf_filename}")
                
                except Exception as e:
                    print(f"  ✗ Error processing {pdf_filename}: {e}")
                    continue
        
        return total_chunks
    
    def build_qa_chain(self, chain_type: str = "stuff"):
        """
        Build LangChain QA chain for querying.
        
        Args:
            chain_type: Type of chain ("stuff", "map_reduce", "refine", "map_rerank")
        """
        if self.vector_store is None:
            self.initialize_vector_store()
        
        # Initialize LLM
        if self.use_gemini:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("⚠ Warning: GOOGLE_API_KEY not found. Using simple retrieval only.")
                self.qa_chain = None
                return None
            
            llm = ChatGoogleGenerativeAI(
                model=self.llm_model,
                temperature=0.7,
                google_api_key=api_key
            )
        else:
            # Could use other LLMs here, or just retrieval
            print("⚠ No LLM configured. Using retrieval-only mode.")
            self.qa_chain = None
            return None
        
        # Create retriever
        retriever = self.vector_store.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True
        )
        
        return self.qa_chain
    
    def query(self, question: str, return_sources: bool = True) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question: Question to ask
            return_sources: Whether to return source documents
        
        Returns:
            Dictionary with answer and optionally sources
        """
        if self.qa_chain is None:
            self.build_qa_chain()
        
        if self.qa_chain is None:
            # Fallback to retrieval-only
            docs = self.vector_store.similarity_search(question, k=5)
            return {
                'answer': f"Found {len(docs)} relevant documents. (LLM not configured - showing retrieval only)",
                'question': question,
                'sources': [
                    {
                        'content': doc.page_content[:500] + "...",
                        'metadata': doc.metadata
                    }
                    for doc in docs
                ]
            }
        
        result = self.qa_chain({"query": question})
        
        response = {
            'answer': result['result'],
            'question': question
        }
        
        if return_sources and 'source_documents' in result:
            response['sources'] = [
                {
                    'content': doc.page_content[:500] + "...",
                    'metadata': doc.metadata
                }
                for doc in result['source_documents']
            ]
        
        return response
    
    def run_full_pipeline(self, zenodo_urls: List[str], skip_existing: bool = True):
        """
        Run the complete pipeline: scrape, process, and vectorize.
        
        Args:
            zenodo_urls: List of Zenodo repository URLs
            skip_existing: Skip already processed PDFs
        """
        print("\n" + "=" * 80)
        print("Syntheverse Hydrogen Holographic RAG Pipeline")
        print("LangChain + FREE Local Embeddings (sentence-transformers)")
        print("=" * 80)
        
        # Load environment variables
        load_dotenv()
        
        # Step 1: Scrape
        scrape_results = self.scrape_zenodo_repositories(zenodo_urls)
        
        # Step 2: Process PDFs
        total_chunks = self.process_pdfs(scrape_results, skip_existing=skip_existing)
        
        # Step 3: Display statistics
        print("\n" + "=" * 80)
        print("Pipeline Complete!")
        print("=" * 80)
        stats = self.vector_store.get_stats()
        print(f"Total chunks in vector store: {stats['total_chunks']}")
        print(f"Unique PDFs processed: {stats['unique_pdfs']}")
        print(f"Collection name: {stats['collection_name']}")
        print(f"Embedding model: {stats['embedding_model']} (FREE, local)")
        print("=" * 80)
        
        print("\n✅ Pipeline complete! You can now query the system.")
        print("💡 All embeddings generated locally - no API costs!")
        return self.vector_store


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LangChain RAG pipeline with FREE local embeddings")
    parser.add_argument('--urls', nargs='+', 
                       default=[
                           "https://zenodo.org/records/17627952",
                           "https://zenodo.org/records/17873290",
                           "https://zenodo.org/records/17873279",
                           "https://zenodo.org/records/17861907"
                       ],
                       help="Zenodo repository URLs")
    parser.add_argument('--collection', default="syntheverse_rag",
                       help="Vector store collection name")
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help="Chunk size")
    parser.add_argument('--chunk-overlap', type=int, default=200,
                       help="Chunk overlap")
    parser.add_argument('--data-dir', default="./data",
                       help="Data directory")
    parser.add_argument('--embedding-model', default="all-MiniLM-L6-v2",
                       help="HuggingFace embedding model (free, local)")
    parser.add_argument('--no-gemini', action='store_true',
                       help="Don't use Gemini LLM (retrieval only)")
    parser.add_argument('--query', action='store_true',
                       help="Start interactive query interface after processing")
    parser.add_argument('--query-only', action='store_true',
                       help="Skip processing and go straight to query interface")
    
    args = parser.parse_args()
    
    pipeline = LangChainRAGPipelineLocal(
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        data_dir=args.data_dir,
        embedding_model=args.embedding_model,
        use_gemini=not args.no_gemini
    )
    
    if args.query_only:
        # Query mode only
        pipeline.initialize_vector_store()
        pipeline.build_qa_chain()
        
        print("\n" + "=" * 80)
        print("Interactive Query Interface")
        print("=" * 80)
        print("Type 'exit' or 'quit' to stop\n")
        
        while True:
            question = input("Question: ").strip()
            if question.lower() in ['exit', 'quit']:
                break
            
            if question:
                result = pipeline.query(question)
                print(f"\nAnswer: {result['answer']}\n")
                if 'sources' in result:
                    print("Sources:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"  {i}. {source['metadata'].get('pdf_filename', 'Unknown')}")
                        print(f"     {source['content']}\n")
    else:
        # Run pipeline
        pipeline.run_full_pipeline(args.urls)
        
        if args.query:
            # Start query interface
            pipeline.build_qa_chain()
            
            print("\n" + "=" * 80)
            print("Interactive Query Interface")
            print("=" * 80)
            print("Type 'exit' or 'quit' to stop\n")
            
            while True:
                question = input("Question: ").strip()
                if question.lower() in ['exit', 'quit']:
                    break
                
                if question:
                    result = pipeline.query(question)
                    print(f"\nAnswer: {result['answer']}\n")
                    if 'sources' in result:
                        print("Sources:")
                        for i, source in enumerate(result['sources'], 1):
                            print(f"  {i}. {source['metadata'].get('pdf_filename', 'Unknown')}")
                            print(f"     {source['content']}\n")

