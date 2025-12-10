"""
Example: Using LangChain RAG Pipeline
Simple example demonstrating how to use the LangChain-based RAG system.
"""

from langchain_rag_pipeline import LangChainRAGPipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  ERROR: OPENAI_API_KEY not found!")
        print("Please set your OpenAI API key in .env file or environment variable.")
        return
    
    # Initialize pipeline
    print("Initializing LangChain RAG Pipeline...")
    pipeline = LangChainRAGPipeline(
        collection_name="syntheverse_rag",
        chunk_size=1000,
        chunk_overlap=200,
        data_dir="./data"
    )
    
    # Example 1: Run full pipeline
    print("\n" + "=" * 80)
    print("Example 1: Running Full Pipeline")
    print("=" * 80)
    
    zenodo_urls = [
        "https://zenodo.org/records/17873290",
        "https://zenodo.org/records/17873279"
    ]
    
    # Uncomment to run the full pipeline:
    # pipeline.run_full_pipeline(zenodo_urls)
    
    # Example 2: Query existing vector store
    print("\n" + "=" * 80)
    print("Example 2: Querying the RAG System")
    print("=" * 80)
    
    # Initialize vector store (if not already done)
    pipeline.initialize_vector_store()
    
    # Build QA chain
    pipeline.build_qa_chain()
    
    # Example queries
    questions = [
        "What is hydrogen holography?",
        "Explain the Syntheverse framework",
        "What are the key findings about fractal intelligence?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        try:
            result = pipeline.query(question)
            print(f"Answer: {result['answer']}")
            if 'sources' in result:
                print(f"Sources: {len(result['sources'])} documents")
        except Exception as e:
            print(f"Error: {e}")
    
    # Example 3: Get statistics
    print("\n" + "=" * 80)
    print("Example 3: Vector Store Statistics")
    print("=" * 80)
    
    stats = pipeline.vector_store.get_stats()
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Unique PDFs: {stats['unique_pdfs']}")
    print(f"Collection: {stats['collection_name']}")


if __name__ == "__main__":
    main()
