"""
Vectorize Parsed Chunks
Vectorizes already-parsed PDF chunks (run this after fixing SQLite).
"""

import os
import json
from pathlib import Path
from vector_store import VectorStore
from config import StorageConfig
from dotenv import load_dotenv


def vectorize_parsed_chunks(collection_name: str = "syntheverse_rag",
                            batch_size: int = 100,
                            data_dir: str = None):
    """
    Vectorize already-parsed PDF chunks.
    
    Args:
        collection_name: Name for the vector store collection
        batch_size: Batch size for vectorization
        data_dir: Base directory for data storage (default: ./data)
    """
    print("=" * 80)
    print("Syntheverse RAG - Vectorization Pipeline")
    print("=" * 80)
    
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  ERROR: OPENAI_API_KEY not found!")
        print("Please set your OpenAI API key in .env file or environment variable.")
        print("Create .env file with: OPENAI_API_KEY=your_key_here")
        return
    
    # Initialize storage configuration
    storage_config = StorageConfig(base_dir=data_dir)
    parsed_dir = storage_config.parsed_dir
    
    if not parsed_dir.exists():
        print(f"\n⚠️  ERROR: Parsed directory not found: {parsed_dir}")
        print("Please run scrape_and_parse.py first to parse PDFs.")
        return
    
    # Find all parsed JSON files
    json_files = list(parsed_dir.glob("*.json"))
    
    if not json_files:
        print(f"\n⚠️  ERROR: No parsed JSON files found in {parsed_dir}")
        print("Please run scrape_and_parse.py first to parse PDFs.")
        return
    
    print(f"\nFound {len(json_files)} parsed PDF files")
    
    # Initialize vector store
    print("\n[Step 1/3] Initializing vector store...")
    try:
        vector_store = VectorStore(collection_name=collection_name, storage_config=storage_config)
    except Exception as e:
        print(f"\n⚠️  ERROR: Failed to initialize vector store: {e}")
        print("\nThis is likely due to SQLite version incompatibility.")
        print("ChromaDB requires SQLite 3.35.0 or higher.")
        print("See SETUP_ISSUES.md for solutions.")
        return
    
    # Get already processed PDFs
    processed_pdfs = vector_store.get_processed_pdfs()
    if processed_pdfs:
        print(f"Found {len(processed_pdfs)} already processed PDF(s), will skip duplicates")
    
    # Load and vectorize chunks
    print("\n[Step 2/3] Loading and vectorizing chunks...")
    total_chunks = 0
    skipped_files = 0
    
    for i, json_file in enumerate(json_files, 1):
        pdf_filename = json_file.stem + ".pdf"
        
        # Skip if already processed
        if pdf_filename in processed_pdfs:
            print(f"[{i}/{len(json_files)}] ⊘ Skipping already processed: {pdf_filename}")
            skipped_files += 1
            continue
        
        print(f"[{i}/{len(json_files)}] Processing: {pdf_filename}")
        
        try:
            # Load parsed chunks
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            if not chunks:
                print(f"  ⚠ No chunks in file")
                continue
            
            # Add to vector store
            vector_store.add_chunks(chunks, batch_size=batch_size)
            total_chunks += len(chunks)
            print(f"  ✓ Vectorized {len(chunks)} chunks")
            
        except Exception as e:
            print(f"  ✗ Error processing {pdf_filename}: {e}")
            continue
    
    # Display statistics
    print("\n[Step 3/3] Vectorization complete!")
    print("\n" + "=" * 80)
    stats = vector_store.get_stats()
    print(f"Total chunks in vector store: {stats['total_chunks']}")
    print(f"Unique PDFs processed: {stats['unique_pdfs']}")
    print(f"Collection name: {stats['collection_name']}")
    if skipped_files > 0:
        print(f"Files skipped (already processed): {skipped_files}")
    print("=" * 80)
    
    print("\n✅ Vectorization complete! You can now use the RAG system.")
    print("Run: python main.py --query-only")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vectorize parsed PDF chunks")
    parser.add_argument(
        '--collection',
        default="syntheverse_rag",
        help="Vector store collection name"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help="Batch size for vectorization"
    )
    parser.add_argument(
        '--data-dir',
        default=None,
        help="Base directory for data storage (default: ./data)"
    )
    
    args = parser.parse_args()
    
    vectorize_parsed_chunks(
        collection_name=args.collection,
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )


