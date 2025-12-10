"""
Vectorize Parsed PDF Chunks
Reads parsed JSON files from the parsed directory, creates embeddings, and stores them in a vector database.
Skips duplicates by checking which PDFs are already processed.
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from langchain_vector_store_local import LangChainVectorStoreLocal


def vectorize_parsed_chunks(parsed_dir: str = "./parsed",
                           output_dir: str = "./vectorized",
                           collection_name: str = "syntheverse_rag",
                           batch_size: int = 100,
                           embedding_model: str = "all-MiniLM-L6-v2"):
    """
    Vectorize all parsed PDF chunks from the parsed directory.
    
    Args:
        parsed_dir: Directory containing parsed JSON files
        output_dir: Directory to save vector database and metadata
        collection_name: Name for the ChromaDB collection
        batch_size: Batch size for vectorization
        embedding_model: OpenAI embedding model to use
    """
    parsed_path = Path(parsed_dir)
    output_path = Path(output_dir)
    
    if not parsed_path.exists():
        raise ValueError(f"Parsed directory not found: {parsed_dir}")
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    chroma_db_path = output_path / "chroma_db"
    metadata_path = output_path / "metadata"
    metadata_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Vectorization Pipeline - Parsed Chunks to Embeddings")
    print("=" * 80)
    print(f"📁 Parsed directory: {parsed_path.absolute()}")
    print(f"📁 Output directory: {output_path.absolute()}")
    print(f"📁 ChromaDB path: {chroma_db_path.absolute()}")
    print(f"🤖 Using LOCAL embeddings (no API calls, free!): {embedding_model}")
    print()
    
    # Find all parsed JSON files
    print("🔍 Scanning for parsed JSON files...", flush=True)
    json_files = list(parsed_path.glob("*.json"))
    
    if not json_files:
        print(f"No parsed JSON files found in {parsed_dir}")
        return
    
    print(f"📚 Found {len(json_files)} parsed PDF file(s)")
    print()
    
    # Initialize vector store with local embeddings
    print("⚙️  Initializing vector store with local embeddings...", flush=True)
    print("  (First time will download model ~80MB, then cached locally)", flush=True)
    try:
        vector_store = LangChainVectorStoreLocal(
            collection_name=collection_name,
            persist_directory=str(chroma_db_path),
            embedding_model=embedding_model
        )
        print("✓ Vector store initialized")
    except Exception as e:
        print(f"⚠️  ERROR: Failed to initialize vector store: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get already processed PDFs
    print("🔍 Checking for already processed PDFs...", flush=True)
    processed_pdfs = vector_store.get_processed_pdfs()
    if processed_pdfs:
        print(f"Found {len(processed_pdfs)} already processed PDF(s), will skip duplicates")
    print()
    
    # Process JSON files
    print("🚀 Starting vectorization...")
    print()
    
    total_processed = 0
    total_skipped = 0
    total_chunks = 0
    processing_stats = []
    
    for i, json_file in enumerate(json_files, 1):
        pdf_filename = json_file.stem + ".pdf"
        
        # Skip if already processed
        if pdf_filename in processed_pdfs:
            print(f"[{i}/{len(json_files)}] ⊘ Skipping (already vectorized): {pdf_filename}", flush=True)
            total_skipped += 1
            continue
        
        print(f"[{i}/{len(json_files)}] 📄 Processing: {pdf_filename}", flush=True)
        
        try:
            # Load parsed chunks
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            if not chunks:
                print(f"  ⚠️  Warning: No chunks in file", flush=True)
                continue
            
            # Vectorize chunks
            print(f"  🔄 Vectorizing {len(chunks)} chunks...", flush=True)
            vector_store.add_chunks(chunks, batch_size=batch_size)
            
            total_chunks += len(chunks)
            total_processed += 1
            
            # Save processing metadata
            processing_stats.append({
                'pdf_filename': pdf_filename,
                'json_file': str(json_file),
                'chunks_count': len(chunks),
                'status': 'success'
            })
            
            print(f"  ✓ Vectorized {len(chunks)} chunks", flush=True)
            
        except Exception as e:
            print(f"  ✗ Error processing {pdf_filename}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            
            processing_stats.append({
                'pdf_filename': pdf_filename,
                'json_file': str(json_file),
                'chunks_count': 0,
                'status': 'error',
                'error': str(e)
            })
            continue
    
    # Save processing metadata
    metadata_file = metadata_path / "vectorization_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_processed': total_processed,
            'total_skipped': total_skipped,
            'total_chunks': total_chunks,
            'collection_name': collection_name,
            'embedding_model': embedding_model,
            'processing_stats': processing_stats
        }, f, indent=2, ensure_ascii=False)
    
    # Get final statistics
    print()
    print("=" * 80)
    print("Vectorization Complete!")
    print("=" * 80)
    
    stats = vector_store.get_stats()
    print(f"Total chunks in vector store: {stats['total_chunks']}")
    print(f"Unique PDFs processed: {stats['unique_pdfs']}")
    print(f"Collection name: {stats['collection_name']}")
    print(f"Persist directory: {stats['persist_directory']}")
    print()
    print(f"New PDFs processed: {total_processed}")
    print(f"PDFs skipped (already processed): {total_skipped}")
    print(f"New chunks vectorized: {total_chunks}")
    print(f"Metadata saved to: {metadata_file}")
    print("=" * 80)
    
    print()
    print("✅ Vectorization complete! The vector database is ready for RAG queries.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vectorize parsed PDF chunks")
    parser.add_argument(
        '--parsed-dir',
        default="./parsed",
        help="Directory containing parsed JSON files (default: ./parsed)"
    )
    parser.add_argument(
        '--output-dir',
        default="./vectorized",
        help="Directory to save vector database and metadata (default: ./vectorized)"
    )
    parser.add_argument(
        '--collection',
        default="syntheverse_rag",
        help="ChromaDB collection name (default: syntheverse_rag)"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help="Batch size for vectorization (default: 100)"
    )
    parser.add_argument(
        '--embedding-model',
        default="all-MiniLM-L6-v2",
        help="HuggingFace embedding model (default: all-MiniLM-L6-v2). Options: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (better quality)"
    )
    
    args = parser.parse_args()
    
    vectorize_parsed_chunks(
        parsed_dir=args.parsed_dir,
        output_dir=args.output_dir,
        collection_name=args.collection,
        batch_size=args.batch_size,
        embedding_model=args.embedding_model
    )

