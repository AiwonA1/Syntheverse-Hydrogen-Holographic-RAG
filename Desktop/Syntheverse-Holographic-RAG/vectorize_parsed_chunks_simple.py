"""
Vectorize Parsed PDF Chunks (Simple File-Based Version)
Reads parsed JSON files, creates embeddings using local models, and saves to JSON files.
No ChromaDB required - works with any SQLite version.
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import hashlib


def vectorize_parsed_chunks(parsed_dir: str = "./parsed",
                           output_dir: str = "./vectorized",
                           embedding_model: str = "all-MiniLM-L6-v2",
                           batch_size: int = 100):
    """
    Vectorize all parsed PDF chunks and save embeddings to JSON files.
    
    Args:
        parsed_dir: Directory containing parsed JSON files
        output_dir: Directory to save vectorized embeddings
        embedding_model: HuggingFace embedding model to use
        batch_size: Batch size for embedding generation
    """
    parsed_path = Path(parsed_dir)
    output_path = Path(output_dir)
    
    if not parsed_path.exists():
        raise ValueError(f"Parsed directory not found: {parsed_dir}")
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_path / "embeddings"
    embeddings_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "metadata"
    metadata_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Vectorization Pipeline - Parsed Chunks to Embeddings (File-Based)")
    print("=" * 80)
    print(f"📁 Parsed directory: {parsed_path.absolute()}")
    print(f"📁 Output directory: {output_path.absolute()}")
    print(f"📁 Embeddings directory: {embeddings_path.absolute()}")
    print(f"🤖 Using LOCAL embeddings (no API calls, free!): {embedding_model}")
    print()
    
    # Initialize local embeddings
    print("⚙️  Loading local embedding model...", flush=True)
    print("  (First time will download model ~80MB, then cached locally)", flush=True)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠️  ERROR: Failed to load embedding model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Find all parsed JSON files
    print()
    print("🔍 Scanning for parsed JSON files...", flush=True)
    json_files = list(parsed_path.glob("*.json"))
    
    if not json_files:
        print(f"No parsed JSON files found in {parsed_dir}")
        return
    
    print(f"📚 Found {len(json_files)} parsed PDF file(s)")
    print()
    
    # Check for already processed files
    processed_files = set()
    if embeddings_path.exists():
        for emb_file in embeddings_path.glob("*.json"):
            processed_files.add(emb_file.stem)
    
    if processed_files:
        print(f"Found {len(processed_files)} already vectorized file(s), will skip duplicates")
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
        output_file = embeddings_path / f"{json_file.stem}.json"
        
        # Skip if already processed
        if json_file.stem in processed_files:
            print(f"[{i}/{len(json_files)}] ⊘ Skipping (already vectorized): {pdf_filename}", flush=True)
            total_skipped += 1
            # Count chunks from existing file
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        total_chunks += len(existing_data)
            except:
                pass
            continue
        
        print(f"[{i}/{len(json_files)}] 📄 Processing: {pdf_filename}", flush=True)
        
        try:
            # Load parsed chunks
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            if not chunks:
                print(f"  ⚠️  Warning: No chunks in file", flush=True)
                continue
            
            # Extract texts for embedding
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings in batches
            print(f"  🔄 Generating embeddings for {len(texts)} chunks...", flush=True)
            all_embeddings = []
            
            for batch_idx in range(0, len(texts), batch_size):
                batch_texts = texts[batch_idx:batch_idx + batch_size]
                batch_embeddings = embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                if (batch_idx // batch_size + 1) % 10 == 0:
                    print(f"    Processed {min(batch_idx + batch_size, len(texts))}/{len(texts)} chunks...", flush=True)
            
            # Combine chunks with embeddings
            vectorized_chunks = []
            for chunk, embedding in zip(chunks, all_embeddings):
                vectorized_chunk = {
                    'text': chunk['text'],
                    'embedding': embedding,
                    'metadata': chunk.get('metadata', {}),
                    'chunk_index': chunk.get('chunk_index', 0)
                }
                vectorized_chunks.append(vectorized_chunk)
            
            # Save vectorized chunks
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(vectorized_chunks, f, indent=2, ensure_ascii=False)
            
            total_chunks += len(vectorized_chunks)
            total_processed += 1
            
            # Save processing metadata
            processing_stats.append({
                'pdf_filename': pdf_filename,
                'json_file': str(json_file),
                'chunks_count': len(vectorized_chunks),
                'embedding_dim': len(embedding) if all_embeddings else 0,
                'status': 'success'
            })
            
            print(f"  ✓ Vectorized {len(vectorized_chunks)} chunks (dim: {len(embedding) if all_embeddings else 0})", flush=True)
            
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
            'embedding_model': embedding_model,
            'embedding_dim': len(all_embeddings[0]) if processing_stats and processing_stats[0].get('embedding_dim', 0) > 0 else 0,
            'processing_stats': processing_stats
        }, f, indent=2, ensure_ascii=False)
    
    # Summary
    print()
    print("=" * 80)
    print("Vectorization Complete!")
    print("=" * 80)
    print(f"New PDFs processed: {total_processed}")
    print(f"PDFs skipped (already processed): {total_skipped}")
    print(f"Total chunks vectorized: {total_chunks}")
    print(f"Embedding model: {embedding_model}")
    if processing_stats and processing_stats[0].get('embedding_dim', 0) > 0:
        print(f"Embedding dimension: {processing_stats[0]['embedding_dim']}")
    print(f"Embeddings saved to: {embeddings_path}")
    print(f"Metadata saved to: {metadata_file}")
    print("=" * 80)
    
    print()
    print("✅ Vectorization complete! Embeddings saved as JSON files.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vectorize parsed PDF chunks (file-based, no ChromaDB)")
    parser.add_argument(
        '--parsed-dir',
        default="./parsed",
        help="Directory containing parsed JSON files (default: ./parsed)"
    )
    parser.add_argument(
        '--output-dir',
        default="./vectorized",
        help="Directory to save vectorized embeddings (default: ./vectorized)"
    )
    parser.add_argument(
        '--embedding-model',
        default="all-MiniLM-L6-v2",
        help="HuggingFace embedding model (default: all-MiniLM-L6-v2). Options: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (better quality)"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help="Batch size for embedding generation (default: 100)"
    )
    
    args = parser.parse_args()
    
    vectorize_parsed_chunks(
        parsed_dir=args.parsed_dir,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size
    )

