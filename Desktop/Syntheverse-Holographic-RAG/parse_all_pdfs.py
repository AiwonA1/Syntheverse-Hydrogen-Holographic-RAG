"""
Parse all PDFs from a directory and save parsed chunks to a parsed subdirectory.
Avoids duplicates by checking if parsed files already exist.
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from langchain_pdf_processor import LangChainPDFProcessor


def parse_all_pdfs(pdf_dir: str, output_dir: str = None, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Parse all PDFs in a directory and save parsed chunks to output directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save parsed JSON files (default: pdf_dir/../parsed)
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters to overlap between chunks
    """
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise ValueError(f"PDF directory not found: {pdf_dir}")
    
    # Set output directory
    if output_dir is None:
        output_dir = pdf_path.parent / "parsed"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PDF Parser - Processing All PDFs")
    print("=" * 80)
    print(f"📁 PDF directory: {pdf_path.absolute()}")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print()
    
    # Find all PDF files
    print("🔍 Scanning for PDF files...", flush=True)
    pdf_files = list(pdf_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF file(s)")
    print()
    
    # Initialize PDF processor
    print("⚙️  Initializing PDF processor...", flush=True)
    processor = LangChainPDFProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_recursive_splitter=True
    )
    print("✓ Processor initialized")
    print()
    
    total_processed = 0
    total_skipped = 0
    total_chunks = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        # Check if already parsed
        json_filename = pdf_file.stem + ".json"
        json_path = output_dir / json_filename
        
        if json_path.exists():
            # Load existing file to check if it's valid
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if existing_data and len(existing_data) > 0:
                    print(f"[{i}/{len(pdf_files)}] ⊘ Skipping (already parsed): {pdf_file.name}", flush=True)
                    total_skipped += 1
                    total_chunks += len(existing_data)
                    continue
            except:
                # If file is corrupted, re-parse it
                pass
        
        # Process PDF
        print(f"[{i}/{len(pdf_files)}] 📄 Processing: {pdf_file.name}", flush=True)
        try:
            chunks = processor.process_pdf(
                str(pdf_file),
                metadata={
                    'source': str(pdf_file),
                    'pdf_filename': pdf_file.name
                }
            )
            
            if not chunks:
                print(f"  ⚠️  Warning: No chunks extracted from {pdf_file.name}", flush=True)
                continue
            
            # Save to JSON file
            print(f"  💾 Saving {len(chunks)} chunks to {json_filename}...", flush=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Saved {len(chunks)} chunks to {json_filename}", flush=True)
            total_processed += 1
            total_chunks += len(chunks)
            
        except Exception as e:
            print(f"  ✗ Error processing {pdf_file.name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print()
    print("=" * 80)
    print("Parsing Complete!")
    print("=" * 80)
    print(f"Total PDFs processed: {total_processed}")
    print(f"Total PDFs skipped (already parsed): {total_skipped}")
    print(f"Total chunks extracted: {total_chunks}")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse all PDFs from a directory")
    parser.add_argument(
        '--pdf-dir',
        default="/Users/macbook/Desktop/Syntheverse-Holographic-RAG/pdfs",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help="Directory to save parsed JSON files (default: pdf_dir/../parsed)"
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help="Maximum characters per chunk (default: 1000)"
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help="Characters to overlap between chunks (default: 200)"
    )
    
    args = parser.parse_args()
    
    parse_all_pdfs(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

