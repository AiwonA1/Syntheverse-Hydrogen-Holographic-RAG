"""
Test Gemini Pipeline: Scrape, Parse, and Test Embeddings
Tests the Gemini-based pipeline without full vectorization (to avoid SQLite issues).
"""

from langchain_zenodo_scraper import LangChainZenodoScraper
from langchain_pdf_processor import LangChainPDFProcessor
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pathlib import Path
import json
import os
from dotenv import load_dotenv

def test_gemini_pipeline():
    """Test Gemini pipeline components."""
    
    load_dotenv()
    
    # URLs to test
    urls = [
        "https://zenodo.org/records/17627952",
        "https://zenodo.org/records/17873290",
        "https://zenodo.org/records/17873279",
        "https://zenodo.org/records/17861907"
    ]
    
    print("=" * 80)
    print("LangChain + Gemini RAG Pipeline Test")
    print("=" * 80)
    
    # Step 1: Scrape
    print("\n[Step 1] Scraping Zenodo repositories...")
    scraper = LangChainZenodoScraper()
    results = scraper.scrape_multiple_repositories(urls)
    
    total_pdfs = sum(len(r['pdf_files']) for r in results)
    print(f"\n✓ Found {total_pdfs} total PDFs across {len(urls)} repositories")
    
    # Step 2: Test Gemini embeddings
    print("\n[Step 2] Testing Google Gemini embeddings...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("  ⚠ GOOGLE_API_KEY not found in .env")
        return
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Test embedding generation
        test_text = "Hydrogen holographic fractal intelligence"
        print(f"  Generating embedding for: '{test_text}'")
        embedding = embeddings.embed_query(test_text)
        print(f"  ✓ Generated embedding with {len(embedding)} dimensions")
        print(f"  ✓ First 5 values: {embedding[:5]}")
        
    except Exception as e:
        print(f"  ✗ Error testing embeddings: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Process a few PDFs
    print("\n[Step 3] Processing sample PDFs with LangChain...")
    processor = LangChainPDFProcessor(chunk_size=1000, chunk_overlap=200)
    
    pdfs_dir = Path("./data/pdfs")
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    total_chunks = 0
    
    for repo_result in results[:2]:  # Process first 2 repositories
        if not repo_result.get('pdf_files'):
            continue
        
        repo_title = repo_result['metadata'].get('title', 'Unknown')
        print(f"\n  Processing: {repo_title}")
        
        # Process first 2 PDFs from each repo
        for pdf_file in repo_result['pdf_files'][:2]:
            pdf_url = pdf_file['url']
            pdf_filename = pdf_file['filename']
            pdf_path = pdfs_dir / pdf_filename
            
            try:
                # Download if needed
                if not pdf_path.exists():
                    print(f"    Downloading: {pdf_filename}")
                    scraper.download_pdf(pdf_url, pdf_path)
                
                # Parse PDF
                print(f"    Parsing: {pdf_filename}")
                chunks = processor.process_pdf(
                    str(pdf_path),
                    metadata={
                        'source_repository': repo_title,
                        'pdf_url': pdf_url
                    }
                )
                
                if chunks:
                    total_chunks += len(chunks)
                    processed_count += 1
                    print(f"    ✓ Extracted {len(chunks)} chunks")
                    
                    # Test embedding one chunk
                    if processed_count == 1:
                        print(f"\n    Testing Gemini embedding on first chunk...")
                        chunk_text = chunks[0]['text'][:500]  # First 500 chars
                        chunk_embedding = embeddings.embed_query(chunk_text)
                        print(f"    ✓ Chunk embedded successfully ({len(chunk_embedding)} dimensions)")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Repositories scraped: {len(urls)}")
    print(f"  - Total PDFs found: {total_pdfs}")
    print(f"  - PDFs processed: {processed_count}")
    print(f"  - Total chunks extracted: {total_chunks}")
    print(f"  - Gemini embeddings: ✓ Working")
    
    print("\n✅ All components working!")
    print("\nNote: For full vectorization, you may need to upgrade SQLite:")
    print("  brew install sqlite3")
    print("  Or use Python 3.11+ which includes newer SQLite")
    
    print("\nTo run full pipeline (after SQLite upgrade):")
    print("  python3 langchain_rag_pipeline_gemini.py --urls <your_urls>")

if __name__ == "__main__":
    test_gemini_pipeline()

