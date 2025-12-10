"""
Test Script: Scrape and Parse Zenodo PDFs (without vectorization)
This demonstrates the LangChain-based scraper and PDF processor.
"""

from langchain_zenodo_scraper import LangChainZenodoScraper
from langchain_pdf_processor import LangChainPDFProcessor
from pathlib import Path
import json

def test_scrape_and_parse():
    """Test scraping and parsing without vectorization."""
    
    # URLs to test
    urls = [
        "https://zenodo.org/records/17627952",
        "https://zenodo.org/records/17873290",
        "https://zenodo.org/records/17873279",
        "https://zenodo.org/records/17861907"
    ]
    
    print("=" * 80)
    print("LangChain Zenodo Scraper & PDF Processor Test")
    print("=" * 80)
    
    # Step 1: Scrape
    print("\n[Step 1] Scraping Zenodo repositories...")
    scraper = LangChainZenodoScraper()
    results = scraper.scrape_multiple_repositories(urls)
    
    # Count PDFs
    total_pdfs = sum(len(r['pdf_files']) for r in results)
    print(f"\n✓ Found {total_pdfs} total PDFs across {len(urls)} repositories")
    
    # Step 2: Test downloading and parsing one PDF
    print("\n[Step 2] Testing PDF download and parsing...")
    
    # Find first PDF
    first_pdf = None
    for repo_result in results:
        if repo_result.get('pdf_files'):
            first_pdf = repo_result['pdf_files'][0]
            repo_title = repo_result['metadata'].get('title', 'Unknown')
            break
    
    if first_pdf:
        pdf_url = first_pdf['url']
        pdf_filename = first_pdf['filename']
        
        print(f"\n  Testing with: {pdf_filename}")
        print(f"  From repository: {repo_title}")
        
        # Download PDF
        pdfs_dir = Path("./data/pdfs")
        pdfs_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdfs_dir / pdf_filename
        
        try:
            if not pdf_path.exists():
                print(f"  Downloading PDF...")
                scraper.download_pdf(pdf_url, pdf_path)
                print(f"  ✓ Downloaded: {pdf_path}")
            else:
                print(f"  ✓ PDF already exists: {pdf_path}")
            
            # Test parsing with LangChain
            print(f"  Parsing PDF with LangChain...")
            processor = LangChainPDFProcessor(chunk_size=1000, chunk_overlap=200)
            chunks = processor.process_pdf(
                str(pdf_path),
                metadata={
                    'source_repository': repo_title,
                    'pdf_url': pdf_url
                }
            )
            
            print(f"  ✓ Extracted {len(chunks)} chunks")
            if chunks:
                print(f"\n  Sample chunk (first 200 chars):")
                print(f"  {chunks[0]['text'][:200]}...")
                print(f"\n  Metadata: {chunks[0]['metadata']}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ⚠ No PDFs found to test")
    
    # Save results
    print("\n[Step 3] Saving scrape results...")
    metadata_dir = Path("./data/metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = metadata_dir / "test_scrape_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved to: {results_file}")
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Repositories scraped: {len(urls)}")
    print(f"  - Total PDFs found: {total_pdfs}")
    print(f"  - Test PDF processed: {'Yes' if first_pdf else 'No'}")
    print("\nNext steps:")
    print("  1. Set OPENAI_API_KEY in .env file")
    print("  2. Run: python langchain_rag_pipeline.py --urls <your_urls>")
    print("  3. Or use: python langchain_rag_pipeline.py --query-only (after processing)")

if __name__ == "__main__":
    test_scrape_and_parse()

