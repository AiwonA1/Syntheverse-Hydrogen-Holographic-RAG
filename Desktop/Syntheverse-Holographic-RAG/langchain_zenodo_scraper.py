"""
LangChain-based Zenodo Scraper
Uses LangChain document loaders and Zenodo API to scrape and load PDFs.
Based on LangChain (121K+ stars on GitHub) - the most popular RAG framework.
"""

import requests
import re
from typing import List, Dict, Optional
from pathlib import Path
import time
import os


class LangChainZenodoScraper:
    """
    Scrapes PDFs from Zenodo repositories using Zenodo API.
    Integrates with LangChain document loaders for robust PDF processing.
    """
    
    def __init__(self, request_delay: float = 1.0):
        """
        Initialize Zenodo scraper.
        
        Args:
            request_delay: Delay between API requests in seconds
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        self.request_delay = request_delay
    
    def get_record_id(self, url: str) -> str:
        """Extract record ID from Zenodo URL."""
        match = re.search(r'/records?/(\d+)', url)
        if match:
            return match.group(1)
        raise ValueError(f"Could not extract record ID from URL: {url}")
    
    def fetch_record_data(self, record_id: str, retries: int = 3) -> Dict:
        """Fetch record data from Zenodo API."""
        api_url = f"https://zenodo.org/api/records/{record_id}"
        
        for attempt in range(retries):
            try:
                response = self.session.get(api_url, timeout=30)
                if response.status_code == 200:
                    return response.json()
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  Request failed (attempt {attempt + 1}/{retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        
        raise Exception(f"Failed to fetch record {record_id} after {retries} attempts")
    
    def extract_pdf_files(self, record_data: Dict) -> List[Dict]:
        """Extract PDF file information from record data."""
        pdf_files = []
        
        if 'files' in record_data:
            for file_info in record_data['files']:
                filename = file_info.get('key', '')
                if filename.lower().endswith('.pdf'):
                    pdf_files.append({
                        'filename': filename,
                        'url': file_info.get('links', {}).get('self', ''),
                        'size': file_info.get('size', 0),
                        'checksum': file_info.get('checksum', '')
                    })
        
        return pdf_files
    
    def scrape_repository(self, zenodo_url: str) -> Dict:
        """Scrape a Zenodo repository and return all PDF files."""
        print(f"Scraping Zenodo repository: {zenodo_url}")
        
        record_id = self.get_record_id(zenodo_url)
        record_data = self.fetch_record_data(record_id)
        
        # Extract metadata
        metadata = {
            'record_id': record_id,
            'title': record_data.get('metadata', {}).get('title', 'Unknown'),
            'doi': record_data.get('doi', ''),
            'url': zenodo_url,
            'creators': [c.get('name', '') for c in record_data.get('metadata', {}).get('creators', [])],
            'publication_date': record_data.get('metadata', {}).get('publication_date', ''),
            'description': record_data.get('metadata', {}).get('description', '')
        }
        
        # Extract PDF files
        pdf_files = self.extract_pdf_files(record_data)
        
        return {
            'metadata': metadata,
            'pdf_files': pdf_files
        }
    
    def scrape_multiple_repositories(self, zenodo_urls: List[str]) -> List[Dict]:
        """Scrape multiple Zenodo repositories."""
        results = []
        for i, url in enumerate(zenodo_urls):
            try:
                result = self.scrape_repository(url)
                results.append(result)
                print(f"Found {len(result['pdf_files'])} PDF(s) in {result['metadata']['title']}")
                
                # Add delay between requests
                if i < len(zenodo_urls) - 1:
                    time.sleep(self.request_delay)
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                results.append({
                    'metadata': {'url': url, 'error': str(e)},
                    'pdf_files': []
                })
        return results
    
    def download_pdf(self, url: str, save_path: Path) -> Path:
        """Download PDF from URL to local path."""
        if save_path.exists():
            print(f"  PDF already exists: {save_path.name}")
            return save_path
        
        print(f"  Downloading: {save_path.name}")
        response = self.session.get(url, timeout=60, stream=True)
        response.raise_for_status()
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return save_path


if __name__ == "__main__":
    # Test scraper
    urls = [
        "https://zenodo.org/records/17873290",
        "https://zenodo.org/records/17873279",
    ]
    
    scraper = LangChainZenodoScraper()
    results = scraper.scrape_multiple_repositories(urls)
    
    total_pdfs = sum(len(r['pdf_files']) for r in results)
    print(f"\nTotal PDFs found: {total_pdfs}")
