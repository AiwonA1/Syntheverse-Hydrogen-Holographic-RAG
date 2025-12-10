"""
Simple PDF Scraper for Zenodo Repositories
Scrapes all PDFs from Zenodo repositories and downloads them locally.
No duplicates - skips already downloaded files.
"""

import requests
import re
from typing import List, Dict, Set
from pathlib import Path
import time
import json
from urllib.parse import urlparse


class ZenodoPDFScraper:
    """Simple scraper for downloading PDFs from Zenodo repositories."""
    
    def __init__(self, download_dir: str = "./pdfs", request_delay: float = 1.0):
        """
        Initialize scraper.
        
        Args:
            download_dir: Directory to save PDFs
            request_delay: Delay between requests in seconds
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        
        # Track downloaded files to avoid duplicates
        self.downloaded_urls: Set[str] = set()
        self.downloaded_files: Set[str] = set()
    
    def get_record_id(self, url: str) -> str:
        """Extract record ID from Zenodo URL."""
        match = re.search(r'/records?/(\d+)', url)
        if match:
            return match.group(1)
        raise ValueError(f"Could not extract record ID from URL: {url}")
    
    def fetch_record_data(self, record_id: str) -> Dict:
        """Fetch record data from Zenodo API."""
        api_url = f"https://zenodo.org/api/records/{record_id}"
        
        response = self.session.get(api_url, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
    
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
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing spaces and dots
        filename = filename.strip(' .')
        # Limit length
        if len(filename) > 200:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:200-len(ext)-1] + '.' + ext if ext else name[:200]
        return filename
    
    def download_pdf(self, url: str, filename: str, file_num: int = None, total_files: int = None) -> bool:
        """
        Download PDF from URL with progress display.
        
        Returns:
            True if downloaded, False if skipped (duplicate or error)
        """
        # Check if URL already downloaded
        if url in self.downloaded_urls:
            return False
        
        # Sanitize filename
        safe_filename = self.sanitize_filename(filename)
        file_path = self.download_dir / safe_filename
        
        # Progress prefix
        if file_num is not None and total_files is not None:
            progress = f"[{file_num}/{total_files}]"
        else:
            progress = ""
        
        # Check if file already exists
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"{progress} ⊘ Already exists: {safe_filename} ({file_size:,} bytes)")
            self.downloaded_urls.add(url)
            self.downloaded_files.add(safe_filename)
            return False
        
        try:
            print(f"{progress} ↓ Downloading: {safe_filename}")
            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            # Get file size from headers if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Write file with progress
            downloaded = 0
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"     Progress: {percent:.1f}% ({mb_downloaded:.2f}MB/{mb_total:.2f}MB)", end='\r', flush=True)
            
            file_size = file_path.stat().st_size
            mb_size = file_size / (1024 * 1024)
            print(f"\n{progress} ✓ Downloaded: {safe_filename} ({mb_size:.2f}MB / {file_size:,} bytes)")
            
            self.downloaded_urls.add(url)
            self.downloaded_files.add(safe_filename)
            return True
            
        except Exception as e:
            print(f"\n{progress} ✗ Error downloading {safe_filename}: {e}")
            return False
    
    def scrape_repository(self, zenodo_url: str) -> Dict:
        """Scrape a Zenodo repository and return PDF file information."""
        print(f"\nScraping: {zenodo_url}")
        
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
        }
        
        # Extract PDF files
        pdf_files = self.extract_pdf_files(record_data)
        
        return {
            'metadata': metadata,
            'pdf_files': pdf_files
        }
    
    def scrape_and_download(self, zenodo_urls: List[str]) -> Dict:
        """
        Scrape multiple Zenodo repositories and download all PDFs.
        
        Returns:
            Dictionary with scrape results and download statistics
        """
        print("=" * 80)
        print("Zenodo PDF Scraper")
        print("=" * 80)
        print(f"📁 Download directory: {self.download_dir.absolute()}")
        print(f"📚 Repositories to scrape: {len(zenodo_urls)}")
        print()
        
        all_results = []
        total_found = 0
        total_downloaded = 0
        total_skipped = 0
        
        for i, url in enumerate(zenodo_urls):
            try:
                # Scrape repository
                result = self.scrape_repository(url)
                all_results.append(result)
                
                pdf_files = result['pdf_files']
                total_found += len(pdf_files)
                
                print(f"  Found {len(pdf_files)} PDF(s) in: {result['metadata']['title']}")
                
                # Download PDFs with progress
                for idx, pdf_file in enumerate(pdf_files, 1):
                    downloaded = self.download_pdf(
                        pdf_file['url'], 
                        pdf_file['filename'],
                        file_num=idx,
                        total_files=len(pdf_files)
                    )
                    if downloaded:
                        total_downloaded += 1
                    else:
                        total_skipped += 1
                    
                    # Rate limiting
                    time.sleep(self.request_delay)
                
                # Delay between repositories
                if i < len(zenodo_urls) - 1:
                    time.sleep(self.request_delay)
                    
            except Exception as e:
                print(f"  ✗ Error scraping {url}: {e}")
                all_results.append({
                    'metadata': {'url': url, 'error': str(e)},
                    'pdf_files': []
                })
        
        # Save results
        results_file = self.download_dir.parent / "scrape_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Summary
        print("\n" + "=" * 80)
        print("Scraping Complete!")
        print("=" * 80)
        print(f"Total PDFs found: {total_found}")
        print(f"Total downloaded: {total_downloaded}")
        print(f"Total skipped (duplicates/existing): {total_skipped}")
        print(f"Download directory: {self.download_dir.absolute()}")
        print(f"Results saved to: {results_file}")
        print("=" * 80)
        
        return {
            'results': all_results,
            'statistics': {
                'total_found': total_found,
                'total_downloaded': total_downloaded,
                'total_skipped': total_skipped,
                'download_dir': str(self.download_dir.absolute())
            }
        }


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape PDFs from Zenodo repositories")
    parser.add_argument('--urls', nargs='+',
                       default=[
                           "https://zenodo.org/records/17627952",
                           "https://zenodo.org/records/17873290",
                           "https://zenodo.org/records/17873279",
                           "https://zenodo.org/records/17861907"
                       ],
                       help="Zenodo repository URLs")
    parser.add_argument('--download-dir', default="./pdfs",
                       help="Directory to save PDFs (default: ./pdfs)")
    parser.add_argument('--delay', type=float, default=1.0,
                       help="Delay between requests in seconds (default: 1.0)")
    
    args = parser.parse_args()
    
    scraper = ZenodoPDFScraper(
        download_dir=args.download_dir,
        request_delay=args.delay
    )
    
    scraper.scrape_and_download(args.urls)


if __name__ == "__main__":
    main()

