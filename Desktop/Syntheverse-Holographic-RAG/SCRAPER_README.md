# Zenodo PDF Scraper

Simple, efficient scraper for downloading all PDFs from Zenodo repositories.

## Features

✅ **Scrapes all PDFs** from multiple Zenodo repositories  
✅ **Progress display** for each file download  
✅ **No duplicates** - automatically skips already downloaded files  
✅ **Rate limiting** - respectful delays between requests  
✅ **Error handling** - continues even if individual files fail  

## Quick Start

### Basic Usage

```bash
python3 scrape_pdfs.py
```

This will scrape the default repositories:
- https://zenodo.org/records/17627952
- https://zenodo.org/records/17873290
- https://zenodo.org/records/17873279
- https://zenodo.org/records/17861907

### Custom URLs

```bash
python3 scrape_pdfs.py --urls \
  https://zenodo.org/records/12345678 \
  https://zenodo.org/records/87654321
```

### Custom Download Directory

```bash
python3 scrape_pdfs.py --download-dir ./my_pdfs
```

### Adjust Rate Limiting

```bash
python3 scrape_pdfs.py --delay 2.0  # 2 second delay between downloads
```

## Output

- **PDFs saved to:** `./pdfs/` (or custom directory)
- **Results saved to:** `scrape_results.json`
- **Progress shown:** Real-time download progress for each file

## Example Output

```
================================================================================
Zenodo PDF Scraper
================================================================================
📁 Download directory: /path/to/pdfs
📚 Repositories to scrape: 4

Scraping: https://zenodo.org/records/17627952
  Found 100 PDF(s) in: Fractal Science and Intelligence Foundational Papers
[1/100] ↓ Downloading: example.pdf
     Progress: 100.0% (0.13MB/0.13MB)
[1/100] ✓ Downloaded: example.pdf (0.13MB / 135,668 bytes)
[2/100] ⊘ Already exists: another.pdf (73,700 bytes)
...
```

## Requirements

```bash
pip install requests
```

That's it! No other dependencies needed.

## How It Works

1. **Scrapes Zenodo API** - Uses Zenodo's REST API to get repository metadata
2. **Extracts PDF URLs** - Finds all PDF files in each repository
3. **Downloads with progress** - Shows real-time download progress
4. **Skips duplicates** - Checks if file already exists before downloading
5. **Saves metadata** - Creates `scrape_results.json` with all repository info

## File Structure

```
.
├── scrape_pdfs.py          # Main scraper script
├── pdfs/                   # Downloaded PDFs (created automatically)
└── scrape_results.json     # Scrape results and metadata
```

## Notes

- PDFs are saved with sanitized filenames (invalid characters replaced)
- Files are checked by URL to avoid duplicate downloads
- Rate limiting prevents overwhelming Zenodo servers
- All errors are logged but don't stop the process
