# Syntheverse Hydrogen Holographic RAG

A comprehensive RAG (Retrieval-Augmented Generation) system for processing and querying scientific PDFs from Zenodo repositories, with a focus on Hydrogen Holographic and Fractal Intelligence research.

## Overview

This system provides a complete pipeline for:
1. **Scraping PDFs** from Zenodo repositories
2. **Parsing PDFs** into searchable text chunks
3. **Vectorizing** chunks for semantic search
4. **Querying** the knowledge base using RAG

## Quick Start Workflow

### Step 1: Scrape PDFs from Zenodo

Download PDFs from Zenodo repositories:

```bash
python scrape_pdfs.py --urls https://zenodo.org/records/17244387
```

**Options:**
- `--urls`: One or more Zenodo repository URLs
- `--download-dir`: Directory to save PDFs (default: `./pdfs`)
- `--delay`: Delay between requests in seconds (default: 1.0)

**Example:**
```bash
# Scrape multiple repositories
python scrape_pdfs.py --urls \
  https://zenodo.org/records/17244387 \
  https://zenodo.org/records/17627952

# Custom download directory
python scrape_pdfs.py --urls https://zenodo.org/records/17244387 \
  --download-dir /path/to/pdfs
```

**Output:**
- PDFs saved to: `./pdfs/` (or custom directory)
- Metadata saved to: `scrape_results.json`

### Step 2: Parse All PDFs

Parse all downloaded PDFs into text chunks:

```bash
python parse_all_pdfs.py --pdf-dir ./pdfs
```

**Options:**
- `--pdf-dir`: Directory containing PDF files (default: `./pdfs`)
- `--output-dir`: Directory to save parsed JSON files (default: `pdf_dir/../parsed`)
- `--chunk-size`: Maximum characters per chunk (default: 1000)
- `--chunk-overlap`: Characters to overlap between chunks (default: 200)

**Example:**
```bash
# Parse PDFs with custom chunk size
python parse_all_pdfs.py \
  --pdf-dir ./pdfs \
  --output-dir ./parsed \
  --chunk-size 1500 \
  --chunk-overlap 300
```

**Output:**
- Parsed chunks saved to: `./parsed/` (one JSON file per PDF)
- Automatically skips already parsed PDFs (no duplicates)

**Features:**
- ✅ Uses LangChain's robust PDF processors
- ✅ Intelligent text chunking with overlap
- ✅ Skips already parsed files (no duplicates)
- ✅ Progress tracking for each PDF
- ✅ Error handling continues on failures

### Step 3: Vectorize and Query (Optional)

After parsing, you can vectorize the chunks and use the RAG pipeline:

```bash
# Vectorize parsed chunks
python vectorize_parsed.py

# Or use the full LangChain RAG pipeline
python langchain_rag_pipeline.py --query-only
```

## Complete Example

```bash
# 1. Scrape PDFs from Zenodo
python scrape_pdfs.py --urls https://zenodo.org/records/17244387

# 2. Parse all PDFs into chunks
python parse_all_pdfs.py --pdf-dir ./pdfs

# 3. (Optional) Vectorize and query
python langchain_rag_pipeline.py --query-only
```

## Directory Structure

```
.
├── scrape_pdfs.py          # Zenodo PDF scraper
├── parse_all_pdfs.py      # PDF parser (scraper → parser workflow)
├── pdfs/                   # Downloaded PDF files
├── parsed/                 # Parsed PDF chunks (JSON files)
├── scrape_results.json    # Scrape metadata
└── data/                  # Vector database and metadata
    ├── chroma_db/         # ChromaDB vector store
    ├── parsed/            # Alternative parsed location
    └── metadata/          # Additional metadata
```

## Features

### PDF Scraper (`scrape_pdfs.py`)
- ✅ Scrapes all PDFs from Zenodo repositories
- ✅ Progress display for each download
- ✅ No duplicates - skips already downloaded files
- ✅ Rate limiting - respectful delays
- ✅ Error handling - continues on failures

### PDF Parser (`parse_all_pdfs.py`)
- ✅ Processes all PDFs in a directory
- ✅ Uses LangChain's robust PDF loaders
- ✅ Intelligent text chunking with overlap
- ✅ Skips already parsed files (no duplicates)
- ✅ Progress tracking and error handling
- ✅ Saves chunks as JSON files

### RAG Pipeline
- ✅ LangChain-based (industry standard)
- ✅ ChromaDB vector store
- ✅ Semantic search and retrieval
- ✅ Question-answering interface

## Requirements

### For Scraping
```bash
pip install requests
```

### For Parsing
```bash
pip install -r requirements_langchain.txt
```

Key dependencies:
- `langchain`
- `langchain-community`
- `pypdf`
- `chromadb`

## Configuration

### PDF Scraper
- Default download directory: `./pdfs`
- Default delay: 1.0 seconds between requests
- Automatically skips duplicate downloads

### PDF Parser
- Default chunk size: 1000 characters
- Default chunk overlap: 200 characters
- Output: JSON files with text chunks and metadata

## Workflow Summary

1. **Scrape**: Download PDFs from Zenodo → `./pdfs/`
2. **Parse**: Extract text chunks from PDFs → `./parsed/`
3. **Vectorize**: Create embeddings and store in vector database
4. **Query**: Ask questions using the RAG system

## Additional Documentation

- **SCRAPER_README.md**: Detailed scraper documentation
- **LANGCHAIN_README.md**: LangChain RAG pipeline documentation
- **QUICK_START_LANGCHAIN.md**: Quick start guide for RAG

## Notes

- PDFs are saved with sanitized filenames
- Parsed files are cached (JSON) to avoid re-parsing
- The parser automatically detects and skips already processed PDFs
- All tools include progress tracking and error handling

## License

See repository for license information.

