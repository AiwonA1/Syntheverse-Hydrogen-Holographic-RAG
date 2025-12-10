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

### Step 3: Vectorize Parsed Chunks

Vectorize all parsed chunks using local embeddings (no API calls required):

```bash
python vectorize_parsed_chunks_simple.py --parsed-dir ./parsed --output-dir ./vectorized
```

**Options:**
- `--parsed-dir`: Directory containing parsed JSON files (default: `./parsed`)
- `--output-dir`: Directory to save vectorized embeddings (default: `./vectorized`)
- `--embedding-model`: HuggingFace embedding model (default: `all-MiniLM-L6-v2`)
- `--batch-size`: Batch size for embedding generation (default: 100)

**Example:**
```bash
# Vectorize with custom settings
python vectorize_parsed_chunks_simple.py \
  --parsed-dir ./parsed \
  --output-dir ./vectorized \
  --embedding-model all-MiniLM-L6-v2 \
  --batch-size 100
```

**Output:**
- Vectorized embeddings saved to: `./vectorized/embeddings/` (one JSON file per PDF)
- Metadata saved to: `./vectorized/metadata/vectorization_metadata.json`
- Automatically skips already vectorized files (no duplicates)

**Features:**
- ✅ Uses local HuggingFace embeddings (no API calls, free!)
- ✅ No ChromaDB/SQLite dependency - saves directly to JSON files
- ✅ Skips already vectorized files (no duplicates)
- ✅ Progress tracking for each PDF
- ✅ Error handling continues on failures

### Step 4: Query (Optional)

After vectorization, you can use the RAG pipeline:

```bash
# Use the full LangChain RAG pipeline
python langchain_rag_pipeline.py --query-only
```

## Complete Example

```bash
# 1. Scrape PDFs from Zenodo
python scrape_pdfs.py --urls https://zenodo.org/records/17244387

# 2. Parse all PDFs into chunks
python parse_all_pdfs.py --pdf-dir ./pdfs

# 3. Vectorize parsed chunks
python vectorize_parsed_chunks_simple.py --parsed-dir ./parsed --output-dir ./vectorized

# 4. (Optional) Query using RAG pipeline
python langchain_rag_pipeline.py --query-only
```

## Directory Structure

```
.
├── scrape_pdfs.py                    # Zenodo PDF scraper
├── parse_all_pdfs.py                # PDF parser
├── vectorize_parsed_chunks_simple.py # Vectorization script (local embeddings)
├── pdfs/                             # Downloaded PDF files
├── parsed/                           # Parsed PDF chunks (JSON files)
├── vectorized/                       # Vectorized embeddings
│   ├── embeddings/                   # Embedding JSON files (one per PDF)
│   └── metadata/                     # Vectorization metadata
├── scrape_results.json              # Scrape metadata
└── data/                            # Vector database and metadata (optional)
    ├── chroma_db/                   # ChromaDB vector store
    ├── parsed/                      # Alternative parsed location
    └── metadata/                    # Additional metadata
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

### Vectorization (`vectorize_parsed_chunks_simple.py`)
- ✅ Vectorizes parsed chunks using local embeddings
- ✅ No API calls required (uses HuggingFace models)
- ✅ Saves embeddings as JSON files (no ChromaDB dependency)
- ✅ Skips already vectorized files (no duplicates)
- ✅ Progress tracking and error handling
- ✅ Supports multiple embedding models

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

### For Parsing and Vectorization
```bash
pip install -r requirements_langchain.txt
```

Key dependencies:
- `langchain`
- `langchain-community`
- `pypdf`
- `sentence-transformers` (for local embeddings)
- `chromadb` (optional, for ChromaDB-based vectorization)

## Configuration

### PDF Scraper
- Default download directory: `./pdfs`
- Default delay: 1.0 seconds between requests
- Automatically skips duplicate downloads

### PDF Parser
- Default chunk size: 1000 characters
- Default chunk overlap: 200 characters
- Output: JSON files with text chunks and metadata

### Vectorization
- Default embedding model: `all-MiniLM-L6-v2` (384 dimensions)
- Alternative models: `all-mpnet-base-v2` (768 dimensions, better quality)
- Output: JSON files with text chunks and embeddings
- No API calls required (uses local HuggingFace models)

## Workflow Summary

1. **Scrape**: Download PDFs from Zenodo → `./pdfs/`
2. **Parse**: Extract text chunks from PDFs → `./parsed/`
3. **Vectorize**: Create embeddings from parsed chunks → `./vectorized/embeddings/`
4. **Query**: Ask questions using the RAG system

## Additional Documentation

- **SCRAPER_README.md**: Detailed scraper documentation
- **LANGCHAIN_README.md**: LangChain RAG pipeline documentation
- **QUICK_START_LANGCHAIN.md**: Quick start guide for RAG

## Notes

- PDFs are saved with sanitized filenames
- Parsed files are cached (JSON) to avoid re-parsing
- Vectorized embeddings are cached (JSON) to avoid re-vectorization
- The parser and vectorizer automatically detect and skip already processed files
- All tools include progress tracking and error handling
- Vectorization uses local embeddings (no API costs, works offline)

## License

See repository for license information.

