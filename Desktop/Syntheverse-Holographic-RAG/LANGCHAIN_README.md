# LangChain-based Syntheverse RAG Pipeline

This implementation uses **LangChain** (121,359+ stars on GitHub) - the most popular and widely-used RAG framework. It provides a robust, production-ready solution for scraping Zenodo PDFs, parsing them, and building a RAG system.

## Why LangChain?

- **Most Popular**: 121K+ GitHub stars, 20K+ forks - the industry standard for RAG
- **Robust PDF Processing**: Advanced document loaders with multiple fallback strategies
- **Better Text Splitting**: RecursiveCharacterTextSplitter optimized for structured documents
- **Seamless Integration**: Built-in ChromaDB integration and RAG chains
- **Production Ready**: Battle-tested by thousands of developers
- **Active Community**: Extensive documentation and community support

## Features

✅ **Zenodo Scraping**: Uses Zenodo API to fetch repository metadata and PDF URLs  
✅ **LangChain PDF Loaders**: PyPDFLoader and UnstructuredPDFLoader for robust text extraction  
✅ **Advanced Text Splitting**: RecursiveCharacterTextSplitter for optimal chunking  
✅ **ChromaDB Integration**: Native LangChain integration with persistent storage  
✅ **RAG Pipeline**: Complete RetrievalQA chain for question-answering  
✅ **Duplicate Detection**: Automatically skips already processed PDFs  

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_langchain.txt
```

### 2. Set OpenAI API Key

Create a `.env` file:

```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

Or export it:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

## Quick Start

### Full Pipeline (Scrape, Parse, Vectorize)

Run the complete pipeline:

```bash
python langchain_rag_pipeline.py
```

This will process the default Zenodo repositories:
- https://zenodo.org/records/17873290
- https://zenodo.org/records/17873279
- https://zenodo.org/records/17861907
- https://zenodo.org/records/17627952

### Custom URLs

```bash
python langchain_rag_pipeline.py --urls \
  https://zenodo.org/records/12345678 \
  https://zenodo.org/records/87654321
```

### Query Interface

After processing, start the interactive query interface:

```bash
python langchain_rag_pipeline.py --query
```

Or skip processing and go straight to querying:

```bash
python langchain_rag_pipeline.py --query-only
```

### Advanced Options

```bash
python langchain_rag_pipeline.py \
  --urls https://zenodo.org/records/17873290 \
  --collection my_collection \
  --chunk-size 1500 \
  --chunk-overlap 300 \
  --data-dir ./my_data \
  --query
```

## Usage as Python Module

```python
from langchain_rag_pipeline import LangChainRAGPipeline

# Initialize pipeline
pipeline = LangChainRAGPipeline(
    collection_name="syntheverse_rag",
    chunk_size=1000,
    chunk_overlap=200
)

# Run full pipeline
zenodo_urls = [
    "https://zenodo.org/records/17873290",
    "https://zenodo.org/records/17873279"
]
pipeline.run_full_pipeline(zenodo_urls)

# Query the system
result = pipeline.query("What is hydrogen holography?")
print(result['answer'])
```

## Components

### 1. LangChainZenodoScraper (`langchain_zenodo_scraper.py`)

Scrapes Zenodo repositories using the Zenodo API:
- Extracts PDF file URLs and metadata
- Handles rate limiting and retries
- Downloads PDFs to local storage

### 2. LangChainPDFProcessor (`langchain_pdf_processor.py`)

Processes PDFs using LangChain's document loaders:
- **PyPDFLoader**: Fast, reliable PDF text extraction
- **UnstructuredPDFLoader**: Advanced processing (optional)
- **RecursiveCharacterTextSplitter**: Optimal chunking for structured documents
- Automatic caching of parsed chunks

### 3. LangChainVectorStore (`langchain_vector_store.py`)

Manages vector storage with LangChain's ChromaDB integration:
- OpenAI embeddings (text-embedding-3-small by default)
- Persistent ChromaDB storage
- Similarity search with metadata filtering
- Automatic duplicate detection

### 4. LangChainRAGPipeline (`langchain_rag_pipeline.py`)

Complete pipeline orchestrator:
- Coordinates scraping, processing, and vectorization
- Builds RetrievalQA chain for question-answering
- Interactive query interface
- Progress tracking and statistics

## Project Structure

```
.
├── langchain_zenodo_scraper.py    # Zenodo scraper
├── langchain_pdf_processor.py     # PDF processing with LangChain
├── langchain_vector_store.py      # Vector store with LangChain
├── langchain_rag_pipeline.py      # Main pipeline script
├── requirements_langchain.txt     # Dependencies
├── LANGCHAIN_README.md            # This file
└── data/                          # Data storage (created automatically)
    ├── pdfs/                      # Downloaded PDFs
    ├── chroma_db/                 # Vector database
    ├── parsed/                    # Cached parsed chunks
    └── metadata/                  # Scrape results
```

## Comparison with Custom Implementation

| Feature | Custom Implementation | LangChain Implementation |
|---------|----------------------|-------------------------|
| **Popularity** | Custom code | 121K+ GitHub stars |
| **PDF Processing** | pdfplumber | PyPDFLoader + UnstructuredPDFLoader |
| **Text Splitting** | Basic chunking | RecursiveCharacterTextSplitter |
| **Vector Store** | Manual ChromaDB | Native LangChain integration |
| **RAG Chain** | Custom implementation | Built-in RetrievalQA |
| **Maintenance** | You maintain | Community maintained |
| **Documentation** | Limited | Extensive |

## Advantages of LangChain

1. **Better Text Extraction**: LangChain's loaders handle edge cases better
2. **Smarter Chunking**: RecursiveCharacterTextSplitter preserves document structure
3. **Easier Integration**: Built-in support for multiple vector stores and LLMs
4. **Future-Proof**: Active development and regular updates
5. **Community Support**: Large community for troubleshooting
6. **Extensibility**: Easy to add new document types or LLM providers

## Troubleshooting

### OpenAI API Key Error

Make sure you've set `OPENAI_API_KEY` in your `.env` file or environment variable.

### PDF Download Errors

Check your internet connection and verify Zenodo URLs are accessible. The scraper includes retry logic.

### Memory Issues

Reduce `--chunk-size` parameter for large PDFs:

```bash
python langchain_rag_pipeline.py --chunk-size 500
```

### SQLite Version Issues

ChromaDB requires SQLite 3.35.0+. If you encounter issues:

```bash
# macOS
brew install sqlite3

# Or use Python 3.11+
brew install python@3.11
python3.11 -m pip install -r requirements_langchain.txt
```

## Example Queries

Once the system is set up, you can ask questions like:

- "What is hydrogen holography?"
- "Explain the Hydrogen Holographic Framework"
- "What are the key findings about fractal intelligence?"
- "How does the Syntheverse work?"
- "What is the relationship between hydrogen and cognitive processes?"

## References

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain) (121K+ stars)
- [Zenodo API Documentation](https://developers.zenodo.org/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

## License

This project is part of the Syntheverse Hydrogen Holographic research initiative.
