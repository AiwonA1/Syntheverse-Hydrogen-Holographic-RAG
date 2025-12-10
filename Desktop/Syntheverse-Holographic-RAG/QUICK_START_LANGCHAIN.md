# Quick Start: LangChain RAG Pipeline

This guide will get you up and running with the LangChain-based RAG system in minutes.

## Why LangChain?

**LangChain** is the most popular RAG framework on GitHub with **121,359+ stars** and **20,010+ forks**. It's the industry standard for building production-ready RAG applications.

## Installation (3 steps)

### 1. Install dependencies

```bash
pip install -r requirements_langchain.txt
```

### 2. Set OpenAI API key

Create a `.env` file:

```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

Get your API key from: https://platform.openai.com/api-keys

### 3. Run the pipeline

```bash
python langchain_rag_pipeline.py
```

That's it! The pipeline will:
1. Scrape Zenodo repositories for PDFs
2. Download and parse PDFs using LangChain
3. Create embeddings and store in ChromaDB
4. Build a queryable RAG system

## Usage Examples

### Basic Usage

```bash
# Run full pipeline with default URLs
python langchain_rag_pipeline.py

# Run with custom Zenodo URLs
python langchain_rag_pipeline.py --urls \
  https://zenodo.org/records/17873290 \
  https://zenodo.org/records/17873279

# Query the system (after processing)
python langchain_rag_pipeline.py --query-only
```

### Interactive Query

After processing, start an interactive session:

```bash
python langchain_rag_pipeline.py --query-only
```

Then ask questions:
```
Question: What is hydrogen holography?
Question: Explain the Syntheverse framework
Question: exit
```

### Python API

```python
from langchain_rag_pipeline import LangChainRAGPipeline

# Initialize
pipeline = LangChainRAGPipeline()

# Run pipeline
pipeline.run_full_pipeline([
    "https://zenodo.org/records/17873290"
])

# Query
result = pipeline.query("What is hydrogen holography?")
print(result['answer'])
```

## What Makes This Better?

| Feature | Custom Code | LangChain |
|---------|-------------|-----------|
| **Popularity** | Custom | 121K+ stars |
| **PDF Processing** | Basic | Advanced loaders |
| **Text Splitting** | Simple | RecursiveCharacterTextSplitter |
| **Maintenance** | You | Community |
| **Documentation** | Limited | Extensive |

## Troubleshooting

### "OPENAI_API_KEY not found"
- Create `.env` file with your API key
- Or export: `export OPENAI_API_KEY=your_key`

### "SQLite version" error
```bash
# macOS
brew install sqlite3

# Or use Python 3.11+
brew install python@3.11
python3.11 -m pip install -r requirements_langchain.txt
```

### PDF download fails
- Check internet connection
- Verify Zenodo URLs are accessible
- The scraper includes automatic retries

## Next Steps

1. **Read the full documentation**: See `LANGCHAIN_README.md`
2. **Customize settings**: Adjust chunk size, overlap, etc.
3. **Explore the code**: Check individual components in the files
4. **Join the community**: LangChain has extensive docs and community support

## Files Created

- `langchain_zenodo_scraper.py` - Zenodo scraper
- `langchain_pdf_processor.py` - PDF processing with LangChain
- `langchain_vector_store.py` - Vector store integration
- `langchain_rag_pipeline.py` - Main pipeline script
- `example_langchain_usage.py` - Usage examples

## Resources

- [LangChain Docs](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain) (121K+ stars)
- [Zenodo API](https://developers.zenodo.org/)
