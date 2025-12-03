# Syntheverse RAG — Colab-ready single-file program
# Copy-paste this entire block into a Google Colab cell and run.

# 0) Install required packages (this cell takes a minute)
!pip install -qU \
    sentence-transformers \
    faiss-cpu \
    pypdf \
    gitpython \
    requests \
    transformers \
    torch \
    accelerate \
    numpy

# 1) Imports
import os
import re
import io
import time
import json
import shutil
import git
import math
import requests
import numpy as np
from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 2) Settings (edit these if desired)
ZENODO_COMMUNITY_SLUGS = [
    "fractal-science-intelligence",
    "ciencia-inteligencia-fractal"
]
GITHUB_URL = "https://github.com/AiwonA1/FractalHydrogenHolography-Validation"
LOCAL_REPO_PATH = "/content/fhh-validation-repo"
DOWNLOAD_FOLDER = "/content/syntheverse_docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
GEN_MODEL_ID = "gpt2"  # small and fast; replace for quality if you have GPU

os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# 3) Zenodo / Git helpers
def get_community_records(community_slug: str, page_size: int = 100) -> List[str]:
    api_url = f"https://zenodo.org/api/records?communities={community_slug}&size={page_size}"
    ids = set()
    page = 1
    while True:
        paged = f"{api_url}&page={page}"
        print(f"[Zenodo] Fetching {community_slug} page {page} ...")
        try:
            r = requests.get(paged, timeout=30)
            r.raise_for_status()
            data = r.json()
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break
            for h in hits:
                ids.add(str(h.get("id")))
            if len(hits) < page_size:
                break
            page += 1
        except Exception as e:
            print(f"[Zenodo] Error: {e}")
            break
    print(f"[Zenodo] Found {len(ids)} records for {community_slug}")
    return list(ids)

def resolve_zenodo_pdf_url(record_id: str) -> Tuple[str, str]:
    api_url = f"https://zenodo.org/api/records/{record_id}"
    try:
        r = requests.get(api_url, timeout=30)
        r.raise_for_status()
        data = r.json()
        for file in data.get("files", []):
            # Some Zenodo records use 'type' or 'contentType'; be permissive
            ftype = file.get("type") or file.get("contentType") or ""
            if "pdf" in str(ftype).lower() or str(file.get("key","")).lower().endswith(".pdf"):
                filename = file.get("key")
                download_url = f"https://zenodo.org/record/{record_id}/files/{filename}"
                return download_url, filename
        return None, None
    except Exception as e:
        # print("resolve error", e)
        return None, None

def download_pdf(url: str, out_path: str) -> bool:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        with requests.get(url, stream=True, headers=headers, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as fd:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        fd.write(chunk)
        return True
    except Exception as e:
        print(f"[Download] Failed {url}: {e}")
        return False

def clone_github(repo_url: str, dest_path: str):
    if os.path.exists(dest_path):
        print("[Git] Repo already exists; removing and recloning.")
        shutil.rmtree(dest_path)
    print(f"[Git] Cloning {repo_url}...")
    git.Repo.clone_from(repo_url, dest_path)
    print("[Git] Done.")

# 4) Text extraction
def extract_text_from_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        text_parts = []
        for p in reader.pages:
            txt = p.extract_text()
            if txt:
                text_parts.append(txt)
        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"[PDF] Error reading {path}: {e}")
        return ""

def extract_texts_from_repo(repo_path: str, suffixes=(".md",".txt",".py")) -> List[Tuple[str,str]]:
    docs = []
    for root, _, files in os.walk(repo_path):
        for fn in files:
            if fn.lower().endswith(suffixes):
                full = os.path.join(root, fn)
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        t = f.read()
                        docs.append((full, t))
                except Exception as e:
                    print(f"[Repo] Could not read {full}: {e}")
    return docs

# 5) Chunking
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks

# 6) Ingest pipeline (Zenodo + Git)
ALL_CHUNKS = []
METADATA = []  # parallel metadata for each chunk

# A: Zenodo
all_ids = set()
for slug in ZENODO_COMMUNITY_SLUGS:
    ids = get_community_records(slug)
    all_ids.update(ids)

print(f"[Ingest] Total unique Zenodo record IDs: {len(all_ids)}")

file_counter = 1
for rec in sorted(all_ids):
    pdf_url, fname = resolve_zenodo_pdf_url(rec)
    if not pdf_url:
        continue
    local_name = os.path.join(DOWNLOAD_FOLDER, f"zenodo_{rec}_{fname or file_counter}.pdf")
    if download_pdf(pdf_url, local_name):
        txt = extract_text_from_pdf(local_name)
        if txt.strip():
            chunks = chunk_text(txt)
            for c in chunks:
                ALL_CHUNKS.append(c)
                METADATA.append({"source": local_name})
        file_counter += 1
    else:
        print(f"[Ingest] Failed to download {pdf_url}")

# B: GitHub
try:
    clone_github(GITHUB_URL, LOCAL_REPO_PATH)
    repo_docs = extract_texts_from_repo(LOCAL_REPO_PATH)
    for path, text in repo_docs:
        if not text.strip():
            continue
        chunks = chunk_text(text)
        for c in chunks:
            ALL_CHUNKS.append(c)
            METADATA.append({"source": path})
except Exception as e:
    print(f"[GitHub] Error: {e}")

print(f"[Ingest] Total text chunks collected: {len(ALL_CHUNKS)}")

# If no documents found, add a tiny placeholder so rest of pipeline doesn't crash
if len(ALL_CHUNKS) == 0:
    ALL_CHUNKS.append("No documents were ingested. Please check Zenodo slugs and GitHub URL.")
    METADATA.append({"source": "none"})

# 7) Embeddings and FAISS index
print("[Embedding] Loading sentence-transformers model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
embed_dim = embed_model.get_sentence_embedding_dimension()
print(f"[Embedding] Model dim = {embed_dim}")

print("[Embedding] Encoding chunks (this may take a moment)...")
batch_size = 32
embeddings = []
for i in range(0, len(ALL_CHUNKS), batch_size):
    batch = ALL_CHUNKS[i:i+batch_size]
    embs = embed_model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
    embeddings.append(embs)
embeddings = np.vstack(embeddings).astype("float32")

# Build FAISS index
print("[FAISS] Building index...")
index = faiss.IndexFlatL2(embed_dim)
index.add(embeddings)
print(f"[FAISS] Indexed {index.ntotal} vectors.")

# 8) Generator pipeline (GPT-2)
print("[Generator] Loading generation pipeline (gpt2)...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_ID)
device = 0 if (torch.cuda.is_available()) else -1
gen_pipe = pipeline(
    "text-generation",
    model=gen_model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    device=device
)
print("[Generator] Ready.")

# 9) Query function: retrieve + generate
def answer_query(question: str, k: int = 3, max_context_tokens: int = 2000) -> dict:
    """
    - Embeds query, searches FAISS for top-k similar chunks.
    - Builds a prompt with the retrieved context chunks.
    - Calls local generator pipeline to produce an answer.
    Returns a dict with answer, sources, and retrieved texts.
    """
    q_emb = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, k)
    retrieved = []
    sources = set()
    for idx in I[0]:
        if idx < 0 or idx >= len(ALL_CHUNKS):
            continue
        retrieved.append(ALL_CHUNKS[idx])
        sources.add(METADATA[idx].get("source", "unknown"))
    # Build context: join retrieved chunks, but trim to max_context_tokens (characters)
    context = "\n\n---\n\n".join(retrieved)
    if len(context) > max_context_tokens:
        context = context[:max_context_tokens] + "\n\n[TRUNCATED]"
    prompt = (
        "You are GINA × LEO × PRU — a retrieval-augmented responder. "
        "Answer the user's question based ONLY on the following retrieved context (do not invent facts). "
        "If the context doesn't contain an answer, say you don't know.\n\n"
        "=== Retrieved Context ===\n"
        f"{context}\n\n"
        "=== Question ===\n"
        f"{question}\n\n"
        "=== Answer (grounded, concise) ===\n"
    )
    # Generate
    out = gen_pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
    raw = out[0]["generated_text"]
    # Postprocess: strip prompt prefix if present
    answer = raw
    if prompt in raw:
        answer = raw.split(prompt, 1)[1].strip()
    # Truncate answer to reasonable length
    answer = answer.strip()
    # Remove anything after a long runaway (keep up to 2000 chars)
    if len(answer) > 2000:
        answer = answer[:2000] + "..."
    return {
        "question": question,
        "answer": answer,
        "sources": list(sources),
        "retrieved_count": len(retrieved)
    }

# 10) Quick interactive demo
print("\n\n=== Demo ready ===")
print("Use the function `answer_query(<your question>)` to query the indexed documents.\n")
print("Example:")
demo_q = "What is the Hydrogen Holographic Matrix (HHM) and how is it described?"
print("→ Running example query (may take a few seconds)...")
demo_result = answer_query(demo_q, k=3)
print("\nQuestion:", demo_result["question"])
print("Answer:\n", demo_result["answer"])
print("Sources:", demo_result["sources"])
print("Retrieved chunks:", demo_result["retrieved_count"])

# Optional: Save the index and metadata to disk for later use
np.save("/content/syntheverse_embeddings.npy", embeddings)
with open("/content/syntheverse_metadata.json", "w", encoding="utf-8") as f:
    json.dump(METADATA, f, ensure_ascii=False, indent=2)
print("\nIndex & metadata saved to /content (embeddings .npy + metadata .json).")

# Optional: FastAPI + ngrok (commented out)
#
# If you want to expose an HTTP endpoint from Colab, uncomment and configure below.
# WARNING: exposing Colab to the public internet carries risk. Use at your own discretion.
#
# !pip install -qU fastapi uvicorn pyngrok
# from fastapi import FastAPI
# from pydantic import BaseModel
# from pyngrok import ngrok
# import nest_asyncio
# nest_asyncio.apply()
#
# app = FastAPI()
#
# class Q(BaseModel):
#     question: str
#
# @app.post("/query")
# def http_query(q: Q):
#     return answer_query(q.question, k=3)
#
# NGROK_TOKEN = "<PUT_YOUR_NGROK_TOKEN_HERE>"
# ngrok.set_auth_token(NGROK_TOKEN)
# tunnel = ngrok.connect(8000)
# print("Public URL:", tunnel.public_url)
# import uvicorn
# uvicorn.run(app, host="0.0.0.0", port=8000)
