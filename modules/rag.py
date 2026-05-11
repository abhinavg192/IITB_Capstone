"""
modules/rag.py
RAG (Retrieval Augmented Generation) Pipeline
Extracts brand voice context from PDF documents using FAISS

Usage:
    from modules.rag import build_index, retrieve_brand_context

Author: Shreyas Shanbag (extracted and modularized by Abhinav Gupta)
"""
import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Embedding model — free, no API cost
# all-MiniLM-L6-v2 is fast, lightweight, good for semantic search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking parameters
CHUNK_SIZE = 500    # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks

# How many chunks to retrieve per query
TOP_K = 3

# ─────────────────────────────────────────────────────────────
# GLOBAL STATE
# In-memory only — no persistence needed for demo
# Fresh index built per session when user uploads PDF
# ─────────────────────────────────────────────────────────────

_embedder = None
_indexes = {}       # brand_name -> faiss index
_chunks = {}        # brand_name -> list of text chunks


# ─────────────────────────────────────────────────────────────
# EMBEDDER INITIALIZATION
# ─────────────────────────────────────────────────────────────

def get_embedder():
    """
    Loads sentence transformer model once per session.
    Subsequent calls return cached model.
    """
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Embedding model loaded!")
    return _embedder


# ─────────────────────────────────────────────────────────────
# TEXT PROCESSING
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Cleans extracted PDF text.
    Removes extra whitespace and line breaks.
    """
    text = text.replace("\n", " ").strip()
    return " ".join(text.split())


def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list:
    """
    Splits text into overlapping chunks for FAISS indexing.
    Overlap ensures context isn't lost at chunk boundaries.

    Args:
        text: cleaned text string
        chunk_size: characters per chunk
        overlap: characters of overlap between chunks

    Returns:
        list of text chunks
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts and cleans text from a PDF file.
    Uses PyPDFLoader from langchain_community.

    Args:
        pdf_path: path to PDF file

    Returns:
        cleaned text string
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        cleaned = [clean_text(doc.page_content) for doc in documents]
        return " ".join(cleaned)
    except ImportError:
        # Fallback to pdfplumber if langchain_community not available
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
            return clean_text(text)
        except ImportError:
            raise ImportError(
                "Please install either langchain-community or pdfplumber: "
                "pip install langchain-community pypdf"
            )


# ─────────────────────────────────────────────────────────────
# INDEX BUILDING
# ─────────────────────────────────────────────────────────────

def build_index(pdf_path: str, brand_name: str) -> None:
    """
    Builds FAISS index from a brand voice PDF.
    Stores index and chunks in memory for retrieval.

    This is called once per user session when they upload
    their brand guidelines PDF.

    Args:
        pdf_path: path to brand voice PDF
        brand_name: name of brand (used as key for retrieval)
    """
    global _indexes, _chunks

    print(f"Building RAG index for {brand_name}...")

    # Step 1 — Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    print(f"  Extracted {len(text)} characters from PDF")

    # Step 2 — Chunk the text
    chunks = chunk_text(text)
    print(f"  Created {len(chunks)} chunks")

    # Step 3 — Embed all chunks
    embedder = get_embedder()
    embeddings = embedder.encode(chunks)
    print(f"  Generated embeddings shape: {embeddings.shape}")

    # Step 4 — Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    # Step 5 — Store in memory
    _indexes[brand_name] = index
    _chunks[brand_name] = chunks

    print(f"✅ RAG index built for {brand_name} "
          f"({len(chunks)} chunks indexed)")


def build_index_from_text(text: str, brand_name: str) -> None:
    """
    Builds FAISS index directly from text string.
    Used when brand guidelines are entered as text in UI
    (alternative to PDF upload).

    Args:
        text: brand guidelines text
        brand_name: name of brand
    """
    global _indexes, _chunks

    # Clean and chunk
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)

    # Embed
    embedder = get_embedder()
    embeddings = embedder.encode(chunks)

    # Build index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    # Store
    _indexes[brand_name] = index
    _chunks[brand_name] = chunks

    print(f"✅ RAG index built for {brand_name} "
          f"({len(chunks)} chunks indexed)")


# ─────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────

def retrieve_brand_context(brand_name: str,
                           topic: str,
                           platform: str = "",
                           tone: str = "",
                           n_results: int = TOP_K) -> str:
    """
    Retrieves most relevant brand voice chunks for a given topic.
    This is the main function called by Abhinav's generate_posts().

    Args:
        brand_name: name of brand (must match build_index() call)
        topic: post topic e.g. "Semrush acquisition"
        platform: target platform e.g. "linkedin"
        tone: desired tone e.g. "professional"
        n_results: number of chunks to retrieve

    Returns:
        string of relevant brand context to inject into prompt
    """
    if brand_name not in _indexes:
        print(f"⚠️ No index found for {brand_name}. "
              f"Call build_index() first.")
        return ""

    # Build a descriptive query combining all context
    # Better query = more relevant chunks retrieved
    if platform and tone:
        query = (f"How should {brand_name} write a {platform} post "
                 f"about {topic} in a {tone} tone?")
    else:
        query = (f"Brand voice and tone guidelines for {brand_name} "
                 f"writing about {topic}")

    # Embed the query
    embedder = get_embedder()
    query_embedding = embedder.encode([query])

    # Search FAISS index
    index = _indexes[brand_name]
    chunks = _chunks[brand_name]

    distances, indices = index.search(
        query_embedding.astype(np.float32),
        k=min(n_results, len(chunks))
    )

    # Retrieve matching chunks
    retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]

    # Join chunks into a single context string
    brand_context = "\n\n".join(retrieved)

    return brand_context


def is_index_built(brand_name: str) -> bool:
    """
    Checks if an index has been built for a brand.
    Used by app.py to decide whether to show upload prompt.
    """
    return brand_name in _indexes


def list_indexed_brands() -> list:
    """
    Returns list of all brands with built indexes.
    """
    return list(_indexes.keys())


def clear_index(brand_name: str = None) -> None:
    """
    Clears index from memory.
    If brand_name is None, clears all indexes.
    """
    global _indexes, _chunks
    if brand_name:
        _indexes.pop(brand_name, None)
        _chunks.pop(brand_name, None)
        print(f"✅ Index cleared for {brand_name}")
    else:
        _indexes = {}
        _chunks = {}
        print("✅ All indexes cleared")


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rag.py <path_to_pdf> [brand_name]")
        print("Example: python rag.py data/brand_guidelines/Adobe_Brand_Voice.pdf Adobe")
        sys.exit(1)

    pdf_path = sys.argv[1]
    brand_name = sys.argv[2] if len(sys.argv) > 2 else "TestBrand"

    # Build index
    build_index(pdf_path, brand_name)

    # Test retrieval
    test_queries = [
        ("product launch announcement", "linkedin", "professional"),
        ("community story", "instagram", "warm"),
        ("corporate acquisition", "linkedin", "strategic"),
    ]

    print("\n" + "="*60)
    print("RETRIEVAL TEST RESULTS")
    print("="*60)

    for topic, platform, tone in test_queries:
        print(f"\nQuery: {topic} | {platform} | {tone}")
        print("-"*40)
        context = retrieve_brand_context(
            brand_name, topic, platform, tone
        )
        print(context[:300] + "..." if len(context) > 300 else context)