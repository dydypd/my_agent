"""RAG tools backed by Qdrant Cloud.

Environment variables (store in .env):
    QDRANT_URL              – Qdrant Cloud cluster URL
                              e.g. https://xyz.us-east4-0.gcp.cloud.qdrant.io:6333
    QDRANT_API_KEY          – Qdrant Cloud API key
    QDRANT_COLLECTION_NAME  – target collection (default: "documents")
    EMBEDDING_PROVIDER      – "openai" | "nvidia" | "huggingface" (default: "openai")
    EMBEDDING_MODEL         – embedding model name
                              OpenAI   → "text-embedding-3-small"
                              Nvidia   → "nvidia/nv-embedqa-e5-v5"
                              HF       → "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_API_KEY          – required when EMBEDDING_PROVIDER=openai
    NVIDIA_API_KEY          – required when EMBEDDING_PROVIDER=nvidia
    RAG_TOP_K               – number of chunks to retrieve (default: 5)
    RAG_SCORE_THRESHOLD     – minimum similarity score 0‑1 (default: 0.0)
"""

from __future__ import annotations

import os
import textwrap
from functools import lru_cache
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            "Check your .env file."
        )
    return value


def _get_embeddings():
    """Return the configured embedding model instance."""
    provider = os.getenv("EMBEDDING_PROVIDER", "nvidia").lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(
            model=model,
            openai_api_key=_get_required_env("OPENAI_API_KEY"),
        )

    if provider == "nvidia":
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        model = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
        return NVIDIAEmbeddings(
            model=model,
            nvidia_api_key=_get_required_env("NVIDIA_API_KEY"),
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        model = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        return HuggingFaceEmbeddings(model_name=model)

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER '{provider}'. "
        "Choose one of: openai, nvidia, huggingface"
    )


@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    """Return a cached Qdrant Cloud client."""
    return QdrantClient(
        url=_get_required_env("QDRANT_URL"),
        api_key=_get_required_env("QDRANT_API_KEY"),
    )




def _ensure_collection_exists(vector_size: int) -> None:
    """Create the Qdrant collection if it does not already exist."""
    client = _get_qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION_NAME", "documents")
    try:
        client.get_collection(collection_name=collection)
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _format_docs(docs: List[Document]) -> str:
    parts: List[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        snippet = textwrap.shorten(doc.page_content, width=2000, placeholder="…")
        parts.append(f"[{i}] source={source}\n{snippet}")
    return "\n\n".join(parts) if parts else "No relevant documents found."


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------

@tool
def qdrant_retriever(query: str) -> str:
    """Search the Qdrant vector store and return the most relevant document chunks.

    Use this tool whenever you need to answer questions that rely on information
    stored in the knowledge base (internal documents, manuals, FAQs, etc.).

    Parameters
    ----------
    query:
        The natural-language question or search phrase to look up.

    Returns
    -------
    str
        Numbered list of the most relevant document chunks with their sources.
    """
    top_k = int(os.getenv("RAG_TOP_K", "5"))

    client = _get_qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION_NAME", "documents")
    embeddings = _get_embeddings()
    
    query_vector = embeddings.embed_query(query)
    
    results = client.retrieve(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )

    docs = []
    for r in results:
        payload = r.payload or {}
        # Support both 'text' and 'page_content'
        text = payload.get("text") or payload.get("page_content") or ""
        
        # Extract metadata from flat structure
        metadata = {k: v for k, v in payload.items() if k not in ["text", "page_content"]}
        # Merge nested metadata if it exists
        if "metadata" in payload and isinstance(payload["metadata"], dict):
            metadata.update(payload["metadata"])

        docs.append(Document(page_content=text, metadata=metadata))

    return _format_docs(docs)


@tool
def qdrant_upsert(texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> str:
    """Add or update document chunks in the Qdrant vector store.

    Use this tool when you need to ingest new information into the knowledge
    base so it becomes available for future retrieval.

    Parameters
    ----------
    texts:
        List of plain-text chunks to embed and store.
    metadatas:
        Optional list of metadata dicts (one per chunk).  Useful fields:
        ``source``, ``page``, ``author``, ``created_at``, etc.
        If omitted, an empty dict is used for every chunk.

    Returns
    -------
    str
        Confirmation message with the number of chunks indexed.
    """
    if not texts:
        return "No texts provided – nothing was indexed."

    if metadatas is None:
        metadatas = [{} for _ in texts]

    if len(metadatas) != len(texts):
        return (
            f"Length mismatch: {len(texts)} texts vs {len(metadatas)} metadatas. "
            "Provide one metadata dict per text chunk."
        )

    embeddings = _get_embeddings()
    vectors = embeddings.embed_documents(texts)
    _ensure_collection_exists(vector_size=len(vectors[0]))

    client = _get_qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION_NAME", "documents")

    from uuid import uuid4
    from qdrant_client.http.models import PointStruct

    points = []
    for i, (text, meta, vec) in enumerate(zip(texts, metadatas, vectors)):
        # Flat payload: text + metadata fields at root
        payload = {"text": text, **meta}
        points.append(PointStruct(id=str(uuid4()), vector=vec, payload=payload))

    client.upsert(collection_name=collection, points=points)
    return f"Successfully indexed {len(texts)} chunk(s) into Qdrant."


@tool
def qdrant_delete_collection() -> str:
    """Delete the entire Qdrant collection configured by QDRANT_COLLECTION_NAME.

    WARNING: This permanently removes all vectors and documents in that
    collection. Use only when you intend to wipe the knowledge base.

    Returns
    -------
    str
        Confirmation or error message.
    """
    client = _get_qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION_NAME", "documents")
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        return f"Collection '{collection}' does not exist – nothing to delete."
    client.delete_collection(collection_name=collection)
    return f"Collection '{collection}' has been deleted."



@tool
def qdrant_show_collection(limit: int = 10) -> str:
    """Show documents stored in the Qdrant collection."""

    client = _get_qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION_NAME", "documents")

    try:
        records, _ = client.scroll(
            collection_name=collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        return f"Error accessing collection: {e}"

    if not records:
        return "Collection is empty."

    result = []
    for i, r in enumerate(records, 1):
        payload = r.payload or {}
        # Support both 'text' and 'page_content'
        text = payload.get("text") or payload.get("page_content") or "No text"
        
        # Extract metadata from flat structure
        metadata = {k: v for k, v in payload.items() if k not in ["text", "page_content"]}
        if "metadata" in payload and isinstance(payload["metadata"], dict):
            metadata.update(payload["metadata"])
            if "metadata" in metadata:
                metadata.pop("metadata")

        result.append(
            f"[{i}] id={r.id}\n"
            f"text={text[:300]}\n"
            f"metadata={metadata}"
        )

    return "\n\n".join(result)


if __name__ == "__main__":
    print("--- SHOW COLLECTION ---")
    print(qdrant_show_collection.invoke(input={"limit": 3}))
    print("\n--- RETRIEVER TEST ---")
    print(qdrant_retriever.invoke(input={"query": "What is my name?"}))