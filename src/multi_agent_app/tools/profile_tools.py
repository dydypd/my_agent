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
def profile_retriever(query: str) -> str:
    """
    Retrieve a user's profile information from the database.
    
    Use this tool whenever the user asks about:
    - Personal information (name, email, age, address)
    - Account details or preferences
    - Any question starting with "who is", "tell me about [person]"
    
    Args:
        query
    
    Returns:
        A dict containing the user's profile data.
    """
    top_k = int(os.getenv("RAG_TOP_K", "5"))

    client = _get_qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION_NAME", "profile")
    embeddings = _get_embeddings()
    
    query_vector = embeddings.embed_query(query)
    
    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    docs = []
    print(results)
    for r in results.points:
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
