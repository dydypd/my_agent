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
import requests
from functools import lru_cache
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool

load_dotenv()
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
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


# util



def flatten_json(data, parent_key="", sep="."):
    items = []

    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key).items())
        else:
            items.append((new_key, v))

    return dict(items)


@tool
def notify_me_discord(data):
    """
    Send a notification to Discord with the provided data.

    Args:
        data (dict): The data to include in the Discord notification.
    """
    fields = []

    for k, v in data.items():
        fields.append({
            "name": str(k),
            "value": str(v),
            "inline": False
        })

    payload = {
        "embeds": [
            {
                "title": "📩 New Contact Added!",
                "fields": fields
            }
        ]
    }

    requests.post(WEBHOOK_URL, json=payload)
    
    
@tool
def get_time():
    """Get the current time."""
    from datetime import datetime
    return datetime.now().isoformat()