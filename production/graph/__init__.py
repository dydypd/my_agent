"""
graph/__init__.py
-----------------
Singleton loader for the compiled LangGraph.

The graph is built once when the Worker process starts.  Subsequent calls
to `get_graph()` return the same cached instance — no module-level side
effects are triggered on every job.

This module adds `src/` to sys.path so that the existing `multi_agent_app`
package (which lives in my_agent/src/) is importable without installing it
as an editable package inside the production image.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

_graph: CompiledStateGraph | None = None


def _ensure_src_on_path() -> None:
    """Add my_agent/src to sys.path if it's not already there."""
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        logger.debug("Added %s to sys.path", src_path)


def get_graph() -> CompiledStateGraph:
    """
    Return the compiled LangGraph singleton.
    Safe to call from multiple coroutines (GIL protects the assignment).
    """
    global _graph
    if _graph is None:
        _ensure_src_on_path()
        # Import lazily so LLM clients and Qdrant connections are not
        # initialised until the worker is actually running.
        from multi_agent_app.graph import graph as _compiled  # type: ignore[import]

        _graph = _compiled
        logger.info("LangGraph compiled and cached: %s", _graph.name)
    return _graph
