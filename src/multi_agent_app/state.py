from __future__ import annotations

from typing import Annotated, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
import operator


class AgentState(TypedDict):
    # ── Conversation core ──────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    # add_messages: append + dedup by id, giữ đúng type sau JSON round-trip

    system_instruction: str
    # Prompt hệ thống — supervisor hoặc input có thể override per-run

    # ── Routing & control ──────────────────────────────────────
    next: str | None
    # Supervisor ghi vào đây để điều hướng: "rag_agent" | "code_agent" | "END"

    current_agent: str | None
    # Node nào đang chạy — dùng để log, debug, và conditional edge

    # ── Tool & memory ──────────────────────────────────────────
    tool_calls: Annotated[list[dict], operator.add]
    # Accumulate pending tool calls — tool_node đọc và xóa từng cái

    scratchpad: str
    # Reasoning trung gian (ReAct / Chain-of-Thought) — không expose ra user

    retrieved_context: Annotated[list[str], operator.add]
    # Kết quả RAG — retriever_node ghi, agent_node đọc để inject vào prompt

    # ── Execution metadata ─────────────────────────────────────
    step_count: int
    # Tăng mỗi vòng lặp — conditional edge dùng để phá vòng lặp vô tận

    error: str | None
    # Node lỗi ghi vào đây — error_handler_node đọc và quyết định retry/abort

    metadata: dict[str, Any]
    # Dữ liệu phụ: user_id, session_id, tenant, feature flags, v.v.

    # ── Final output ───────────────────────────────────────────
    final_answer: str | None
    # Câu trả lời cuối — chỉ ghi 1 lần bởi node cuối, không dùng reducer

    artifacts: Annotated[list[dict], operator.add]
    # File, chart, structured data agent tạo ra trong quá trình xử lý