"""
命理古籍研究助手 — FastAPI HTTP 接口
--------------------------------------
启动方式：uvicorn src.api.main:app --reload
接口文档：http://localhost:8000/docs
"""
import os
import sys
from pathlib import Path

# 项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from src.agent.graph import mingli_graph

app = FastAPI(
    title="命理古籍研究助手 API",
    description="基于 RAG + LangGraph 的子平八字命理知识问答接口",
    version="1.0.0",
)


# ── 请求 / 响应模型 ──────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' 或 'assistant'")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户当前输入")
    history: list[ChatMessage] = Field(default=[], description="历史对话（可选）")
    use_rag: bool = Field(default=True, description="是否启用古籍检索")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="助手回答")
    sources: list[str] = Field(default=[], description="引用的古籍来源列表")
    bazi_str: str = Field(default="", description="排盘详情（仅八字问题时有值）")
    query_type: str = Field(default="", description="意图类型：chat / knowledge / bazi")


# ── 接口 ────────────────────────────────────────────────────

@app.get("/health")
def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    多轮命理问答接口

    - 传入 history 可保持多轮上下文
    - use_rag=false 时跳过古籍检索，由 LLM 直接回答
    """
    history = [{"role": m.role, "content": m.content} for m in req.history]

    initial_state = {
        "user_query":   req.message,
        "chat_history": history,
        "use_rag":      req.use_rag,
        "query_type":   "",
        "needs_bazi":   False,
        "birth_info":   {},
        "search_query": "",
        "bazi_str":     "",
        "chunks":       [],
        "draft_answer": "",
        "final_answer": "",
    }

    result = mingli_graph.invoke(initial_state)

    # 整理古籍来源
    sources = []
    seen = set()
    for c in result.get("chunks", []):
        loc = f"《{c['source']}》"
        if c.get("chapter"):
            loc += f" · {c['chapter']}"
        if loc not in seen:
            seen.add(loc)
            sources.append(loc)

    return ChatResponse(
        answer=result.get("final_answer", ""),
        sources=sources,
        bazi_str=result.get("bazi_str", ""),
        query_type=result.get("query_type", ""),
    )
