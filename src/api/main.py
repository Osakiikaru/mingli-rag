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
from langchain_core.messages import HumanMessage, AIMessage
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
    message:    str = Field(..., description="用户当前输入")
    history:    list[ChatMessage] = Field(default=[], description="历史对话（可选）")
    use_rag:    bool = Field(default=True,  description="是否启用古籍检索")
    use_critic: bool = Field(default=False, description="是否启用 Self-Critique 来源核验")


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
    # 构建 LangGraph 消息历史
    lc_messages = []
    for m in req.history:
        if m.role == "user":
            lc_messages.append(HumanMessage(content=m.content))
        else:
            lc_messages.append(AIMessage(content=m.content))
    lc_messages.append(HumanMessage(content=req.message))

    initial_state = {
        "messages":   lc_messages,
        "use_rag":    req.use_rag,
        "use_critic": req.use_critic,
        "chunks":     [],
        "bazi_str":   "",
    }

    result = mingli_graph.invoke(initial_state)

    # 提取最终回答（最后一条没有 tool_calls 的 AIMessage）
    final_answer = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            final_answer = msg.content
            break

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
        answer=final_answer,
        sources=sources,
        bazi_str=result.get("bazi_str", ""),
        query_type="",   # ReAct 架构不再显式标注意图类型
    )
