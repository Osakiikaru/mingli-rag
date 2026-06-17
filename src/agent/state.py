"""
Agent 状态定义（ReAct 版）
--------------------------
ReAct 架构下状态极度精简：
- messages 是核心，通过 add_messages reducer 自动追加（不覆盖）
- chunks / bazi_str 额外保存，供 Streamlit 界面展示来源和排盘详情
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages:   Annotated[list, add_messages]  # 完整消息历史（HumanMessage/AIMessage/ToolMessage）
    use_rag:    bool                            # 是否启用古籍检索（False=纯LLM直接回答）
    use_critic: bool                            # 是否启用 Self-Critique（默认 False）
    chunks:     list                            # 最近一次 search_tool 的原始片段（供 UI 展示来源）
    bazi_str:   str                             # 最近一次 bazi_tool 的排盘结果（供 UI 展示）
