"""
LangGraph 图组装
----------------
三条路径：
  chat     → chat_node → END
  knowledge → retriever → generator → critic → END
  personal  → bazi → query_rewriter → retriever → generator → critic → END
"""

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    intent_parser_node,
    chat_node,
    bazi_node,
    query_rewriter_node,
    retriever_node,
    generator_node,
    critic_node,
)


def _route_after_intent(state: AgentState) -> str:
    """意图解析后三叉路由"""
    qt = state.get("query_type", "knowledge")
    if qt == "chat":
        return "chat"
    return "bazi" if state.get("needs_bazi") else "retrieval"


def build_graph():
    g = StateGraph(AgentState)

    # 注册节点
    g.add_node("intent_parser",       intent_parser_node)
    g.add_node("chat_node",           chat_node)
    g.add_node("bazi_node",           bazi_node)
    g.add_node("query_rewriter_node", query_rewriter_node)
    g.add_node("retriever_node",      retriever_node)
    g.add_node("generator_node",      generator_node)
    g.add_node("critic_node",         critic_node)

    # 入口
    g.set_entry_point("intent_parser")

    # 三叉路由
    g.add_conditional_edges(
        "intent_parser",
        _route_after_intent,
        {
            "chat":      "chat_node",
            "bazi":      "bazi_node",
            "retrieval": "retriever_node",
        },
    )

    # chat 路径：直接结束
    g.add_edge("chat_node", END)

    # personal 路径：排盘 → 重构词 → 检索
    g.add_edge("bazi_node",           "query_rewriter_node")
    g.add_edge("query_rewriter_node", "retriever_node")

    # 共用尾链：检索 → 生成 → 审查 → 结束
    g.add_edge("retriever_node", "generator_node")
    g.add_edge("generator_node", "critic_node")
    g.add_edge("critic_node",    END)

    return g.compile()


mingli_graph = build_graph()
