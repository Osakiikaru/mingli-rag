"""
LangGraph ReAct 图组装
----------------------
结构：agent ↔ tools 循环，直到 LLM 不再调用工具为止

  agent_node → (有工具调用) → tools_node → 回到 agent_node
             → (无工具调用) → END
"""

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import agent_node, tools_node, critic_node, should_continue


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("agent",  agent_node)
    g.add_node("tools",  tools_node)
    g.add_node("critic", critic_node)

    g.set_entry_point("agent")

    g.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "critic": "critic", "end": END},
    )
    g.add_edge("tools",  "agent")
    g.add_edge("critic", END)

    return g.compile()


mingli_graph = build_graph()
