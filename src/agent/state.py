"""
Agent 状态定义
-------------
贯穿整个 LangGraph 流程的数据结构。
每个节点读取需要的字段，写入自己负责的字段。
"""

from typing import TypedDict


class AgentState(TypedDict):
    # ── 输入 ──────────────────────────────────────
    user_query: str          # 用户原始问题
    chat_history: list       # 最近几轮对话历史 [{role, content}, ...]
    use_rag: bool            # 是否启用古籍检索（False = 纯LLM直接回答）

    # ── 意图解析结果 ──────────────────────────────
    query_type: str          # "knowledge"（知识问答）| "personal"（个人命理）
    needs_bazi: bool         # 是否需要排盘（当前或历史中有生辰信息）
    birth_info: dict         # 提取的生辰信息 {year, month, day, hour, gender}
    search_query: str        # 精炼后用于检索的关键词

    # ── 排盘结果 ──────────────────────────────────
    bazi_str: str            # 格式化的四柱八字字符串

    # ── 检索结果 ──────────────────────────────────
    chunks: list             # hybrid+rerank 召回的 Top-5 chunks

    # ── 生成结果 ──────────────────────────────────
    draft_answer: str        # 生成节点的初稿
    final_answer: str        # 经 Self-Critique 后的最终答案
