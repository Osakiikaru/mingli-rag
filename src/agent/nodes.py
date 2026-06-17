"""
LangGraph ReAct 节点实现
------------------------
架构从"7节点3路由"简化为"2节点1循环"：

  agent_node  — LLM 推理，决定调用工具还是直接回答
  tools_node  — 执行工具调用，同时更新 chunks/bazi_str 供 UI 展示

工具列表：
  bazi_tool   — lunar-python 八字排盘（确定性算法）
  search_tool — BGE-M3 向量 + BM25 + RRF 混合检索
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.tools.bazi import get_bazi

load_dotenv()

# ── LLM 客户端 ────────────────────────────────────────────────────────
_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    openai_api_key=os.getenv("NAGA_API_KEY"),
    openai_api_base="https://api.naga.ac/v1",
    request_timeout=60,
    temperature=0.3,
)

# ── 检索器（懒加载单例）──────────────────────────────────────────────
_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieval.hybrid_retriever import HybridRetriever
        _retriever = HybridRetriever()
    return _retriever


# ════════════════════════════════════════════════════════════
# 系统提示词
# ════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一位精通子平八字命理的古籍研究助手，熟读《子平真诠》《滴天髓》\
《渊海子平》《三命通会》《穷通宝鉴》等经典。

【工具使用规则】
1. 用户提供出生年月日时 → 先调用 bazi_tool 排盘，再调用 search_tool 检索相关古籍
2. 连续追问同一命盘 → 排盘结果已在消息历史中，无需重复排盘，直接 search_tool 检索
3. 纯命理知识问题 → 直接调用 search_tool 检索，无需排盘
4. 日常闲聊 → 无需调用任何工具，直接自然回答

【检索词规范】（影响召回质量，请严格遵守）
- 有八字时，使用"[日元][月令]生"或"[月令][日元]"格式
  例：丁火辰月生、三月丁火、壬水三秋生、三秋壬水
  这是穷通宝鉴章节命名格式，能精准匹配章节标题
- 禁止单独输出干支组合（如单独的"甲辰""壬午"）
  这些词也出现在时柱/年柱的章节名中，会误召回无关内容
- 使用古籍术语：七杀/偏官、调候、格局、用神、印绶、食神制杀等

【自适应重搜】
- 调用 search_tool 后，快速判断返回片段是否与问题相关
- 若明显不符（如问月令却返回了时柱章节，或日元不匹配），
  换用不同关键词再次调用 search_tool，最多重试 1 次

【回答规范】
- 引用古籍时注明来源，格式：（出自《书名·章节》）；章节名只能来自 search_tool 返回的原文标题，不得凭记忆杜撰
- 语言现代中文，清晰详尽；根据问题复杂度决定字数，知识性回答通常400-800字，八字分析可适当更长"""

SYSTEM_PROMPT_NO_RAG = """你是一位精通子平八字命理的古籍研究助手，熟读各类命理经典。

（古籍检索功能当前已关闭）

【工具使用规则】
- 只有 bazi_tool 可用：当用户提供出生年月日时，调用它排盘
- 命理知识问题：根据自身学识直接回答，无需引用古籍
- 日常闲聊：自然交流即可

回答清晰易懂，根据问题决定长短。"""


# ════════════════════════════════════════════════════════════
# 内部执行函数（供 tools_node 调用，同时获取原始数据）
# ════════════════════════════════════════════════════════════

def _run_bazi(year: int, month: int, day: int,
              hour: int, gender: str = "男") -> tuple[str, str]:
    """执行排盘，返回 (LLM可读文本, bazi_str供UI展示)"""
    try:
        result = get_bazi(year=year, month=month, day=day, hour=hour, gender=gender)
        bazi_str = (
            f"四柱：{result['年柱']['天干']}{result['年柱']['地支']}年  "
            f"{result['月柱']['天干']}{result['月柱']['地支']}月  "
            f"{result['日柱']['天干']}{result['日柱']['地支']}日  "
            f"{result['时柱']['天干']}{result['时柱']['地支']}时\n"
            f"日元：{result['日元']}  性别：{result['性别']}\n"
            f"起运：{result.get('起运', '未知')}  {result.get('顺逆', '')}"
        )
        da_yun = result.get("大运列表", [])
        if da_yun:
            dy_parts = [
                f"{dy['干支']}({dy['起运岁']}-{dy['结束岁']}岁/"
                f"{dy['起运年']}-{dy['结束年']})"
                for dy in da_yun
            ]
            bazi_str += "\n大运：" + "  ".join(dy_parts)
        return bazi_str, bazi_str
    except Exception as e:
        err = f"排盘失败：{e}"
        return err, ""


def _run_search(query: str, use_rag: bool = True) -> tuple[str, list]:
    """执行检索，返回 (LLM可读格式化文本, raw_chunks供UI展示)"""
    if not use_rag:
        return "（古籍检索已关闭）", []

    retriever = _get_retriever()
    chunks = retriever.search(query, top_k=5, mode="hybrid", rerank=False)

    parts = []
    for i, c in enumerate(chunks, 1):
        loc = c["source"]
        if c.get("chapter"):
            loc += f"·{c['chapter']}"
        if c.get("section"):
            loc += f"·{c['section']}"
        parts.append(f"[{i}] 《{loc}》\n{c['original'][:400]}")

    text = "\n\n".join(parts) or "未检索到相关内容"
    return text, chunks


# ════════════════════════════════════════════════════════════
# LangChain Tool 定义（LLM 看到的接口描述）
# ════════════════════════════════════════════════════════════

@tool
def bazi_tool(year: int, month: int, day: int,
              hour: int, gender: str = "男") -> str:
    """
    计算八字四柱大运排盘。当用户提供出生年月日时辰时调用。
    hour 时辰对照：子=0或23, 丑=1-2, 寅=3-4, 卯=5-6, 辰=7-8, 巳=9-10,
                   午=11-12, 未=13-14, 申=15-16, 酉=17-18, 戌=19-20, 亥=21-22
    返回：四柱（年月日时柱）、日元、起运信息、大运列表
    """
    text, _ = _run_bazi(year, month, day, hour, gender)
    return text


@tool
def search_tool(query: str) -> str:
    """
    检索命理古籍（子平真诠、滴天髓、穷通宝鉴、三命通会等）中的相关原文片段。
    有八字时 query 应使用"[日元][月令]生"格式，如"丁火辰月生 三月丁火 调候用神"。
    返回：相关古籍原文及书名章节信息。
    """
    text, _ = _run_search(query)
    return text


TOOLS = [bazi_tool, search_tool]


# ════════════════════════════════════════════════════════════
# Agent 节点（主推理）
# ════════════════════════════════════════════════════════════

def agent_node(state: AgentState) -> dict:
    """主推理节点：LLM 决定调用哪些工具，或直接给出最终回答"""
    use_rag = state.get("use_rag", True)

    # use_rag=False 时不绑定 search_tool，LLM 看不到它就不会调用
    tools      = TOOLS if use_rag else [bazi_tool]
    sys_prompt = SYSTEM_PROMPT if use_rag else SYSTEM_PROMPT_NO_RAG

    llm_with_tools = _llm.bind_tools(tools)
    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ════════════════════════════════════════════════════════════
# Tools 执行节点
# ════════════════════════════════════════════════════════════

def tools_node(state: AgentState) -> dict:
    """
    执行上一步 agent_node 发出的工具调用。
    同时将排盘结果和检索片段写入 state，供 Streamlit UI 展示。
    """
    last_msg   = state["messages"][-1]
    use_rag    = state.get("use_rag", True)
    tool_msgs  = []
    new_chunks   = state.get("chunks", [])
    new_bazi_str = state.get("bazi_str", "")

    for tc in last_msg.tool_calls:
        name = tc["name"]
        args = tc["args"]

        if name == "bazi_tool":
            text, bazi = _run_bazi(**args)
            new_bazi_str = bazi
            tool_msgs.append(
                ToolMessage(content=text, tool_call_id=tc["id"], name=name)
            )

        elif name == "search_tool":
            text, chunks = _run_search(args.get("query", ""), use_rag)
            new_chunks = chunks
            tool_msgs.append(
                ToolMessage(content=text, tool_call_id=tc["id"], name=name)
            )

    return {
        "messages":  tool_msgs,
        "chunks":    new_chunks,
        "bazi_str":  new_bazi_str,
    }


# ════════════════════════════════════════════════════════════
# critic_node（Self-Critique，默认关闭）
# ════════════════════════════════════════════════════════════

CRITIQUE_PROMPT = """你是一个命理回答质量审查员。

用户问题：{query}

实际检索到的古籍片段（这是唯一合法的引用来源）：
{chunks_text}

待审查的回答：
{draft}

审查步骤：
1. 扫描回答中所有带书名或章节的引用论断（含"出自《xxx》""《xxx》载""《xxx·章节》指出"等格式）
2. 对照检索片段逐条核实：
   a. 能找到该书对应内容，但章节名不符 → 保留内容，只保留书名，删去具体章节名
   b. 完全找不到对应内容，属于凭记忆引用 → 删除该条引用标注，将论断改为无引用的一般性表述
3. 无任何引用标注的一般性命理知识 → 全部保留，不做处理

直接输出修正后的最终回答，不要输出审查过程和说明。"""


def critic_node(state: AgentState) -> dict:
    """Self-Critique：核验回答中的章节引用是否来自实际检索结果（use_critic=True 时生效）"""
    # 提取最终草稿（最后一条 AIMessage）
    draft = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            draft = msg.content
            break

    # 提取用户问题（最后一条 HumanMessage）
    from langchain_core.messages import HumanMessage
    query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    # 格式化检索片段
    chunks_text_parts = []
    for i, c in enumerate(state.get("chunks", []), 1):
        loc = c["source"]
        if c.get("chapter"):
            loc += f"·{c['chapter']}"
        if c.get("section"):
            loc += f"·{c['section']}"
        chunks_text_parts.append(f"[{i}] 《{loc}》：{c['original'][:200]}")
    chunks_text = "\n".join(chunks_text_parts) or "（无检索片段）"

    prompt = CRITIQUE_PROMPT.format(
        query=query,
        chunks_text=chunks_text,
        draft=draft,
    )
    response = _llm.invoke(prompt)
    # 用修正后的回答替换最后一条 AIMessage
    return {"messages": [AIMessage(content=response.content.strip())]}


# ════════════════════════════════════════════════════════════
# 路由函数
# ════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> str:
    """
    路由逻辑：
    - 最后消息有工具调用 → tools
    - 没有工具调用 + use_critic=True + 有检索片段 → critic
    - 其他 → end
    """
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    if state.get("use_critic", False) and state.get("chunks"):
        return "critic"
    return "end"
