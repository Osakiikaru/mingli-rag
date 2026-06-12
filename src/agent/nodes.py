"""
LangGraph 各节点实现
--------------------
节点列表：
  1. intent_parser    — 解析用户意图（chat / knowledge / personal），提取生辰，精炼检索词
  2. chat_node        — 普通对话，直接 LLM 回答，不查古籍
  3. bazi_node        — 调用排盘工具，输出四柱八字 + 大运
  4. query_rewriter   — 排盘后，根据日元月令重构专业古籍检索词
  5. retriever_node   — 混合检索 + Reranker，召回 Top-5 古籍片段
  6. generator_node   — DeepSeek 读古籍生成回答
  7. critic_node      — Self-Critique：验证回答是否有据可查
"""

import json
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.agent.state import AgentState
from src.tools.bazi import get_bazi

load_dotenv()

# ── 系统提示词（所有节点共用人格）───────────────────────────
SYSTEM_PROMPT = (
    "你是一位精通子平八字命理的古籍研究助手，熟读《子平真诠》《滴天髓》"
    "《渊海子平》《三命通会》《穷通宝鉴》等经典。你博学而亲切，"
    "既能严谨引用古籍原文解答命理问题，也能轻松陪用户日常聊天。"
    "回答命理问题时务必注明古籍来源；日常对话则自然交流，不必强行扯到命理。"
)

# ── LLM 客户端 ───────────────────────────────────────────────
_BASE_KWARGS = dict(
    model="deepseek-v4-flash",
    openai_api_key=os.getenv("NAGA_API_KEY"),
    openai_api_base="https://api.naga.ac/v1",
    request_timeout=60,
)
# 分析类节点：低温，严谨
_llm = ChatOpenAI(**_BASE_KWARGS, temperature=0.3)
# 对话类节点：高温，自然
_llm_chat = ChatOpenAI(**_BASE_KWARGS, temperature=0.7)

# ── 检索器（懒加载单例）─────────────────────────────────────
_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.retrieval.hybrid_retriever import HybridRetriever
        _retriever = HybridRetriever()
    return _retriever


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

def _format_history(chat_history: list) -> str:
    """将完整对话历史格式化为文本供 LLM 读取（不截断）"""
    if not chat_history:
        return "（无历史对话）"
    lines = []
    for msg in chat_history:
        role    = "用户" if msg["role"] == "user" else "助手"
        content = str(msg.get("content", ""))
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# Node 1: 意图解析
# ════════════════════════════════════════════════════════════

INTENT_PROMPT = """你是一个命理问答系统的意图解析器。分析用户的问题，提取关键信息。

对话历史（完整上下文）：
{history_text}

用户当前问题：{query}

请严格按以下 JSON 格式返回（不要输出任何其他内容）：
{{
  "query_type": "chat"或"knowledge"或"personal",
  "needs_bazi": true或false,
  "birth_info": {{
    "year": 公历年份整数,
    "month": 月份整数1-12,
    "day": 日期整数1-31,
    "hour": 小时整数0-23（子时=0，丑时=1，寅时=3，卯时=5，辰时=7，巳时=9，午时=11，未时=13，申时=15，酉时=17，戌时=19，亥时=21），
    "gender": "男"或"女"
  }},
  "search_query": "提炼出的命理检索关键词，3-8个词，聚焦核心命理问题"
}}

规则：
- query_type="chat"：日常问候、情感表达、与命理无关的闲聊（如"你好""谢谢""讲个笑话"）
- query_type="knowledge"：命理理论、格局解释、古籍知识等通用问题（不涉及某人八字）
- query_type="personal"：用户询问自身或他人的命运、运势、格局等个人分析问题
- 【重要】当用户使用「我的命盘」「我的盘」「我的八字」「此造」「这个命」「帮我看」「我的格局」「我的大运」等指代性词语时，即使当前消息没有生辰数字，也必须判定为 query_type="personal"，并从对话历史中提取生辰
- needs_bazi=true：当前消息或对话历史中能找到明确的出生年月日；若历史中有多个人的生辰，提取与当前问题最相关的那个人的生辰
- 若对话历史中已讨论过某人命盘，用户当前问题又与该命盘相关（如问大运、案例对比、喜用神等），则 needs_bazi=true，从历史提取生辰
- 若 needs_bazi=false，birth_info 所有字段填 null
- 性别未提及时默认 "男"
- query_type="chat" 时，search_query 填空字符串""
- search_query 去掉出生信息，只保留命理问题关键词"""


def intent_parser_node(state: AgentState) -> dict:
    """解析用户意图，结合完整对话历史提取生辰和检索词"""
    history_text = _format_history(state.get("chat_history", []))
    prompt = INTENT_PROMPT.format(
        query=state["user_query"],
        history_text=history_text,
    )
    response = _llm.invoke(prompt)
    text = response.content.strip()

    # 去掉可能的 markdown 代码块
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {
            "query_type": "knowledge",
            "needs_bazi": False,
            "birth_info": {},
            "search_query": state["user_query"],
        }

    return {
        "query_type":   parsed.get("query_type", "knowledge"),
        "needs_bazi":   parsed.get("needs_bazi", False),
        "birth_info":   parsed.get("birth_info") or {},
        "search_query": parsed.get("search_query", state["user_query"]),
    }


# ════════════════════════════════════════════════════════════
# Node 2: 普通对话（chat 路径）
# ════════════════════════════════════════════════════════════

CHAT_PROMPT = """对话历史：
{history_text}

用户：{query}

请自然地回答，字数根据问题决定，不要强行扯到命理。"""


def chat_node(state: AgentState) -> dict:
    """普通聊天节点，直接 LLM 回答，不查古籍"""
    history_text = _format_history(state.get("chat_history", []))
    prompt = CHAT_PROMPT.format(
        history_text=history_text,
        query=state["user_query"],
    )
    response = _llm_chat.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return {"final_answer": response.content.strip()}


# ════════════════════════════════════════════════════════════
# Node 3: 排盘
# ════════════════════════════════════════════════════════════

def bazi_node(state: AgentState) -> dict:
    """调用 lunar-python 排盘，返回格式化四柱 + 大运字符串"""
    bi = state["birth_info"]
    try:
        result = get_bazi(
            year=int(bi["year"]),
            month=int(bi["month"]),
            day=int(bi["day"]),
            hour=int(bi["hour"]),
            gender=bi.get("gender", "男"),
        )

        # 四柱基础
        bazi_str = (
            f"四柱：{result['年柱']['天干']}{result['年柱']['地支']}年  "
            f"{result['月柱']['天干']}{result['月柱']['地支']}月  "
            f"{result['日柱']['天干']}{result['日柱']['地支']}日  "
            f"{result['时柱']['天干']}{result['时柱']['地支']}时\n"
            f"日元：{result['日元']}  性别：{result['性别']}\n"
            f"起运：{result.get('起运', '未知')}  {result.get('顺逆', '')}"
        )

        # 大运列表
        da_yun = result.get("大运列表", [])
        if da_yun:
            dy_parts = []
            for dy in da_yun:
                dy_parts.append(
                    f"{dy['干支']}({dy['起运岁']}-{dy['结束岁']}岁/"
                    f"{dy['起运年']}-{dy['结束年']})"
                )
            bazi_str += "\n大运：" + "  ".join(dy_parts)

    except Exception as e:
        bazi_str = f"排盘失败：{e}"

    return {"bazi_str": bazi_str}


# ════════════════════════════════════════════════════════════
# Node 4: 排盘后查询重构
# ════════════════════════════════════════════════════════════

QUERY_REWRITE_PROMPT = """你是一位命理古籍检索专家。根据用户问题和八字排盘结果，生成最适合检索古籍的专业短语。

用户问题：{query}
八字排盘：{bazi_str}

要求：
- 输出5-8个命理专业词汇或短语，用空格分隔
- 必须用"[日元][月令]生"或"[月令][日元]"的短语形式表达日元与月令的关系
  例如：丁火辰月生、三月丁火、壬水秋月生、三秋壬水
  （这是穷通宝鉴等古籍的章节命名格式，能精确匹配章节标题）
- 禁止单独输出四柱的干支组合（如单独的"甲辰""壬午"），
  因为这些词也出现在时柱/年柱章节名中，会造成误召回
- 不要使用"事业运势""发展方向"这类现代口语
- 聚焦用户问题的核心命理概念（格局、用神、调候等）

只输出关键词和短语，不要任何其他内容："""


def query_rewriter_node(state: AgentState) -> dict:
    """排盘后，根据日元+月令+用户问题重构专业古籍检索词"""
    prompt = QUERY_REWRITE_PROMPT.format(
        query=state["user_query"],
        bazi_str=state["bazi_str"],
    )
    response = _llm.invoke(prompt)
    return {"search_query": response.content.strip()}


# ════════════════════════════════════════════════════════════
# Node 5: 检索
# ════════════════════════════════════════════════════════════

def retriever_node(state: AgentState) -> dict:
    """混合检索召回 Top-5 相关古籍片段（use_rag=False 时跳过）"""
    if not state.get("use_rag", True):
        return {"chunks": []}
    retriever = _get_retriever()
    chunks = retriever.search(state["search_query"], top_k=5, mode="hybrid", rerank=False)
    return {"chunks": chunks}


# ════════════════════════════════════════════════════════════
# Node 6: 生成回答
# ════════════════════════════════════════════════════════════

GENERATE_PROMPT = """你是一位精通子平八字命理的专家，根据提供的古籍原文回答用户问题。

对话历史（保持连贯性）：
{history_text}

用户问题：{query}
{bazi_section}
── 检索到的相关古籍片段 ──────────────────────────
{chunks_text}
──────────────────────────────────────────────────

要求：
1. 只根据上方古籍片段作答，不要自行发挥没有依据的内容
2. 引用时注明来源，格式：（出自《书名·章节》），章节名必须来自上方片段中出现的原文标题，不得引用记忆中的章节名
3. 回答用现代中文，清晰易懂，600-1000字，内容充实详尽
4. 结合对话历史，保持回答的连贯性和上下文一致性
5. 如有大运信息，结合大运走势进行分析
6. 如果古籍片段与问题相关性不足，如实说明，并给出力所能及的分析"""

GENERATE_PROMPT_NO_RAG = """你是一位精通子平八字命理的专家，凭借自身学识直接回答用户问题。

对话历史（保持连贯性）：
{history_text}

用户问题：{query}
{bazi_section}
要求：
1. 根据你掌握的命理知识作答，观点清晰，言之有物
2. 回答用现代中文，清晰易懂，不必拘于字数
3. 结合对话历史，保持回答的连贯性和上下文一致性
4. 如有大运信息，结合大运走势进行分析"""


def generator_node(state: AgentState) -> dict:
    """基于检索结果和对话历史生成命理分析回答"""
    history_text = _format_history(state.get("chat_history", []))

    bazi_section = ""
    if state.get("bazi_str"):
        bazi_section = f"\n命主八字信息：\n{state['bazi_str']}\n"

    use_rag = state.get("use_rag", True)

    if use_rag:
        chunks_text_parts = []
        for i, c in enumerate(state.get("chunks", []), 1):
            loc = c["source"]
            if c.get("chapter"):
                loc += f"·{c['chapter']}"
            if c.get("section"):
                loc += f"·{c['section']}"
            preview = c["original"][:400]
            chunks_text_parts.append(f"[{i}] 《{loc}》\n{preview}")
        chunks_text = "\n\n".join(chunks_text_parts) or "未检索到相关内容"

        prompt = GENERATE_PROMPT.format(
            history_text=history_text,
            query=state["user_query"],
            bazi_section=bazi_section,
            chunks_text=chunks_text,
        )
    else:
        prompt = GENERATE_PROMPT_NO_RAG.format(
            history_text=history_text,
            query=state["user_query"],
            bazi_section=bazi_section,
        )

    response = _llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return {"draft_answer": response.content.strip()}


# ════════════════════════════════════════════════════════════
# Node 7: Self-Critique
# ════════════════════════════════════════════════════════════

CRITIQUE_PROMPT = """你是一个命理回答质量审查员。

用户问题：{query}

古籍片段（检索来源）：
{chunks_text}

待审查的回答：
{draft}

请检查：回答中的命理论断是否都能在古籍片段中找到依据？
- 如果基本有据可查：直接输出原回答，末尾加一行「【来源核验：通过】」
- 如果有明显无中生有的内容：指出哪些内容缺乏依据，输出修正后的回答，末尾加「【来源核验：已修正】」

直接输出最终回答，不要输出审查过程。"""


def critic_node(state: AgentState) -> dict:
    """Self-Critique：验证回答是否有古籍依据（use_rag=False 时直接透传）"""
    if not state.get("use_rag", True):
        return {"final_answer": state.get("draft_answer", "")}

    chunks_text_parts = []
    for i, c in enumerate(state.get("chunks", []), 1):
        loc = c["source"]
        if c.get("chapter"):
            loc += f"·{c['chapter']}"
        chunks_text_parts.append(f"[{i}] 《{loc}》：{c['original'][:200]}")
    chunks_text = "\n".join(chunks_text_parts)

    prompt = CRITIQUE_PROMPT.format(
        query=state["user_query"],
        chunks_text=chunks_text,
        draft=state["draft_answer"],
    )

    response = _llm.invoke(prompt)
    return {"final_answer": response.content.strip()}
