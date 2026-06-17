"""
命理古籍研究助手 — Streamlit 多轮对话界面（ReAct 版）
------------------------------------------------------
运行方式：streamlit run app.py --server.fileWatcherType none
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# LangSmith 追踪：必须在 LangGraph 导入前加载 .env
from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph import mingli_graph

# ── 页面配置 ────────────────────────────────────────────────
st.set_page_config(
    page_title="命理古籍研究助手",
    page_icon="📚",
    layout="wide",
)

st.title("📚 命理古籍研究助手")
st.caption("基于《子平真诠》《滴天髓》等古籍的命理学习与研究工具 · 每条回答均有古籍出处")

# ── 会话状态初始化 ───────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # [{role, content, sources, bazi_str}, ...]

# ── 侧边栏 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 使用说明")
    st.markdown(
        "- **知识问答**：直接提问，例如：\n"
        "  「七杀格如何制化？」\n\n"
        "- **个人命理**：提供生辰后提问，例如：\n"
        "  「我1990年3月15日午时出生，男，事业运势如何？」\n\n"
        "- **连续对话**：提供过生辰后，后续提问无需重复填写，\n"
        "  助手会自动从历史中读取。\n\n"
        "- **日常聊天**：也可以随意聊天，不限于命理话题。"
    )
    st.divider()

    # RAG 开关
    use_rag = st.toggle(
        "📚 启用古籍检索",
        value=True,
        help="关闭后由 DeepSeek 直接凭自身知识回答，不检索古籍",
    )
    if not use_rag:
        st.caption("⚠️ 古籍检索已关闭，回答不含古籍引用")

    # Self-Critique 开关（默认关闭，开启后多一次 LLM 调用核验引用）
    use_critic = st.toggle(
        "🔍 来源核验（Self-Critique）",
        value=False,
        help="开启后对回答进行二次核验，删除无古籍依据的引用，但会增加约10秒延迟",
    )

    st.divider()

    # 导出对话
    if st.session_state.chat_history:
        def _export_md() -> str:
            lines = ["# 命理古籍研究助手 — 对话记录\n"]
            for msg in st.session_state.chat_history:
                role = "**用户**" if msg["role"] == "user" else "**助手**"
                lines.append(f"{role}：\n\n{msg['content']}\n")
                if msg.get("sources"):
                    lines.append("📖 古籍来源：" + "、".join(msg["sources"]) + "\n")
                if msg.get("bazi_str"):
                    lines.append(f"🔢 排盘详情：\n```\n{msg['bazi_str']}\n```\n")
                lines.append("---\n")
            return "\n".join(lines)

        st.download_button(
            label="📥 导出对话（MD）",
            data=_export_md(),
            file_name="mingli_chat.md",
            mime="text/markdown",
        )

    if st.button("🗑️ 清空对话"):
        st.session_state.chat_history = []
        st.rerun()

# ── 展示历史对话 ─────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("sources"):
                with st.expander("📖 古籍来源"):
                    for source in msg["sources"]:
                        st.markdown(f"- {source}")
            if msg.get("bazi_str"):
                with st.expander("🔢 排盘详情"):
                    st.code(msg["bazi_str"], language=None)

# ── 用户输入 ─────────────────────────────────────────────────
if user_input := st.chat_input("有什么想聊的？命理问题或日常闲聊均可"):

    with st.chat_message("user"):
        st.markdown(user_input)

    # 构建 LangGraph 消息历史（HumanMessage + AIMessage 交替）
    lc_messages = []
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))
    lc_messages.append(HumanMessage(content=user_input))

    initial_state = {
        "messages":   lc_messages,
        "use_rag":    use_rag,
        "use_critic": use_critic,
        "chunks":     [],
        "bazi_str":   "",
    }

    # ── 流式处理 + 进度展示 ────────────────────────────────
    final_answer = ""
    final_chunks  = []
    final_bazi    = ""
    sources       = []

    with st.chat_message("assistant"):
        with st.status("正在处理...", expanded=False) as status:
            for chunk in mingli_graph.stream(initial_state):
                node_name = list(chunk.keys())[0]
                node_data = chunk.get(node_name, {})

                if node_name == "agent":
                    msgs = node_data.get("messages", [])
                    if msgs:
                        last = msgs[-1]
                        if hasattr(last, "tool_calls") and last.tool_calls:
                            # LLM 正在调用工具——显示调的是什么
                            names = [tc["name"] for tc in last.tool_calls]
                            if "bazi_tool" in names:
                                status.update(label="🔢 排盘中...")
                            if "search_tool" in names:
                                status.update(label="📚 查阅古籍...")
                        else:
                            # 没有工具调用 = 最终回答
                            final_answer = last.content
                            status.update(label="✅ 生成回答")

                elif node_name == "tools":
                    if node_data.get("bazi_str"):
                        final_bazi = node_data["bazi_str"]
                    if node_data.get("chunks") is not None:
                        final_chunks = node_data["chunks"]

                elif node_name == "critic":
                    status.update(label="🔍 来源核验...")
                    msgs = node_data.get("messages", [])
                    if msgs:
                        final_answer = msgs[-1].content

            status.update(label="完成", state="complete", expanded=False)

        st.markdown(final_answer)

        # 整理古籍来源
        seen = set()
        for c in final_chunks:
            loc = f"《{c['source']}》"
            if c.get("chapter"):
                loc += f" · {c['chapter']}"
            if loc not in seen:
                seen.add(loc)
                sources.append(loc)

        if sources:
            with st.expander("📖 古籍来源"):
                for s in sources:
                    st.markdown(f"- {s}")

        if final_bazi:
            with st.expander("🔢 排盘详情"):
                st.code(final_bazi, language=None)

    # 更新对话历史
    st.session_state.chat_history.append({
        "role":    "user",
        "content": user_input,
    })
    st.session_state.chat_history.append({
        "role":     "assistant",
        "content":  final_answer,
        "sources":  sources,
        "bazi_str": final_bazi,
    })
