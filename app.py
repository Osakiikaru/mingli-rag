"""
命理古籍研究助手 — Streamlit 多轮对话界面
------------------------------------------
运行方式：streamlit run app.py --server.fileWatcherType none
"""
# ⚠️ 禁用 tokenizers 并行，避免 Windows 上 sentence_transformers 导入崩溃
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# ⚠️ LangSmith 追踪：必须在 LangGraph 导入前加载 .env，否则追踪不生效
from dotenv import load_dotenv
load_dotenv()  # 读取 .env 中的 LANGCHAIN_API_KEY / LANGCHAIN_TRACING_V2 等

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from src.agent.graph import mingli_graph

# ── 页面配置 ────────────────────────────────────────────────
st.set_page_config(
    page_title="命理古籍研究助手",
    page_icon="📚",
    layout="wide",
)

st.title("📚 命理古籍研究助手")
st.caption("基于《子平真诠》《滴天髓》等古籍的命理学习与研究工具 · 每条回答均有古籍出处")

# ── 节点进度标签 ─────────────────────────────────────────────
_NODE_LABELS = {
    "intent_parser":       "解析意图",
    "chat_node":           "思考中",
    "bazi_node":           "排盘中",
    "query_rewriter_node": "生成检索词",
    "retriever_node":      "查阅古籍",
    "generator_node":      "生成回答",
    "critic_node":         "来源核验",
}

# ── 会话状态初始化 ───────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── 侧边栏 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 使用说明")
    st.markdown(
        "- **知识问答**：直接提问，例如：\n"
        "  「七杀格如何制化？」\n\n"
        "- **个人命理**：提供生辰后提问，例如：\n"
        "  「我1990年3月15日午时出生，男，事业运势如何？」\n\n"
        "- **连续对话**：提供过生辰后，后续提问无需重复填写，\n"
        "  系统会自动从历史中读取。\n\n"
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

    # 传全量历史（不切片）
    full_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history
    ]

    initial_state = {
        "user_query":   user_input,
        "chat_history": full_history,
        "use_rag":      use_rag,
        "query_type":   "",
        "needs_bazi":   False,
        "birth_info":   {},
        "search_query": "",
        "bazi_str":     "",
        "chunks":       [],
        "draft_answer": "",
        "final_answer": "",
    }

    sources   = []
    bazi_str  = ""
    answer    = ""
    final_state = {}

    with st.chat_message("assistant"):
        # 用 st.status 展示分步进度
        with st.status("正在处理...", expanded=False) as status:
            for chunk in mingli_graph.stream(initial_state):
                node_name = list(chunk.keys())[0]
                label = _NODE_LABELS.get(node_name, node_name)
                status.update(label=f"✅ {label}")
                final_state.update(chunk.get(node_name, {}))
            status.update(label="完成", state="complete", expanded=False)

        answer = final_state.get("final_answer", "")
        st.markdown(answer)

        # 整理古籍来源
        seen = set()
        for c in final_state.get("chunks", []):
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

        bazi_str = final_state.get("bazi_str", "")
        if bazi_str:
            with st.expander("🔢 排盘详情"):
                st.code(bazi_str, language=None)

    # 更新对话历史
    st.session_state.chat_history.append({
        "role":    "user",
        "content": user_input,
    })
    st.session_state.chat_history.append({
        "role":     "assistant",
        "content":  answer,
        "sources":  sources,
        "bazi_str": bazi_str,
    })
