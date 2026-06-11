# 命理测算RAG — 最终收尾计划（第四周）

> 目标：本周结项，推上 GitHub，可以在面试中完整展示
> 核心原则：做完比做好重要，每件事都要能在面试中说出"我为什么这么做"

---

## 当前状态（截至计划制定时）

### ✅ 已完成
- 7本古籍 → 1313个chunk（BM25双字段 + ChromaDB向量索引）
- 混合检索：BM25 + BGE-M3向量 + RRF融合 + CrossEncoder精排
- LangGraph Agent：3路路由，7个节点（意图→排盘→查询重构→检索→生成→自我核验）
- Streamlit 多轮对话界面（进度反馈、导出功能）
- 20道测试题 + ground truth
- 4档消融实验（LLM-as-Judge，正在运行中）

### ❌ 原计划遗漏
- Self-Critique 消融对比（第5档）
- FastAPI HTTP 接口
- LangSmith 全链路追踪
- GitHub README + 架构图
- `.gitignore`（API key 不能提交）

---

## 任务优先级（按重要性排序）

| 优先级 | 任务 | 预计耗时 | 面试价值 |
|--------|------|----------|---------|
| P0 | 等消融实验跑完，整理结果 | 0.5h | 核心数据 |
| P0 | GitHub README + 架构图 | 3-4h | 面试官第一眼看的东西 |
| P0 | .gitignore，确保 API key 不上传 | 10min | 安全意识 |
| P1 | Self-Critique 消融对比（第5档） | 1-2h | 证明设计有效 |
| P1 | LangSmith 接入 | 2-3h | 可观测性技术栈 |
| P2 | FastAPI 接口 | 2-3h | 生产化意识 |
| P3 | chunk 质量抽查 | 1h | 数据质量意识 |

---

## 每个任务详细说明

---

### P0｜Self-Critique 消融对比

**是什么**：在现有4档消融实验里加第5档——关掉critic节点，对比有无Self-Critique时 Faithfulness 的变化。

**怎么做**：

1. 在 `evaluate_ragas.py` 的 EVAL_CONFIGS 里加第5个配置：
```python
{"name": "hybrid_rerank_no_critic", "label": "Hybrid+Rerank (无Self-Critique)", 
 "mode": "hybrid", "rerank": True}
```

2. 在 `run_config()` 里加一个参数 `skip_critic=True`，生成时直接用 generator_node 的输出不经过 critic_node

3. 跑这一档的 Faithfulness + Answer Relevancy（需要 `--no-gen` 不能跳过）

**面试怎么说**：
> "消融实验里我专门对比了有无Self-Critique的Faithfulness，有critic节点时从X提升到Y，说明它确实在过滤没有古籍依据的论断。"

---

### P0｜GitHub README

**必须包含的内容**：

```
# 命理古籍研究助手

## 项目简介（2-3句话）

## 系统架构图
（手画或用 draw.io/excalidraw，PNG 插入）
节点：用户输入 → 意图解析 → [三路路由] → ... → 输出

## 技术栈
- 排盘：lunar-python（确定性算法，100%准确）
- 检索：BM25(jieba) + BGE-M3向量 + RRF融合 + CrossEncoder精排
- Agent编排：LangGraph（有状态图，7节点3路由）
- 向量库：ChromaDB
- 评估：LLM-as-Judge（自研，基于DeepSeek）
- 界面：Streamlit
- 追踪：LangSmith
- API：FastAPI

## 消融实验结果（表格）
（把 data/eval_results/summary.md 的内容放这里）

## 项目亮点
1. Tool Calling 排盘：...
2. 两层存储（原文+注解）：...
3. Self-Critique 防幻觉：...
4. 完整评估体系：...

## 快速开始
pip install -r requirements.txt
cp .env.example .env  # 填入 API key
streamlit run app.py
```

**架构图建议用这个工具**：https://excalidraw.com（免费，画完导出PNG）

---

### P1｜LangSmith 接入

**是什么**：LangChain 官方的 LLM 应用追踪平台。每次对话都能在网页上看到完整调用链——哪个节点花了多久、每个 LLM 调用的 prompt 和 response、token 消耗。

**注册地址**：https://smith.langchain.com（免费账号）

**怎么接**：只需在 `.env` 里加4行环境变量，LangGraph 自动发送 trace，代码不用改：

```bash
# .env 里加：
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=你的key（从网站获取）
LANGCHAIN_PROJECT=命理测算RAG
```

同时在 `app.py` 最顶部（环境变量那里）加上：
```python
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
```

**验证**：跑一次 Streamlit 问一个问题，去 LangSmith 网站看 trace 有没有出现。

**面试怎么说**：
> "我接入了 LangSmith 做全链路追踪，可以看到每个节点的延迟和token消耗。比如检索节点平均Xms，生成节点平均Xms，这帮助我定位了性能瓶颈在哪里。"

最好截一张 LangSmith trace 的截图放进 README。

---

### P2｜FastAPI 接口

**是什么**：把 LangGraph Agent 包装成 HTTP API，让其他程序可以调用，是生产环境的标准架构。

**目标接口**（只需实现这一个就够）：
```
POST /chat
Content-Type: application/json

{
  "message": "七杀格如何制化？",
  "session_id": "user_001",
  "history": []
}

→ 返回：
{
  "answer": "...",
  "sources": ["《子平真诠》·论七杀", ...],
  "bazi_str": "",
  "query_type": "knowledge"
}
```

**文件位置**：新建 `src/api/main.py`

**关键代码骨架**：
```python
from fastapi import FastAPI
from pydantic import BaseModel
from src.agent.graph import mingli_graph

app = FastAPI(title="命理古籍研究助手 API")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    history: list = []

@app.post("/chat")
def chat(req: ChatRequest):
    initial_state = {
        "user_query": req.message,
        "chat_history": req.history,
        # ... 其他字段
    }
    result = mingli_graph.invoke(initial_state)
    return {
        "answer": result["final_answer"],
        "sources": [...],
        "query_type": result["query_type"],
    }
```

**运行方式**：`uvicorn src.api.main:app --reload`

**面试怎么说**：
> "Streamlit 是演示用的前端，FastAPI 是真正的后端接口，符合前后端分离的生产架构。如果要接入微信/飞书等平台，只需要对接这个 API 就行。"

---

### P3｜.gitignore 和安全检查

```gitignore
# .gitignore 必须包含：
.env
data/chroma_db/
models/
__pycache__/
*.pyc
data/eval_results/
```

**上传前检查**：
```
git grep -r "NAGA_API_KEY" --name-only  # 确保没有文件包含真实 key
git grep -r "ng-" --name-only           # 确保没有 key 前缀泄露
```

---

## 收尾 Checklist

```
[ ] 消融实验全量结果出来（--no-gen 版本）
[ ] 补 Self-Critique 对比（第5档，需要生成）
[ ] 注册 LangSmith，接入 .env，验证 trace
[ ] 新建 FastAPI main.py，测试 /chat 接口
[ ] 写 README（架构图 + 技术栈 + 消融结果表格）
[ ] 检查 .gitignore，确认 .env 不在 git 里
[ ] git push 到 GitHub
[ ] 用手机打开 GitHub 项目页，模拟面试官视角检查一遍
```

---

## 面试时的技术栈清单（最终版）

说出这些，面试官会觉得技术面很广：

| 类别 | 技术 | 能说的亮点 |
|------|------|-----------|
| Agent编排 | LangGraph | 有状态图，节点职责单一，路由清晰 |
| 排盘 | lunar-python | 确定性算法，解决LLM排盘不准的痛点 |
| Embedding | BGE-M3 | 中文效果最佳的开源模型 |
| 稀疏检索 | BM25 + jieba | 自定义命理词典防术语拆散 |
| 融合算法 | RRF | 两路检索无需调权重的融合方案 |
| 精排 | CrossEncoder (BGE-Reranker) | 候选20取5，精度进一步提升 |
| 向量库 | ChromaDB | 轻量本地部署，无需独立服务 |
| 防幻觉 | Self-Critique节点 | 有实验数据证明Faithfulness提升 |
| 追踪 | LangSmith | 全链路可观测，节点级延迟可见 |
| 后端 | FastAPI | 生产级HTTP接口，前后端分离 |
| 评估 | LLM-as-Judge（自研） | 理解指标原理，不只是调包 |
| 界面 | Streamlit | 快速原型，多轮对话+进度反馈 |

---

## 关于"古籍太少"的口头应对

> "RAG 技术的核心价值不在于语料量，而在于**可追溯性**——每条回答都能定位到具体古籍的章节，这在知识严谨性要求高的专业场景里是必须的。项目当前用7本古籍完整演示了从检索到评估的全技术链路，语料可以持续扩充而不需要改动任何代码。"

---

*计划制定时间：2026-06-10*
*预计结项时间：本周末*
