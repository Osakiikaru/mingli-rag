# 命理古籍研究助手

> 基于 RAG + LangGraph Agent 的子平八字命理知识问答系统

一个将7本古典命理典籍结构化入库、通过混合检索召回相关原文、由 LLM 生成有据可查回答的端到端 RAG 应用。核心价值在于**可追溯性**——每条回答都能定位到具体古籍章节，而非 LLM 的参数记忆。

---

## 系统架构

```
用户输入
   │
   ▼
┌──────────────────────────────────────────────┐
│  LangGraph Agent（有状态图，7节点3路由）        │
│                                               │
│  intent_parser ──── 三路路由 ────────────────│
│       │                                       │
│       ├── [chat]      → chat_node             │
│       │                    ↓                  │
│       ├── [bazi]      → bazi_node             │
│       │                (lunar-python排盘)      │
│       │                    ↓                  │
│       └── [knowledge] → query_rewriter_node   │
│                              ↓                │
│                         retriever_node        │
│                    ┌─────────────────┐        │
│                    │  BGE-M3 向量     │        │
│                    │  BM25 + jieba   │        │
│                    │  RRF 融合        │        │
│                    │  CrossEncoder   │        │
│                    └─────────────────┘        │
│                              ↓                │
│                         generator_node        │
│                              ↓                │
│                         critic_node           │
│                        (Self-Critique)        │
└──────────────────────────────────────────────┘
   │
   ▼
Streamlit 多轮对话界面（进度反馈 + 导出）
```

> 架构图详见 `docs/architecture.png`

---

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **Agent 编排** | LangGraph | 有状态图，节点职责单一，条件路由 |
| **排盘工具** | lunar-python | 确定性算法，100% 准确，解决 LLM 排盘错误问题 |
| **稀疏检索** | BM25 + jieba | 自定义命理词典 + 查询同义词扩展（三秋↔秋月） |
| **向量检索** | BGE-M3（本地） | BAAI 出品，中文最优开源向量模型，1024维 |
| **融合算法** | RRF（k=60） | 无需归一化的多路检索融合，无额外超参数 |
| **精排** | BGE-Reranker-v2-m3 | CrossEncoder 架构，20候选→top-5 |
| **向量库** | ChromaDB | 轻量本地部署，无需独立服务 |
| **防幻觉** | Self-Critique 节点 | 生成后二次核验，Faithfulness +5.5%（实验验证） |
| **LLM** | DeepSeek-v4-flash | Agent 生成节点，中文命理理解优秀 |
| **评估** | 自研 LLM-as-Judge | 4指标自定义实现，不依赖 ragas 库 |
| **界面** | Streamlit | 多轮对话，分步进度反馈，历史导出 |
| **后端** | FastAPI | HTTP 接口，前后端分离，支持第三方集成 |
| **追踪** | LangSmith | 全链路可观测，节点级延迟与 token 消耗 |

---

## 消融实验结果

**评估集**：20道子平命理知识问答题，人工标注 ground truth  
**评估方式**：自研 LLM-as-Judge（DeepSeek-v4-pro 作为判别模型）

### 检索配置对比（Context 指标）

| 配置 | Context Precision | Context Recall | 说明 |
|------|:-----------------:|:--------------:|------|
| BM25-only | 0.460 | 0.505 | 纯关键词检索，古籍术语歧义导致大面积失效 |
| Vector-only | 0.690 | 0.695 | 语义向量，显著弥补 BM25 盲区 |
| **Hybrid** | **0.710** | **0.710** | BM25 + 向量 + RRF 融合，两路互补 ✅ |
| Hybrid+Rerank | 0.700 | 0.685 | CrossEncoder 精排，因领域偏移略降 |

**关键发现**：BM25 在"壬水秋月"等查询上 CP=0.00（古籍用"三秋壬水"，字符无重叠）；向量检索通过语义理解有效弥补，相对提升 +50%。Reranker 在文言文领域出现偏移，是已知局限。

### Self-Critique 效果验证（生成质量指标）

| 配置 | Faithfulness | Answer Relevancy | 说明 |
|------|:------------:|:----------------:|------|
| Hybrid（无 Self-Critique） | 0.905 | 1.000 | 基线 |
| **Hybrid（有 Self-Critique）** | **0.955** | 0.975 | +5.5% ✅ |

**结论**：critic_node 将 Faithfulness 提升 +5.5%，验证 Self-Critique 在减少无古籍依据论断上的有效性。Answer Relevancy 微降（1.000→0.975）揭示了 Faithfulness 与完整性之间的固有张力。

---

## 项目亮点

### 1. Tool Calling 确定性排盘
LLM 直接推算八字误差率极高（天干地支计算有严格规则）。系统通过 LangGraph Tool Node 调用 `lunar-python` 库，以确定性算法完成排盘，再将排盘结果注入检索上下文，实现"精准排盘 + 古籍知识"的结合。

### 2. 两层存储结构（原文 + 注解）
每个 chunk 同时存储：
- `original`：古籍原文（文言文）→ 展示给用户，保证权威性
- `annotation`：DeepSeek 生成的现代白话注解 → 用于 BGE-M3 向量编码，解决文言文语义理解偏差

### 3. BM25 查询同义词扩展
针对古籍与现代汉语的表达差异（"秋月" vs "三秋"，"七杀" vs "偏官"），在查询阶段做实时扩展，无需修改索引，BM25 覆盖率显著提升。

### 4. Self-Critique 防幻觉（实验验证有效）
生成回答后追加一次 LLM 核验调用，逐句比对回答与检索 chunk，删除无依据论断。消融实验验证 Faithfulness +5.5%，并量化了精确性与完整性的权衡关系。

### 5. 自研评估体系（不依赖 ragas）
完全理解并手动实现 Context Precision、Context Recall、Faithfulness、Answer Relevancy 四个指标，摆脱第三方库依赖，适配古籍中文场景，使用更强的 Pro 模型作为 judge（与生成模型分离，避免自评估偏差）。

---

## 快速开始

### 环境要求

```
Python 3.10+
NVIDIA GPU（可选，用于 Reranker 加速）
```

### 安装

```bash
git clone https://github.com/your-username/mingli-rag.git
cd mingli-rag
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
# 编辑 .env，填入以下内容：
# NAGA_API_KEY=你的API密钥（从 naga.ac 获取）
# LANGCHAIN_API_KEY=你的LangSmith密钥（可选，用于追踪）
```

### 启动对话界面

```bash
streamlit run app.py
```

### 启动 API 服务

```bash
uvicorn src.api.main:app --reload
# POST http://localhost:8000/chat
```

### 运行消融实验

```bash
# 检索指标（快速，约 20 分钟）
python scripts/evaluate_ragas.py --no-gen

# 生成质量指标（含 Self-Critique 对比）
python scripts/evaluate_ragas.py --config hybrid_no_critic
python scripts/evaluate_ragas.py --config hybrid_with_critic
```

---

## 项目结构

```
命理测算RAG/
├── app.py                          # Streamlit 多轮对话界面
├── src/
│   ├── agent/
│   │   ├── graph.py                # LangGraph 状态图定义
│   │   ├── nodes.py                # 7个节点实现
│   │   └── state.py                # AgentState 定义
│   ├── retrieval/
│   │   ├── hybrid_retriever.py     # BM25 + 向量 + RRF 混合检索
│   │   ├── bm25_retriever.py       # BM25 检索器（含同义词扩展）
│   │   ├── reranker.py             # CrossEncoder 精排
│   │   └── direct_encoder.py       # BGE-M3 / Reranker 直接加载（无 sentence-transformers）
│   ├── tools/
│   │   └── bazi.py                 # lunar-python 八字排盘工具
│   └── api/
│       └── main.py                 # FastAPI HTTP 接口
├── scripts/
│   ├── evaluate_ragas.py           # 消融实验（自研 LLM-as-Judge）
│   └── chunker.py                  # 古籍切分脚本
├── data/
│   ├── eval_questions.json         # 20 道评测题 + ground truth
│   └── eval_results/               # 消融实验结果（JSON + Markdown）
└── docs/
    └── architecture.png            # 系统架构图
```

---

## 语料说明

| 古籍 | 内容 | Chunk 数 |
|------|------|---------|
| 子平真诠 | 格局论命核心典籍 | ~200 |
| 滴天髓 | 命理通论，哲学性强 | ~150 |
| 穷通宝鉴 | 日干逐月取用神（调候） | ~200 |
| 三命通会 | 综合命理百科 | ~300 |
| 渊海子平 | 早期子平命理经典 | ~150 |
| 千里命稿 | 近代命理实战案例 | ~200 |
| 格局论命 | 格局取用专著 | ~100 |
| **合计** | | **1313 chunks** |

---

## 关于"古籍数量较少"

RAG 的核心价值不在于语料量，而在于**可追溯性**——每条回答都能定位到具体古籍的章节，在知识严谨性要求高的专业场景里这是必须的。当前7本古籍完整演示了从检索到评估的全技术链路，语料可以持续扩充而不需要改动任何代码。

---

## License

MIT
