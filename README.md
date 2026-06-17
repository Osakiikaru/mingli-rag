# 命理古籍研究助手

> 基于 RAG + LangGraph ReAct Agent 的子平八字命理知识问答系统

将 7 本古典命理典籍结构化入库，通过混合检索召回相关原文，由 LLM 生成有据可查的回答。核心价值在于**可追溯性**——每条回答都能定位到具体古籍章节，而非 LLM 的参数记忆。

---

## 系统架构

```
用户输入
   │
   ▼
┌──────────────────────────────────────────────┐
│  LangGraph ReAct Agent（2节点循环）            │
│                                               │
│  ┌────────────┐  工具调用   ┌──────────────┐ │
│  │ agent_node │ ─────────▶ │  tools_node  │ │
│  │ (DeepSeek) │ ◀───────── │              │ │
│  └────────────┘  工具结果   │  bazi_tool   │ │
│        │                   │  search_tool │ │
│        │ 无工具调用          └──────────────┘ │
│        ▼                                      │
│  [critic_node]  ← use_critic=True 时启用      │
│  Self-Critique：核验引用来源，默认关闭          │
└──────────────────────────────────────────────┘
   │
   ▼
Streamlit 多轮对话界面（流式进度 + 历史导出）
```

**两个工具：**
- `bazi_tool`：调用 lunar-python 确定性算法完成八字四柱大运排盘
- `search_tool`：BGE-M3 向量 + BM25 + RRF 混合检索古籍原文

---

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **Agent 编排** | LangGraph | ReAct 2节点循环，条件路由到 tools / critic / END |
| **排盘工具** | lunar-python | 确定性算法，解决 LLM 八字推算误差问题 |
| **向量检索** | BGE-M3（本地） | BAAI 开源，中文最优，DirectEncoder 直接加载（float16 CUDA） |
| **稀疏检索** | BM25Okapi + jieba | 命理自定义词典 + 查询同义词扩展（69条，秋月↔三秋、七杀↔偏官等） |
| **融合算法** | RRF（k=60） | 无需归一化的多路检索融合 |
| **精排** | BGE-Reranker-v2-m3 | CrossEncoder，消融实验验证有效（需 6GB+ VRAM）|
| **向量库** | ChromaDB | 轻量本地部署，无需独立服务 |
| **防幻觉** | Self-Critique 节点 | 生成后核验引用，Faithfulness +2.1%（0.943→0.963，消融实验验证） |
| **LLM** | DeepSeek-v4-flash | Agent 生成，中文命理理解优秀 |
| **评估** | 自研 LLM-as-Judge | 4指标手动实现，Judge 与生成模型分离，避免自评估偏差 |
| **界面** | Streamlit | 多轮对话，`st.status` 分步进度反馈 |
| **部署** | Docker + docker-compose | 容器化，volume 挂载模型与数据 |

---

## 消融实验结果

**评估集**：30 道子平命理知识问答题，人工标注 ground truth，覆盖格局、十神、日干取用、行运等核心话题  
**评估方式**：自研 LLM-as-Judge（DeepSeek-v4-pro 作为判别模型，top_k=5）  
**注**：Answer Relevancy 在本任务存在天花板效应（均值≥0.97），以下仅展示有区分度的三项指标

| 配置 | Context Precision | Context Recall | Faithfulness |
|------|:-----------------:|:--------------:|:------------:|
| BM25-only | 0.527 | 0.620 | 0.877 |
| Vector-only | 0.733 | 0.777 | 0.898 |
| **Hybrid（BM25+向量+RRF）** | **0.767** | **0.803** | 0.943 |
| **Hybrid + Self-Critique** | 0.740 | 0.790 | **0.963** |

**关键发现：**
1. **语义检索 vs BM25**：向量检索相比纯 BM25 精度提升 **+39%**（CP 0.527→0.733），因为古籍用"三秋壬水"而用户查"壬水秋月"，字符无重叠，BM25 完全失效
2. **混合检索优于纯向量**：Hybrid CP=0.767 > Vector-only CP=0.733（+4.6%），RRF 融合有效利用 BM25 的词频信号，补足语义检索在精确术语匹配上的短板
3. **Self-Critique 提升忠实度，精度略有取舍**：Hybrid+Critic Faithfulness 0.963，相比 Hybrid（0.943）提升 **+2.1%**，代价是 CP 从 0.767 降至 0.740——Critic 过滤掉部分无直接引文依据的表述，回答更保守但更可信

---

## 项目亮点

### 1. ReAct Agent 自主工具调用
LangGraph 2节点循环：`agent_node` 由 LLM 决策（排盘/检索/直接回答），`tools_node` 执行工具并将结果写入 state。用户连续追问同一命盘时，排盘结果保留在消息历史中，LLM 自动判断无需重复排盘。

### 2. 两层存储结构（原文 + 注解）
每个 chunk 同时存储：
- `original`：古籍原文（文言文）→ 展示给用户，保证权威性
- `annotation`：DeepSeek 生成的现代白话注解 → 用于 BGE-M3 向量编码，解决文言文语义理解偏差

### 3. 命理同义词查询扩展（69条）
在查询阶段实时扩展，无需修改索引：
```
秋月 → 三秋 / 申月 / 酉月 / 戌月
七杀 ↔ 偏官
偏印 ↔ 枭神 ↔ 枭印
正印 ↔ 印绶
食伤 → 食神 + 伤官（集合展开）
从格 → 从旺格 + 从强格 + 从弱格 + 从杀格 + 从财格
```

### 4. Self-Critique 防幻觉（实验验证有效）
生成回答后追加一次 LLM 核验：逐条比对引用论断与检索 chunk，删除无依据内容或降级为"一般性表述"。消融实验验证 Faithfulness +2.1%（0.943→0.963），可通过 `use_critic=True` 启用。

### 5. 自研评估体系
手动实现 Context Precision、Context Recall、Faithfulness、Answer Relevancy 四个 RAGAS 指标，摆脱第三方库依赖，适配古籍中文场景，30题评测集覆盖主要命理知识点。

---

## 快速开始

### 环境要求

```
Python 3.10+
NVIDIA GPU（可选，BGE-M3 用于加速；Reranker 需要 6GB+ VRAM）
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
# 编辑 .env，填入：
# NAGA_API_KEY=你的API密钥（从 naga.ac 获取，兼容 OpenAI 格式）
# LANGCHAIN_API_KEY=你的LangSmith密钥（可选，用于链路追踪）
```

### 构建索引（首次运行）

```bash
# 构建向量索引（需要 BGE-M3 模型，首次自动下载）
python src/retrieval/build_index.py

# 构建 BM25 索引
python -c "from src.retrieval.bm25_retriever import BM25Retriever; BM25Retriever.build()"
```

### 启动对话界面

```bash
streamlit run app.py
```

### Docker 部署

```bash
# 启动 API 服务
docker-compose up api

# 同时启动 Streamlit 界面
docker-compose --profile ui up
```

### 运行消融实验

```bash
# 全量评测（4配置 × 30题，含生成，约 40-60 分钟）
python scripts/evaluate_ragas.py --no-resume

# 仅评测检索指标（跳过生成，约 15 分钟）
python scripts/evaluate_ragas.py --no-resume --no-gen

# 单配置调试（前2题）
python scripts/evaluate_ragas.py --dry-run --config hybrid
```

---

## 项目结构

```
命理测算RAG/
├── app.py                          # Streamlit 多轮对话界面
├── Dockerfile                      # 容器化部署
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── agent/
│   │   ├── graph.py                # LangGraph 状态图（ReAct 2节点 + critic）
│   │   ├── nodes.py                # agent_node / tools_node / critic_node
│   │   └── state.py                # AgentState 定义
│   ├── retrieval/
│   │   ├── hybrid_retriever.py     # BM25 + 向量 + RRF 混合检索
│   │   ├── bm25_retriever.py       # BM25（jieba + 同义词扩展）
│   │   ├── reranker.py             # CrossEncoder 精排（需 6GB+ VRAM）
│   │   ├── direct_encoder.py       # BGE-M3 / Reranker 直接加载（无 sentence-transformers）
│   │   └── build_index.py          # 向量索引构建脚本
│   ├── tools/
│   │   └── bazi.py                 # lunar-python 八字四柱大运排盘
│   └── api/
│       └── main.py                 # FastAPI HTTP 接口
├── scripts/
│   ├── evaluate_ragas.py           # 消融实验（自研 LLM-as-Judge，4配置）
│   └── chunker.py                  # 古籍切分脚本
└── data/
    ├── eval_questions.json         # 30 道评测题 + ground truth
    ├── dict/
    │   └── mingli_dict.txt         # jieba 命理自定义词典
    └── processed/                  # 切分后的 chunk JSON 文件
```

---

## 语料说明

| 古籍 | 内容 | Chunk 数（约） |
|------|------|:------:|
| 子平真诠 | 格局论命核心典籍 | ~200 |
| 滴天髓 | 命理通论，哲学性强 | ~150 |
| 穷通宝鉴 | 日干逐月取用神（调候） | ~200 |
| 三命通会 | 综合命理百科 | ~300 |
| 渊海子平 | 早期子平命理经典 | ~150 |
| 千里命稿 | 近代命理实战案例 | ~200 |
| 格局论命 | 格局取用专著 | ~100 |
| **合计** | | **1313 chunks** |

---

## 设计反思

### 参数记忆 vs 检索记忆

项目完成后做了一组对照实验：关闭 RAG，让 LLM 直接回答命理问题。结果发现纯 LLM 回答流畅度更高——DeepSeek 的训练数据已覆盖这 7 本公开古籍。

| 类型 | 存储位置 | 适合什么 |
|------|---------|---------|
| 参数记忆 | 模型权重 | 通用规律、原理、定义 |
| 检索记忆 | 向量库 | 具体事实、私有数据、需精确溯源的内容 |

RAG 在本项目的核心价值不是"提供 LLM 不知道的知识"，而是**提供可核查的古籍原文依据**——每条回答锚定在真实文本片段上，用户可以验证原文，而非依赖无法追溯的模型记忆。

### 忠实度与精度的取舍（Self-Critique 的代价）

实验结果显示，Hybrid+Critic 的 Faithfulness（0.963）高于 Hybrid（0.943），但 Context Precision 略有下降（0.767→0.740）。

根因：Self-Critique 的机制是删除或降级"无直接古籍引文支撑"的论断。这类论断有时是 LLM 基于检索结果做出的合理推断，本身并非错误——但删除后回答更保守，召回的相关 chunk 在回答中被使用的比例也相对减少。

权衡：对"可溯源性"要求极高的场景（学术引用、原文核验）启用 Self-Critique；对日常问答场景，Hybrid 已能在精度与流畅度之间取得更好平衡。

### Reranker 的硬件门槛

BGE-Reranker-v2-m3 精排在离线消融实验中验证有效（Context Precision +0.013），但同时加载 BGE-M3 和 Reranker 共需约 2.2GB 显存，叠加推理激活占用后超出 4GB 消费级 GPU 上限，生产环境默认关闭。工业部署建议使用独立 Reranker 服务或 8GB+ GPU。

### HyDE 与同义词扩展的取舍

HyDE（Hypothetical Document Embeddings）通过 LLM 生成"假设性回答"再编码，能缩小白话查询与文言文档的语义鸿沟。本项目用**查询同义词扩展**（69条命理术语映射）替代，在零额外延迟的前提下达到类似效果，适合术语体系封闭的垂直领域。

---

## License

MIT
