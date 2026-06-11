# 命理 RAG 项目 — 4 周保姆版 To-Do List

> 配套文档：`mingli_rag_lean_plan.md`（设计方案）
> 本文档：每日任务清单 + 技术名词翻译
> 日期：2026-05-25

---

## 📦 Day 0：开工前准备（半天）

### Task 0.1 注册账号 + 拿 API Key

| 服务 | 用处 | 怎么拿 |
|---|---|---|
| **DeepSeek** | 主力 LLM（意图识别、查询重写、最终回答） | platform.deepseek.com，充 10 块够用很久 |
| **Claude** | 注解生成器（古文理解强） | 已有 |
| **SiliconFlow** | Reranker API | siliconflow.cn 免费注册，新用户送额度 |
| **LangSmith** | Agent 全链路追踪可视化 | smith.langchain.com 免费 |

> **什么是 API Key？** 就像家门钥匙，每次调用 API 时附上它，对方才知道是你、给你计费。Key 不要传 GitHub！用 `.env` 文件存。

### Task 0.2 装 Python 环境

```bash
conda create -n mingli python=3.11
conda activate mingli

pip install lunar-python rank-bm25 chromadb sentence-transformers
pip install langgraph langchain langchain-core langsmith langchain-openai
pip install fastapi uvicorn streamlit pdfplumber jieba ragas python-dotenv
```

> **chromadb** = 向量数据库；**sentence-transformers** = 跑 BGE-M3 的框架；**rank-bm25** = BM25 算法库；**langgraph** = Agent 编排框架（核心）。

### Task 0.3 建项目骨架

```
命理测算RAG/
├── .env                    # 存 API Key
├── data/
│   ├── classics_raw/       # 古籍原始 txt/pdf
│   ├── classics_chunks/    # 切好块的 JSON
│   └── mingli_terms.txt    # jieba 命理词典
├── src/
│   ├── tools/bazi.py       # 排盘工具
│   ├── data/chunker.py     # 切分脚本
│   ├── retriever/          # 检索相关
│   ├── graph/              # LangGraph 节点
│   └── api/main.py         # FastAPI
├── tests/test_bazi.py
├── ui/app.py               # Streamlit
└── requirements.txt
```

---

## 🗓️ Week 1：地基（排盘 + 数据）

### Day 1：排盘工具 + 边界测试

**做什么：**
1. 写 `src/tools/bazi.py`，照抄 lean plan 模块 1 的代码
2. 写 `tests/test_bazi.py`，至少 5 个 case：自己八字、毛泽东、夜子时、立春前后、闰月
3. 跑 `pytest tests/`，全绿就过关

**为啥这么干：** 八字算错，后面所有分析都错。**先把地基钉死。**

**技术解释：**
- **lunar-python**：开源 Python 库，输入公历 → 输出农历、八字、节气。纯算法，不联网，100% 确定性。
- **pytest**：Python 测试框架。写 `def test_xxx()`，它自动找出来跑。
- **夜子时**：晚上 23:00-24:00 在传统八字里算"次日的子时"——LLM 经常错的点，必须测。

**完成长啥样：** 终端跑 `pytest`，看到 `5 passed`。

---

### Day 2：下载 + 转换古籍

**做什么：**
1. 去 `github.com/mymmsc/books` 把 6 本古籍下载下来（`国学/` 目录）
2. 去 `github.com/cautionsign/bazi-1` 拿千里命稿
3. **txt 类**（三命通会、渊海子平）：写清洗脚本，去掉 `o` 章节标记等杂质
4. **pdf 类**（其余 5 本）：用 pdfplumber 转 txt
   ```python
   import pdfplumber
   def pdf_to_txt(pdf_path):
       with pdfplumber.open(pdf_path) as pdf:
           return "\n".join(p.extract_text() or "" for p in pdf.pages)
   ```
5. **每本抽 3 段人工看一眼**，确认 OCR 没乱码

**技术解释：**
- **pdfplumber**：Python 的 PDF 提取库，比 PyPDF2 更准，能保留版式。

**完成长啥样：** `data/classics_raw/` 下有 7 个 `.txt` 文件，每个肉眼可读。

---

### Day 3：切分古籍（chunk）

**做什么：** 写 `src/data/chunker.py`，把 7 本 txt 切成"块"，每块 50-200 字，存成 JSON。

**为啥要切：** Embedding 模型一次能塞的字数有限（一般 512 token），而且**精确检索的关键是"块小"**——块越小，命中越精准。

**技术解释：**
- **chunk（块）**：把长文本按规则切成小段，每段独立向量化。
- **为啥不 overlap**：文言文每句都是独立论断（"官印相生，贵格也"），机械重叠反而拼出语义噪声。

**完成长啥样：** `data/classics_chunks/` 下 7 个 JSON，每个长这样：
```json
[
  {"id": "stt_0001", "source": "三命通会", "original": "...", "annotation": ""},
  ...
]
```

---

### Day 4-5：整理 jieba 命理词典 ⭐ 你来主导

**做什么：** 在 `data/mingli_terms.txt` 里手工列 200-300 个命理专有词，每行一个：
```
七杀 5 n
伤官 5 n
日元 5 n
正官格 5 n
...
```
（数字是词频权重，5 够用；`n` 是名词）

**为啥关键：** 默认中文分词会把"伤官"拆成"伤"+"官"，"七杀"拆成"七"+"杀"，BM25 检索就完全失效了。**你领域知识强，别人替不了。**

**技术解释：**
- **jieba**：中文分词库，能加自定义词典让它把专有词当一个整体。
- **BM25**：一种古老但极有效的关键词检索算法（Google 搜索都在用），基于词频统计算相关性，对**精确字符匹配**非常强。

**完成长啥样：** 200-300 行的词典文件。**建议边整理边分类**（十神类、神煞类、格局类、节气类...）——这本身就是简历可讲的"做了领域词典的体系化整理"。

---

### Day 6：搭 ChromaDB + 向量化

**做什么：**
1. 下载 BGE-M3 模型（首次 `SentenceTransformer("BAAI/bge-m3")` 自动下，约 2GB）
2. 把所有 chunk 内容（注解还没生成，先用 `original` 临时跑通）向量化
3. 存进 ChromaDB

**技术解释：**
- **Embedding**：把一段文字变成一串数字（向量，比如 1024 维）。**意思接近的文字，向量也接近。**这是语义搜索的基础。
- **BGE-M3**：北京智源研究院开源的 Embedding 模型，中文界标杆。
- **ChromaDB**：本地向量数据库，存向量 + 元数据，支持相似度搜索。**不需要装独立服务，pip 装完就能用。**

**完成长啥样：** 跑测试 `collection.query(query_texts=["甲木"], n_results=5)`，能返回 5 条相关 chunk。

---

### Day 7：Buffer

留给前面任何超时的任务，或补测试。

---

## 🗓️ Week 2：检索链路 + 注解

### Day 1-2：批量生成注解

**做什么：** 写脚本循环调 Claude API，给每个 chunk 生成 150 字以内的现代白话注解。每天抽检 10-20 条，质量差的手工改。

**为啥重要：** 这是**两层存储**的核心。Claude 生成注解 → 注解送给 embedding → LLM 拿到原文+注解。

**技术解释：**
- **两层存储**：原文（古文，给 LLM）+ 注解（白话，给 embedding）。既保留古籍权威性，又解决文言文向量化效果差的问题。
- **API 批量调用**：for 循环 + sleep 控制速率即可，注意 Claude 有 rate limit。

**完成长啥样：** 350-700 条 chunk 都填上了 `annotation` 字段。

---

### Day 3：BM25 模块

**做什么：** 抄 lean plan 模块 3 的 `HybridRetriever`，先单独跑 BM25 部分。

**验证：** "伤官生财" 这个 query 能查出包含整词的 chunk，没被拆开。

---

### Day 4：RRF 融合

**做什么：** 实现 `_rrf_merge` 方法，把向量结果和 BM25 结果合并。

**技术解释：**
- **RRF（Reciprocal Rank Fusion，倒数排名融合）**：简单粗暴但效果出奇好的融合算法。公式：`score(d) = Σ 1/(k + rank)`。一个文档在两个排名里都靠前，最终分就高。k=60 是经验值。
- **为啥融合**：BM25 抓精确字符（"甲木"不和"乙木"混），向量抓语义（"发财"能匹"财星透出"）。**两路互补，召回率显著提升**——这是简历能写"提升 23%"的来源。

**验证：** 同一 query，对比"纯 BM25"、"纯向量"、"RRF 融合"三种返回的 Top-10，能看出融合后更平衡。

---

### Day 5-6：接 SiliconFlow Reranker

**做什么：** 注册 SiliconFlow → 拿 Key → 抄 lean plan 模块 4 的代码。

**技术解释：**
- **Reranker（重排序器）**：检索召回的 Top-20 再过一遍精排模型，挑出真正最相关的 Top-5。
- **跟向量检索的区别**：向量检索是"两两独立打分"，Reranker 是"Cross-Encoder"，query 和文档同时进模型，**精度高得多但慢**，所以只对 Top-K 用。
- **为啥走 API**：1650Ti 4GB 显存跑 BGE-Reranker 会爆显存，SiliconFlow 免费额度够用。

**验证：** 给 query → 拿 RRF 出的 20 条 → 调 Reranker → Top-5 主观上确实更精准。

---

### Day 7：Buffer，整条检索链端到端跑通

---

## 🗓️ Week 3：LangGraph Agent 编排

### Day 1-2：搭 LangGraph 骨架

**做什么：** 照 lean plan 模块 5 的代码，把 6 个节点框架搭起来（每个节点先 mock，随便返回 dict 让流程跑通）。

**技术解释：**
- **LangGraph**：基于状态机的 Agent 编排框架。每个"节点"是个 Python 函数，节点之间用"边"连接，整个流程跑一个共享的 State（TypedDict）。**比 LangChain 的链式调用更适合复杂控制流。**
- **State**：跨节点的共享内存，每个节点读一部分、写一部分。

**完成长啥样：** 跑 `graph.invoke({"user_input": "test"})`，能依次打出每个节点被调用的日志。

---

### Day 3：意图识别 + 查询重写

**做什么：** 把这两个节点的 mock 换成真 LLM 调用（用 DeepSeek）。

**为啥要查询重写：** 用户问"我今年能发财吗"，古籍里没人这么说话，要翻译成"流年财星透出 伤官生财 财库开合"才能精准命中。

---

### Day 4：Tool Calling 排盘集成

**做什么：** 把 Day 1 写的 `bazi_calculator` 函数注册成 LangChain `@tool`，挂到 LangGraph 的 `calc_bazi` 节点。

**技术解释：**
- **Tool Calling**：LLM 接收一个工具列表，自己决定**啥时候调用**、**传啥参数**。LLM 不能算八字，但能识别"用户说了生日"、知道"应该调排盘工具"，然后把结构化参数传过去。

---

### Day 5：Self-Critique 引用核查节点 ⭐ MVP 核心亮点

**做什么：** 抄 lean plan 模块 5.5 的 `citation_verify_node`，挂到 LangGraph 里 `generate_answer` 之后、`update_memory` 之前。

**为啥关键：** 这是**主动防幻觉**机制——LLM 生成完答案后，再过一次"事实核查"，把没有古籍依据的论断标 ⚠️。RAGAS Faithfulness 指标的直接拉手，简历能讲"我不是被动测幻觉，而是主动防幻觉"。

**技术解释：**
- **Self-Critique**：让 LLM 当自己的审稿人。一次 prompt 同时做三件事：提取论断 → 逐条核查 → 输出修正版回答 + 未支持论断列表。
- **为啥用单次 LLM 而不是多步**：成本低、延迟低，质量足够用。多步分解是 Future Work 的"改写回路"。
- **State 新增字段**：`verified_answer`（标注后的最终回答）、`unsupported_claims`（未支持论断列表）。

**完成长啥样：** 跑一个会"编造"的 query（故意问古籍里没的细节），看回答里出现 "⚠️ 此处古籍未直接支持" 标注。

---

### Day 6：会话记忆 + LangSmith

**做什么：**
1. **会话记忆**：抄 lean plan 模块 8 的 `MemoryManager`。命理咨询天然多轮，滑动窗口保留最近 N 轮 + 超长压缩摘要，八字本身永久保留。
2. **LangSmith**：在 `.env` 里加：
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=你的key
LANGCHAIN_PROJECT=mingli-rag
```
然后跑一次 graph，去 smith.langchain.com 看自动出现的链路图。

**为啥是面试加分项：** 面试官想看"你怎么调试 Agent"——掏出 LangSmith 截图，每个节点输入输出耗时清清楚楚，**比任何描述都有说服力**。

---

### Day 7：全流程跑通

输入一个真实问题 → 走完全部节点 → 输出分析结果。

---

## 🗓️ Week 4：评估 + 收尾

### Day 1：构建 25 条测试集

- 10 条：从《千里命稿》《滴天髓阐微》挑历史人物案例 + 大师原文当 ground_truth
- 10 条：找 10 个公开八字的近代名人（毛、李嘉诚、王永庆等）
- 5 条：直接让 Claude 生成参考答案当 ground_truth

**这是 Week 4 最费时间的一天，老老实实做。**

---

### Day 2-3：跑 5 档 Ablation ⭐ 项目核心数据

**做什么：** 同一套 25 条测试集，分别跑 5 种配置：
1. 纯 BM25
2. 纯向量
3. BM25 + 向量 + RRF（无 Reranker）
4. 全套检索（含 Reranker，无 Self-Critique）
5. 全套 + Self-Critique 引用核查 ⭐ 最终配置

**核心看点：** 配置 4 vs 配置 5 的 **Faithfulness** 差异——这一列数字直接证明 Self-Critique 节点的价值。

**技术解释：**
- **Ablation Study（消融实验）**：ML 论文里的标准做法——逐个去掉某个组件，看效果掉多少，证明每个组件的贡献。**这张表是简历最硬的数据，面试官一看就知道你做了真功夫。**
- **RAGAS 4 大指标**：
  - **Context Precision**：检索到的内容有多少真相关
  - **Context Recall**：相关内容有多少被检出
  - **Faithfulness**：回答有没有编造（防幻觉）
  - **Answer Relevancy**：回答切不切题

**完成长啥样：** 一张 5 行 4 列的表格（5 配置 × 4 指标），每个格子有数字。Self-Critique 那一行在 Faithfulness 列应该明显高于配置 4——这就是你的"主动防幻觉"卖点的硬数据。

---

### Day 4：FastAPI 接口

**做什么：** 把 LangGraph 包成 HTTP API。

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/ask")
def ask(req: dict):
    return graph.invoke(req)
```

**为啥：** Streamlit 直接调 Python 函数也能跑，但**做成 API 显得工程化**，前后端解耦。

---

### Day 5-6：Streamlit Demo

**做什么：** 抄 lean plan 模块 10 的代码，本地能流畅演示就够。

**技术解释：**
- **Streamlit**：Python 写 Web 界面的最简框架，不用学 HTML/CSS。`pip install streamlit` 后 `streamlit run app.py` 就有网页。**面试前一定本地实际演示一遍，别现场翻车。**

---

### Day 7：收尾文档

1. **GitHub README** 必备元素：
   - 项目简介（3 句话）
   - 架构图（Excalidraw 或 draw.io 画一张干净的）
   - 4 档 Ablation 对比表
   - 演示截图（Streamlit 跑起来的样子）
   - Future Work（抄 lean plan §九）

2. **简历项目描述定稿**：把 lean plan §七 模板里的 `XX` 替换成真实数字。

---

## 🎯 每周末"过关检查"

- **Week 1 末**：`pytest tests/` 全绿；检索通路打通（输入查询 → 返回 Top-20）
- **Week 2 末**：输入"今年感情如何"→ Reranker 返回 Top-5 古籍段落，主观看着相关
- **Week 3 末**：LangGraph 全流程跑通，LangSmith 能看到完整链路图
- **Week 4 末**：4 档 Ablation 表填完；Streamlit Demo 能流畅演示；README 写好

---

## 💪 最后建议

1. **每天结束花 5 分钟记日记**：今天写了啥、遇到啥坑、怎么解决的。**面试讲故事全靠这个**。
2. **遇到卡 1 小时以上的问题就来问我**，不要硬刚 —— 时间是最贵的。
3. **每周末跑一次"过关检查"**，没过的话宁可砍功能也不要拖到下周，**节奏比完美重要**。
4. **GitHub 每天 push**，1 是不丢代码，2 是 commit 历史就是"做过项目"的最佳证据。

---

## 📂 参考文档

- 设计方案：`mingli_rag_lean_plan.md`（含所有代码示例）
- 原始打磨记录：`C:\Users\Admin\.claude\plans\mingli-rag-ultimate-plan-agent-radiant-babbage.md`
- 旧的完整版（已废弃）：`mingli_rag_ultimate_plan.md`

---

开干吧！💯
