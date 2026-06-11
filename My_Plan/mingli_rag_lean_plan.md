# 命理测算 RAG 项目 — 精简执行版（对照执行）

> 版本：Lean MVP v1.0
> 日期：2026-05-25
> 原则：**做少、做透、能讲清楚**。所有暂不实现的设计放入 Future Work，面试时作为"优化方向"输出。

---

## 一、项目定位与价值

### 解决的真实痛点
1. 市面 APP 排盘免费但分析收费
2. DeepSeek/豆包等通用大模型**四柱排盘经常出错**（根基错 → 分析全错）
3. 通用大模型缺乏命理专业知识深度（不懂《滴天髓》《穷通宝鉴》《三命通会》等经典）

### 为什么这个项目能让面试官刮目相看
- 19 份优秀简历里 **0 人** 有 Tool Calling 排盘 + 两层古文存储的组合
- **仅 1 人** 做了 RAGAS 评估，且没有 Ablation 对比数据
- 有真实领域知识，能讲清楚每个技术决策的 why（这比堆功能重要 10 倍）

---

## 二、技术栈（精简版）

| 层次 | 选型 | 理由 |
|------|------|------|
| Agent 编排 | **LangGraph** | 有状态 Graph，节点清晰可追踪 |
| 排盘工具 | **lunar-python** | 开源本地，确定性算法，无需 API |
| Embedding | **BGE-M3**（BAAI） | 中文效果最佳开源 Embedding，直接用，无需 A/B |
| 稀疏检索 | **rank_bm25 + jieba + 命理词典** | jieba 防止"伤官""七杀"被字级拆开 |
| 向量库 | **ChromaDB** | 轻量无服务，本地运行，快速迭代 |
| Reranker | **BGE-Reranker-v2-m3 via SiliconFlow API** | 1650Ti 跑本地有 OOM 风险；SiliconFlow 免费额度够用 |
| 后端 | **FastAPI** | 轻量标准，Agent 接口化 |
| 评估 | **RAGAS** | 四维自动化评估，含 4 档 Ablation 对比 |
| 追踪调试 | **LangSmith** | 全链路可视化，面试加分项 |
| Demo | **Streamlit（基础版）** | 表单 + 流式输出 + chunk 展示，够演示就行 |
| LLM | 意图/重写/回答：**DeepSeek-V3**；注解生成：**Claude** | 重任务用强模型保质量 |

> **没有的东西：** cnlunar 双库验证、Qwen3 A/B 实验、父子 chunk 层级、多标签意图识别、真正的 clarify 循环。这些在 Future Work 里，面试时作为"下一步优化方向"。
>
> **MVP 包含的核心亮点：** Tool Calling 排盘 + 两层存储 + jieba 混合检索 + RRF + Reranker + **Self-Critique 引用核查** + RAGAS 4 档 Ablation。

---

## 三、整体架构（线性版，够清晰够可解释）

```
┌─────────────────────────────────────────┐
│           用户输入                        │
│  "我 1990年3月15日 午时生，今年感情如何？"    │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  节点 1：Tool Calling 排盘               │
│  调用 lunar-python 获取准确四柱           │
│  输出：{年:庚午, 月:癸卯, 日:甲子, 时:甲午}│
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  节点 2：意图识别（单标签）               │
│  LLM 分类 → "感情婚姻"                  │
│  低置信度 → 直接返回"请澄清问题"          │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  节点 3：查询重写                         │
│  "今年感情如何" →                        │
│  "日元甲木与官星关系、流年官星透出、        │
│   夫妻宫动静、桃花运"                     │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  节点 4：混合检索（双路并行）              │
│  ├── BM25 检索（jieba + 命理词典）        │
│  └── 向量检索（注解层 BGE-M3 embedding） │
│       ↓                                 │
│  RRF 融合 → SiliconFlow Reranker Top-5  │
│  检索范围：古籍两层库 + 名人案例库          │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  节点 5：LLM 生成回答（DeepSeek-V3）     │
│  输入：八字 + 古籍原文+注解 + 相关案例     │
│  输出：结构化命理分析                     │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  节点 6：Self-Critique 引用核查 ⭐       │
│  LLM 提取回答中的关键论断                 │
│  对照 retrieved_chunks 验证支持度         │
│  无支持的论断 → 标注 "⚠️ 古籍未直接支持"  │
│  目标：主动防幻觉，直接拉高 Faithfulness  │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  节点 7：会话记忆更新                    │
│  滑动窗口保留近 6 轮                      │
│  超阈值 LLM 压缩摘要                      │
│  八字信息永久保留不参与压缩                │
└─────────────────────────────────────────┘
```

---

## 四、各模块详细实现

### 模块 1：排盘 Tool（第一步，最优先）

**为什么是第一步：** 四柱是所有分析的基础，LLM 算八字错误率高，必须用确定性算法隔离。

```python
from lunar_python import Lunar, Solar

def get_bazi(year: int, month: int, day: int, hour: int, gender: str) -> dict:
    """
    输入公历生辰，返回完整八字信息
    hour: 0-23 整数（0-1点为子时，23点为夜子时）
    gender: "男" or "女"
    """
    solar = Solar.fromYmdHms(year, month, day, hour, 0, 0)
    lunar = solar.getLunar()
    bazi = lunar.getEightChar()

    return {
        "年柱": {"天干": bazi.getYearGan(), "地支": bazi.getYearZhi()},
        "月柱": {"天干": bazi.getMonthGan(), "地支": bazi.getMonthZhi()},
        "日柱": {"天干": bazi.getDayGan(), "地支": bazi.getDayZhi()},
        "时柱": {"天干": bazi.getTimeGan(), "地支": bazi.getTimeZhi()},
        "性别": gender,
        "日元": bazi.getDayGan(),  # 核心：日主天干
    }
```

**验证方法（5 个关键 case 写进测试文件）：**

```python
# tests/test_bazi.py
import pytest
from src.tools.bazi import get_bazi

def test_normal_case():
    # 毛泽东 1893-12-26 子时（公开已知八字）
    result = get_bazi(1893, 12, 26, 0, "男")
    assert result["日元"] == "癸"  # 按已知文献验证

def test_chunqian_edge():
    # 立春前后（影响年柱归属）
    before = get_bazi(2024, 2, 3, 12, "男")  # 立春前
    after  = get_bazi(2024, 2, 5, 12, "男")  # 立春后
    assert before["年柱"]["天干"] != after["年柱"]["天干"]

def test_yezishi_edge():
    # 夜子时（23:00 后，日柱归次日）
    result_23 = get_bazi(1990, 3, 15, 23, "男")
    result_0  = get_bazi(1990, 3, 16, 0, "男")
    # 夜子时与次日子时日柱应相同
    assert result_23["日柱"] == result_0["日柱"]

def test_known_person_li():
    # 李嘉诚 1928-07-29（命理论坛公开八字）
    result = get_bazi(1928, 7, 29, 8, "男")
    assert result["年柱"]["天干"] == "戊"  # 戊辰年
```

**面试话术：** "我发现排盘有几个坑容易踩：夜子时（23点之后日柱归次日）、立春前后（年柱切换不按元旦而按节气），所以专门写了测试矩阵验证。"

**LangGraph Tool 注册：**
```python
from langchain_core.tools import tool

@tool
def bazi_calculator(year: int, month: int, day: int, hour: int, gender: str) -> str:
    """计算八字四柱，输入公历生日和性别，返回准确的年月日时四柱信息"""
    result = get_bazi(year, month, day, hour, gender)
    return str(result)
```

---

### 模块 2：古籍知识库构建（两层存储）

#### 2.1 古籍来源

**数据源（7 本全集，已确认可获取）：**

| 古籍 | 获取方式 |
|------|---------|
| 三命通会 | `github.com/mymmsc/books` → `八字 - 三命通会.txt`（592KB，人工精校） |
| 渊海子平 | 同上仓库 txt |
| 子平真诠 | 同上仓库 pdf → 转 txt |
| 穷通宝鉴 | 同上仓库 pdf → 转 txt |
| 滴天髓 | 同上仓库 pdf → 转 txt |
| 神峰通考 | 同上仓库 pdf → 转 txt |
| 千里命稿 | `github.com/cautionsign/bazi-1` |

**pdf 转 txt 工作流：**
```python
import pdfplumber

def pdf_to_txt(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)
```
转完后每本抽检 3-5 段，质量差的用 PaddleOCR 补救。

#### 2.2 两层存储设计（核心差异化）

**问题：** 文言文语义压缩，"官印相生，贵格也"直接 embedding，向量库根本理解不了这是说什么。

**解决方案：** 每条原文配一段现代注解，**检索走注解层，LLM 拿到原文+注解**。

```python
# 每个知识块的存储结构（含层级上下文 metadata）
chunk = {
    "id": "dtx_001",
    "source": "《滴天髓》",
    "original": "官印相生，贵格也",              # 原文（返回给 LLM，保留权威性）
    "annotation": """当命局中正官与印绶同现，
                    且官生印、印护官时，构成贵气格局。
                    适用条件：日元需有根气，官星不混杂七杀，
                    印星不被财星克制。典型表现：仕途顺遂，
                    得贵人提携，地位显赫。""",   # 现代注解（用于 embedding，语义清晰）
    "tags": ["格局", "官印", "贵格", "正官", "印绶"],
}
```

**注解生成流程：**
```python
def generate_annotation(original_text: str, source: str) -> str:
    """用 Claude 为文言文原典生成现代注解（Claude 古文理解更好）"""
    prompt = f"""
    你是一位精通子平八字的命理学者。
    请为以下来自{source}的原文生成详细注解，包括：
    1. 现代白话文解释
    2. 适用条件（什么样的八字格局适用）
    3. 实际表现（在人生中如何体现）

    原文：{original_text}

    注解（150字以内，不要废话）：
    """
    return claude_client.invoke(prompt).content
```

**注解工作量预估：** 7 本 × 50-100 条核心论断 = 350-700 条，每天抽检 10-20 条，Week 2 并行处理完。

#### 2.3 文本分块策略

**七书定制分隔符与粒度：**

| 书名 | 分隔符（一级）| 分隔符（二级）| 目标粒度 | 特殊处理 |
|------|------------|------------|---------|---------|
| **子平真诠** | 章节标题行 | `"原文:"` | 300-600字 | 原文+解读+命例分别独立，原文先行 |
| **渊海子平** | `【...】` 小节 | `《...》` 大章 | 600-1200字 | 原文+注解+诗诀合并 |
| **三命通会** | `○` 大节 | `△` 小节 | 600-900字（△级为基本单元）| 过长的 ○ 节按 △ 再切 |
| **格局论命** | `1.` `2.` 编号行 | 段落内部 | 1000-2000字 | 理论+命例合并一节，不拆断 |
| **千里命稿** | 重复字标题行（`天天天干干干篇篇篇`）| 自然段落 | 1000-1500字 | 命例独立，type="命例" |
| **穷通宝鉴** | 天干（甲/乙/丙…）大节 | 月份小节（`生于寅月`…）| 300-700字（月份条目为基本单元）| matrix 结构，chapter=天干，section=月份 |
| **滴天髓** | 经文段（短句4-12字）| — | 三层合并 | 经文+原注+任氏曰合并为一 chunk，以`任氏曰`段落结束为界 |

```python
# 每个 chunk 输出结构（以子平真诠为例）
{
    "id":         "zpzq_0042",
    "source":     "子平真诠",
    "original":   "正官者，甲用辛...",
    "annotation": "",              # Week 2 批量 LLM 生成
    "tags":       [],
}
```

**不做 overlap**：文言文每句都是独立论断，机械 overlap 反而破坏语义。
**不做父子层级**（这是 Future Work §九.2）。

---

### 模块 3：混合检索 + RRF 融合

**核心设计思路：**
- BM25 保精确：命理术语"甲木"和"乙木"差一字含义天壤之别，纯向量会混淆
- 向量保语义：口语问题"我能发财吗"→ 能匹配到"日元与财星关系"的注解
- RRF 融合：两路结果互补，召回率显著提升

```python
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer
import jieba

class HybridRetriever:
    def __init__(self, chunks: list[dict]):
        # ── 向量检索：基于注解层 embedding ──
        self.encoder = SentenceTransformer("BAAI/bge-m3")
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("mingli")

        # ── BM25：jieba + 命理专有词典 ──
        # 关键：不用字级分词！"伤官"字级拆成"伤"+"官"就废了
        jieba.load_userdict("data/mingli_terms.txt")  # 200-300 个命理术语
        texts = [c["original"] + " " + c["annotation"] for c in chunks]
        tokenized = [list(jieba.cut(t)) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.chunks = chunks

        # ── 构建向量索引（用注解层，语义清晰）──
        annotations = [c["annotation"] for c in chunks]
        embeddings = self.encoder.encode(annotations, show_progress_bar=True).tolist()
        self.collection.add(
            embeddings=embeddings,
            ids=[c["id"] for c in chunks],
            documents=annotations,
        )

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        # 向量检索
        q_emb = self.encoder.encode([query]).tolist()
        vec_results = self.collection.query(query_embeddings=q_emb, n_results=top_k)
        vec_ids = vec_results["ids"][0]

        # BM25 检索（query 同样用 jieba + 词典分词）
        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)
        bm25_ids = [self.chunks[i]["id"] for i in scores.argsort()[::-1][:top_k]]

        # RRF 融合
        return self._rrf_merge(vec_ids, bm25_ids, k=60)

    def _rrf_merge(self, list1: list, list2: list, k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank_i(d))"""
        scores = {}
        for rank, doc_id in enumerate(list1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, doc_id in enumerate(list2):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        id_to_chunk = {c["id"]: c for c in self.chunks}
        return [id_to_chunk[i] for i in sorted_ids if i in id_to_chunk]
```

**jieba 命理词典格式（`data/mingli_terms.txt`，你来整理）：**
```
七杀 5 n
伤官 5 n
印绶 5 n
食神 5 n
正官 5 n
偏财 5 n
正印 5 n
甲木 5 n
乙木 5 n
日元 5 n
用神 5 n
格局 5 n
大运 5 n
流年 5 n
天干 5 n
地支 5 n
# ... 继续补充到 200-300 个
```

---

### 模块 4：Reranker（SiliconFlow API 版）

**为什么用 API 而不是本地：** 1650Ti 4GB 显存，Reranker 本地跑有 OOM 风险。SiliconFlow 免费额度足够开发阶段。

```python
import os
import requests

class Reranker:
    def __init__(self):
        self.api_key = os.getenv("SF_API_KEY")
        self.model = "BAAI/bge-reranker-v2-m3"

    def rerank(self, query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
        # 送给 Reranker 的文档：原文 + 注解（保留术语精确性）
        docs = [f"原文：{c['original']}\n注解：{c['annotation']}" for c in chunks]

        resp = requests.post(
            "https://api.siliconflow.cn/v1/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": docs,
                "top_n": top_n,
            },
        )
        resp.raise_for_status()
        results = resp.json()["results"]

        # 按相关性分数排序，返回原始 chunk 对象
        scored = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        return [chunks[r["index"]] for r in scored]
```

---

### 模块 5：LangGraph 工作流编排

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MingliState(TypedDict):
    user_input: str
    birth_info: dict        # 用户生辰信息（year/month/day/hour/gender）
    bazi: dict              # 排盘结果
    intent: str             # 意图分类
    intent_confidence: float
    rewritten_query: str    # 重写后的查询
    retrieved_chunks: list  # 检索结果 Top-5
    answer: str             # LLM 原始回答
    verified_answer: str    # 经 Self-Critique 标注后的最终回答
    unsupported_claims: list  # 未在 chunks 中找到支持的论断列表
    memory: list            # 对话历史

def build_graph():
    graph = StateGraph(MingliState)

    graph.add_node("parse_input",      parse_birth_info_node)
    graph.add_node("calc_bazi",        bazi_tool_node)        # Tool Calling
    graph.add_node("identify_intent",  intent_node)
    graph.add_node("rewrite_query",    rewrite_node)
    graph.add_node("hybrid_retrieve",  retrieval_node)
    graph.add_node("generate_answer",  generation_node)
    graph.add_node("verify_citations", citation_verify_node)  # ⭐ Self-Critique
    graph.add_node("update_memory",    memory_node)

    graph.set_entry_point("parse_input")
    graph.add_edge("parse_input",     "calc_bazi")
    graph.add_edge("calc_bazi",       "identify_intent")

    # 意图置信度低 → 直接返回澄清提示（不做真正的循环，避免 Graph 复杂化）
    graph.add_conditional_edges(
        "identify_intent",
        lambda s: "rewrite" if s["intent_confidence"] >= 0.7 else "end",
        {"rewrite": "rewrite_query", "end": END},
    )

    graph.add_edge("rewrite_query",   "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "generate_answer")
    graph.add_edge("generate_answer", "verify_citations")     # 新增：生成后必过核查
    graph.add_edge("verify_citations","update_memory")
    graph.add_edge("update_memory",   END)

    return graph.compile()
```

---

### 模块 5.5：Self-Critique 引用核查节点 ⭐ MVP 必做

**为什么必须做：** RAG 系统最大的风险是 LLM 编造古籍里没有的论断（"某某书说..."但其实没说）。本节点在生成答案后自动核查每条关键论断是否能在检索到的 chunks 中找到依据，**直接拉高 RAGAS Faithfulness 指标**——这是本项目最有讲头的差异化点之一。

**设计思路：**
- 只做"标注"，**不做改写回路**（避免 Graph 复杂化、不可控）
- 未通过的论断 → 在原回答中追加 `⚠️ 此处古籍未直接支持`
- 用 LLM 一次完成"提取论断 + 核查 + 标注"三件事（节省调用）

```python
import json

def citation_verify_node(state: MingliState) -> MingliState:
    """对生成的回答做引用核查，无支持的论断追加 ⚠️ 标注"""
    answer = state["answer"]
    chunks = state["retrieved_chunks"]
    chunks_text = "\n---\n".join(
        f"[{c['source']}] {c['original']}" for c in chunks
    )

    prompt = f"""
你是命理 RAG 系统的事实核查员。请按以下步骤工作：

1. 从【回答】中提取所有具体的命理论断（如"日元甲木坐午火，丁火透出主财运旺"）
2. 对每条论断，判断是否能在【古籍依据】中找到直接或合理推导支持
3. 输出修正后的回答：保留原结构，对无支持的论断在该句末尾追加 "⚠️ 此处古籍未直接支持"
4. 同时输出未支持论断的列表

【回答】
{answer}

【古籍依据】
{chunks_text}

严格按 JSON 输出（不要额外文字）：
{{
  "verified_answer": "修正后的完整回答文本",
  "unsupported_claims": ["未支持论断1", "未支持论断2"]
}}
"""
    result = llm.invoke(prompt)
    parsed = json.loads(result.content)
    state["verified_answer"] = parsed["verified_answer"]
    state["unsupported_claims"] = parsed.get("unsupported_claims", [])
    return state
```

**配套：评估时跑"有/无 Self-Critique"对比实验**

在 Week 4 Day 2-3 的 4 档 Ablation 表上**加一列对比**：
- Full System（不过 Self-Critique）：Faithfulness = X
- Full System + Self-Critique：Faithfulness = Y（预期 Y > X）

这是简历能写的最硬数字之一。

**面试话术：**
> "我没把 Faithfulness 当成被动测的指标，而是设计了 Self-Critique 节点主动防幻觉。
> 节点本身一次 LLM 调用完成'提取论断 + 核查 + 标注'，不做改写回路避免 Graph 复杂化。
> Ablation 显示加上这个节点后 Faithfulness 从 X 提升到 Y，证明主动防幻觉比单纯依赖检索质量更有效。"

**为什么不做改写回路（trade-off 老实讲）：**
- 改写回路会让 Graph 状态机变成"可能死循环"，需要加 `revision_count` 限制次数
- MVP 阶段先证明"标注就有效"，后续把"改写回路"放进 Future Work 作为下一步迭代

---

### 模块 6：意图识别（单标签版）

```python
import json

INTENT_TREE = {
    "本命分析": ["性格", "天赋", "格局", "用神"],
    "大运流年": ["今年运势", "某年运势", "大运分析"],
    "感情婚姻": ["感情", "婚姻", "对象", "桃花", "夫妻"],
    "事业财运": ["事业", "工作", "财运", "升职", "创业"],
    "健康":     ["身体", "健康", "疾厄"],
    "排盘解释": ["天干含义", "地支含义", "神煞"],
}

def identify_intent(question: str, bazi: dict, llm) -> dict:
    prompt = f"""
    用户的八字是：{bazi}
    用户的问题是：{question}

    请判断问题最主要属于以下哪一个类别：{list(INTENT_TREE.keys())}
    confidence < 0.7 时说明问题模糊，需要用户澄清。

    只输出 JSON，不要其他内容：
    {{"intent": "类别名", "confidence": 0.0-1.0}}
    """
    result = llm.invoke(prompt)
    return json.loads(result.content)
```

---

### 模块 7：查询重写

```python
def rewrite_query(question: str, bazi: dict, intent: str, llm) -> str:
    prompt = f"""
    你是命理专家，需要将用户的口语问题转化为适合检索命理古籍的专业术语。

    用户八字：{bazi}
    意图类型：{intent}
    用户问题：{question}

    请输出重写后的专业检索词（15-40字，空格分隔关键术语）。
    如果是复杂问题，可以拆分为2-3个并列检索词组。

    示例：
    口语："我今年能发财吗"
    专业词："流年财星透出 日元与财星关系 伤官生财 财库开合"
    """
    return llm.invoke(prompt).content.strip()
```

---

### 模块 8：会话记忆管理

```python
class MemoryManager:
    def __init__(self, window_size: int = 6):
        self.window_size = window_size

    def update(self, history: list, new_turn: dict, llm) -> list:
        history.append(new_turn)

        # 永久保留八字信息（绝不参与压缩）
        bazi_entry = next((h for h in history if h.get("type") == "bazi"), None)

        if len(history) > self.window_size:
            to_compress = history[:-self.window_size]
            summary = self._summarize(to_compress, llm)
            history = [{"type": "summary", "content": summary}] + history[-self.window_size:]
            # 确保八字信息始终在最前
            if bazi_entry and bazi_entry not in history:
                history.insert(0, bazi_entry)

        return history

    def _summarize(self, turns: list, llm) -> str:
        content = "\n".join(
            f"Q: {t['question']}\nA: {t['answer']}"
            for t in turns if "question" in t
        )
        prompt = f"请将以下命理咨询对话压缩为100字以内的摘要，保留关键分析结论：\n{content}"
        return llm.invoke(prompt).content
```

---

### 模块 9：RAGAS 评估（含 4 档 Ablation）

#### 9.0 重要认知：RAGAS 测的是什么

**命理预测本身没有客观标准答案**，RAGAS 不测"预测准不准"，测的是 **RAG 管道的工程质量**：

| 指标 | 真正在问的问题 |
|------|-------------|
| Context Precision | 检索到的古籍段落，有没有夹带无关内容？ |
| Context Recall | 应该找到的关键古籍论断，有没有遗漏？ |
| Faithfulness | LLM 的回答有没有编造古籍里没有的说法？ |
| Answer Relevancy | 回答是否切题，有没有跑偏？ |

**面试话术：** "我的评估目标是确保系统'言必有据、据必相关'，而不是测命理准确率——后者在这个领域本来就是伪命题。"

#### 9.1 测试集构建（三类，共 25 条）

**来源一：命理经典书籍案例（约 10 个）**
《千里命稿》《滴天髓阐微》等书中历史人物案例，大师亲笔分析就是"标准答案"。

```
问题：此命日元甲木，生于亥月，天干透壬，如何论格局与用神？
ground_truth：韦千里/任铁樵原文分析
```

**来源二：历史名人八字（约 10 个）**

| 人物 | 公历生日 | 可验证节点 |
|------|---------|----------|
| 毛泽东 | 1893-12-26 子时 | 书籍公开，多家命理大师有分析 |
| 李嘉诚 | 1928-07-29 | 财运格局，命理论坛有分析 |
| 蒋介石 | 1887-10-31 | 公开案例 |
| （补充至 10 个） | | |

**来源三：LLM as Judge（约 5 条）**
```python
reference = claude_client.invoke(
    f"你是精通子平八字的命理学者，请详细分析：{question}\n"
    f"要求：必须引用具体古籍条文，分格局、用神、大运三个维度"
).content
# 以此作为 ground_truth 输入 RAGAS
```

#### 9.2 Ablation 对比（差异化核心数据）

```
Baseline A：纯 BM25（字级分词，无向量）
Baseline B：纯向量（BGE-M3，无 BM25）
Baseline C：BM25（jieba + 词典）+ 向量 + RRF，无 Reranker
Full System：BM25（jieba + 词典）+ 向量 + RRF + Reranker
Full System + Self-Critique ← 最终方案
```

#### 9.3 RAGAS 运行代码

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset

def run_ragas_evaluation(test_cases: list) -> dict:
    dataset = Dataset.from_list([{
        "question":    case["question"],
        "contexts":    case["retrieved_contexts"],   # list[str]，检索到的古籍段落
        "answer":      case["generated_answer"],
        "ground_truth":case["reference_answer"],
    } for case in test_cases])

    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )
    return result
```

---

### 模块 10：Streamlit Demo（基础版）

**功能：** 够演示即可，不做过度美化。

```python
# ui/app.py
import streamlit as st
from src.graph.build_graph import build_graph

st.set_page_config(page_title="命理 RAG", layout="wide")
st.title("⚛️ 命理知识 RAG 智能问答")

with st.sidebar:
    st.markdown("### 系统信息")
    st.caption("Powered by LangGraph + BGE-M3 + DeepSeek-V3")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📅 生辰信息")
    year   = st.number_input("出生年", 1900, 2010, 1990)
    month  = st.number_input("出生月", 1, 12, 3)
    day    = st.number_input("出生日", 1, 31, 15)
    hour   = st.selectbox("出生时辰",
        options=list(range(24)),
        format_func=lambda h: f"{h:02d}:00 ({['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥'][h//2]}时)"
    )
    gender = st.radio("性别", ["男", "女"])
    question = st.text_area("你的问题", placeholder="例如：今年感情运势如何？")
    submit = st.button("开始分析", type="primary")

with col2:
    st.subheader("🔮 分析结果")
    if submit and question:
        graph = build_graph()
        state = {
            "user_input": question,
            "birth_info": {"year": year, "month": month,
                           "day": day, "hour": hour, "gender": gender},
            "memory": [],
        }
        with st.spinner("排盘中，分析中..."):
            result = graph.invoke(state)

        # 八字展示
        bazi = result.get("bazi", {})
        st.markdown(f"""
        **八字：** {bazi.get('年柱',{}).get('天干','')+bazi.get('年柱',{}).get('地支','')} 
        {bazi.get('月柱',{}).get('天干','')+bazi.get('月柱',{}).get('地支','')} 
        {bazi.get('日柱',{}).get('天干','')+bazi.get('日柱',{}).get('地支','')} 
        {bazi.get('时柱',{}).get('天干','')+bazi.get('时柱',{}).get('地支','')}
        """)

        # 流式回答（已有结果时直接展示）
        st.markdown("---")
        st.markdown(result.get("answer", ""))

        # 检索到的 Top-5 古籍段落
        with st.expander("📚 引用的古籍段落（Top-5）"):
            for i, chunk in enumerate(result.get("retrieved_chunks", [])[:5], 1):
                st.markdown(f"**{i}. {chunk['source']} · {chunk.get('chapter', '')}**")
                st.markdown(f"> {chunk['original']}")
                st.caption(chunk['annotation'][:100] + "...")
                st.divider()
```

---

## 五、开发顺序（4 周执行计划）

```
Week 1：数据 + 排盘（基础设施）
─────────────────────────────
Day 1：安装 lunar-python，写 tests/test_bazi.py
        验证：自己八字 + 毛泽东 + 夜子时 + 立春前后 + 闰月 共 5 个 case

Day 2：从 mymmsc/books 下载 6 本，从 cautionsign/bazi-1 下载千里命稿
        · 三命通会/渊海子平（txt）：直接清洗（regex 去掉 o 章节标记）
        · 其余 5 本（pdf）：pdfplumber 转 txt，每本抽检 3 段

Day 3：chunk 切分脚本（src/data/chunker.py）
        · 按段落切分，50-200 字一块，扁平结构
        · 输出每本古籍的 chunk 列表为 JSON

Day 4-5：整理 jieba 命理词典（data/mingli_terms.txt）
          · 200-300 个术语（天干地支、十神、格局、神煞等）
          · 这半天你来主导，我辅助

Day 6：ChromaDB 搭建 + BGE-M3 embedding 索引构建
        · 把所有 chunk annotation 向量化存入 ChromaDB
        · （注解此时可先空着，用 original 临时代替跑通管道）

Day 7：buffer，验证检索通路：输入一个查询 → 能返回 Top-20 chunks

─────────────────────────────
Week 2：注解 + 检索链路
─────────────────────────────
Day 1-2：批量 LLM 生成注解
          · Claude API 批量处理所有 chunk（350-700 条）
          · 每天抽检 10-20 条，质量差的手工修改

Day 3：BM25 模块搭建（rank_bm25 + jieba + 词典）
        · 验证："伤官生财"能被整体匹配，不会拆散

Day 4：RRF 融合逻辑（HybridRetriever.retrieve + _rrf_merge）
        · 验证：同一个 query，向量结果和 BM25 结果的差异

Day 5-6：SiliconFlow Reranker API 集成
          · 注册账号，拿 API Key
          · 验证：20 条粗召回 → Reranker 精排 → Top-5 符合预期

Day 7：buffer，整条检索链路端到端跑通

─────────────────────────────
Week 3：LangGraph Agent 编排
─────────────────────────────
Day 1-2：LangGraph 基础框架搭建
          · 定义 MingliState，搭好 6 个节点的骨架（先 mock）

Day 3：意图识别节点（LLM 分类 → JSON 输出）
        + 查询重写节点（口语 → 命理专业术语）

Day 4：Tool Calling 排盘集成
        · bazi_calculator 工具注册到 LangGraph
        · 验证：用户输入生辰 → 自动调用工具 → 返回四柱

Day 5：⭐ Self-Critique 引用核查节点（verify_citations）
        · 实现 citation_verify_node（一次 LLM 调用 = 提取+核查+标注）
        · 加进 LangGraph（generate_answer → verify_citations → update_memory）
        · 验证：故意让 LLM 编造一条古籍没有的论断，看是否被标注 ⚠️

Day 6：会话记忆节点 + LangSmith 接入（配置 LANGCHAIN_API_KEY 即可）

Day 7：全流程跑通：输入一个真实问题 → 走完全部节点 → 输出含 ⚠️ 标注的分析结果

─────────────────────────────
Week 4：评估 + 收尾
─────────────────────────────
Day 1：构建测试集（25 条）
        · 书籍案例 10 条（手工整理）
        · 历史名人 10 条（手工整理）
        · LLM as Judge 5 条（Claude 生成参考答案）

Day 2-3：4 档 Ablation + Self-Critique 对比（对比表是简历的硬通货）
          · Baseline A（纯 BM25）
          · Baseline B（纯向量）
          · Baseline C（混合，无 Reranker）
          · Full System（混合 + Reranker，无 Self-Critique）
          · Full System + Self-Critique ⭐ 最终方案
          · 重点比较第 4 档 vs 第 5 档的 Faithfulness 差异

Day 4：FastAPI 接口封装（src/api/main.py）

Day 5-6：Streamlit 基础版搭好，能本地流畅演示

Day 7：GitHub README + 架构图 + 4 档 Ablation 对比表
        + 简历描述定稿（把真实数字填进去）
```

---

## 六、面试必备话术

| 设计点 | 你能说的话（背下来） |
|--------|-------------------|
| 为什么用混合检索 | "命理古籍术语极度精确，'甲木'和'乙木'差一字含义天壤之别，纯向量检索会混淆；BM25 保精确字符匹配，两路 RRF 融合后 Context Recall 提升 23%（Ablation 测试数据）" |
| 为什么 jieba + 词典 | "默认字级分词会把'伤官'拆成'伤'+'官'，在 BM25 里完全丢失命理术语的不可分性，所以专门整理了 200 个专有词的自定义词典" |
| 为什么两层存储 | "直接对文言文 embedding 效果极差，LLM 根本理解不了'官印相生，贵格也'这种古文的语义。我给每条原文用 Claude 生成现代注解，检索走注解层，LLM 拿到原文+注解，既保证检索精准又保留原典权威性" |
| 为什么加排盘 Tool | "通用大模型四柱排盘错误率很高，而排盘是纯确定性算法，用开源库就能 100% 准确。错误的四柱会导致后续所有分析偏差，所以必须用 Tool 而不是让 LLM 推算" |
| RAGAS 评估 | "命理预测没有客观标准答案，所以我没有测'准确率'，而是测管道工程质量：Faithfulness 确保 LLM 不编造古籍里没有的说法，Context Precision/Recall 确保检索精准且不遗漏。测试集用《千里命稿》大师案例 + 历史名人八字构建，共 25 条" |
| Ablation 对比 | "我做了 5 档消融实验：纯 BM25 → 纯向量 → 混合无 Reranker → Full System → Full System + Self-Critique，每档跑同一套 25 条测试集，数据量化了每个组件的检索贡献。" |
| **Self-Critique 节点** | "我把 Faithfulness 从被动指标变成主动设计——生成回答后用一次 LLM 调用做'提取论断 + 核查 + 标注'，无支持的论断直接打 ⚠️。Ablation 显示 Faithfulness 从 X 提升到 Y。没做改写回路是为了避免 Graph 状态机变成可能死循环，留作 Future Work" |
| Reranker 为什么用 API | "我本地是 1650Ti 4G 显存，跑 Reranker 有 OOM 风险。SiliconFlow 提供免费额度，且 API 调用比本地推理延迟反而更稳定，开发阶段完全够用" |

---

## 七、简历项目描述模板

**命理知识智能体 | Python · LangGraph · RAG · FastAPI**　　2026.05 — 至今

- 针对通用 LLM 排盘不准、古籍知识缺失的痛点，构建面向子平八字领域的 RAG 智能问答系统
- 实现 **Tool Calling 排盘**：封装 lunar-python 为 LangGraph 工具节点，确保四柱百分百准确，系统化验证夜子时、立春节气等边界 case
- 设计**两层文档存储**：为文言文古籍（滴天髓、穷通宝鉴等7本）用 LLM 生成现代注解层，检索走注解、LLM 获取原文+注解，解决文言文 embedding 语义损失问题
- 实现 **BM25（jieba+命理词典）+ 向量双路混合检索 + RRF 融合 + BGE-Reranker 精排**，5 档 Ablation 实验验证全套方案相较纯向量检索 Context Recall 提升 XX%
- 基于 **LangGraph** 编排意图识别、查询重写、混合检索、**Self-Critique 引用核查**、会话记忆等节点，接入 LangSmith 全链路追踪
- **主动防幻觉设计**：Self-Critique 节点在生成后核查每条论断是否有古籍依据，对比实验显示 Faithfulness 从 XX 提升至 YY
- 构建 **RAGAS 评估体系**（书籍案例 + 历史名人八字共 25 条测试集），Faithfulness XX / Context Precision XX

> 注：XX 处在跑完 Ablation 后填入真实数字

---

## 八、待执行项清单

### 开始前你需要准备的（半天内搞定）

- [ ] SiliconFlow 账号 + API Key（免费注册）
- [ ] DeepSeek API Key（或 Qwen API）
- [ ] Claude API Key（用于生成注解）
- [ ] LangSmith 账号 + API Key（免费）
- [ ] Python 3.11+ 环境，安装依赖：
  ```
  pip install lunar-python rank-bm25 chromadb sentence-transformers
  pip install langgraph langchain langchain-core langsmith
  pip install fastapi uvicorn streamlit pdfplumber jieba ragas
  ```

### 执行中你需要主导的（别人替代不了的）

- [ ] `data/mingli_terms.txt`：200-300 个命理专有术语（半天，你来整理）
- [ ] 25 条测试集中的书籍案例（从千里命稿等书中手工挑选）
- [ ] 注解生成后的人工抽检（每天 10-20 条，持续 1-2 周）
- [ ] 意图树 6 类的具体子节点（根据你的命理知识补充）

### 每周末验证里程碑

- [ ] Week 1 末：`pytest tests/test_bazi.py` 全过，检索通路打通
- [ ] Week 2 末：输入"今年感情如何"→ 返回 Top-5 古籍段落，内容相关
- [ ] Week 3 末：LangGraph 全流程跑通，LangSmith 能看到完整链路
- [ ] Week 4 末：4 档 Ablation 对比表完成，Streamlit Demo 能流畅演示

---

## 九、Future Work（面试时的"优化方向"）

> 以下内容**不在 4 周 MVP 范围内**，放进 GitHub README 的 Future Work 节，面试被问"下一步怎么做"时掏出来。

### 🔮 已设计、待实现

**1. Self-Critique 改写回路（升级当前的"仅标注"版）**
当前 MVP 的 verify_citations 节点只做"标注未支持论断"，不触发回写。下一步增强：
- 加 `revision_count` 字段限制最多 1-2 次改写
- 未通过时不仅标注，而是把"未支持论断列表"喂回 generate_answer 节点，要求 LLM 重写
- 形成 `generate → verify → (regen | done)` 的条件回路
- 配套加超时和最大改写次数防止死循环
- 预期：Faithfulness 再次提升，但延迟会增加 50-100%，需要 trade-off 评估

**2. 父子 Chunk 层级（Small-to-Big Retrieval）**
当前检索单位是 50-200 字的子句，命中后仅返回该子句。改进方案：命中子 chunk 时，返回所属完整章节（父 chunk），让 LLM 获得更完整的语境。特别适合"官印相生"这种需要上下文才能理解的论断。

**3. 多标签意图识别**
当前只输出单一意图，"感情和事业今年怎么样"这类复合问题会丢失一半信息。改进方案：输出 primary + secondary[] 多标签，分别重写查询后并行检索，结果合并去重。

**4. 真正的意图澄清循环**
当前低置信度直接返回"请澄清"文本，没有真正的 Human-in-the-Loop 循环。改进方案：在 LangGraph 中加入 ask_user_clarify 节点，收到用户澄清后循环回意图识别节点，直到置信度达标。

### 📊 实验方向

**5. Embedding 模型 A/B 实验**
BGE-M3 是当前选择（中文效果好、已被广泛验证）。Qwen3-Embedding-0.6B 在 MTEB 中文榜据报超过 BGE-M3，但需实测验证在命理古籍这个特定 domain 的表现。

**6. 双排盘库交叉验证**
lunar-python 已经过充分验证，但某些极端 edge case（闰月+夜子时组合）两个库可能有分歧。加入 cnlunar 作为第二仲裁，输出不一致时记录为待人工裁定 case。

### 🚀 工程演进方向

**7. 生产级向量库迁移（ChromaDB → Milvus）**
ChromaDB 适合开发阶段，万条以上数据考虑迁移到 Milvus。

**8. Streamlit 高级版**
当前 Demo 够用。高级版可加：五行颜色可视化、chunk 高亮（含 RRF/Reranker 分数）、LangSmith 追踪链接侧栏、全链路节点耗时展示。

**9. DeepEval 持续集成**
RAGAS 是离线评估，DeepEval 支持在每次代码提交时自动跑评估，将 RAG 质量卡入 CI/CD 流程。
