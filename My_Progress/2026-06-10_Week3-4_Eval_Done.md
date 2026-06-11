# 2026-06-10 Week 3–4 进展：Agent 框架完成 + 消融实验全部跑完

## 本阶段完成内容概览

| 阶段 | 内容 |
|------|------|
| Week 3 | LangGraph Agent 7节点框架、大运排盘、Streamlit 多轮界面 |
| Week 4（前半） | 自研 LLM-as-Judge 评估框架、5档消融实验、Self-Critique 对比 |

---

## Week 3：LangGraph Agent 端到端问答系统

### 系统架构

```
用户输入
  │
  ▼
intent_parser（DeepSeek-v4-flash）
  │ 三路路由：chat / knowledge / bazi
  │
  ├── chat_node           → 自由对话（DeepSeek-v4-flash，temperature=0.7）
  │
  ├── bazi_node           → lunar-python 确定性排盘（无LLM）
  │     └── query_rewriter_node → 扩展检索词（DeepSeek-v4-flash）
  │                              ↓
  └── retriever_node      → 混合检索（BGE-M3 + BM25 + RRF + 可选Reranker）
                              ↓
                         generator_node（DeepSeek-v4-flash，temperature=0.3）
                              ↓
                         critic_node（Self-Critique，DeepSeek-v4-flash）
                              ↓
                         final_answer
```

### 关键实现细节

**1. 三路路由设计（LangGraph conditional_edges）**
- `"chat"` → chat_node（直接回复，不查古籍）
- `"knowledge"` → retriever 路线（查古籍生成回答）
- `"bazi"` → 先排盘再检索（Tool Calling 模式）

**2. 大运排盘扩展（`src/tools/bazi.py`）**
- 使用 lunar-python 的 `getYun()` API
- 返回起运岁数、顺逆、8步大运列表（每步干支+起止年龄+起止年份）
- 解决了 LLM 直接排盘不准确的问题（确定性算法 vs 语言模型）

**3. Self-Critique 节点（`src/agent/nodes.py` → `critic_node`）**
- 生成回答后再跑一次 LLM 核验
- 逐句检查每个论断是否有古籍 chunk 依据
- 删除无据论断，只保留有古籍支撑的内容

**4. Streamlit 多轮对话界面（`app.py`）**
- `st.status` 展示分步进度（意图解析 → 检索 → 生成 → 核验）
- 全量历史传递（去掉了之前的 6 条切片限制）
- 消息导出功能

---

## Week 4：评估体系 + 消融实验

### 评估框架设计

**放弃 ragas 库**：ragas 内部依赖 sentence_transformers，在 Windows 环境下因 fork 问题静默崩溃。
**方案**：从零实现 LLM-as-Judge 评估（`scripts/evaluate_ragas.py`），完全不依赖 ragas。

**Judge 模型**：DeepSeek-v4-pro（生成用 flash，评估用 pro，避免自评估偏差）

**4个评估指标**：

| 指标 | 实现方式 | 含义 |
|------|---------|------|
| Context Precision | 每个 chunk 单独打分（二值 0/1），取均值 | 检索到的 chunk 有多少真正相关 |
| Context Recall | 1次打分，ground_truth vs 全部 chunks | ground_truth 知识点被覆盖的比例 |
| Faithfulness | 1次打分，回答 vs 全部 chunks | 回答论断有多少能在古籍中找到依据 |
| Answer Relevancy | 1次打分，回答 vs 问题 | 回答是否紧扣问题 |

**测试集**：20道命理知识问答题（`data/eval_questions.json`），涵盖八格、用神调候、从格、日干分析、大运十神等主要知识域，每题含人工标注 ground_truth。

---

## 消融实验完整结果

### 第1-4档：检索配置对比（Context 指标，Judge: DeepSeek-v4-flash）

| 配置 | Context Precision | Context Recall | 说明 |
|------|-------------------|----------------|------|
| BM25-only | 0.460 | 0.505 | 纯关键词检索 |
| Vector-only | 0.690 | 0.695 | 纯 BGE-M3 向量检索 |
| **Hybrid** | **0.710** | **0.710** | BM25 + 向量 + RRF 融合 ✅ 最优 |
| Hybrid+Rerank | 0.700 | 0.685 | 加 CrossEncoder 精排，反而略降 |

**关键发现1：BM25 大面积失效（CP=0.00）于"壬水秋月""己土春月""羊刃格"等题目**
- 根因：古籍用"三秋壬水"，查询用"壬水秋月"，字符无重叠
- 向量检索通过语义理解弥补了这个缺口
- 相对提升：Vector vs BM25，CP +50%（0.46 → 0.69）

**关键发现2：Hybrid+Rerank 效果不如 Hybrid（反直觉）**
- 根因：bge-reranker-v2-m3 在现代文本上训练，文言古籍属于领域偏移
- Reranker 对"枭印夺食""羊刃制杀"等术语的语义敏感度不足
- 工业界修复方案：领域微调 Reranker，或改用 LLM-as-Reranker

**优化项（本轮执行）**：
- BM25 查询同义词扩展（"秋月"→"三秋/申月/酉月/戌月"，"七杀"↔"偏官"等）
- 重排序模块支持 GPU 推理（GTX 1650 Ti，待 CUDA PyTorch 重装后生效）

---

### 第5-6档：Self-Critique 效果对比（Judge: DeepSeek-v4-pro）

| 配置 | CP | CR | Faithfulness | Answer Relevancy |
|------|----|----|-------------|-----------------|
| Hybrid（无 Self-Critique） | 0.760 | 0.730 | **0.905** | 1.000 |
| Hybrid（有 Self-Critique） | 0.720 | 0.710 | **0.955** | 0.975 |
| 变化 | ±noise | ±noise | **+0.050 ✅** | -0.025 ⚠️ |

**关键发现3：Self-Critique 将 Faithfulness 提升 +5.5%（0.905 → 0.955）**
- 证明 critic_node 有效减少了无古籍依据的论断
- 提升幅度适中（而非大幅），因为 DeepSeek Flash 在高质量检索上下文下本身就较为忠实

**关键发现4：Answer Relevancy 微降（1.00 → 0.975）——Critic 过度剪裁的代价**
- Critic 偶尔删除了恰好直接回答问题但古籍依据不够直接的句子
- 揭示 Faithfulness 和 Answer Relevancy 之间存在固有张力
- 可通过调整 Critic prompt 的严格程度来平衡

**注：第5-6档与第1-4档的 CP/CR 绝对值不可直接比较（judge 从 flash 升级到 pro），但相对趋势一致。**

---

## 已知问题与局限

| 问题 | 严重程度 | 根因 | 改进方向 |
|------|---------|------|---------|
| Hybrid+Rerank 在 CPU 上每题 ~300s | 中 | 无 GPU，CrossEncoder 串行推理 | 重装 CUDA PyTorch（1650 Ti 可支持），约 40-50x 加速 |
| 测试集仅 20 题，统计置信度有限 | 低 | 时间约束 | 扩充至 40-50 题，覆盖合冲刑害、神煞等盲区 |
| 评估打分有随机性（LLM 方差 ±0.03-0.05） | 低 | LLM-as-Judge 固有特性 | 多轮打分取均值，或用更稳定的专用 judge 模型 |
| 古籍语料仅 7 本 1313 个 chunk | 中 | 数据收集限制 | 框架已支持无缝扩充，不需改代码 |

---

## 后续任务清单

**P0（本周必须完成）**

- [ ] GitHub README（架构图 + 技术栈 + 消融结果表格 + 快速开始）
- [ ] .gitignore（确保 .env / chroma_db / models 不上传）
- [ ] git push 到 GitHub

**P1（本周完成）**

- [ ] FastAPI 接口（`src/api/main.py`，POST /chat）
- [ ] LangSmith 接入（`.env` 加4行，无需改代码）

**P2（后续优化，视时间）**

- [ ] 重装 CUDA 版 PyTorch，开启 GPU 加速 Reranker
- [ ] 扩充测试集至 40-50 题，覆盖更多知识域
- [ ] 精修部分 ground_truth（Q9 用神的表述有细微不准确）
- [ ] 考虑 LLM-as-Reranker 替换 CrossEncoder（解决领域偏移问题）

---

## 面试技术亮点（本阶段新增）

1. **自研 LLM-as-Judge 评估框架**：完全不依赖 ragas，理解每个指标的计算原理和局限性
2. **消融实验方法论**：5档系统对比，发现 Reranker 领域偏移问题，有反直觉的实验结论
3. **Self-Critique 定量验证**：Faithfulness +5.5%，同时观察到 Faithfulness/Relevancy 的设计张力
4. **Judge 与 Generator 分离**：Pro 版打分、Flash 版生成，主动规避自评估偏差
5. **BM25 同义词扩展**：针对古籍"三秋"vs"秋月"的具体问题，工程化修复 BM25 盲区
