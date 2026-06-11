
## RAGAS 消融实验结果（LLM-as-Judge）
> 2026-06-10 20:12  |  top_k=5

| 配置             | Context Precision  | Context Recall     | Faithfulness       | Answer Relevancy   |
| ---------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| BM25-only        | —                  | —                  | —                  | —                  |
| Vector-only      | —                  | —                  | —                  | —                  |
| Hybrid           | —                  | —                  | —                  | —                  |
| Hybrid+Rerank    | —                  | —                  | —                  | —                  |
| Hybrid (无Self-Critique) | —                  | —                  | —                  | —                  |
| Hybrid (有Self-Critique) | 0.7200             | 0.7100             | 0.9550             | 0.9750             |

### 指标说明
| 指标 | 含义 |
|------|------|
| Context Precision | 召回的 chunk 中真正相关的比例 |
| Context Recall    | ground_truth 知识点被覆盖的比例 |
| Faithfulness      | 回答论断有古籍依据的比例 |
| Answer Relevancy  | 回答紧扣问题的程度 |