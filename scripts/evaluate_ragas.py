"""
scripts/evaluate_ragas.py
=========================
消融实验：比较 4 种检索配置的评估指标

检索配置：
  1. BM25-only        — 纯关键词检索（jieba 分词）
  2. Vector-only      — 纯向量检索（BGE-M3）
  3. Hybrid           — BM25 + 向量 + RRF 融合
  4. Hybrid+Rerank    — Hybrid 后再用 CrossEncoder 精排

评估指标（LLM-as-Judge，全部基于 DeepSeek）：
  - Context Precision  检索精度：召回的 chunk 中，有多少真正相关
  - Context Recall     检索召回：ground_truth 所需知识有多少被覆盖
  - Faithfulness       生成忠实度：回答论断有多少能在 chunk 里找到依据
  - Answer Relevancy   回答相关性：回答是否紧扣问题

运行方式：
  python scripts/evaluate_ragas.py              # 全量 20题×4配置
  python scripts/evaluate_ragas.py --dry-run    # 只跑前2题验证流程
  python scripts/evaluate_ragas.py --no-gen     # 只跑 Context 指标（跳过生成）
  python scripts/evaluate_ragas.py --config hybrid_rerank  # 单配置
"""

# ── 必须在所有 import 之前 ──────────────────────────────────
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────
EVAL_CONFIGS = [
    {"name": "bm25_only",          "label": "BM25-only",              "mode": "bm25",   "rerank": False, "use_critic": False},
    {"name": "vector_only",        "label": "Vector-only",            "mode": "vector", "rerank": False, "use_critic": False},
    {"name": "hybrid",             "label": "Hybrid",                 "mode": "hybrid", "rerank": False, "use_critic": False},
    {"name": "hybrid_rerank",      "label": "Hybrid+Rerank",          "mode": "hybrid", "rerank": True,  "use_critic": False},
    {"name": "hybrid_no_critic",   "label": "Hybrid (无Self-Critique)", "mode": "hybrid", "rerank": False, "use_critic": False},
    {"name": "hybrid_with_critic", "label": "Hybrid (有Self-Critique)", "mode": "hybrid", "rerank": False, "use_critic": True},
]
TOP_K       = 5
RESULTS_DIR = ROOT / "data" / "eval_results"

# ─────────────────────────────────────────────────────────────
# LLM 评判提示词
# ─────────────────────────────────────────────────────────────

_PRECISION_PROMPT = """你是一位命理知识评估专家。

用户问题：{question}

以下是检索到的古籍片段：
{chunk}

请判断：这段古籍内容是否与用户问题直接相关，能帮助回答该问题？

只输出一个数字：1（相关）或 0（不相关）"""

_RECALL_PROMPT = """你是一位命理知识评估专家。

用户问题：{question}

参考答案（ground truth）：
{ground_truth}

检索到的古籍片段（合并）：
{contexts}

请判断：参考答案中的核心知识点，在检索到的古籍片段中是否基本都能找到？

评分标准（输出0.0-1.0之间的数字）：
- 1.0：参考答案的所有要点都在古籍片段中有据可查
- 0.7：大部分要点有依据，少数缺失
- 0.5：约一半要点有依据
- 0.3：只有少数要点有依据
- 0.0：古籍片段与参考答案基本无关

只输出一个数字（如 0.8），不要任何解释："""

_FAITHFULNESS_PROMPT = """你是一位命理知识评估专家。

用户问题：{question}

系统回答：
{answer}

古籍片段（检索来源）：
{contexts}

请评估：系统回答中的命理论断，有多少比例能在古籍片段中找到明确依据？

评分标准（输出0.0-1.0之间的数字）：
- 1.0：所有论断都有古籍依据
- 0.7：大部分论断有依据，少量自由发挥
- 0.5：约一半论断有据可查
- 0.3：只有少数论断有依据
- 0.0：回答基本是无据推断

只输出一个数字（如 0.8），不要任何解释："""

_RELEVANCY_PROMPT = """你是一位命理知识评估专家。

用户问题：{question}

系统回答：
{answer}

请评估：这个回答是否紧扣用户问题，没有跑题或大量无关内容？

评分标准（输出0.0-1.0之间的数字）：
- 1.0：回答完全针对问题，内容高度相关
- 0.7：主要内容相关，有少量偏题
- 0.5：部分相关，有明显偏题
- 0.3：回答较为笼统，与问题关联不紧密
- 0.0：回答基本没有回应问题

只输出一个数字（如 0.8），不要任何解释："""

_GENERATE_PROMPT = """你是一位精通子平八字命理的专家，根据提供的古籍原文回答用户问题。

用户问题：{question}

── 检索到的相关古籍片段 ─────────────────────────────
{contexts}
─────────────────────────────────────────────────────

要求：
1. 只根据上方古籍片段作答
2. 引用时注明来源（出自《书名》）
3. 用现代中文回答，300-500字"""

_CRITIC_PROMPT = """你是一位命理知识核验专家。

用户问题：{question}

系统生成的回答：
{answer}

古籍片段（检索来源）：
{contexts}

请逐一核查回答中的命理论断：
- 有古籍依据的：保留
- 无法在上方古籍片段中找到直接支撑的：删除或标注"存疑"

只输出修改后的回答，保持行文通顺，不要任何解释："""


# ─────────────────────────────────────────────────────────────
# LLM 调用
# ─────────────────────────────────────────────────────────────

def _build_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="deepseek-v4-pro",   # Judge 用 Pro，生成用 Flash，避免自评估偏差
        openai_api_key=os.getenv("NAGA_API_KEY"),
        openai_api_base="https://api.naga.ac/v1",
        temperature=0.0,
        request_timeout=60,
    )


def _parse_score(text: str) -> float:
    """从 LLM 输出里提取 0-1 之间的浮点数"""
    import re
    text = text.strip()
    m = re.search(r"([01](?:\.\d+)?|\.\d+)", text)
    if m:
        v = float(m.group(1))
        return max(0.0, min(1.0, v))
    return 0.5   # 解析失败给中间值


def _llm_score(llm, prompt: str) -> float:
    try:
        resp = llm.invoke(prompt)
        return _parse_score(resp.content)
    except Exception as e:
        print(f"      ⚠️  LLM 评分失败：{e}")
        return 0.5


# ─────────────────────────────────────────────────────────────
# 单题评估
# ─────────────────────────────────────────────────────────────

def evaluate_one(
    llm,
    question: str,
    ground_truth: str,
    chunks: list[dict],
    answer: str,
    skip_gen: bool,
) -> dict:
    """计算单题的 4 个指标"""
    contexts_text = "\n\n".join(
        f"[{i+1}] 《{c['source']}》\n{c['original'][:400]}"
        for i, c in enumerate(chunks)
    )

    # Context Precision：逐 chunk 打分取平均
    if chunks:
        prec_scores = []
        for c in chunks:
            chunk_text = f"《{c['source']}》\n{c['original'][:400]}"
            p = _PRECISION_PROMPT.format(question=question, chunk=chunk_text)
            prec_scores.append(_llm_score(llm, p))
        context_precision = sum(prec_scores) / len(prec_scores)
    else:
        context_precision = 0.0

    # Context Recall：一次性打分
    p = _RECALL_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        contexts=contexts_text or "（无检索结果）",
    )
    context_recall = _llm_score(llm, p)

    faithfulness    = None
    answer_relevancy = None

    if not skip_gen and answer:
        p = _FAITHFULNESS_PROMPT.format(
            question=question,
            answer=answer,
            contexts=contexts_text or "（无检索结果）",
        )
        faithfulness = _llm_score(llm, p)

        p = _RELEVANCY_PROMPT.format(question=question, answer=answer)
        answer_relevancy = _llm_score(llm, p)

    return {
        "context_precision":  round(context_precision, 4),
        "context_recall":     round(context_recall, 4),
        "faithfulness":       round(faithfulness, 4)     if faithfulness     is not None else None,
        "answer_relevancy":   round(answer_relevancy, 4) if answer_relevancy is not None else None,
    }


# ─────────────────────────────────────────────────────────────
# 生成回答
# ─────────────────────────────────────────────────────────────

def generate_answer(llm, question: str, chunks: list[dict]) -> str:
    contexts = "\n\n".join(
        f"[{i+1}] 《{c['source']}》\n{c['original'][:400]}"
        for i, c in enumerate(chunks)
    )
    try:
        resp = llm.invoke(_GENERATE_PROMPT.format(question=question, contexts=contexts))
        return resp.content.strip()
    except Exception as e:
        return f"（生成失败：{e}）"


def generate_with_critic(llm, question: str, chunks: list[dict]) -> str:
    """先生成回答，再由 Critic 过滤无古籍依据的论断，模拟 LangGraph 中的 critic_node。"""
    # Step 1: 生成初稿
    answer = generate_answer(llm, question, chunks)
    if not answer or answer.startswith("（生成失败"):
        return answer

    # Step 2: Critic 核验，删除无依据论断
    contexts = "\n\n".join(
        f"[{i+1}] 《{c['source']}》\n{c['original'][:400]}"
        for i, c in enumerate(chunks)
    )
    prompt = _CRITIC_PROMPT.format(
        question=question,
        answer=answer,
        contexts=contexts,
    )
    try:
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception as e:
        print(f"      ⚠️  Critic 调用失败，退回原始回答：{e}")
        return answer  # Critic 失败则保留原始回答


# ─────────────────────────────────────────────────────────────
# 单配置实验
# ─────────────────────────────────────────────────────────────

def run_config(
    config: dict,
    questions: list[dict],
    retriever,
    llm,
    skip_gen: bool,
    dry_run: bool,
) -> list[dict]:
    qs = questions[:2] if dry_run else questions
    label = config["label"]
    print(f"\n{'='*55}")
    print(f"  配置：{label}")
    print(f"  题数：{len(qs)}{'（dry-run）' if dry_run else ''}")
    print(f"{'='*55}")

    records = []
    for i, q in enumerate(qs, 1):
        print(f"  [{i:02d}/{len(qs)}] {q['topic']} — {q['question'][:28]}...", flush=True)

        # 检索
        t0 = time.time()
        try:
            chunks = retriever.search(
                q["question"],
                top_k=TOP_K,
                mode=config["mode"],
                rerank=config["rerank"],
            )
        except Exception as e:
            print(f"       ⚠️  检索失败：{e}")
            chunks = []
        t_ret = round(time.time() - t0, 2)

        # 生成
        answer = ""
        if not skip_gen:
            if config.get("use_critic", False):
                answer = generate_with_critic(llm, q["question"], chunks)
            else:
                answer = generate_answer(llm, q["question"], chunks)

        # 评估
        scores = evaluate_one(llm, q["question"], q["ground_truth"], chunks, answer, skip_gen)
        print(
            f"       检索{t_ret}s | CP={scores['context_precision']:.2f}"
            f" CR={scores['context_recall']:.2f}"
            + (f" F={scores['faithfulness']:.2f} AR={scores['answer_relevancy']:.2f}" if not skip_gen else ""),
            flush=True,
        )

        records.append({
            "id":           q["id"],
            "topic":        q["topic"],
            "question":     q["question"],
            "ground_truth": q["ground_truth"],
            "answer":       answer,
            "n_chunks":     len(chunks),
            "sources":      [f"{c['source']}·{c.get('chapter','')}" for c in chunks],
            **scores,
        })

    return records


# ─────────────────────────────────────────────────────────────
# 汇总输出
# ─────────────────────────────────────────────────────────────

def aggregate(records: list[dict], skip_gen: bool) -> dict:
    keys = ["context_precision", "context_recall"]
    if not skip_gen:
        keys += ["faithfulness", "answer_relevancy"]
    result = {}
    for k in keys:
        vals = [r[k] for r in records if r.get(k) is not None]
        result[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return result


def save_and_print(all_scores: dict, all_records: dict, skip_gen: bool):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 保存逐题详情
    for name, records in all_records.items():
        path = RESULTS_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    # 保存分数
    scores_path = RESULTS_DIR / "scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)

    # 打印 Markdown 表格
    cols = ["Context Precision", "Context Recall"]
    if not skip_gen:
        cols += ["Faithfulness", "Answer Relevancy"]
    keys = ["context_precision", "context_recall"]
    if not skip_gen:
        keys += ["faithfulness", "answer_relevancy"]

    w = 18
    header = "| 配置             | " + " | ".join(c.ljust(w) for c in cols) + " |"
    sep    = "| " + "-"*16 + " | " + " | ".join("-"*w for _ in cols) + " |"

    lines = [
        "",
        "## RAGAS 消融实验结果（LLM-as-Judge）",
        f"> {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  top_k={TOP_K}",
        "",
        header,
        sep,
    ]
    for cfg in EVAL_CONFIGS:
        name   = cfg["name"]
        label  = cfg["label"]
        scores = all_scores.get(name, {})
        vals   = " | ".join(f"{scores.get(k, '—'):.4f}".ljust(w) if isinstance(scores.get(k), float) else "—".ljust(w) for k in keys)
        lines.append(f"| {label.ljust(16)} | {vals} |")

    lines += [
        "",
        "### 指标说明",
        "| 指标 | 含义 |",
        "|------|------|",
        "| Context Precision | 召回的 chunk 中真正相关的比例 |",
        "| Context Recall    | ground_truth 知识点被覆盖的比例 |",
        "| Faithfulness      | 回答论断有古籍依据的比例 |",
        "| Answer Relevancy  | 回答紧扣问题的程度 |",
    ]

    summary = "\n".join(lines)
    print("\n" + summary)
    out = RESULTS_DIR / "summary.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\n✅ 结果已保存到 data/eval_results/")


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true", help="只跑前2题")
    parser.add_argument("--no-gen",    action="store_true", help="跳过生成，只评估 Context 指标")
    parser.add_argument("--config",    type=str, default=None, help="单配置运行")
    parser.add_argument("--no-resume", action="store_true", help="忽略缓存强制重跑")
    args = parser.parse_args()

    configs = EVAL_CONFIGS
    if args.config:
        configs = [c for c in EVAL_CONFIGS if c["name"] == args.config]
        if not configs:
            print(f"❌ 未知配置：{args.config}，可选：{[c['name'] for c in EVAL_CONFIGS]}")
            sys.exit(1)

    print("=" * 55)
    print("  命理测算RAG — 消融实验（LLM-as-Judge）")
    print(f"  配置数：{len(configs)} | no-gen={args.no_gen} | dry-run={args.dry_run}")
    print("=" * 55)

    # 加载题目
    with open("data/eval_questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"\n✅ 加载 {len(questions)} 道评测题")

    # 初始化检索器
    print("\n⏳ 初始化检索器...", flush=True)
    try:
        from src.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        print("✅ 检索器就绪", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        sys.exit(1)

    # 初始化 LLM
    print("\n⏳ 初始化 LLM...", flush=True)
    llm = _build_llm()
    print("✅ LLM 就绪", flush=True)

    # 逐配置运行
    all_scores  = {}
    all_records = {}

    for cfg in configs:
        name = cfg["name"]

        # 断点续跑
        cache_path = RESULTS_DIR / f"{name}.json"
        if not args.no_resume and not args.dry_run and cache_path.exists():
            print(f"\n⏩ [{cfg['label']}] 读取缓存（--no-resume 强制重跑）")
            with open(cache_path, encoding="utf-8") as f:
                records = json.load(f)
        else:
            records = run_config(cfg, questions, retriever, llm, args.no_gen, args.dry_run)
            if not args.dry_run:
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)

        all_scores[name]  = aggregate(records, args.no_gen)
        all_records[name] = records

    save_and_print(all_scores, all_records, args.no_gen)
    print("\n🎉 实验完成！")


if __name__ == "__main__":
    main()
