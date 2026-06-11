"""
Week 2 Day 1-2: 批量生成 annotation
------------------------------------
通过 naga.ac（OpenAI 兼容接口）并发调用 DeepSeek V4 Flash，
为全部 chunks 生成现代汉语注解，写回 JSON 后重建向量索引。

用法：
  python scripts/generate_annotations.py           # 正常运行
  python scripts/generate_annotations.py --dry-run  # 只统计不发请求

费用估算（DeepSeek V4 Flash，naga.ac 价格）：
  1313 条 × 400 tokens 输入 × $0.07/1M ≈ $0.037
  1313 条 × 150 tokens 输出 × $0.14/1M ≈ $0.028
  合计约 $0.065 ≈ ¥0.47，可忽略不计
"""

import sys
import json
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── 配置 ──────────────────────────────────────────────
PROCESSED_DIR    = Path("data/processed")
CHECKPOINT_FILE  = Path("data/annotation_checkpoint.json")

MODEL            = "deepseek-v4-flash"   # 备用: "deepseek-v3.2"
MAX_TOKENS       = 350
TEMPERATURE      = 0.3
MAX_WORKERS      = 8     # 并发请求数，naga.ac 建议不超过 10
SAVE_EVERY       = 100   # 每完成 N 条自动保存一次（断点续传）
MAX_ORIGINAL_LEN = 1200  # 超长 chunk 截断，避免单条 token 过多
# ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是子平八字命理学专家。请为下面的古籍片段生成简洁的现代汉语注解，帮助检索系统准确理解该片段的命理含义。

注解要求：
1. 用现代白话文概括古文的核心命理含义，控制在 120~200 字
2. 保留并简短解释关键术语（如七杀、食神、正印、用神、格局、月令、日主等）
3. 若为命例，指出案例的八字格局类型和判断逻辑
4. 若为月令数据或表格，说明该月令对各天干五行强弱的影响规律
5. 若为基础理论，提炼核心规则或概念的命理意义

直接输出注解文字，不要输出任何前缀、标题或解释。"""


# ── 工具函数 ──────────────────────────────────────────

def load_all_chunks() -> tuple[dict[str, dict], dict[str, Path]]:
    all_chunks: dict[str, dict] = {}
    chunk_to_file: dict[str, Path] = {}
    for jf in sorted(PROCESSED_DIR.glob("*_chunks.json")):
        if "baseline" in jf.name:
            continue
        data = json.loads(jf.read_text(encoding="utf-8"))
        for chunk in data:
            all_chunks[chunk["id"]] = chunk
            chunk_to_file[chunk["id"]] = jf
    return all_chunks, chunk_to_file


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    return {"done": []}


def save_checkpoint(cp: dict):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps(cp, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_user_content(chunk: dict) -> str:
    original = chunk["original"]
    if len(original) > MAX_ORIGINAL_LEN:
        original = original[:MAX_ORIGINAL_LEN] + "……（以下略）"
    lines = [f"书名：{chunk['source']}"]
    if chunk.get("chapter"):
        lines.append(f"章节：{chunk['chapter']}")
    if chunk.get("section"):
        lines.append(f"小节：{chunk['section']}")
    lines.append(f"类型：{chunk.get('type', '理论')}")
    lines.append(f"\n原文：\n{original}")
    return "\n".join(lines)


def annotate_one(client: OpenAI, chunk_id: str, chunk: dict) -> tuple[str, str | None]:
    """
    发送单条请求。
    返回 (chunk_id, annotation)，失败返回 (chunk_id, None)
    """
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_content(chunk)},
            ],
        )
        return chunk_id, resp.choices[0].message.content.strip()
    except Exception as e:
        return chunk_id, None


def write_annotations_back(annotations: dict[str, str], chunk_to_file: dict[str, Path]):
    """按文件分组写回，减少 I/O"""
    file_updates: dict[Path, dict[str, str]] = {}
    for cid, ann in annotations.items():
        jf = chunk_to_file[cid]
        file_updates.setdefault(jf, {})[cid] = ann

    for jf, updates in file_updates.items():
        data = json.loads(jf.read_text(encoding="utf-8"))
        for chunk in data:
            if chunk["id"] in updates:
                chunk["annotation"] = updates[chunk["id"]]
        jf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  💾 {jf.name}：写入 {len(updates)} 条")


# ── 主流程 ────────────────────────────────────────────

def main(dry_run: bool = False):
    # ── Step 1: 加载数据 + checkpoint ──
    print("── Step 1: 加载 chunks ──")
    all_chunks, chunk_to_file = load_all_chunks()
    cp = load_checkpoint()
    done_set: set[str] = set(cp["done"])

    # 跳过已完成 + 已有手动注解的
    todo = [
        (cid, chunk)
        for cid, chunk in all_chunks.items()
        if cid not in done_set and not chunk.get("annotation", "").strip()
    ]

    print(f"  总 chunks : {len(all_chunks)}")
    print(f"  已完成    : {len(done_set)}")
    print(f"  待处理    : {len(todo)}\n")

    if not todo:
        print("✅ 全部完成，运行 build_index.py 重建索引即可")
        return

    if dry_run:
        avg_in, avg_out = 400, 150
        cost_usd = (
            len(todo) * avg_in  / 1_000_000 * 0.07
            + len(todo) * avg_out / 1_000_000 * 0.14
        )
        print(f"[dry-run] 待提交 {len(todo)} 条")
        print(f"          预估费用 ≈ ${cost_usd:.3f} USD（DeepSeek V4 Flash）")
        return

    # ── Step 2: 初始化客户端 ──
    client = OpenAI(
        base_url="https://api.naga.ac/v1",
        api_key=os.getenv("NAGA_API_KEY"),
    )

    # ── Step 3: 并发请求 ──
    print(f"── Step 2: 并发生成 annotation（{MAX_WORKERS} 线程）──")
    start_time = time.time()

    pending_annotations: dict[str, str] = {}
    failed: list[str] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(annotate_one, client, cid, chunk): cid
            for cid, chunk in todo
        }

        for future in as_completed(futures):
            cid, annotation = future.result()

            if annotation is not None:
                pending_annotations[cid] = annotation
                done_set.add(cid)
                completed += 1
            else:
                failed.append(cid)

            # 每 SAVE_EVERY 条保存一次（断点续传保险）
            if len(pending_annotations) >= SAVE_EVERY:
                write_annotations_back(pending_annotations, chunk_to_file)
                cp["done"] = sorted(done_set)
                save_checkpoint(cp)
                elapsed = time.time() - start_time
                print(f"  [{completed}/{len(todo)}] 已用 {elapsed:.0f}s，自动保存")
                pending_annotations.clear()

    # ── Step 4: 写回剩余结果 ──
    if pending_annotations:
        print("\n── Step 3: 写回剩余 annotation ──")
        write_annotations_back(pending_annotations, chunk_to_file)

    cp["done"] = sorted(done_set)
    save_checkpoint(cp)

    # ── 汇总 ──
    elapsed = time.time() - start_time
    print(f"\n{'✅' if not failed else '⚠️'} 完成 {len(done_set)}/{len(all_chunks)} 条，"
          f"失败 {len(failed)} 条，耗时 {elapsed:.0f}s")
    if failed:
        print(f"   失败 ID（重跑脚本可补）：{failed[:5]}{'...' if len(failed)>5 else ''}")
    else:
        print("\n下一步：python src/retrieval/build_index.py")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
