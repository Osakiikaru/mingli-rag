"""
清洗 data/raw/ 下的 txt 文件，去除水印、URL、乱码行。
清洗后覆盖原文件，并打印前后字数对比。
用法：python scripts/clean_texts.py
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

RAW_DIR = Path("data/raw")

SKIP_FILES = {
    "胡一鸣八字命理.txt",
    "胡一鸣老师八字结缘高级面授班笔记.txt",
}


def is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False  # 空行保留，后续统一处理
    # URL 或网站水印
    if re.search(r'https?://|www\.|\.cn|\.com|\.net', s, re.IGNORECASE):
        return True
    # 重复字符（wwwww... / ccczzz...）
    if re.search(r'(.)\1{4,}', s):
        return True
    # 纯英文字母行（非命理内容）
    if re.match(r'^[a-zA-Z\s\.]+$', s) and len(s) > 3:
        return True
    # 极短行（1-2个字，多为页码或排版符号）
    if len(s) <= 2:
        return True
    return False


def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if not is_noise_line(line):
            cleaned.append(line)
    # 合并多个连续空行为一个
    result = re.sub(r'\n{3,}', '\n\n', '\n'.join(cleaned))
    return result.strip()


if __name__ == "__main__":
    txts = [f for f in RAW_DIR.glob("*.txt") if f.name not in SKIP_FILES and f.name != ".gitkeep"]

    if not txts:
        print("没有找到需要清洗的 txt 文件。")
        sys.exit(0)

    print(f"找到 {len(txts)} 个文件，开始清洗...\n")
    for path in sorted(txts):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw)
        before = len(raw.replace('\n', '').replace(' ', ''))
        after  = len(cleaned.replace('\n', '').replace(' ', ''))
        removed_pct = (1 - after / before) * 100 if before else 0
        path.write_text(cleaned, encoding="utf-8")
        print(f"  {path.name}")
        print(f"    清洗前：{before:,} 字  →  清洗后：{after:,} 字  （去除 {removed_pct:.1f}%）")
        # 抽检前3段
        paras = [p.strip() for p in cleaned.split('\n') if len(p.strip()) > 15][:3]
        for p in paras:
            print(f"    ✦ {p[:60]}")
        print()

    print("清洗完成。")
