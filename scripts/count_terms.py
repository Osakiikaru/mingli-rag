# scripts/count_terms.py
import json, re
from pathlib import Path
from collections import Counter

# 读词典
terms_path = Path("data/mingli_terms.txt")
terms = []
for line in terms_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        word = line.split()[0]
        terms.append(word)

# 读所有 chunk 的 original 字段
all_text = ""
for jf in Path("data/processed").glob("*_chunks.json"):
    chunks = json.loads(jf.read_text(encoding="utf-8"))
    for c in chunks:
        all_text += c.get("original", "")

# 统计每个词出现次数
counter = Counter()
for term in terms:
    count = len(re.findall(re.escape(term), all_text))
    if count > 0:
        counter[term] = count

# 输出：按频率降序
print(f"{'词语':<12} {'出现次数':>8}")
print("-" * 22)
for word, cnt in counter.most_common():
    print(f"{word:<12} {cnt:>8}")

# 输出从未出现的词（可能是冗余词条）
zero_terms = [t for t in terms if counter[t] == 0]
print(f"\n从未出现的词条（{len(zero_terms)} 个，可考虑删除）：")
print(", ".join(zero_terms))