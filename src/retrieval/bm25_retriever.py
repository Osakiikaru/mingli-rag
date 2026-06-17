"""
BM25 稀疏检索器
--------------
对 annotation 字段做 jieba 分词，建立 BM25Okapi 索引。
annotation 是现代白话文，分词效果远好于原始古文。

用法：
  from src.retrieval.bm25_retriever import BM25Retriever
  r = BM25Retriever.load()          # 加载已有索引
  results = r.search("冬月甲木", top_k=20)
"""

import json
import pickle
from pathlib import Path

import jieba
import jieba.analyse
from rank_bm25 import BM25Okapi

# ── 配置 ──────────────────────────────────────────────
PROCESSED_DIR  = Path("data/processed")
INDEX_PATH     = Path("data/bm25_index.pkl")
MINGLI_DICT    = Path("data/dict/mingli_dict.txt")   # 命理自定义词典（可选）
# ──────────────────────────────────────────────────────

# ── 命理同义词扩展表 ──────────────────────────────────
# 查询里出现左边的词时，自动把右边的词附加进查询 token 列表
# 目的：弥补古籍用"三秋"而用户查询用"秋月"等表达差异
#
# 共 69 个条目，覆盖：
#   季节/月份 (20) | 天干别名 (10) | 十神别称 (7) | 格局简称 (15)
#   日主别称 (3)  | 忌神别称 (2) | 基础术语互换 (8) | 术语异体 (4)
QUERY_SYNONYMS: dict[str, list[str]] = {

    # ── 季节 ↔ 古籍三X表达（白话/古籍双向）──────────
    "春月": ["三春", "寅月", "卯月", "辰月"],
    "夏月": ["三夏", "巳月", "午月", "未月"],
    "秋月": ["三秋", "申月", "酉月", "戌月"],
    "冬月": ["三冬", "亥月", "子月", "丑月"],
    "三春": ["春月"],
    "三夏": ["夏月"],
    "三秋": ["秋月"],
    "三冬": ["冬月"],

    # ── 地支月份 → 季节（解决"辰月"查不到"三春"的问题）──
    "寅月": ["春月", "三春"],
    "卯月": ["春月", "三春"],
    "辰月": ["春月", "三春", "三月"],  # 辰是春末，穷通宝鉴写"三月"
    "巳月": ["夏月", "三夏"],
    "午月": ["夏月", "三夏"],
    "未月": ["夏月", "三夏"],
    "申月": ["秋月", "三秋"],
    "酉月": ["秋月", "三秋"],
    "戌月": ["秋月", "三秋"],
    "亥月": ["冬月", "三冬"],
    "子月": ["冬月", "三冬"],
    "丑月": ["冬月", "三冬"],

    # ── 天干常见别名（穷通宝鉴章节标题用法）──────────
    "甲木": ["甲"],
    "乙木": ["乙"],
    "丙火": ["丙"],
    "丁火": ["丁"],
    "戊土": ["戊"],
    "己土": ["己"],
    "庚金": ["庚"],
    "辛金": ["辛"],
    "壬水": ["壬"],
    "癸水": ["癸"],

    # ── 十神别称互换 ───────────────────────────────────
    # 偏印三种叫法：偏印 / 枭神 / 枭印
    "偏印": ["枭神", "枭印"],
    "枭神": ["偏印", "枭印"],
    "枭印": ["偏印", "枭神"],
    # 正印 ↔ 印绶（子平真诠基本互用）
    "正印": ["印绶"],
    "印绶": ["正印"],
    # 七杀 ↔ 偏官
    "七杀": ["偏官"],
    "偏官": ["七杀"],
    # 羊刃 ↔ 阳刃（异体字）
    "羊刃": ["阳刃"],
    "阳刃": ["羊刃"],

    # ── 格局全称 ↔ 简称（古籍常用简称）──────────────
    "正官格": ["官格"],
    "官格":   ["正官格"],
    "七杀格": ["杀格"],
    "杀格":   ["七杀格"],
    "食神格": ["食格"],
    "食格":   ["食神格"],
    "印绶格": ["印格", "正印格"],
    "正印格": ["印绶格", "印格"],
    "印格":   ["印绶格", "正印格"],
    "伤官格": ["伤格"],
    "伤格":   ["伤官格"],
    "羊刃格": ["刃格", "月刃格"],
    "刃格":   ["羊刃格", "月刃格"],
    "月刃格": ["羊刃格", "刃格"],
    "财格":   ["正财格", "偏财格"],   # 古籍"财格"泛指
    "建禄":   ["建禄格"],
    "建禄格": ["建禄"],

    # ── 日主三种说法打通 ───────────────────────────────
    "日主": ["日元", "日干"],
    "日元": ["日主", "日干"],
    "日干": ["日主", "日元"],

    # ── 忌神 ↔ 仇神（不同古籍用语不同）──────────────
    "忌神": ["仇神"],
    "仇神": ["忌神"],

    # ── 八字 ↔ 四柱（用户说八字，古籍说四柱）─────────
    "八字": ["四柱"],
    "四柱": ["八字"],

    # ── 月令 ↔ 月建（不同古籍用词不同）──────────────
    "月令": ["月建"],
    "月建": ["月令"],

    # ── 集合词扩展（提升食伤/比劫相关查询召回）──────
    "食伤": ["食神", "伤官"],   # 食伤是二者总称
    "比劫": ["比肩", "劫财"],   # 比劫是二者总称

    # ── 从格扩展 ───────────────────────────────────────
    "从格": ["从旺格", "从强格", "从弱格", "从杀格", "从财格"],

    # ── 格局简称（泛用）───────────────────────────────
    "格局": ["格"],
}
# ──────────────────────────────────────────────────────


def _setup_jieba():
    """加载命理自定义词典（如果存在）"""
    if MINGLI_DICT.exists():
        jieba.load_userdict(str(MINGLI_DICT))


def _tokenize(text: str) -> list[str]:
    """jieba 精确模式分词，过滤单字和空白"""
    return [w for w in jieba.cut(text) if len(w.strip()) > 1]


def _expand_query(query: str) -> list[str]:
    """
    查询同义词扩展：先分词，再把命理同义词附加进 token 列表。
    例如查询"壬水生于秋月"→ tokens = ["壬水", "生于", "秋月"] + ["三秋", "申月", "酉月", "戌月"]
    扩展后 BM25 能匹配到穷通宝鉴"三秋壬水"章节的内容。
    """
    tokens = _tokenize(query)
    extra: list[str] = []
    for token in tokens:
        if token in QUERY_SYNONYMS:
            extra.extend(QUERY_SYNONYMS[token])
    # 也检查整个查询字符串（捕获未被 jieba 切出的组合词）
    for term, synonyms in QUERY_SYNONYMS.items():
        if term in query and term not in tokens:
            extra.extend(synonyms)
    return tokens + extra


def _load_all_chunks() -> list[dict]:
    chunks = []
    for jf in sorted(PROCESSED_DIR.glob("*_chunks.json")):
        if "baseline" in jf.name:
            continue
        chunks.extend(json.loads(jf.read_text(encoding="utf-8")))
    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[dict], bm25: BM25Okapi):
        self.chunks = chunks          # 原始 chunk 列表，顺序与 bm25 对应
        self.bm25   = bm25

    # ── 构建 ──────────────────────────────────────────

    @classmethod
    def build(cls, save: bool = True) -> "BM25Retriever":
        """从 data/processed 读取所有 chunks，建立 BM25 索引"""
        _setup_jieba()
        print("── 构建 BM25 索引 ──")

        chunks = _load_all_chunks()
        print(f"  读取 {len(chunks)} 条 chunks")

        # 对 annotation 分词（annotation 为空则 fallback 到 original）
        corpus = []
        for c in chunks:
            text = c.get("annotation", "").strip() or c["original"]
            corpus.append(_tokenize(text))

        bm25 = BM25Okapi(corpus)
        print(f"  BM25 索引构建完成（词表大小：{len(bm25.idf)} 个词）")

        retriever = cls(chunks, bm25)
        if save:
            retriever.save()
        return retriever

    # ── 持久化 ────────────────────────────────────────

    def save(self, path: Path = INDEX_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)
        print(f"  ✅ BM25 索引已保存：{path}")

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> "BM25Retriever":
        if not path.exists():
            print(f"  索引不存在，重新构建...")
            return cls.build()
        _setup_jieba()
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"  ✅ BM25 索引已加载：{len(data['chunks'])} 条")
        return cls(data["chunks"], data["bm25"])

    # ── 检索 ──────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        返回 top_k 条结果，每条包含：
          chunk_id, source, chapter, original, annotation, bm25_score
        查询时自动做同义词扩展（秋月→三秋、七杀→偏官等），弥补古籍表达差异。
        """
        tokens = _expand_query(query)   # ← 替换为扩展版，包含同义词
        scores = self.bm25.get_scores(tokens)

        # 取分数最高的 top_k 个索引
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            c = self.chunks[idx]
            results.append({
                "chunk_id":   c["id"],
                "source":     c["source"],
                "chapter":    c.get("chapter", ""),
                "section":    c.get("section", ""),
                "original":   c["original"],
                "annotation": c.get("annotation", ""),
                "bm25_score": float(scores[idx]),
            })
        return results
