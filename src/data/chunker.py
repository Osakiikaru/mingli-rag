"""
古籍文本切分模块 —— 语义定制版（实验性，2026-05-25）

设计决策：
  1. 书籍定制化分隔符：7本书各用专属分隔逻辑
  2. 原文 + 注解/解读 合并在同一 chunk，不拆断
  3. 命例独立成块，type="命例"；理论内容 type="理论"
  4. 标题行不单独成 chunk，写入 chapter/section metadata 传播给所有子 chunk
     （层级上下文传播 / Hierarchical Context Propagation）
  5. 滴天髓：经文 + 原注 + 任氏曰 三层合并为一 chunk
  6. 对照基线（Week 4 Ablation 用）：split_by_chars() 纯字数切分版本

chunk 结构：
  {
    "id":         "zpzq_0042",
    "source":     "子平真诠",
    "chapter":    "论正官",      # 从标题行提取，传播到该章所有子 chunk
    "section":    "",           # 二级标题（穷通宝鉴、三命通会 有二级）
    "type":       "理论",       # "理论" | "命例"
    "original":   "...",
    "annotation": "",           # Week 2 批量 LLM 生成，此处留空
    "tags":       [],
  }

用法：
  python src/data/chunker.py         # 直接运行，处理全部7本书
"""

import json
import random
import re
from pathlib import Path

# ── 路径与书名映射 ─────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

SOURCE_MAP = {
    "八字 - 三命通会.txt":   ("三命通会", "scth"),
    "八字 - 渊海子平.txt":   ("渊海子平", "yhzp"),
    "《子平真诠》本义.txt":  ("子平真诠", "zpzq"),
    "八字 - 格局论命.txt":   ("格局论命", "gjlm"),
    "千里命稿.txt":          ("千里命稿", "qlmg"),
    "滴天髓.txt":            ("滴天髓",   "dts"),
    "穷通宝鉴.txt":          ("穷通宝鉴", "qtbj"),
}

# ── 全局参数 ──────────────────────────────────────────────────────────────────
MIN_CHARS    = 50
MAX_CHARS    = 600
SENTENCE_END = re.compile(r'(?<=[。！？；])')


# ══════════════════════════════════════════════════════════════════════════════
# 通用工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _make_chunk(idx: int, prefix: str, source: str, text: str,
                chapter: str = "", section: str = "",
                type_: str = "理论") -> dict:
    """构建标准 chunk 字典。"""
    return {
        "id":         f"{prefix}_{idx:04d}",
        "source":     source,
        "chapter":    chapter,
        "section":    section,
        "type":       type_,
        "original":   text.strip(),
        "annotation": "",
        "tags":       [],
    }


def _split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """在句末标点处切分过长段落，尽量每段 <= max_chars。"""
    if len(text) <= max_chars:
        return [text]
    parts = SENTENCE_END.split(text)
    chunks, buf = [], ""
    for part in parts:
        if len(buf) + len(part) <= max_chars:
            buf += part
        else:
            if len(buf) >= MIN_CHARS:
                chunks.append(buf)
            buf = part
    if len(buf) >= MIN_CHARS:
        chunks.append(buf)
    return chunks if chunks else [text]  # 兜底：无法切分时整段返回


def _flush_buf(buf: str, idx: int, prefix: str, source: str,
               chapter: str = "", section: str = "",
               type_: str = "理论",
               max_chars: int = MAX_CHARS) -> tuple[list[dict], int]:
    """将 buf 切分并输出 chunk 列表，返回 (新chunks列表, 新idx)。"""
    text = buf.strip().replace('\n', '')
    result = []
    for seg in _split_long(text, max_chars):
        if len(seg) >= MIN_CHARS:
            result.append(_make_chunk(idx, prefix, source, seg,
                                      chapter=chapter, section=section,
                                      type_=type_))
            idx += 1
    return result, idx


# ══════════════════════════════════════════════════════════════════════════════
# 通用段落切分（fallback，当专属 splitter 不可用时）
# ══════════════════════════════════════════════════════════════════════════════

def _generic_split(text: str, source: str, prefix: str) -> list[dict]:
    """按双换行切段，合并过短，拆分过长。无任何书籍特化逻辑。"""
    paras = [p.strip().replace('\n', '')
             for p in re.split(r'\n{2,}', text) if p.strip()]
    # 合并过短段
    merged, buf = [], ""
    for p in paras:
        if len(p) < MIN_CHARS:
            buf += p
        else:
            merged.append(buf + p) if buf else merged.append(p)
            buf = ""
    if buf:
        merged.append(buf)

    chunks, idx = [], 0
    for para in merged:
        new_chunks, idx = _flush_buf(para, idx, prefix, source)
        chunks.extend(new_chunks)
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 书籍专属 splitter
# ══════════════════════════════════════════════════════════════════════════════

# ── 子平真诠（zpzq）───────────────────────────────────────────────────────────
_ZPZQ_CHAPTER  = re.compile(r'^(?:[一二三四五六七八九十百]+[、]\s?)?论[^\n，。]{1,20}$')
_ZPZQ_YUANWEN  = re.compile(r'^原文[：:]')                  # 原文：...
_ZPZQ_MINGLI   = re.compile(
    r'^(命例|例[一二三四五六七八九十\d]|【例'
    r'|例如[^，。\n]{1,15}命[：:]'     # 例如于右任命：
    r'|另有[^，。\n]{1,15}命[：:]'     # 另有某男命：
    r'|还有[^，。\n]{1,15}命[：:]'     # 还有一个男命：
    r'|再如[^，。\n]{1,15}命[：:]'     # 再如某男命：
    r')'
)

def _split_zpzq(text: str, source: str, prefix: str) -> list[dict]:
    """
    子平真诠：
      - 章节标题（"论..."短行）→ chapter
      - "原文："开头 → 新的 type=理论 chunk
      - "命例"/"例X" 开头 → 新的 type=命例 chunk
    """
    lines = text.splitlines()
    chunks, idx = [], 0
    current_chapter = ""
    current_type    = "理论"
    buf = ""

    def flush(c_type=None):
        nonlocal idx, buf
        c_type = c_type or current_type
        new_c, idx = _flush_buf(buf, idx, prefix, source,
                                chapter=current_chapter, type_=c_type)
        chunks.extend(new_c)
        buf = ""

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # 章节标题（"论正官"类短行）
        if _ZPZQ_CHAPTER.match(s):
            if buf:
                flush()
            current_chapter = s
            current_type    = "理论"
            continue

        # 命例段落
        if _ZPZQ_MINGLI.match(s):
            if buf:
                flush()
            current_type = "命例"
            buf = s + "\n"
            continue

        # 原文段落 → 重置为理论
        if _ZPZQ_YUANWEN.match(s):
            if buf:
                flush()
            current_type = "理论"
            buf = s + "\n"
            continue

        buf += s + "\n"

    if buf:
        flush()
    return chunks


# ── 渊海子平（yhzp）──────────────────────────────────────────────────────────
_YHZP_SECTION = re.compile(r'^《([^》]{2,35})》')   # 《论 XX》/ 《干支体象》等行首章节标题
_YHZP_CASE_ID = re.compile(                        # 行内四柱干支 【乙丑，乙亥，壬申，乙巳】
    r'【([甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥，、\s]{5,30})】'
)
_GANZHI_SET = set('甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥')


def _is_ganzhi_title(name: str) -> bool:
    """判断【...】内容是否为四柱干支（命例标识），区别于书名/篇名。"""
    clean = re.sub(r'[，、\s　]', '', name)
    return len(clean) >= 4 and all(c in _GANZHI_SET for c in clean)


def _is_chapter_section(title: str) -> bool:
    """判断《》内容是否为需切分的章节标题（排除《诗诀》《伤官说》等小节）。"""
    clean = title.replace(' ', '')
    if clean == '诗诀':                            return False  # 诗歌注解小节
    if len(clean) <= 2:                            return False  # 《甲》《子》等单字
    if clean.endswith('说') and len(clean) <= 5:   return False  # 《伤官说》等
    return True


def _split_yhzp(text: str, source: str, prefix: str) -> list[dict]:
    """
    渊海子平 v2：逐行处理
    - 《论XX》/ 《干支体象》等行首 → 章节分界（type=理论）
    - 独立【篇名】行 → 大章节；独立【干支四柱】行 → 命例分界
    - 行内含【干支四柱】 → 就地切出命例 chunk
    - 《诗诀》《伤官说》等小节合并在当前章节 chunk 中
    """
    lines = text.splitlines()
    chunks, idx = [], 0
    current_chapter = source
    current_type = "理论"
    buf = ""

    def flush():
        nonlocal idx, buf
        t = buf.strip()
        if not t:
            buf = ""
            return
        new_c, idx = _flush_buf(t, idx, prefix, source,
                                chapter=current_chapter,
                                type_=current_type,
                                max_chars=1000)
        chunks.extend(new_c)
        buf = ""

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # ① 独立【篇名/四柱】行
        if re.match(r'^【[^】]{1,30}】$', s):
            flush()
            name = s[1:-1]
            if _is_ganzhi_title(name):
                current_chapter = name
                current_type    = "命例"
            else:
                current_chapter = name
                current_type    = "理论"
            continue

        # ② 行首《...》章节标题
        m = _YHZP_SECTION.match(s)
        if m:
            title = m.group(1).replace(' ', '')
            if _is_chapter_section(title):
                flush()
                current_chapter = title
                current_type    = "理论"
                remainder = s[m.end():].strip()
                if remainder:
                    buf = remainder + "\n"
            else:
                buf += s + "\n"   # 《诗诀》等小节继续合并到当前 chunk
            continue

        # ③ 行内含【干支四柱】 → 就地切分命例
        case_m = _YHZP_CASE_ID.search(s)
        if case_m:
            before = s[:case_m.start()].strip()
            if before:
                buf += before + "\n"
            flush()
            current_chapter = case_m.group(1).replace(' ', '')
            current_type    = "命例"
            after = s[case_m.end():].strip()
            buf = (after + "\n") if after else ""
            continue

        # ④ 普通内容行
        buf += s + "\n"

    if buf.strip():
        flush()
    return chunks


# ── 三命通会（scth）──────────────────────────────────────────────────────────
def _split_scth(text: str, source: str, prefix: str) -> list[dict]:
    """
    三命通会：○ 一级标题 → chapter；△ 二级标题 → section。
    以 △ 级别为基本切分单元（目标 600-900字）。
    """
    lines = text.splitlines()
    chunks, idx = [], 0
    current_chapter = ""
    current_section = ""
    buf = ""

    def flush():
        nonlocal idx, buf
        new_c, idx = _flush_buf(buf, idx, prefix, source,
                                chapter=current_chapter,
                                section=current_section,
                                max_chars=900)
        chunks.extend(new_c)
        buf = ""

    for line in lines:
        s = line.strip()
        if not s:
            continue

        if s.startswith('○'):
            if buf:
                flush()
            current_chapter = s.lstrip('○').strip()
            current_section = ""
            continue

        if s.startswith('△'):
            if buf:
                flush()
            current_section = s.lstrip('△').strip()
            continue

        buf += s + "\n"

    if buf:
        flush()
    return chunks


# ── 格局论命（gjlm）──────────────────────────────────────────────────────────
_GJLM_SECTION = re.compile(
    r'^[一二三四五六七八九十百\d]+[、.．。\s]'
    r'[（(]?[一-鿿]{2,10}[）)]?$'
)
_GJLM_MINGLI  = re.compile(r'^(命例|案例|例[一二三四五六七八九十\d：:])')

def _split_gjlm(text: str, source: str, prefix: str) -> list[dict]:
    """
    格局论命：编号章节（一、/1.）作为分界。
    理论与命例按关键词区分，允许单节最长 1500 字。
    """
    lines = text.splitlines()
    chunks, idx = [], 0
    current_chapter = ""
    current_type    = "理论"
    buf = ""

    def flush():
        nonlocal idx, buf
        new_c, idx = _flush_buf(buf, idx, prefix, source,
                                chapter=current_chapter,
                                type_=current_type,
                                max_chars=1500)
        chunks.extend(new_c)
        buf = ""

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # 编号章节标题
        if _GJLM_SECTION.match(s):
            if buf:
                flush()
            current_chapter = s
            current_type    = "理论"
            continue

        # 命例段落
        if _GJLM_MINGLI.match(s):
            if buf:
                flush()
            current_type = "命例"
            buf = s + "\n"
            continue

        buf += s + "\n"

    if buf:
        flush()
    return chunks


# ── 千里命稿（qlmg）──────────────────────────────────────────────────────────
def _normalize_repeated_title(line: str) -> str | None:
    """
    检测并还原重复字标题。
    "天天天干干干篇篇篇" → "天干篇"（PDF 扫描 artifact）
    返回 None 表示不是重复字标题。
    """
    s = line.strip()
    if not (3 <= len(s) <= 30):
        return None
    collapsed = re.sub(r'(.)\1+', r'\1', s)
    # 折叠后<=8字 且 原始长度≥折叠后2倍 → 判定为重复字标题
    if len(collapsed) <= 8 and len(s) >= len(collapsed) * 2:
        return collapsed
    return None


_QLMG_MINGLI = re.compile(r'^(造[一二三四五六七八九十\d]|命例[一二三四五六七八九十\d]|【造)')

def _split_qlmg(text: str, source: str, prefix: str) -> list[dict]:
    """
    千里命稿：重复字标题行作为章节分界。
    "造一"/"造二"等命例条目独立成块，type="命例"。
    """
    lines = text.splitlines()
    chunks, idx = [], 0
    current_chapter = ""
    current_type    = "理论"
    buf = ""

    def flush():
        nonlocal idx, buf
        new_c, idx = _flush_buf(buf, idx, prefix, source,
                                chapter=current_chapter,
                                type_=current_type,
                                max_chars=1500)
        chunks.extend(new_c)
        buf = ""

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # 重复字标题检测
        normalized = _normalize_repeated_title(s)
        if normalized:
            if buf:
                flush()
            current_chapter = normalized
            current_type    = "理论"
            continue

        # 命例段落（造X）
        if _QLMG_MINGLI.match(s):
            if buf:
                flush()
            current_type = "命例"
            buf = s + "\n"
            continue

        buf += s + "\n"

    if buf:
        flush()
    return chunks


# ── 穷通宝鉴（qtbj）──────────────────────────────────────────────────────────
# 实际格式：
#   "三春甲木总论"  → 章节总论（独立行）
#   "三春甲木"      → 章节标题（独立行）
#   "正月甲木，..."  → 月份小节，标题嵌在行首
_QTBJ_CHAPTER = re.compile(
    r'^(?:三[春夏秋冬][甲乙丙丁戊己庚辛壬癸][木火土金水]?(?:总论)?'
    r'|五行总论'
    r'|[甲乙丙丁戊己庚辛壬癸][木火土金水]?总论)$'
)
_QTBJ_SECTION = re.compile(
    r'^((?:正|二|三|四|五|六|七|八|九|十[一二]?)月'
    r'[甲乙丙丁戊己庚辛壬癸][木火土金水]?)[，,]'
)

def _split_qtbj(text: str, source: str, prefix: str) -> list[dict]:
    """
    穷通宝鉴：矩阵结构。
      chapter = 季节+天干（三春甲木 / 三夏甲木 ...）
      section = 月份+天干（正月甲木 / 二月甲木 ...）
    月份小节行首含标题，整行作为正文内容，section 提取到 metadata。
    """
    lines = text.splitlines()
    chunks, idx = [], 0
    current_chapter = ""
    current_section = ""
    buf = ""

    def flush():
        nonlocal idx, buf
        # 有 section（月份小节）才是矩阵行 → type=表格；无 section（总论/五行介绍）→ type=理论
        chunk_type = "表格" if current_section else "理论"
        new_c, idx = _flush_buf(buf, idx, prefix, source,
                                chapter=current_chapter,
                                section=current_section,
                                type_=chunk_type,
                                max_chars=700)
        chunks.extend(new_c)
        buf = ""

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # 季节+天干章节标题（"三春甲木" / "三春甲木总论"）
        if _QTBJ_CHAPTER.match(s):
            if buf:
                flush()
            # 去掉"总论"后缀，提取干净的 chapter 名
            ch = re.sub(r'总论$', '', s).strip()
            current_chapter = ch if ch else s
            current_section = ""
            continue

        # 月份+天干小节（"正月甲木，..."），标题嵌在行首
        m = _QTBJ_SECTION.match(s)
        if m:
            if buf:
                flush()
            current_section = m.group(1)
            buf = s + "\n"   # 整行（含header）都作为正文
            continue

        buf += s + "\n"

    if buf:
        flush()
    return chunks


# ── 滴天髓（dts）────────────────────────────────────────────────────────────
# 实际格式（逐行，非双换行分段）：
#   通神论 / 六亲论   → 大章节标题
#   一、天道           → 小节标题（数字+、+名称）
#   欲识三元万法宗...  → 经文（紧跟小节标题的下一行）
#   原注：...          → 原注内容（可多行）
#   任氏曰：...        → 任氏曰内容（可多行）
# 文件开头为目录区（1-66行），目录只有标题行，无经文/原注/任氏曰，
# 解析时自动跳过（flush 时若内容为空则 no-op）。
_DTS_CHAPTER = re.compile(r'^(通神论|六亲论)$')
_DTS_SECTION = re.compile(r'^[一二三四五六七八九十]+[、]\s?[一-鿿]{2,10}$')
_DTS_YUANZHU = re.compile(r'^原注[：:]')
_DTS_RENSHI  = re.compile(r'^任氏曰[：:]')


def _split_dts(text: str, source: str, prefix: str) -> list[dict]:
    """
    滴天髓：逐行解析，跳过目录区，合并 经文+原注+任氏曰 为一 chunk。
    遇到下一个小节标题 → flush 当前单元。
    """
    lines = [l.strip() for l in text.splitlines()]
    chunks, idx = [], 0
    current_chapter = ""
    current_section = ""
    jingwen = ""
    yuanzhu = ""
    renshi  = ""
    # state: "start" | "chapter" | "sec_hdr" | "jingwen" | "yuanzhu" | "renshi"
    state = "start"

    def flush():
        nonlocal idx, jingwen, yuanzhu, renshi
        if not any([jingwen, yuanzhu, renshi]):
            return
        parts = [p for p in [jingwen, yuanzhu, renshi] if p]
        merged = "\n".join(parts).strip()
        if len(merged) >= MIN_CHARS:
            for seg in _split_long(merged, max_chars=800):
                if len(seg) >= MIN_CHARS:
                    chunks.append(_make_chunk(idx, prefix, source, seg,
                                              chapter=current_chapter,
                                              section=current_section))
                    idx += 1
        jingwen = yuanzhu = renshi = ""

    for line in lines:
        if not line:
            continue

        # 大章节（通神论 / 六亲论）
        if _DTS_CHAPTER.match(line):
            flush()
            current_chapter = line
            current_section = ""
            state = "chapter"
            continue

        # 小节标题（一、天道 / 二十三、震兑 ...）
        if _DTS_SECTION.match(line):
            flush()
            current_section = line
            state = "sec_hdr"
            continue

        # 原注
        if _DTS_YUANZHU.match(line):
            yuanzhu += ("" if not yuanzhu else "\n") + line
            state = "yuanzhu"
            continue

        # 任氏曰
        if _DTS_RENSHI.match(line):
            renshi += ("" if not renshi else "\n") + line
            state = "renshi"
            continue

        # 其余行：按当前 state 分流
        if state == "sec_hdr":
            # 紧跟小节标题的第一行 = 经文
            jingwen = line
            state = "jingwen"
        elif state == "jingwen":
            jingwen += "\n" + line
        elif state == "yuanzhu":
            yuanzhu += "\n" + line
        elif state == "renshi":
            renshi += "\n" + line
        # state in ("start", "chapter"): 目录区或章节标题后，跳过

    flush()
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 主切分入口
# ══════════════════════════════════════════════════════════════════════════════

_SPLITTERS: dict = {
    "zpzq": _split_zpzq,
    "yhzp": _split_yhzp,
    "scth": _split_scth,
    "gjlm": _split_gjlm,
    "qlmg": _split_qlmg,
    "qtbj": _split_qtbj,
    "dts":  _split_dts,
}


def split_into_chunks(text: str, source: str, prefix: str) -> list[dict]:
    """根据书籍 prefix 选择对应 splitter，返回 chunk 列表。"""
    splitter = _SPLITTERS.get(prefix, _generic_split)
    return splitter(text, source, prefix)


# ══════════════════════════════════════════════════════════════════════════════
# 对照基线：纯字数切分（Week 4 Ablation 用）
# ══════════════════════════════════════════════════════════════════════════════

def split_by_chars(text: str, source: str, prefix: str,
                   min_chars: int = MIN_CHARS,
                   max_chars: int = MAX_CHARS) -> list[dict]:
    """
    Ablation 对照组：纯按字数 + 句末标点切分，不考虑语义结构。
    无 chapter/section/type 信息（全部为空/"理论"）。
    输出到 {prefix}_chunks_baseline.json。
    """
    text_flat = re.sub(r'\n+', '', text)
    parts = SENTENCE_END.split(text_flat)
    chunks, buf, idx = [], "", 0
    for part in parts:
        if len(buf) + len(part) <= max_chars:
            buf += part
        else:
            if len(buf) >= min_chars:
                chunks.append(_make_chunk(idx, prefix, source, buf))
                idx += 1
            buf = part
    if len(buf) >= min_chars:
        chunks.append(_make_chunk(idx, prefix, source, buf))
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 批量处理入口
# ══════════════════════════════════════════════════════════════════════════════

def chunk_all(raw_dir: Path = RAW_DIR,
              out_dir: Path = PROCESSED_DIR,
              also_baseline: bool = True) -> dict[str, int]:
    """
    处理全部7本古籍：
      - 语义切分 → {prefix}_chunks.json
      - 字数基线 → {prefix}_chunks_baseline.json（also_baseline=True时）
    返回 {书名: chunk数量} 统计字典。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    for filename, (source, prefix) in SOURCE_MAP.items():
        path = raw_dir / filename
        if not path.exists():
            print(f"  [跳过] {filename} 不存在")
            continue

        # 自动检测编码（格局论命等 GBK 文件 fallback）
        text = None
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                text = path.read_text(encoding=enc, errors="strict")
                break
            except (UnicodeDecodeError, ValueError):
                continue
        if text is None:
            text = path.read_text(encoding="utf-8", errors="ignore")
            print(f"  [警告] {filename} 编码检测全部失败，强制 UTF-8")

        # ── 语义切分（主策略）
        chunks = split_into_chunks(text, source, prefix)
        (out_dir / f"{prefix}_chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        stats[source] = len(chunks)

        # ── 字数基线（Ablation 对照组）
        baseline_count = 0
        if also_baseline:
            baseline = split_by_chars(text, source, prefix)
            (out_dir / f"{prefix}_chunks_baseline.json").write_text(
                json.dumps(baseline, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            baseline_count = len(baseline)

        # ── 抽检信息
        type_dist = {}
        for c in chunks:
            t = c.get("type", "理论")
            type_dist[t] = type_dist.get(t, 0) + 1

        chapters = {c.get("chapter", "") or "（无标题）" for c in chunks}

        bl_info = f" / {baseline_count}（字数基线）" if also_baseline else ""
        print(f"\n  ✓ {source}: {len(chunks)} chunks（语义）{bl_info}")
        print(f"    类型分布: {type_dist}  |  章节数: {len(chapters)}")

        # 随机抽 3 个样本
        samples = random.sample(chunks, min(3, len(chunks)))
        for s in samples:
            ch  = f"[{s['chapter'][:8]}]" if s['chapter'] else ""
            sec = f"[{s['section'][:6]}]" if s['section'] else ""
            preview = s["original"][:55].replace('\n', ' ')
            print(f"    [{s['id']}]{ch}{sec}[{s['type']}] {preview}…")

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# 直接运行入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("开始切分古籍（语义定制版 + 字数基线）...\n")
    stats = chunk_all()
    total = sum(stats.values())
    print(f"\n合计：{total} 个 chunk（语义切分），输出目录：data/processed/")
    print("\n各书分布：")
    for src, cnt in stats.items():
        print(f"  {src}: {cnt}")
