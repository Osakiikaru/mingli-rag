"""
把 data/raw/ 下的 PDF 古籍转换为 txt，输出到同目录。
用法：python scripts/pdf_to_txt.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import pdfplumber

RAW_DIR = Path("data/raw")


def pdf_to_txt(pdf_path: Path) -> Path:
    out_path = pdf_path.with_suffix(".txt")
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    content = "\n".join(pages_text)
    out_path.write_text(content, encoding="utf-8")
    print(f"  ✓ {pdf_path.name} → {out_path.name}  ({len(content)} 字)")
    return out_path


def sample_check(txt_path: Path, n: int = 3):
    """抽检前 n 段，目视判断转换质量"""
    text = txt_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 20]
    print(f"\n  【{txt_path.stem} 抽检 {n} 段】")
    for p in paragraphs[:n]:
        print(f"    {p[:80]}")


if __name__ == "__main__":
    pdfs = list(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print("data/raw/ 下没有 PDF 文件，请先下载古籍。")
        sys.exit(0)

    print(f"找到 {len(pdfs)} 个 PDF，开始转换...\n")
    for pdf_path in pdfs:
        txt_path = pdf_to_txt(pdf_path)
        sample_check(txt_path)

    print("\n全部完成。目视检查抽检内容，乱码严重的用 PaddleOCR 补救。")
