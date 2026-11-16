"""Xuất report/BAO_CAO.docx từ report/BAO_CAO.md bằng pandoc (qua pypandoc_binary) và chỉnh định dạng bằng python-docx.

Chạy (từ thư mục gốc repo; không cần cài gì vào repo):
    uv run --no-project --python 3.11 --with pypandoc_binary --with python-docx python report/build_docx.py

Định dạng: font Times New Roman 13 cho toàn văn (tiêu đề đậm, cùng font), bảng có viền đơn, hình nhúng (đường dẫn hình trong .md là
tương đối từ report/), công thức LaTeX thành công thức Word. Mục lục do pandoc sinh (cần cập nhật số trang khi mở bằng Word: nhấn F9).
"""

import re
import sys
import tempfile
from pathlib import Path

import pypandoc
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "report" / "BAO_CAO.md"
OUT = ROOT / "report" / "BAO_CAO.docx"
FONT, SIZE = "Times New Roman", 13


def set_font(style, size=None, bold=None, color_black=True):
    f = style.font
    f.name = FONT
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if color_black:
        from docx.shared import RGBColor
        f.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]


def make_reference(path):
    """reference.docx của pandoc, đã đổi font mọi kiểu."""
    import subprocess
    with open(path, "wb") as f:
        f.write(subprocess.run([pypandoc.get_pandoc_path(), "--print-default-data-file", "reference.docx"], check=True, capture_output=True).stdout)
    doc = Document(str(path))
    for st in doc.styles:
        try:
            if st.type in (1, 2):                                  # đoạn, ký tự
                set_font(st, color_black=True)
        except Exception:
            pass
    set_font(doc.styles["Normal"], SIZE)
    for n, size in (("Title", 20), ("Heading 1", 18), ("Heading 2", 15), ("Heading 3", 14), ("Heading 4", 13)):
        if n in [s.name for s in doc.styles]:
            set_font(doc.styles[n], size, bold=True)
    for n in ("Body Text", "First Paragraph", "Compact", "Block Text", "Table Caption", "Image Caption", "Captioned Figure"):
        if n in [s.name for s in doc.styles]:
            set_font(doc.styles[n], SIZE)
    for n in ("Source Code", "Verbatim Char"):
        if n in [s.name for s in doc.styles]:
            doc.styles[n].font.name = "Courier New"
            doc.styles[n].font.size = Pt(11)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin = Cm(3), Cm(2), Cm(2.5), Cm(2.5)
    doc.save(str(path))


def border_tables(doc):
    """Viền đơn cho mọi bảng, chữ bảng 11 pt (bảng rộng), hàng tiêu đề đậm."""
    for t in doc.tables:
        tbl = t._tbl
        pr = tbl.tblPr
        for old in pr.findall(qn("w:tblBorders")):
            pr.remove(old)
        b = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), "4")
            e.set(qn("w:space"), "0")
            e.set(qn("w:color"), "000000")
            b.append(e)
        pr.append(b)
        for r, row in enumerate(t.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.space_before = Pt(1)
                    for run in p.runs:
                        run.font.name = FONT
                        run.font.size = Pt(11)
                        if r == 0:
                            run.font.bold = True


def fit_images(doc, max_width_cm=16.0):
    """Hình rộng hơn khổ trang thì co lại, giữ tỉ lệ; căn giữa đoạn chứa hình."""
    maxw = Cm(max_width_cm)
    for shape in doc.inline_shapes:
        if shape.width > maxw:
            ratio = maxw / shape.width
            shape.width, shape.height = int(shape.width * ratio), int(shape.height * ratio)
    for p in doc.paragraphs:
        if p._p.findall(".//" + qn("w:drawing")):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    text = SRC.read_text(encoding="utf-8")
    text = re.sub(r"```mermaid\n.*?```", "![Hình 4.1](../reports/figures/gd4_hybrid_diagram.png)", text, flags=re.S)       # mermaid không hiển thị được trong Word
    text = re.sub(r"^## Mục lục\n.*?\n(?=\*\*Quy ước)", "", text, flags=re.S | re.M)          # bỏ mục lục tay (pandoc sinh mục lục thật)
    with tempfile.TemporaryDirectory() as d:
        ref = Path(d) / "reference.docx"
        make_reference(ref)
        tmp = Path(d) / "tmp.docx"
        pypandoc.convert_text(text, "docx", format="markdown+pipe_tables+tex_math_dollars+yaml_metadata_block", outputfile=str(tmp),
                              extra_args=[f"--reference-doc={ref}", f"--resource-path={ROOT / 'report'}", "--toc", "--toc-depth=2",
                                          "--metadata", "lang=vi-VN"])
        doc = Document(str(tmp))
    border_tables(doc)
    fit_images(doc)
    doc.save(str(OUT))
    n_img = len(doc.inline_shapes)
    print(f"Đã ghi {OUT} ({OUT.stat().st_size / 1e6:.1f} MB): {len(doc.tables)} bảng, {n_img} hình nhúng")
    return 0


if __name__ == "__main__":
    sys.exit(main())
