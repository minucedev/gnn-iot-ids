"""Xuất report/SLIDE.pptx từ report/SLIDE_DAN_Y.md bằng python-pptx.

Chạy (từ thư mục ngoài repo, để uv không dò .venv bị Windows chặn):
    uv run --no-project --python 3.11 --with python-pptx --with pillow python <repo>/report/build_pptx.py

Cú pháp đọc từ dàn ý: "### Slide N. Tiêu đề", các dòng "- ..." là ý chính (**đậm** giữ lại), dòng "**Hình:** ..." chứa đường dẫn hình trong dấu
huyền (tương đối từ report/). Hình tồn tại thì chèn bên phải (tối đa 2 hình mỗi slide, xếp dọc); không có hình thì ý chiếm cả bề ngang.
Mỗi slide ghi nguồn số liệu ở chân trang: BAO_CAO.md. Slide 1 là trang bìa.
"""

import re
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
SRC = REPORT / "SLIDE_DAN_Y.md"
OUT = REPORT / "SLIDE.pptx"
FONT = "Times New Roman"
NAVY, GREY = RGBColor(0x1F, 0x3A, 0x5F), RGBColor(0x55, 0x55, 0x55)
W, H = Inches(13.333), Inches(7.5)


def parse(text):
    slides = []
    for block in re.split(r"^### Slide ", text, flags=re.M)[1:]:
        head, _, body = block.partition("\n")
        m = re.match(r"(\d+)\.\s*(.*)", head.strip())
        title = re.sub(r"\s*\*\*\[.*?\]\*\*", "", m.group(2)).strip()
        bullets, figs, in_fig = [], [], False
        for line in body.splitlines():
            if line.startswith("---") or line.startswith("## "):
                break
            if line.startswith("- "):
                bullets.append(line[2:].strip())
            elif line.startswith("**Hình:**"):
                figs = re.findall(r"`([^`]+\.png)`", line)
        slides.append({"n": int(m.group(1)), "title": title, "bullets": bullets, "figs": figs})
    return slides


def add_runs(par, text, size, color=None):
    """Chèn văn bản vào đoạn, **đậm** thành run đậm, bỏ dấu huyền."""
    text = text.replace("`", "")
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if not part:
            continue
        r = par.add_run()
        r.text = part
        r.font.name, r.font.size, r.font.bold = FONT, Pt(size), (i % 2 == 1)
        if color is not None:
            r.font.color.rgb = color


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    return tf


def title_bar(slide, title, n):
    bar = slide.shapes.add_shape(1, 0, 0, W, Inches(1.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    tf = textbox(slide, Inches(0.5), Inches(0.1), Inches(12.3), Inches(0.85), MSO_ANCHOR.MIDDLE)
    add_runs(tf.paragraphs[0], title, 28, RGBColor(255, 255, 255))
    tf = textbox(slide, Inches(0.5), Inches(7.05), Inches(10), Inches(0.35))
    add_runs(tf.paragraphs[0], "Nguồn số liệu: report/BAO_CAO.md và các file trong reports/", 10, GREY)
    tf = textbox(slide, Inches(12.0), Inches(7.05), Inches(0.9), Inches(0.35))
    tf.paragraphs[0].alignment = PP_ALIGN.RIGHT
    add_runs(tf.paragraphs[0], str(n), 11, GREY)


def place_images(slide, paths, x, y, w, h):
    """Xếp tối đa 2 hình dọc trong khung (x, y, w, h), giữ tỉ lệ, căn giữa."""
    paths = paths[:2]
    gap = Inches(0.15)
    cell_h = (h - gap * (len(paths) - 1)) // len(paths)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            iw, ih = im.size
        scale = min(w / iw, cell_h / ih)
        pw, ph = int(iw * scale), int(ih * scale)
        slide.shapes.add_picture(str(p), x + (w - pw) // 2, y + i * (cell_h + gap) + (cell_h - ph) // 2, pw, ph)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    slides = parse(SRC.read_text(encoding="utf-8"))
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]
    missing = []
    for s in slides:
        sl = prs.slides.add_slide(blank)
        if s["n"] == 1:
            bg = sl.shapes.add_shape(1, 0, 0, W, H)
            bg.fill.solid()
            bg.fill.fore_color.rgb = NAVY
            bg.line.fill.background()
            tf = textbox(sl, Inches(0.9), Inches(1.6), Inches(11.5), Inches(2.4), MSO_ANCHOR.MIDDLE)
            add_runs(tf.paragraphs[0], "Ứng dụng mạng Lightweight GNN gợi ý phát hiện xâm nhập IoT thời gian thực", 38, RGBColor(255, 255, 255))
            tf = textbox(sl, Inches(0.9), Inches(4.3), Inches(11.5), Inches(2.2))
            lines = ["Nhóm: Nguyễn Minh Phúc (nhóm trưởng), Lê Trung Phong, Văn Bá Phạm Tấn",
                     "Phát hiện IP đang tấn công trong mạng IoT từ luồng NetFlow, bằng GNN nhẹ (hybrid GAT + LightGBM)",
                     "Mã nguồn: minucedev/gnn-iot-ids"]
            for i, t in enumerate(lines):
                par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                par.space_after = Pt(8)
                add_runs(par, t, 20, RGBColor(0xDD, 0xE6, 0xF2))
            continue
        title_bar(sl, f"{s['n']}. {s['title']}", s["n"])
        figs = []
        for f in s["figs"]:
            p = (REPORT / f).resolve()
            if p.exists():
                figs.append(p)
            else:
                missing.append((s["n"], f))
        text_w = Inches(6.4) if figs else Inches(12.3)
        size = 19 if figs else 22
        total = sum(len(b) for b in s["bullets"])
        if total > 520:
            size -= 2
        if total > 700:
            size -= 2
        tf = textbox(sl, Inches(0.5), Inches(1.3), text_w, Inches(5.6))
        for i, b in enumerate(s["bullets"]):
            par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            par.space_after = Pt(10)
            add_runs(par, "• " + b, size)
        if figs:
            place_images(sl, figs, Inches(7.1), Inches(1.3), Inches(5.8), Inches(5.6))
    prs.save(str(OUT))
    print(f"Đã ghi {OUT} ({OUT.stat().st_size / 1e6:.1f} MB): {len(prs.slides)} slide")
    for n, f in missing:
        print(f"  slide {n}: không thấy hình {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
