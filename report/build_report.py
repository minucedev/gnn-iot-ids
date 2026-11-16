"""Dựng bản báo cáo hoàn chỉnh: report/BAO_CAO_NOP.docx (và PDF qua Word, xem export_pdf.ps1) từ report/BAO_CAO.md.

Bản này KHÔNG trích đường dẫn file, tên notebook, mã nhiệm vụ nội bộ (task, GĐ, D1...D13), không có các dòng "Nguồn: ..." và không có
phụ lục truy vết nguồn. Các nhận xét có nội dung (ví dụ "Std bằng 0 vì ba seed cho cùng kết quả") được giữ lại thành ghi chú dưới bảng.
Tài liệu tham khảo học thuật (Crossref, arXiv...) được giữ lại; thêm --no-references để bỏ cả mục này.

Chạy (từ thư mục ngoài repo, để uv không dò .venv bị Windows chặn):
    uv run --no-project --python 3.11 --with pypandoc_binary --with python-docx python <repo>/report/build_report.py [--no-references] [--check]
  --check: chỉ in các chỗ còn sót tham chiếu nội bộ, không ghi file.
"""

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "report" / "BAO_CAO.md"
OUT = ROOT / "report" / "BAO_CAO_NOP.docx"
FONT, SIZE = "Times New Roman", 13
TITLE = "Ứng dụng mạng Lightweight GNN gợi ý phát hiện xâm nhập IoT thời gian thực"

PATH = re.compile(r"`[^`]*(?:/|\.csv|\.json|\.png|\.py|\.md|\.pkl|\.pt)[^`]*`")

# ---------------------------------------------------------------- làm sạch markdown


def clean_source_line(line):
    """Dòng chú thích "*Nguồn: ...*": bỏ các đoạn trỏ tới file, giữ phần giải thích còn lại thành "*Ghi chú: ...*"."""
    i = line.find("*Nguồn:")
    if i < 0:
        return line
    head, rest = line[:i], line[i + len("*Nguồn:"):]
    if rest.rstrip().endswith("*"):
        rest = rest.rstrip()[:-1]
    segs = re.split(r";\s+|(?<=[.)])\s+(?=[A-ZÀ-Ỹ\"])", rest.strip())
    kept = [s.strip() for s in segs if s.strip() and not PATH.search(s) and not re.match(r"\(?notebook\b", s.strip())]
    if not kept:
        return head.rstrip()
    kept = [k[0].upper() + k[1:] for k in kept]
    txt = " ".join(k if k.endswith((".", ")")) else k + "." for k in kept)
    return (head + "*" + txt + "*").rstrip()


PHU_LUC_B = """## Phụ lục B. Ghi chú về số liệu

1. Một số số liệu baseline trên tập val của NF-ToN-IoT-v3 được đọc từ hai lần chạy khác nhau cho kết quả khác nhau (ví dụ LightGBM val 0,9351 so với 0,8984). Báo cáo dùng bản khớp với nhật ký tìm kiếm tham số của chính baseline (đã kiểm lại); nguyên nhân chênh lệch giữa hai lần đọc chưa được làm rõ.
2. Phân tích lỗi ở mục 5.7 dùng bảng phân tích lỗi của hybrid chính (4.565 node sai trên ToN, 3,7%; 922 trên BoT, 8,6%), vì tỉ lệ lỗi này khớp accuracy test của hybrid chính (96,30% và 91,20%).
3. Số cửa sổ của NF-ToN-IoT-v3 là 66.693 (số đồ thị dùng để chia tập và huấn luyện); con số 66.722 là số đo khi thống kê kích thước cửa sổ, trước bước lọc.
4. Dòng "tái lập" và các dòng thời gian thực của Bảng 6.1 dùng seed 42 và ngưỡng đã đóng băng; các dòng còn lại là trung bình 3 seed.
"""

GIAI_DOAN = {"B": "Dựng đồ thị NF-BoT-IoT-v3", "C": "EDA", "D": "Baseline ML và MLP", "E": "GNN đơn trên NF-BoT-IoT-v3", "F": "Hybrid GAT + LightGBM",
             "G": "Ablation, hybrid nhẹ, hiệu chỉnh ngưỡng", "H": "Đóng băng, chạy test, tổng hợp", "I": "AHP, mô phỏng thời gian thực",
             "J": "AHP ràng buộc, phân tích đợt, chẩn đoán RAM, tái lập"}


def rewrite_appendix_c(text):
    a = text.index("## Phụ lục C.")
    b = text.index("## Tài liệu tham khảo")
    sec = text[a:b]
    rows = [l for l in sec.splitlines() if l.startswith("| ")]
    out = ["## Phụ lục C. Tài nguyên tính toán", "",
           "Các lần chạy chính trên Kaggle (bỏ các lần chạy hỏng, mỗi lần chạy vài giây). Tổng thời gian các lần chạy có ghi nhật ký: khoảng 11,66 giờ, "
           "trong đó 4,01 giờ GPU (Tesla T4); tổng này chưa gồm các lần chạy của bước AHP, thời gian thực và tái lập (đều CPU, 0 giờ GPU).", "",
           "**Bảng C.1. Các lần chạy trên Kaggle.**", "",
           "| Nội dung | Tác vụ | Thiết bị | Thời gian (s) | RAM đỉnh (GB) |", "|---|---|---|---|---|"]
    for r in rows[2:]:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        gd, nb, dev, secs, ram = cells[0], cells[1], cells[2], cells[3], cells[4]
        out.append(f"| {GIAI_DOAN.get(gd, gd)} | {nb.replace('`', '')} | {dev} | {secs} | {ram} |")
    return text[:a] + "\n".join(out) + "\n\n---\n\n" + text[b:]


def rewrite_references(text, keep):
    a = text.index("## Tài liệu tham khảo")
    if not keep:
        return text[:a].rstrip() + "\n"
    sec = text[a:]
    rows = [l for l in sec.splitlines() if re.match(r"\| \d+ \|", l)]
    out = ["## Tài liệu tham khảo", "", "Tài liệu được trích dẫn theo số thứ tự trong dấu ngoặc vuông.", ""]
    for r in rows:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        n, ref, doi = cells[0], cells[1], cells[2]
        ref = ref.rstrip(".")
        tail = "" if doi.startswith("không có") else f" {('DOI: ' + doi) if doi[:2] == '10' else doi}."
        out.append(f"\\[{n}\\] {ref}.{tail}\n")
    return text[:a] + "\n".join(out)


RULES = [
    # (cũ, mới): phải có mặt đúng một lần trở lên, nếu không script dừng để bạn xem lại
    ("(quyết định D4, đã bỏ vì hạn nộp)", "(đã bỏ vì hạn nộp)"),
    ("chạy trên Kaggle (D12); máy cá nhân chỉ sửa mã và tài liệu", "chạy trên Kaggle; máy cá nhân chỉ sửa mã và tài liệu"),
    ("(giả thuyết lúc thiết kế, quyết định D9 ngày 12/11/2026)", "(giả thuyết lúc thiết kế, ngày 12/11/2026)"),
    (" (`reports/tables/prelim_split_v1/`)", ""),
    (" (kế hoạch mục 5c)", ""),
    (" Phiếu của từng bộ ở `docs/dataset_cards/`.", ""),
    ("(D3, điều chỉnh sau khi thử 12 cấu hình)", "(điều chỉnh sau khi thử 12 cấu hình)"),
    ("(quyết định D13, 15/11/2026)", "(15/11/2026)"),
    ("CPU 4 nhân; D12)", "CPU 4 nhân)"),
    (" (D7 điều chỉnh)", ""),
    ("phương án B đã cân nhắc ở D9", "phương án B đã được cân nhắc"),
    ("(không tính vào F1-macro theo D5)", "(không tính vào F1-macro)"),
    ("(GĐ A–E)", ""),
    (" (GĐ F)", ""),
    (" (GĐ G)", ""),
    (" (GĐ H)", ""),
    ("ở GĐ H (1,88%", "ở bước đánh giá cuối (1,88%"),
    ("**RAM (task 8.1).**", "**RAM.**"),
    ("Đặc tả ban đầu (task 5.7):", "Đặc tả ban đầu:"),
    ("Tiêu chí công nhận (task 5.4):", "Tiêu chí công nhận:"),
    ("**Attention của GAT** (task 5.3, tiêu chí 4):", "**Attention của GAT:**"),
    ("(task 6.4)", ""),
    (" (notebook `15a`)", ""),
    (" (notebook `15h`)", ""),
    ("(notebook `15a`, trên val;", "(trên val;"),
    (" (notebook có `run_info.json`)", ""),
    (", và notebook `07-train-gnn` của GĐ A chạy trên T4 nhưng không có `run_info.json` nên giờ GPU của nó chưa được tính; con số 4,09 là cận dưới",
     ", và một lần chạy GNN đơn đầu tiên trên T4 không có nhật ký nên giờ GPU của nó chưa được tính; con số 4,09 là cận dưới"),
    (" trên 20 notebook có `run_info.json`", " trên 20 lần chạy có nhật ký"),
    (" Chi tiết ở [Phụ lục C](#phụ-lục-c-các-notebook-kaggle-và-giờ-gpu).", " Chi tiết ở Phụ lục C."),
    ("giờ GPU của notebook `07` không được ghi", "giờ GPU của một lần chạy đầu không được ghi"),
    ("(`.csv`, `.json`, `.png`)", ""),
    (" Kiểm tra môi trường trên Kaggle (notebook `00-env-check`):", " Môi trường Kaggle:"),
    ("; phiên bản ghi trong `requirements.txt`", ""),
    ("Mọi mô hình dùng chung `split.json`, cùng 3 seed (42, 43, 44) và cùng module `src/evaluate.py` (kết quả ghi vào `reports/tables/results.csv` với 20 cột).",
     "Mọi mô hình dùng chung một cách chia tập, cùng 3 seed (42, 43, 44) và cùng một mô-đun đánh giá."),
    (" bằng tag `freeze-v1` (commit `abcb5f8`)", ""),
    (" (`targets.csv`)", ""),
    (" (điểm đã nêu ở `freeze_list.md` mục 4)", ""),
    ("(`summary_test.csv`); không chạy lại test.", "; không chạy lại test."),
    (", `docs/ahp_matrix.json`)", ")"),
    ("`scripts/detector.py` phát lại", "Bộ phát hiện phát lại"),
    ("gom cửa sổ bằng `src/stream_features.py`,", "gom cửa sổ bằng mô-đun gom cửa sổ trực tuyến,"),
    (" (`scripts/undetected_episodes.py`, kết quả `reports/tables/undetected_episodes.json`; kết luận phát hiện hay không của từng đợt khớp với `windows.csv` của lần chạy thời gian thực)",
     " (kết luận phát hiện hay không của từng đợt khớp với kết quả của lần chạy thời gian thực)"),
    ("(notebook `20b`, `20c`, `20d`, `scripts/ram_diag.py`, 4.000 cửa sổ mỗi phép đo)", "(4.000 cửa sổ mỗi phép đo)"),
    ("**bộ đo trong `detector.py` giữ", "**bộ đo trong bộ phát hiện giữ"),
    ("ghi thẳng `windows.csv` từng dòng", "ghi thẳng bảng thống kê từng cửa sổ ra tệp từng dòng"),
    ("`src/stream_features.py` không có chỗ giữ", "mô-đun gom cửa sổ không có chỗ giữ"),
    ("; `reports/tables/alerts_sample.csv` là mẫu 1.000 dòng cảnh báo (t_close, window, ip, pred_class, p_attack, p_class, proc_ms)",
     "; mẫu 1.000 dòng cảnh báo gồm thời điểm đóng cửa sổ, cửa sổ, IP, lớp dự đoán, xác suất tấn công, xác suất lớp dự đoán và thời gian xử lý"),
    (" (xem README, mục \"Demo thời gian thực\")", ""),
    ("**Chạy lại `aggregate_h.py` từ output đã lưu (notebook `21_reproduce`, Kaggle CPU)**", "**Chạy lại bước tổng hợp từ kết quả đã lưu**"),
    ("chạy lại `aggregate_h.py` từ output đã lưu (notebook `21_reproduce`, Kaggle CPU)", "chạy lại bước tổng hợp kết quả từ các kết quả đã lưu (Kaggle CPU)"),
    ("(`aggregate_h.py`) từ các output đã lưu", "từ các kết quả đã lưu"),
    ("và Tấn đã có notebook tiền xử lý và EDA. Nhóm không dùng đầu ra của notebook đó", "và Tấn đã có bước tiền xử lý và EDA. Nhóm không dùng đầu ra của bước đó"),
    (" ở `reports/kaggle/gnn-iot-ids-12-eda/gd2_<bộ>_flow_stats.csv`;", ";"),
    (" (`oofavg`)", ""),
    (" (`stack`)", ""),
    ("từ trường `classes_missing_in_fit` trong `hybrid_search_oofavg_<bộ>.json`", "từ danh sách lớp vắng mặt khi huấn luyện từng GAT nhóm trong kết quả tìm kiếm của hybrid"),
    (" (`hybrid_gat_lgbm_oofavg_il`)", ""),
    ("Các notebook còn lại (đồ thị", "Các lần chạy còn lại (đồ thị"),
    (" được lưu và vẽ ở `gd6_attention_<bộ>.png`:", " được lưu và vẽ lại:"),
    (" trong phép đo `20c`)", " trong phép đo)"),
    ("RAM đỉnh 19,5 GB của notebook là do", "RAM đỉnh 19,5 GB của lần chạy là do"),
    ("chênh sau do chúng tôi tính từ `summary_test.csv` theo quy ước ở mục 2.6, `conclusions.csv` không có hàng này", "chênh sau do chúng tôi tính theo quy ước ở mục 2.6"),
    ("(Đối chiếu này do chúng tôi suy từ `summary_test.csv`; `targets.csv` chỉ tính cho hybrid.)", "(Đối chiếu này do chúng tôi suy từ bảng kết quả test; phần đánh giá chỉ tiêu chỉ tính cho hybrid.)"),
]

GENERIC = [
    (r"\s*\*\(Nguồn:.*?\)\*", ""),                                   # *(Nguồn: ...)* trong câu
    (r"\s*\((?:D\d{1,2})\)", ""),                                    # (D5), (D1)...
    (r"\s*\(task \d+\.\d+[^)]*\)", ""),
    (r"(?m)^\| `(\d[^`]*)`", r"| \1"),                                # tên tác vụ ở cột đầu bảng
]


def clean_markdown(text, keep_refs=True):
    # bỏ đầu trang (tiêu đề, nhóm, mã nguồn), trạng thái, mục lục tay
    a = text.index("## Tóm tắt")
    text = text[a:]
    text = re.sub(r"^\*\*Quy ước trình bày\.\*\*.*$", "", text, flags=re.M)   # ở trước Tóm tắt trong file gốc; xử lý riêng bên dưới
    for pat, rep in GENERIC:
        text = re.sub(pat, rep, text, flags=re.S if "Nguồn" in pat else 0)
    text = "\n".join(clean_source_line(l) for l in text.splitlines())
    unused = []
    for old, new in RULES:
        if old == "Nguồn: ":
            continue
        if old not in text:
            unused.append(old[:70])
            continue
        text = text.replace(old, new)
    if unused:
        print("Quy tắc không khớp (có thể đã được xử lý bởi quy tắc chung):")
        for u in unused:
            print("  -", u)
    # bỏ hàng viết tắt GĐ
    text = re.sub(r"^\| GĐ \| .*\n", "", text, flags=re.M)
    # phụ lục
    a = text.index("## Phụ lục B.")
    b = text.index("## Phụ lục C.")
    text = text[:a] + PHU_LUC_B + "\n" + text[b:]
    text = rewrite_appendix_c(text)
    text = rewrite_references(text, keep_refs)
    text = re.sub(r"(?<=\S)  +(?=\S)", " ", text)
    text = re.sub(r"(?<=[^\s|]) +\.(?=\s|$)", ".", text)
    text = text.replace("tải về file kết quả nhỏ.", "tải về các file kết quả nhỏ.")
    # hình: bỏ chú thích tự động của pandoc (đã có dòng "Hình x.y." bên dưới)
    text = re.sub(r"!\[[^\]]*\]\(", "![](", text)
    # sơ đồ mermaid -> hình
    text = re.sub(r"```mermaid\n.*?```", "![](../reports/figures/gd4_hybrid_diagram.png)", text, flags=re.S)
    return text


def front_matter():
    return f"""::: {{custom-style="CoverSmall"}}
BÁO CÁO CUỐI KỲ
:::

::: {{custom-style="CoverTitle"}}
{TITLE}
:::

::: {{custom-style="CoverSmall"}}
Phát hiện IP đang tấn công trong mạng IoT từ luồng NetFlow bằng mạng nơ-ron đồ thị nhẹ (hybrid GAT + LightGBM)
:::

::: {{custom-style="CoverInfo"}}
Nhóm thực hiện: Nguyễn Minh Phúc (nhóm trưởng), Lê Trung Phong, Văn Bá Phạm Tấn
:::

::: {{custom-style="CoverInfo"}}
Mã nguồn: github.com/minucedev/gnn-iot-ids
:::

::: {{custom-style="CoverInfo"}}
Tháng 11 năm 2026
:::

```{{=openxml}}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

::: {{custom-style="TocTitle"}}
MỤC LỤC
:::

```{{=openxml}}
<w:p><w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> TOC \\o "1-2" \\h \\z \\u </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>Mục lục sẽ được cập nhật khi mở bằng Word (nhấn F9).</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>
```

"""


def convert_quy_uoc():
    return ("**Quy ước trình bày.** Số thập phân dùng dấu phẩy. \"±\" là độ lệch chuẩn của 3 lần chạy với 3 seed (42, 43, 44). "
            "Hình và bảng đánh số theo chương (Hình 3.2, Bảng 5.1). Báo cáo nêu cả các chỉ tiêu đạt và chưa đạt; các nhận định chỉ đúng trong điều kiện nhất định đều ghi rõ điều kiện đó.\n\n")


# ---------------------------------------------------------------- docx
def build(keep_refs=True, check=False):
    sys.stdout.reconfigure(encoding="utf-8")
    raw = SRC.read_text(encoding="utf-8")
    text = clean_markdown(raw, keep_refs)
    # chèn "Quy ước trình bày" (đã viết lại) ngay sau Tóm tắt? -> đặt trước Danh mục viết tắt
    text = text.replace("## Danh mục từ viết tắt và ký hiệu", convert_quy_uoc() + "## Danh mục từ viết tắt và ký hiệu", 1)
    residual = []
    for n, l in enumerate(text.splitlines(), 1):
        if PATH.search(l) or re.search(r"\btask \d|\bGĐ [A-J]\b|\bD\d{1,2}\b|notebook|TASKS|\.md\b|README", l):
            residual.append((n, l[:200]))
    print(f"Còn {len(residual)} dòng có tham chiếu nội bộ:")
    for n, l in residual:
        print(f"  {n}: {l}")
    if check:
        Path(tempfile.gettempdir(), "bao_cao_nop_clean.md").write_text(text, encoding="utf-8")
        return 0
    import pypandoc
    from docx import Document

    with tempfile.TemporaryDirectory() as d:
        ref = Path(d) / "reference.docx"
        make_reference(ref)
        tmp = Path(d) / "tmp.docx"
        md = front_matter() + text
        Path(d, "full.md").write_text(md, encoding="utf-8")
        pypandoc.convert_text(md, "docx", format="markdown-fancy_lists+pipe_tables+tex_math_dollars+fenced_divs+raw_attribute+bracketed_spans", outputfile=str(tmp),
                              extra_args=[f"--reference-doc={ref}", f"--resource-path={ROOT / 'report'}", "--shift-heading-level-by=-1", "--metadata", "lang=vi-VN"])
        doc = Document(str(tmp))
    postprocess(doc)
    doc.save(str(OUT))
    print(f"Đã ghi {OUT} ({OUT.stat().st_size / 1e6:.1f} MB): {len(doc.tables)} bảng, {len(doc.inline_shapes)} hình")
    return 0


def set_font(style, size=None, bold=None, italic=None, color=None):
    from docx.shared import Pt, RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    f = style.font
    f.name = FONT
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    f.color.rgb = color if color is not None else RGBColor(0, 0, 0)
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
    import subprocess
    import pypandoc
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt, RGBColor

    with open(path, "wb") as f:
        f.write(subprocess.run([pypandoc.get_pandoc_path(), "--print-default-data-file", "reference.docx"], check=True, capture_output=True).stdout)
    doc = Document(str(path))
    for st in doc.styles:
        try:
            if st.type in (1, 2):
                set_font(st)
        except Exception:
            pass
    NAVY = RGBColor(0x1F, 0x3A, 0x5F)
    set_font(doc.styles["Normal"], SIZE)
    for n, size in (("Heading 1", 18), ("Heading 2", 15), ("Heading 3", 13.5), ("Heading 4", 13)):
        st = doc.styles[n]
        set_font(st, size, bold=True, color=NAVY)
        st.paragraph_format.space_before = Pt(16 if n == "Heading 1" else 12)
        st.paragraph_format.space_after = Pt(8)
        st.paragraph_format.keep_with_next = True
    for n in ("Body Text", "First Paragraph"):
        st = doc.styles[n]
        set_font(st, SIZE)
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        st.paragraph_format.line_spacing = 1.25
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.space_before = Pt(0)
    st = doc.styles["Compact"]
    set_font(st, SIZE)
    st.paragraph_format.line_spacing = 1.15
    st.paragraph_format.space_after = Pt(3)
    for n in ("Source Code", "Verbatim Char"):
        if n in [s.name for s in doc.styles]:
            doc.styles[n].font.name = "Courier New"
            doc.styles[n].font.size = Pt(11)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin = Cm(3), Cm(2), Cm(2.5), Cm(2.5)
    doc.save(str(path))


def postprocess(doc):
    from docx.enum.style import WD_STYLE_TYPE
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    NAVY = RGBColor(0x1F, 0x3A, 0x5F)
    styles = {s.name: s for s in doc.styles}

    def ensure(name, size, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=6, color=None):
        st = styles.get(name) or doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        st.base_style = doc.styles["Normal"]
        set_font(st, size, bold=bold, italic=italic, color=color)
        st.paragraph_format.alignment = align
        st.paragraph_format.space_before, st.paragraph_format.space_after = Pt(before), Pt(after)
        return st

    ensure("CoverSmall", 15, bold=True, before=36, after=18, color=NAVY)
    ensure("CoverTitle", 28, bold=True, before=70, after=26, color=NAVY)
    ensure("CoverInfo", 14, before=10, after=4)
    ensure("TocTitle", 18, bold=True, before=0, after=14, color=NAVY)
    # đoạn tham khảo
    ensure("RefEntry", 12, align=WD_ALIGN_PARAGRAPH.LEFT, after=4)

    body = doc.element.body
    # --- tiêu đề chương: sang trang mới
    for p in doc.paragraphs:
        if p.style.name == "Heading 1":
            p.paragraph_format.page_break_before = True
        txt = p.text.strip()
        # chú thích bảng, hình
        if re.match(r"^(Bảng|Hình) [\dA-Z]+(\.\d+)?\.", txt):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.size = Pt(12)
            if txt.startswith("Bảng"):
                p.paragraph_format.keep_with_next = True
        elif p._p.findall(".//" + qn("w:drawing")):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
        elif txt.startswith("Ghi chú:") or (p.runs and p.runs[0].italic and len(txt) > 0 and p.style.name in ("Body Text", "First Paragraph") and len(txt) < 400
                                            and all(r.italic for r in p.runs if r.text.strip())):
            for r in p.runs:
                r.font.size = Pt(11)
        if re.match(r"^\[\d+\]", txt) and p.style.name in ("Body Text", "First Paragraph", "Compact"):
            p.style = doc.styles["RefEntry"]
            p.paragraph_format.left_indent = Cm(1.0)
            p.paragraph_format.first_line_indent = Cm(-1.0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    # --- hình: co về khổ trang
    maxw, maxh = Cm(15.5), Cm(19)
    for shape in doc.inline_shapes:
        r = min(1.0, maxw / shape.width, maxh / shape.height)
        if r < 1.0:
            shape.width, shape.height = int(shape.width * r), int(shape.height * r)
    # --- bảng
    for t in doc.tables:
        tbl = t._tbl
        pr = tbl.tblPr
        for tag in ("w:tblBorders", "w:tblLayout", "w:tblW", "w:tblCellMar"):
            for old in pr.findall(qn(tag)):
                pr.remove(old)
        w = OxmlElement("w:tblW")
        w.set(qn("w:w"), "5000")
        w.set(qn("w:type"), "pct")
        pr.append(w)
        lay = OxmlElement("w:tblLayout")
        lay.set(qn("w:type"), "autofit")
        pr.append(lay)
        b = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), "4")
            e.set(qn("w:space"), "0")
            e.set(qn("w:color"), "808080")
            b.append(e)
        pr.append(b)
        mar = OxmlElement("w:tblCellMar")
        for side, v in (("top", 40), ("left", 70), ("bottom", 40), ("right", 70)):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:w"), str(v))
            e.set(qn("w:type"), "dxa")
            mar.append(e)
        pr.append(mar)
        for tc in tbl.iter(qn("w:tcW")):
            tc.set(qn("w:type"), "auto")
            tc.set(qn("w:w"), "0")
        ncols = len(t.columns)
        size = 11 if ncols <= 4 else 10 if ncols <= 6 else 9 if ncols <= 8 else 8.5
        for ri, row in enumerate(t.rows):
            trpr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit")
            trpr.append(cs)
            if ri == 0:
                th = OxmlElement("w:tblHeader")
                trpr.append(th)
            for cell in row.cells:
                if ri == 0:
                    tcpr = cell._tc.get_or_add_tcPr()
                    for old in tcpr.findall(qn("w:shd")):
                        tcpr.remove(old)
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear")
                    shd.set(qn("w:color"), "auto")
                    shd.set(qn("w:fill"), "DCE6F2")
                    tcpr.append(shd)
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.space_before = Pt(1)
                    p.paragraph_format.line_spacing = 1.0
                    for r in p.runs:
                        r.font.name = FONT
                        r.font.size = Pt(size)
                        if ri == 0:
                            r.font.bold = True
    # --- khoảng cách giữa bảng và đoạn ngay sau bảng
    for el in list(body):
        if el.tag == qn("w:tbl"):
            nxt = el.getnext()
            if nxt is not None and nxt.tag == qn("w:p"):
                ppr = nxt.find(qn("w:pPr"))
                if ppr is None:
                    ppr = OxmlElement("w:pPr")
                    nxt.insert(0, ppr)
                for old in ppr.findall(qn("w:spacing")):
                    ppr.remove(old)
                sp = OxmlElement("w:spacing")
                sp.set(qn("w:before"), "140")
                ppr.append(sp)
    # --- đầu trang và chân trang
    sec = doc.sections[0]
    sec.different_first_page_header_footer = True
    hp = sec.header.paragraphs[0]
    hp.text = TITLE
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in hp.runs:
        r.font.size = Pt(10)
        r.font.italic = True
        r.font.name = FONT
    ppr = hp._p.get_or_add_pPr()
    bd = OxmlElement("w:pBdr")
    bt = OxmlElement("w:bottom")
    for k, v in (("val", "single"), ("sz", "4"), ("space", "1"), ("color", "808080")):
        bt.set(qn(f"w:{k}"), v)
    bd.append(bt)
    ppr.append(bd)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    run.font.name, run.font.size = FONT, Pt(11)
    for kind, txt in (("begin", None), (None, " PAGE "), ("end", None)):
        if kind:
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), kind)
        else:
            e = OxmlElement("w:instrText")
            e.set(qn("xml:space"), "preserve")
            e.text = txt
        run._r.append(e)
    # --- mở file thì cập nhật trường (mục lục)
    settings = doc.settings.element
    uf = OxmlElement("w:updateFields")
    uf.set(qn("w:val"), "true")
    settings.append(uf)


if __name__ == "__main__":
    sys.exit(build(keep_refs="--no-references" not in sys.argv, check="--check" in sys.argv))
