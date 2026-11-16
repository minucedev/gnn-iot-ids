"""Vẽ sơ đồ hybrid chính (Hình 4.1) thành reports/figures/gd4_hybrid_diagram.png, thay cho khối mermaid khi xuất DOCX và PPTX.

Chạy (từ thư mục ngoài repo):  uv run --no-project --python 3.11 --with matplotlib python <repo>/report/make_diagram.py
Chỉ là hình minh họa kiến trúc do nhóm vẽ, không chứa số liệu thực nghiệm.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "reports" / "figures" / "gd4_hybrid_diagram.png"
plt.rcParams["font.family"] = ["DejaVu Sans"]
FS = 10.5


def box(ax, x, y, w, h, text, fc="#E8EEF7", ec="#1F3A5F"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS)
    return (x, y, w, h)


def pt(r, s):
    x, y, w, h = r
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[s]


def arrow(ax, a, b, side_a="r", side_b="l"):
    ax.add_patch(FancyArrowPatch(pt(a, side_a), pt(b, side_b), arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#333333"))


fig, ax = plt.subplots(figsize=(9, 4.3))
ax.set_xlim(0, 9)
ax.set_ylim(0, 4.3)
ax.axis("off")
H1, H2 = 2.9, 0.55
A = box(ax, 0.05, H1, 1.5, 0.85, "Flow\nNetFlow")
B = box(ax, 2.0, H1, 1.5, 0.85, "Cửa sổ\n5 giây")
C = box(ax, 3.95, H1, 2.2, 0.85, "Đồ thị\nnode = IP\ncạnh = flow")
D = box(ax, 6.6, H1, 2.3, 0.85, "18 đặc trưng node\n(đã chuẩn hóa)", fc="#FFF1DC")
E = box(ax, 0.05, H2, 1.9, 0.85, "5 GAT nhóm\n(out-of-fold\nxen kẽ)", fc="#E3F2E1")
F = box(ax, 2.35, H2, 1.4, 0.85, "Embedding\n64 chiều", fc="#E3F2E1")
G = box(ax, 4.15, H2, 1.2, 0.85, "Ghép\n82 cột", fc="#F3E4F1")
Hh = box(ax, 5.75, H2, 1.3, 0.85, "LightGBM", fc="#F3E4F1")
I = box(ax, 7.45, H2, 1.5, 0.85, "P(5 lớp)\nargmax hoặc\nngưỡng", fc="#FBE3E3")
arrow(ax, A, B)
arrow(ax, B, C)
arrow(ax, C, D)
arrow(ax, C, E, "b", "t")
arrow(ax, D, G, "b", "t")
arrow(ax, E, F)
arrow(ax, F, G)
arrow(ax, G, Hh)
arrow(ax, Hh, I)
ax.text(4.5, 4.15, "Hybrid chính: xác suất của 5 góc nhìn (mỗi GAT nhóm một góc) được lấy trung bình", fontsize=10, ha="center", color="#555555")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(OUT)
