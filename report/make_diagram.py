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


def box(ax, x, y, w, h, text, fc="#E8EEF7", ec="#1F3A5F"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)
    return (x, y, w, h)


def arrow(ax, a, b, side_a="r", side_b="l"):
    def pt(r, s):
        x, y, w, h = r
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[s]
    ax.add_patch(FancyArrowPatch(pt(a, side_a), pt(b, side_b), arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#333333"))


fig, ax = plt.subplots(figsize=(13.7, 4.6))
ax.set_xlim(0, 13.7)
ax.set_ylim(0, 4.6)
ax.axis("off")
A = box(ax, 0.1, 1.9, 1.5, 0.9, "Flow\nNetFlow")
B = box(ax, 2.0, 1.9, 1.5, 0.9, "Cửa sổ\n5 giây")
C = box(ax, 3.9, 1.9, 2.0, 0.9, "Đồ thị:\nnode = IP, cạnh = flow")
D = box(ax, 6.4, 3.2, 2.2, 0.9, "18 đặc trưng node\n(đã chuẩn hóa)", fc="#FFF1DC")
E = box(ax, 6.4, 0.5, 2.2, 0.9, "5 GAT nhóm\n(out-of-fold xen kẽ)", fc="#E3F2E1")
F = box(ax, 9.0, 0.5, 1.5, 0.9, "Embedding\n64 chiều", fc="#E3F2E1")
G = box(ax, 9.0, 1.9, 1.5, 0.9, "Ghép\n82 cột", fc="#F3E4F1")
H = box(ax, 10.9, 1.9, 1.1, 0.9, "LightGBM", fc="#F3E4F1")
I = box(ax, 12.5, 1.9, 1.1, 0.9, "P(5 lớp)\nargmax /\nngưỡng", fc="#FBE3E3")
for a, b in ((A, B), (B, C)):
    arrow(ax, a, b)
arrow(ax, C, D, "r", "l")
arrow(ax, C, E, "r", "l")
arrow(ax, E, F)
arrow(ax, D, G, "r", "t")
arrow(ax, F, G, "t", "b")
arrow(ax, G, H)
arrow(ax, H, I)
ax.text(6.5, 4.45, "Hybrid chính (hybrid_gat_lgbm_oofavg_il): xác suất của 5 góc nhìn được lấy trung bình", fontsize=9.5, ha="left", color="#555555")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(OUT)
