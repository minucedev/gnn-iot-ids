"""Hình của AHP (task 7.2) từ ahp_result.json: trọng số, điểm phương án, độ nhạy.

Chạy:  python -I scripts/ahp_plots.py <ahp_result.json> <thư mục hình>
Ghi:   gd7_ahp_weights.png, gd7_ahp_scores.png, gd7_ahp_sensitivity.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PERTURB = 0.20
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#9D755D", "#BAB0AC"]


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_weights(res, path):
    ids = res["criteria"]
    lab = [res["criteria_labels"][c] for c in ids]
    w = [res["weights"][c] for c in ids]
    g = [res["weights_geometric_mean"][c] for c in ids]
    order = sorted(range(len(ids)), key=lambda k: w[k])
    fig, ax = plt.subplots(figsize=(8, 4))
    y = range(len(ids))
    ax.barh(y, [w[k] for k in order], color="#4C78A8", label="Vector riêng (dùng để xếp hạng)")
    ax.scatter([g[k] for k in order], y, color="#F58518", marker="D", zorder=3, label="Trung bình nhân (kiểm)")
    for yi, k in zip(y, order):
        ax.text(w[k] + 0.008, yi, f"{w[k]:.3f}", va="center", fontsize=9)
    ax.set_yticks(list(y), [lab[k] for k in order])
    ax.set_xlabel("Trọng số (tổng = 1)")
    c = res["consistency"]
    ax.set_title(f"Trọng số tiêu chí AHP (λmax = {c['lambda_max']:.3f}, CI = {c['ci']:.4f}, CR = {c['cr']:.4f} < {c['cr_limit']})")
    ax.set_xlim(0, max(w) * 1.18)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_scores(res, path):
    ids = res["criteria"]
    alts = res["alternatives"]
    order = res["ranking"][::-1]                                        # tốt nhất ở trên cùng
    fig, ax = plt.subplots(figsize=(9, 5))
    left = [0.0] * len(order)
    for k, cid in enumerate(ids):
        part = [res["weights"][cid] * alts[n]["scores"][cid] for n in order]
        ax.barh(range(len(order)), part, left=left, color=COLORS[k], label=f"{res['criteria_labels'][cid]} (w = {res['weights'][cid]:.2f})")
        left = [a + b for a, b in zip(left, part)]
    for yi, n in enumerate(order):
        ax.text(left[yi] + 0.01, yi, f"{alts[n]['total']:.3f}", va="center", fontsize=9, fontweight="bold" if n == res["recommended"] else None)
    ax.set_yticks(range(len(order)), [alts[n]["label"] for n in order])
    ax.set_xlabel("Điểm AHP = Σ trọng số × điểm chuẩn hóa")
    ax.set_title(f"Điểm phương án (NF-ToN-IoT-v3, test, ngưỡng hiệu chỉnh); gợi ý: {res['recommended_label']}")
    ax.set_xlim(0, max(left) * 1.12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_sensitivity(res, path):
    ids = res["criteria"]
    names = list(res["alternatives"])
    top = res["recommended"]
    w = [res["weights"][c] for c in ids]
    S = {n: [res["alternatives"][n]["scores"][c] for c in ids] for n in names}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
    for i, cid in enumerate(ids):
        ax = axes.flat[i]
        rest = 1 - w[i]
        R = {n: sum(w[k] * S[n][k] for k in range(len(w)) if k != i) / rest for n in names}
        ts = [x / 200 for x in range(201)]
        for k, n in enumerate(names):
            ax.plot(ts, [t * S[n][i] + (1 - t) * R[n] for t in ts], color=COLORS[k], lw=2.6 if n == top else 1.2,
                    label=res["alternatives"][n]["label"], zorder=3 if n == top else 2)
        ax.axvspan(w[i] * (1 - PERTURB), w[i] * (1 + PERTURB), color="gray", alpha=0.18, label="±20% trọng số")
        ax.axvline(w[i], color="black", lw=1, ls="--")
        tp = res["sensitivity"]["tipping"][cid]
        for d in ("up", "down"):
            tc = tp[d]["top_change"]
            if tc:
                ax.axvline(tc["weight"], color="red", lw=1, ls=":")
                ax.annotate(f"đổi hạng 1 → {res['alternatives'][tc['new_top']]['label']}\n(w = {tc['weight']:.2f}, {tc['relative_change']:+.0%})",
                            (tc["weight"], 0.02), xytext=(4 if d == "up" else -4, 0), textcoords="offset points", fontsize=7, color="red",
                            ha="left" if d == "up" else "right", va="bottom", rotation=90)
        ax.set_title(f"Trọng số {res['criteria_labels'][cid]} (gốc {w[i]:.2f})", fontsize=10)
        ax.set_xlabel("Trọng số của tiêu chí này (các tiêu chí còn lại co giãn theo tỉ lệ)", fontsize=7)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].axis("off")
    axes.flat[5].legend(handles, labels, loc="center", fontsize=9)
    axes[0, 0].set_ylabel("Điểm AHP")
    axes[1, 0].set_ylabel("Điểm AHP")
    fig.suptitle("Độ nhạy: điểm AHP theo trọng số từng tiêu chí (đường đỏ: điểm làm phương án đứng đầu đổi)")
    return _save(fig, path)


def _stacked(ax, order, scores, w, ids, labels, criteria_labels, hatch_names=()):
    """Thanh chồng điểm AHP (trọng số × điểm chuẩn hóa) theo tiêu chí; `hatch_names` vẽ nhạt có gạch (phương án bị loại)."""
    left = [0.0] * len(order)
    for k, cid in enumerate(ids):
        for yi, n in enumerate(order):
            p = w[cid] * scores[n][cid]
            cut = n in hatch_names
            ax.barh(yi, p, left=left[yi], color=COLORS[k], alpha=0.35 if cut else 1.0, hatch="//" if cut else None,
                    edgecolor="white", label=f"{criteria_labels[cid]} (w = {w[cid]:.2f})" if yi == 0 else None)
            left[yi] += p
    ax.set_yticks(range(len(order)), [labels[n] + (" (loại)" if n in hatch_names else "") for n in order])
    return left


def plot_constrained(res, path):
    """Hai kịch bản cạnh nhau: (a) không ràng buộc, 8 phương án (phương án vượt ràng buộc vẽ nhạt); (b) sàng lọc rồi xếp hạng."""
    ids = res["criteria"]
    cl = res["criteria_labels"]
    w = res["weights"]
    labels = {n: a["label"] for n, a in res["alternatives"].items()}
    a, b = res["scenario_a_unconstrained"], res["scenario_b_constrained"]
    cons = res["constraints"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
    order_a = a["ranking"][::-1]
    left = _stacked(axes[0], order_a, a["scores"], w, ids, labels, cl, hatch_names=set(b["excluded"]))
    for yi, n in enumerate(order_a):
        axes[0].text(left[yi] + 0.01, yi, f"{a['totals'][n]:.3f}", va="center", fontsize=9, fontweight="bold" if n == a["top"] else None)
    axes[0].set_xlim(0, max(left) * 1.12)
    axes[0].set_title(f"(a) Không ràng buộc: gợi ý {a['top_label']}", fontsize=11)
    order_b = b["ranking"][::-1]
    left = _stacked(axes[1], order_b, b["scores"], w, ids, labels, cl)
    for yi, n in enumerate(order_b):
        axes[1].text(left[yi] + 0.01, yi, f"{b['totals'][n]:.3f}", va="center", fontsize=9, fontweight="bold" if n == b["top"] else None)
    axes[1].set_xlim(0, max(left) * 1.12)
    limit = f"{cons['max_params']:,}".replace(",", ".")
    axes[1].set_title(f"(b) Ràng buộc nhẹ (≤ {limit} tham số, ≤ {cons['max_size_kb']} KB): gợi ý {b['top_label']}", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Điểm AHP = Σ trọng số × điểm chuẩn hóa")
        ax.grid(axis="x", alpha=0.3)
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=9, loc="lower center", ncol=len(lab), frameon=False)
    fig.suptitle("AHP trên NF-ToN-IoT-v3 (test, ngưỡng hiệu chỉnh), cùng ma trận đã duyệt; phương án gạch chéo bị loại ở kịch bản (b)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    res = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    if "scenario_b_constrained" in res:                                      # ahp_constrained.json
        print(plot_constrained(res, out / "gd7_ahp_constrained.png"))
        return 0
    for fn, name in ((plot_weights, "gd7_ahp_weights.png"), (plot_scores, "gd7_ahp_scores.png"), (plot_sensitivity, "gd7_ahp_sensitivity.png")):
        print(fn(res, out / name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
