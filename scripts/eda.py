"""EDA và trực quan hóa đồ thị cho NF-ToN-IoT-v3 và NF-BoT-IoT-v3 (task 2.1, 2.2, 2.3, GĐ C).

Chạy trên Kaggle (CPU), mọi hàm vẽ dùng chung cho 2 bộ:

    python -I scripts/eda.py run   <bộ> <csv> <thư_mục_đồ_thị> <outdir> [--smoke]
    python -I scripts/eda.py merge <outdir> <bộ> [<bộ> ...]

`run` ghi vào <outdir>: gd2_<bộ>_<tên>.png (dpi ≥ 150), gd2_<bộ>_flow_stats.csv và eda_summary_<bộ>.json
(số liệu kèm từng hình và nhận xét 1–2 câu sinh từ chính số liệu đó). `merge` gộp các file đó thành eda_summary.json,
vẽ hình quy trình flow → cửa sổ → đồ thị (gd2_chung_quy_trinh.png) và ghi gd2_homophily.csv.

Mức flow đọc CSV gốc theo khối bằng pyarrow (hai lượt): lượt 1 đếm nhãn và số flow theo phút (đủ toàn bộ dữ liệu);
lượt 2 tính min/max/% thiếu (đủ toàn bộ) và lấy hai mẫu ngẫu nhiên: mẫu đều 2 % (median, tương quan) và mẫu cân bằng
theo nhãn gốc, tối đa BALANCED_CAP flow mỗi nhãn (boxplot, mutual information). EDA chỉ mô tả dữ liệu, không dùng để
chọn cấu hình mô hình.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.build_graph import GROUP_ORDER, to_group  # noqa: E402

SEED = 42
DPI = 160
BLOCK = 64 << 20                 # kích thước khối đọc CSV (byte)
BALANCED_CAP = 30_000            # mẫu cân bằng: tối đa số flow mỗi nhãn gốc
UNIFORM_FRAC = 0.02              # mẫu đều: tỉ lệ flow
CORR_ROWS = 100_000              # số dòng dùng cho ma trận tương quan
MI_CAP = 5_000                   # mutual information: tối đa số flow mỗi nhãn gốc
MAX_NODES_DRAW = 300             # hình đồ thị mẫu: tối đa số node
PARTS = ("train", "val", "test")

DATASET_TITLES = {"nf_ton_iot_v3": "NF-ToN-IoT-v3", "nf_bot_iot_v3": "NF-BoT-IoT-v3"}
TS_COL, LABEL_COL = "FLOW_START_MILLISECONDS", "Attack"
NON_FEATURES = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Label", "Attack"}
BOX_FEATURES = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS",
                "SRC_TO_DST_AVG_THROUGHPUT", "SRC_TO_DST_IAT_AVG", "L4_DST_PORT"]
UNITS = {"IN_BYTES": "byte", "OUT_BYTES": "byte", "IN_PKTS": "gói", "OUT_PKTS": "gói",
         "FLOW_DURATION_MILLISECONDS": "ms", "DURATION_IN": "ms", "DURATION_OUT": "ms",
         "SRC_TO_DST_AVG_THROUGHPUT": "bit/s", "DST_TO_SRC_AVG_THROUGHPUT": "bit/s",
         "SRC_TO_DST_SECOND_BYTES": "byte/s", "DST_TO_SRC_SECOND_BYTES": "byte/s",
         "SRC_TO_DST_IAT_AVG": "ms", "SRC_TO_DST_IAT_MIN": "ms", "SRC_TO_DST_IAT_MAX": "ms",
         "SRC_TO_DST_IAT_STDDEV": "ms", "DST_TO_SRC_IAT_AVG": "ms", "DST_TO_SRC_IAT_MIN": "ms",
         "DST_TO_SRC_IAT_MAX": "ms", "DST_TO_SRC_IAT_STDDEV": "ms",
         "L4_SRC_PORT": "số hiệu cổng", "L4_DST_PORT": "số hiệu cổng", "MIN_TTL": "hop", "MAX_TTL": "hop",
         "LONGEST_FLOW_PKT": "byte", "SHORTEST_FLOW_PKT": "byte", "MIN_IP_PKT_LEN": "byte", "MAX_IP_PKT_LEN": "byte",
         "TCP_WIN_MAX_IN": "byte", "TCP_WIN_MAX_OUT": "byte"}

# Okabe–Ito (an toàn cho người mù màu); cùng một màu cho một lớp ở mọi hình
CLASS_COLORS = {"Normal": "#0072B2", "DoS_DDoS": "#D55E00", "Botnet_Backdoor": "#E69F00",
                "Truy_cap_trai_phep": "#009E73", "Khac": "#CC79A7"}
CLASS_WRAP = {"Normal": "Normal", "DoS_DDoS": "DoS_\nDDoS", "Botnet_Backdoor": "Botnet_\nBackdoor",
              "Truy_cap_trai_phep": "Truy_cap_\ntrai_phep", "Khac": "Khac"}
RAW_OTHER_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#CC79A7", "#000000",
                    "#8C564B", "#17BECF", "#BCBD22"]

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 9, "figure.titlesize": 13,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})


# ---------------------------------------------------------------- tiện ích
def vn(x, d=0):
    """Số theo cách viết của tiếng Việt: 1.234.567 và 0,87."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:,.{d}f}".replace(",", "\0").replace(".", ",").replace("\0", ".")


def pct(x, d=1):
    return f"{vn(100 * x, d)} %"


def jc(o):
    """Chuyển numpy/NaN về kiểu JSON."""
    if isinstance(o, dict):
        return {str(k): jc(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jc(v) for v in o]
    if isinstance(o, np.ndarray):
        return jc(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


def raw_colors(labels):
    """Màu cho nhãn gốc: Benign dùng màu của Normal, nhãn khác lấy lần lượt từ bảng màu."""
    out, i = {}, 0
    for lab in labels:
        if lab.lower() in ("benign", "normal"):
            out[lab] = CLASS_COLORS["Normal"]
        else:
            out[lab] = RAW_OTHER_COLORS[i % len(RAW_OTHER_COLORS)]
            i += 1
    return out


class Recorder:
    """Lưu hình, kiểm tra nhãn trục, gom số liệu và nhận xét của từng hình."""

    def __init__(self, outdir, dataset):
        self.outdir, self.dataset, self.figures = Path(outdir), dataset, []

    def save(self, fig, key, title, level, numbers, comment, title_set=True):
        if title_set:
            fig.suptitle(title)
        fname = f"gd2_{self.dataset}_{key}.png"
        axes_info, problems = [], []
        for i, ax in enumerate(fig.axes):
            if not ax.axison or ax.get_label() == "<colorbar>":
                continue
            xl, yl = ax.get_xlabel(), ax.get_ylabel()
            axes_info.append({"x": xl, "y": yl})
            if not xl or not yl:
                problems.append(f"trục {i} thiếu nhãn")
        fig.savefig(self.outdir / fname, dpi=DPI)
        plt.close(fig)
        self.figures.append(jc({"id": f"{self.dataset}/{key}", "dataset": self.dataset, "level": level, "file": fname,
                                "title": title, "dpi": DPI, "axes": axes_info, "axis_label_problems": problems,
                                "numbers": numbers, "comment": comment}))
        print(f"  hình {fname}" + (f"  CHÚ Ý: {problems}" if problems else ""))


def describe(a):
    a = np.asarray(a, dtype="float64")
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


# ================================================================ MỨC FLOW
def read_header(csv):
    with open(csv, encoding="utf-8-sig") as f:
        return [c.strip() for c in f.readline().rstrip("\r\n").split(",")]


def _reader(csv, columns, types):
    import pyarrow.csv as pacsv
    return pacsv.open_csv(csv, read_options=pacsv.ReadOptions(block_size=BLOCK),
                          convert_options=pacsv.ConvertOptions(include_columns=columns, column_types=types))


def scan_labels_time(csv, max_batches=None):
    """Lượt 1: số flow theo nhãn gốc và theo (phút, nhãn) trên toàn bộ file."""
    import pyarrow as pa
    reader = _reader(csv, [TS_COL, LABEL_COL], {TS_COL: pa.float64(), LABEL_COL: pa.string()})
    counts, per_min = Counter(), Counter()
    n = 0
    for i, b in enumerate(reader):
        if max_batches and i >= max_batches:
            break
        ts = b.column(TS_COL).to_numpy(zero_copy_only=False)
        lab = b.column(LABEL_COL).to_pandas().astype(str).str.strip()
        counts.update(lab.value_counts().to_dict())
        ok = np.isfinite(ts)
        g = pd.DataFrame({"m": np.floor(ts[ok] / 60000.0).astype(np.int64), "l": lab.to_numpy()[ok]}) \
            .groupby(["m", "l"]).size()
        for (m, l), c in g.items():
            per_min[(int(m), str(l))] += int(c)
        n += len(ts)
    return dict(counts), per_min, n


def scan_features(csv, feats, label_counts, max_batches=None):
    """Lượt 2: min/max/thiếu toàn bộ; mẫu đều 2 % và mẫu cân bằng theo nhãn gốc."""
    import pyarrow as pa
    types = {f: pa.float64() for f in feats}
    types[LABEL_COL] = pa.string()
    reader = _reader(csv, feats + [LABEL_COL], types)
    rng = np.random.default_rng(SEED)
    k = len(feats)
    mn, mx = np.full(k, np.inf), np.full(k, -np.inf)
    n_nan, n_inf, n_neg = (np.zeros(k, dtype=np.int64) for _ in range(3))
    p_lab = {lab: min(1.0, BALANCED_CAP / c) for lab, c in label_counts.items()}
    uni, bal, bal_lab = [], [], []
    n = 0
    for i, b in enumerate(reader):
        if max_batches and i >= max_batches:
            break
        X = np.empty((b.num_rows, k))
        for j, f in enumerate(feats):
            X[:, j] = b.column(f).to_numpy(zero_copy_only=False)
        lab = b.column(LABEL_COL).to_pandas().astype(str).str.strip()
        fin = np.isfinite(X)
        n_nan += np.isnan(X).sum(0)
        n_inf += np.isinf(X).sum(0)
        n_neg += (X < 0).sum(0)
        mn = np.minimum(mn, np.where(fin, X, np.inf).min(0))
        mx = np.maximum(mx, np.where(fin, X, -np.inf).max(0))
        uni.append(X[rng.random(len(X)) < UNIFORM_FRAC])
        keep = rng.random(len(X)) < lab.map(p_lab).to_numpy(dtype="float64", na_value=0.0)
        bal.append(X[keep])
        bal_lab.append(lab.to_numpy()[keep])
        n += len(X)
    U = np.concatenate(uni)
    B = pd.DataFrame(np.concatenate(bal), columns=feats)
    B["attack"] = np.concatenate(bal_lab)
    B["group"] = to_group(B["attack"])
    stats = pd.DataFrame({"feature": feats, "unit": [UNITS.get(f, "") for f in feats],
                          "min": np.where(np.isfinite(mn), mn, np.nan),
                          "median": np.nanmedian(np.where(np.isfinite(U), U, np.nan), axis=0),
                          "max": np.where(np.isfinite(mx), mx, np.nan),
                          "n_missing": n_nan + n_inf, "pct_missing": 100.0 * (n_nan + n_inf) / max(n, 1),
                          "n_negative": n_neg})
    return stats, U, B, n


def log1p_pos(a):
    """log(1 + x) với giá trị âm kẹp về 0 và NaN/vô cực về 0 (đặc trưng NetFlow đều ≥ 0)."""
    a = np.nan_to_num(np.asarray(a, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    return np.log1p(np.clip(a, 0, None))


def fig_labels(rec, ds_title, raw_counts):
    raw = sorted(raw_counts.items(), key=lambda kv: -kv[1])
    total = sum(raw_counts.values())
    grp = {g: 0 for g in GROUP_ORDER}
    for lab, c in raw:
        grp[str(to_group([lab])[0])] += c
    rc = raw_colors([k for k, _ in raw])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, names, vals, cols, ttl, ylab in (
            (axes[0], [k for k, _ in raw], [v for _, v in raw], [rc[k] for k, _ in raw], "(a) Nhãn gốc", "Nhãn gốc (cột Attack)"),
            (axes[1], list(grp), list(grp.values()), [CLASS_COLORS[g] for g in grp], "(b) 5 lớp của đề tài", "Lớp")):
        ypos = np.arange(len(names))[::-1]
        ax.barh(ypos, [max(v, 1) for v in vals], color=cols, height=0.7)
        ax.set_xscale("log")
        ax.set_yticks(ypos, names)
        ax.set_xlim(1, max(vals) * 30)
        for y, v in zip(ypos, vals):
            ax.text(max(v, 1) * 1.15, y, vn(v) if v else "0 (vắng mặt)", va="center", fontsize=9)
        ax.set_xlabel("Số flow (thang log)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
    big, small = raw[0], raw[-1]
    absent = [g for g, c in grp.items() if c == 0]
    c1 = (f"{ds_title} có {vn(total)} flow thuộc {len(raw)} nhãn gốc: nhãn lớn nhất {big[0]} chiếm {pct(big[1] / total)}, "
          f"nhãn nhỏ nhất {small[0]} chỉ {vn(small[1])} flow (lệch {vn(big[1] / small[1])} lần).")
    c2 = (f"Theo 5 lớp thì Normal chiếm {pct(grp['Normal'] / total)}" +
          (f"; lớp vắng mặt: {', '.join(absent)}." if absent else "; đủ cả 5 lớp.") +
          (" Mất cân bằng mạnh nên cần macro-F1 và trọng số lớp." if big[1] / small[1] >= 10 else ""))
    rec.save(fig, "label_dist", f"{ds_title}: phân bố nhãn (số flow, thang log)", "flow",
             {"total_flows": total, "raw_counts": raw_counts, "group_counts": grp,
              "imbalance_max_over_min": big[1] / small[1], "absent_classes": absent}, f"{c1} {c2}")
    return grp, total


def fig_time(rec, ds_title, per_min, raw_counts):
    df = pd.DataFrame([(m, l, c) for (m, l), c in per_min.items()], columns=["m", "l", "c"])
    # độ rộng ô: nhỏ nhất (phút) để số ô có flow ≤ 300
    w = next((w for w in (1, 5, 10, 30, 60, 180, 360, 720, 1440, 2880, 10080) if (df["m"] // w).nunique() <= 300), 10080)
    df["b"] = df["m"] // w
    piv = df.pivot_table(index="b", columns="l", values="c", aggfunc="sum", fill_value=0)
    order = [l for l, _ in sorted(raw_counts.items(), key=lambda kv: -kv[1]) if l in piv.columns]
    piv = piv[order].sort_index()
    x = pd.to_datetime(piv.index.to_numpy() * w * 60, unit="s")
    rc = raw_colors(order)
    wl = f"{w} phút" if w < 60 else (f"{w // 60} giờ" if w < 1440 else f"{w // 1440} ngày")
    # trục x chỉ gồm các ô có flow; khoảng trống > 2 giờ bị nén thành một vạch đứt để thấy được các đợt ngắn
    bins = piv.index.to_numpy()
    diff = np.diff(bins)
    brk = np.r_[False, diff * w > 120]
    step = np.where(brk[1:], 4, diff)
    pos = np.r_[0, np.cumsum(step)]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout="constrained")
    bottom = np.zeros(len(piv))
    for lab in order[::-1]:
        axes[0].bar(pos, piv[lab].to_numpy(), width=1.0, bottom=bottom, color=rc[lab], label=lab)
        bottom += piv[lab].to_numpy()
    axes[0].set_ylabel(f"Số flow mỗi {wl} (xếp chồng)")
    axes[0].set_title("(a) Xếp chồng theo nhãn gốc")
    for lab in order:
        v = piv[lab].to_numpy().astype(float)
        full = np.full(int(pos[-1]) + 1, np.nan)
        full[pos] = np.where(v > 0, v, np.nan)
        axes[1].plot(np.arange(len(full)), full, ".-", ms=3, lw=0.7, color=rc[lab], label=lab)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(f"Số flow mỗi {wl} (thang log)")
    axes[1].set_title("(b) Từng nhãn riêng, thang log")
    seg_start = np.r_[0, np.flatnonzero(brk)]
    cand = set(seg_start.tolist())
    for a, b in zip(seg_start, np.r_[seg_start[1:], len(bins)]):
        cand |= set(range(int(a), int(b), 60))
    kept = []                               # bỏ nhãn cách nhãn trước < 7 % chiều rộng trục để không đè nhau
    for j in sorted(cand):
        if not kept or pos[j] - pos[kept[-1]] >= 0.07 * pos[-1]:
            kept.append(j)
    cand = kept
    for ax in axes:
        for j in np.flatnonzero(brk):
            ax.axvline(pos[j] - 2, color="#999999", ls="--", lw=0.8)
        ax.set_xticks([pos[j] for j in cand], [f"{pd.Timestamp(int(bins[j]) * w * 60, unit='s'):%d/%m}\n{pd.Timestamp(int(bins[j]) * w * 60, unit='s'):%H:%M}" for j in cand],
                      fontsize=8)
        ax.set_xlabel("Thời điểm bắt đầu flow (UTC); vạch đứt = khoảng trống > 2 giờ đã bị nén")
        ax.set_xlim(-2, pos[-1] + 2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper", title="Nhãn gốc")
    t0, t1 = x[0], x[-1] + pd.Timedelta(minutes=w)
    span_h = (t1 - t0).total_seconds() / 3600
    coverage = len(piv) / (int(piv.index.max() - piv.index.min()) + 1)
    n_days = len(set(x.date))
    attack = [l for l in order if l.lower() not in ("benign", "normal")]
    total = sum(raw_counts.values())
    att_share = 1 - sum(raw_counts[l] for l in order if l not in attack) / total
    active = {l: float((piv[l] > 0).mean()) for l in attack}
    rare = min(active, key=active.get) if active else None
    peak_lab = max(attack, key=lambda l: piv[l].max()) if attack else None
    c1 = (f"Dữ liệu trải từ {t0:%d/%m/%Y %H:%M} đến {t1:%d/%m/%Y %H:%M} UTC ({vn(span_h, 1)} giờ, {n_days} ngày lịch có flow); "
          f"{pct(coverage, 0)} số ô {wl} trong khoảng này có flow" + (", phần còn lại là khoảng trống." if coverage < 0.95 else "."))
    c2 = ""
    if rare:
        c2 = (f"Flow tấn công chiếm {pct(att_share)}; nhãn {peak_lab} có đỉnh {vn(piv[peak_lab].max())} flow mỗi {wl}, "
              f"nhãn {rare} có mặt trong {pct(active[rare], 0)} số ô có flow" +
              (", tức đến theo từng đợt xen kẽ lưu lượng khác." if active[rare] < 0.5 else ", tức xuất hiện gần như suốt thời gian."))
    rec.save(fig, "flows_over_time", f"{ds_title}: số flow theo thời gian, tô màu theo loại tấn công", "flow",
             {"t_start_utc": str(t0), "t_end_utc": str(t1), "span_hours": span_h, "bin_minutes": w, "n_bins_active": len(piv),
              "coverage": coverage, "n_calendar_days": n_days, "attack_share": att_share,
              "active_share_per_attack_label": active,
              "peak_flows_per_bin": {l: int(piv[l].max()) for l in order}}, f"{c1} {c2}")


def fig_box(rec, ds_title, B, feats):
    boxf = [f for f in BOX_FEATURES if f in feats]
    groups = [g for g in GROUP_ORDER if (B["group"] == g).any()]
    ncol = 4
    nrow = int(np.ceil(len(boxf) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.3 * nrow + 0.6), layout="constrained")
    med = {}
    for ax, f in zip(np.ravel(axes), boxf):
        data = [log1p_pos(B.loc[B["group"] == g, f].to_numpy()) for g in groups]
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.65,
                        medianprops={"color": "black", "lw": 1.4})
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(CLASS_COLORS[g])
            patch.set_alpha(0.9)
        ax.set_xticks([])
        ax.set_xlabel("Lớp (màu theo chú thích)")
        ax.set_ylabel(f"log(1 + x), x tính bằng {UNITS.get(f, 'đơn vị gốc')}")
        ax.set_title(f)
        med[f] = {g: float(np.median(d)) for g, d in zip(groups, data)}
    for ax in np.ravel(axes)[len(boxf):]:
        ax.set_axis_off()
    fig.legend([Patch(fc=CLASS_COLORS[g]) for g in groups], groups, loc="outside lower center", ncol=len(groups),
               title="Lớp (hộp = phân vị 25–75, râu = 5–95, vạch đen = median)")
    spread = {f: max(m.values()) - min(m.values()) for f, m in med.items()}
    top = max(spread, key=spread.get)
    hi = max(med[top], key=med[top].get)
    lo = min(med[top], key=med[top].get)
    n_sep = sum(1 for s in spread.values() if s > 1.0)
    c1 = (f"Đặc trưng {top} tách các lớp rõ nhất: median là {vn(np.expm1(med[top][hi]), 1)} {UNITS.get(top, '')} ở lớp {hi} "
          f"so với {vn(np.expm1(med[top][lo]), 1)} {UNITS.get(top, '')} ở lớp {lo}.")
    c2 = (f"{n_sep}/{len(boxf)} đặc trưng có median giữa các lớp lệch hơn 1 đơn vị log (khoảng 2,7 lần), "
          f"nên các thống kê thô đã mang tín hiệu phân biệt lớp (mẫu cân bằng theo nhãn gốc).")
    rec.save(fig, "boxplot_log_features", f"{ds_title}: log(1 + x) của các đặc trưng chính theo lớp", "flow",
             {"features": boxf, "classes": groups, "median_log1p_by_class": med, "spread_of_medians": spread,
              "sample_rows": int(len(B)), "sample_per_group": {g: int((B['group'] == g).sum()) for g in groups}},
             f"{c1} {c2}")


def fig_corr(rec, ds_title, U, feats):
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(U), size=min(CORR_ROWS, len(U)), replace=False)
    df = pd.DataFrame(log1p_pos(U[idx]), columns=feats)
    const = [c for c in feats if df[c].nunique() <= 1]
    df = df.drop(columns=const)
    corr = df.corr(method="spearman").fillna(0.0)
    d = 1 - np.abs(corr.to_numpy())
    np.fill_diagonal(d, 0)
    order = leaves_list(linkage(squareform(np.clip((d + d.T) / 2, 0, None), checks=False), "average"))
    corr = corr.iloc[order, order]
    names = list(corr.columns)
    fig, ax = plt.subplots(figsize=(12.5, 10.5), layout="constrained")
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("Đặc trưng (xếp theo cụm tương quan)")
    ax.set_ylabel("Đặc trưng (xếp theo cụm tương quan)")
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label("Hệ số tương quan Spearman (không đơn vị)")
    c = corr.to_numpy()
    iu = np.triu_indices(len(names), 1)
    pairs = sorted(zip(np.abs(c[iu]), c[iu], iu[0], iu[1]), reverse=True)
    high = [p for p in pairs if p[0] >= 0.95]
    top_pairs = [{"a": names[i], "b": names[j], "rho": float(r)} for _, r, i, j in pairs[:5]]
    c1 = (f"{len(high)}/{len(pairs)} cặp đặc trưng có |ρ| ≥ 0,95 (cao nhất: {top_pairs[0]['a']} và {top_pairs[0]['b']}, "
          f"ρ = {vn(top_pairs[0]['rho'], 3)}).")
    c2 = (f"Trung bình |ρ| = {vn(float(np.abs(c[iu]).mean()), 2)}; {len(const)} đặc trưng không đổi trong mẫu bị bỏ khỏi hình, "
          f"nên nhiều đặc trưng trùng thông tin và có thể gộp bớt." if const else
          f"Trung bình |ρ| = {vn(float(np.abs(c[iu]).mean()), 2)}, nhiều đặc trưng trùng thông tin nên có thể gộp bớt.")
    rec.save(fig, "corr_heatmap", f"{ds_title}: tương quan Spearman giữa các đặc trưng số (log(1 + x))", "flow",
             {"n_features": len(names), "sample_rows": int(len(df)), "constant_features_dropped": const,
              "pairs_abs_rho_ge_0.95": len(high), "n_pairs": len(pairs), "top_pairs": top_pairs,
              "mean_abs_rho": float(np.abs(c[iu]).mean())}, f"{c1} {c2}")


def fig_mi(rec, ds_title, B, feats):
    from sklearn.feature_selection import mutual_info_classif
    parts = [B[B["attack"] == a].sample(min(int((B["attack"] == a).sum()), MI_CAP), random_state=SEED)
             for a in B["attack"].unique()]
    sub = pd.concat(parts)
    groups = [g for g in GROUP_ORDER if (sub["group"] == g).any()]
    y = sub["group"].map({g: i for i, g in enumerate(groups)}).to_numpy()
    X = log1p_pos(sub[feats].to_numpy())
    mi = mutual_info_classif(X, y, discrete_features=False, n_neighbors=3, random_state=SEED)
    s = pd.Series(mi, index=feats).sort_values(ascending=False)
    show = s.head(25)[::-1]
    colors = ["#D55E00" if f in s.index[:3] else "#7F9FBF" for f in show.index]
    fig, ax = plt.subplots(figsize=(9.5, 8), layout="constrained")
    ax.barh(range(len(show)), show.to_numpy(), color=colors)
    ax.set_yticks(range(len(show)), show.index)
    for i, v in enumerate(show.to_numpy()):
        ax.text(v + s.max() * 0.01, i, vn(v, 3), va="center", fontsize=8)
    ax.set_xlim(0, s.max() * 1.12)
    ax.set_xlabel("Mutual information với nhãn 5 lớp (nat)")
    ax.set_ylabel("Đặc trưng (25 đặc trưng mạnh nhất; 3 cột cam = mạnh nhất)")
    top3 = [{"feature": f, "mi": float(s[f])} for f in s.index[:3]]
    weak = int((s < 0.01).sum())
    id_like = [t["feature"] for t in top3 if "PORT" in t["feature"] or "PROTO" in t["feature"]]
    c1 = "Ba đặc trưng mạnh nhất: " + ", ".join(f"{t['feature']} (MI = {vn(t['mi'], 3)} nat)" for t in top3) + "."
    c2 = (f"{weak}/{len(s)} đặc trưng có MI < 0,01 nat" +
          (f"; {', '.join(id_like)} mã hóa dịch vụ bị nhắm tới nên có thể phản ánh cấu hình testbed hơn là hành vi." if id_like else "."))
    rec.save(fig, "mutual_information", f"{ds_title}: mutual information của đặc trưng flow với nhãn", "flow",
             {"top3": top3, "mi_all": {k: float(v) for k, v in s.items()}, "n_weak_below_0.01": weak,
              "sample_rows": int(len(sub)), "classes": groups}, f"{c1} {c2}")


def run_flow(ds, csv, rec, outdir, smoke):
    title = DATASET_TITLES[ds]
    t0 = time.time()
    header = read_header(csv)
    feats = [c for c in header if c not in NON_FEATURES]
    mb = 3 if smoke else None
    print(f"[{ds}] {len(feats)} đặc trưng số; lượt 1 (nhãn, thời gian)")
    raw_counts, per_min, n1 = scan_labels_time(csv, mb)
    print(f"[{ds}] lượt 1 xong: {vn(n1)} dòng, {time.time() - t0:.0f} s; lượt 2 (đặc trưng, lấy mẫu)")
    stats, U, B, n2 = scan_features(csv, feats, raw_counts, mb)
    assert n1 == n2, f"hai lượt đọc khác số dòng: {n1} và {n2}"
    print(f"[{ds}] lượt 2 xong: {time.time() - t0:.0f} s; mẫu đều {len(U):,}, mẫu cân bằng {len(B):,}")
    grp, total = fig_labels(rec, title, raw_counts)
    fig_time(rec, title, per_min, raw_counts)
    fig_box(rec, title, B, feats)
    fig_corr(rec, title, U, feats)
    fig_mi(rec, title, B, feats)
    stats.to_csv(Path(outdir) / f"gd2_{ds}_flow_stats.csv", index=False, float_format="%.6g")
    n_const = int((stats["min"] == stats["max"]).sum())
    n_miss = int((stats["n_missing"] > 0).sum())
    n_neg = int((stats["n_negative"] > 0).sum())
    comment = (f"{len(feats)} đặc trưng số: {n_miss} đặc trưng có giá trị thiếu (tối đa {vn(stats['pct_missing'].max(), 3)} %), "
               f"{n_const} đặc trưng không đổi, {n_neg} đặc trưng có giá trị âm. Median lấy từ mẫu đều "
               f"{vn(len(U))} flow, min, max và % thiếu tính trên toàn bộ {vn(total)} flow.")
    table = {"title": f"{title}: thống kê đặc trưng số (min, median, max, % thiếu)", "file": f"gd2_{ds}_flow_stats.csv",
             "n_features": len(feats), "n_with_missing": n_miss, "n_constant": n_const, "n_with_negative": n_neg,
             "median_source": f"mẫu đều {len(U)} flow", "rows": stats.to_dict(orient="records"), "comment": comment}
    return {"n_flows": total, "n_features": len(feats), "raw_counts": raw_counts, "group_counts": grp,
            "flow_stats_table": table, "seconds": round(time.time() - t0, 1)}


# ================================================================ MỨC ĐỒ THỊ
def homophily_from_mix(M, node_cnt):
    """Edge homophily, class-insensitive homophily (Lim và cs., 2021) và adjusted homophily (Platonov và cs., 2023).

    M[a, b] = số cạnh có hướng từ node lớp a tới node lớp b; node_cnt[k] = số node lớp k. Cạnh được xem là
    vô hướng cho các độ đo theo lớp (đúng với cách GNN truyền thông điệp hai chiều, models.py `bidirectional`).
    """
    M = np.asarray(M, dtype="float64")
    node_cnt = np.asarray(node_cnt, dtype="float64")
    tot = M.sum()
    S = M + M.T
    deg = S.sum(1)
    h_edge = float(np.trace(M) / tot) if tot else float("nan")
    h_k = np.where(deg > 0, np.diag(S) / np.maximum(deg, 1), np.nan)
    frac = node_cnt / node_cnt.sum()
    ok = (node_cnt > 0) & (deg > 0)
    n_cls = int((node_cnt > 0).sum())
    h_ci = float(np.maximum(h_k[ok] - frac[ok], 0).sum() / (n_cls - 1)) if n_cls > 1 else float("nan")
    pbar = deg / deg.sum() if deg.sum() else deg
    h_adj = float((h_edge - (pbar ** 2).sum()) / (1 - (pbar ** 2).sum())) if (pbar ** 2).sum() < 1 else float("nan")
    return {"edge_homophily": h_edge, "class_insensitive_homophily": h_ci, "adjusted_homophily": h_adj,
            "random_edge_homophily": float((frac ** 2).sum()), "class_wise_homophily": h_k, "node_fraction": frac}


def collect_graph_arrays(parts, n_classes):
    """Duyệt mọi đồ thị: kích thước, bậc vào/ra, ma trận trộn lớp giữa hai đầu cạnh."""
    graphs, part_id = [], []
    for pi, p in enumerate(PARTS):
        graphs += list(parts[p])
        part_id += [pi] * len(parts[p])
    G, C = len(graphs), n_classes
    n_nodes, n_edges = np.zeros(G, np.int64), np.zeros(G, np.int64)
    cls_cnt, max_in, max_out = (np.zeros((G, C), np.int64) for _ in range(3))
    mix = np.zeros((len(PARTS), C, C), np.int64)
    ys, ins, outs, nparts = [], [], [], []
    for gi, g in enumerate(graphs):
        n = g.num_nodes
        ei, y = g.edge_index.numpy(), g.y.numpy()
        o, i = np.bincount(ei[0], minlength=n), np.bincount(ei[1], minlength=n)
        n_nodes[gi], n_edges[gi] = n, ei.shape[1]
        cls_cnt[gi] = np.bincount(y, minlength=C)
        for k in np.flatnonzero(cls_cnt[gi]):
            m = y == k
            max_in[gi, k], max_out[gi, k] = i[m].max(), o[m].max()
        mix[part_id[gi]] += np.bincount(y[ei[0]] * C + y[ei[1]], minlength=C * C).reshape(C, C)
        ys.append(y), ins.append(i), outs.append(o), nparts.append(np.full(n, part_id[gi], np.int8))
    return {"graphs": graphs, "part": np.array(part_id), "n_nodes": n_nodes, "n_edges": n_edges, "cls_cnt": cls_cnt,
            "max_in": max_in, "max_out": max_out, "mix": mix, "y": np.concatenate(ys), "in_deg": np.concatenate(ins),
            "out_deg": np.concatenate(outs), "node_part": np.concatenate(nparts)}


def fig_graph_size(rec, ds_title, A):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    numbers = {"n_graphs": int(len(A["n_nodes"]))}
    comment_bits = {}
    for ax, key, name in ((axes[0], "n_nodes", "node"), (axes[1], "n_edges", "cạnh")):
        v = A[key]
        bins = np.logspace(0, np.log10(max(v.max(), 2) * 1.05), 45)
        ax.hist(v, bins=bins, color="#56789C", edgecolor="white", lw=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        med, p95, p99 = np.median(v), np.percentile(v, 95), np.percentile(v, 99)
        for val, ls, lab in ((med, "-", "median"), (p95, "--", "p95"), (p99, ":", "p99")):
            ax.axvline(val, color="#D55E00", ls=ls, lw=1.3, label=f"{lab} = {vn(val)}")
        ax.legend(title=f"Max = {vn(v.max())}")
        ax.set_xlabel(f"Số {name} mỗi cửa sổ 5 s (thang log)")
        ax.set_ylabel("Số cửa sổ (thang log)")
        ax.set_title(f"({'a' if key == 'n_nodes' else 'b'}) Số {name} mỗi cửa sổ")
        numbers[key] = {"mean": float(v.mean()), "median": float(med), "p95": float(p95), "p99": float(p99),
                        "max": int(v.max()), "min": int(v.min())}
        comment_bits[key] = (med, p99, v.max())
    share_small = float((A["n_nodes"] <= MAX_NODES_DRAW).mean())
    numbers["share_windows_le_300_nodes"] = share_small
    numbers["edges_per_node_mean"] = float(A["n_edges"].sum() / A["n_nodes"].sum())
    mn, me = comment_bits["n_nodes"], comment_bits["n_edges"]
    c1 = (f"Đồ thị rất nhỏ: median {vn(mn[0])} node và {vn(me[0])} cạnh mỗi cửa sổ, p99 là {vn(mn[1])} node và {vn(me[1])} cạnh, "
          f"lớn nhất {vn(mn[2])} node.")
    c2 = (f"{pct(share_small, 2)} cửa sổ có ≤ {MAX_NODES_DRAW} node nên vẽ được nguyên cửa sổ; mỗi node có trung bình "
          f"{vn(numbers['edges_per_node_mean'], 2)} cạnh, đủ nhỏ để GNN nhẹ 2 lớp chạy theo thời gian thực.")
    rec.save(fig, "graph_size", f"{ds_title}: phân bố số node và số cạnh mỗi cửa sổ", "graph", numbers, f"{c1} {c2}")


def _ccdf(d):
    s = np.sort(d)
    xs = np.unique(s[s > 0])
    return xs, 1 - np.searchsorted(s, xs, "left") / len(s)


def fig_degree(rec, ds_title, A, classes):
    y = A["y"]
    present = [k for k in range(len(classes)) if (y == k).any()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    stats = {"in": {}, "out": {}}
    for ax, key, deg, name in ((axes[0], "in", A["in_deg"], "vào (số IP gửi tới node)"),
                               (axes[1], "out", A["out_deg"], "ra (số IP node gửi tới)")):
        for k in present:
            d = deg[y == k]
            xs, ys = _ccdf(d)
            ax.plot(xs, ys, color=CLASS_COLORS[classes[k]], lw=1.8, label=f"{classes[k]} (n = {vn(len(d))})")
            stats[key][classes[k]] = describe(d)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel(f"Bậc {name} k (thang log)")
        ax.set_ylabel("Tỉ lệ node có bậc ≥ k (thang log)")
        ax.set_title(f"({'a' if key == 'in' else 'b'}) Bậc {'vào' if key == 'in' else 'ra'} theo lớp")
        stats[key + "_share_ge5"] = stats.get(key + "_share_ge5", {})
        stats[key + "_share_ge5"].update({classes[k]: float((deg[y == k] >= 5).mean()) for k in present})
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, title="Lớp của node")

    def cmp(key, att):
        if att not in stats[key] or "Normal" not in stats[key]:
            return None
        a, b = stats[key][att], stats[key]["Normal"]
        return {"mean_attack": a["mean"], "mean_normal": b["mean"], "ratio_mean": a["mean"] / max(b["mean"], 1e-9),
                "median_attack": a["median"], "median_normal": b["median"], "p99_attack": a["p99"],
                "p99_normal": b["p99"], "max_attack": a["max"], "max_normal": b["max"]}

    fan_in, fan_out = cmp("in", "DoS_DDoS"), cmp("out", "Truy_cap_trai_phep")

    def sentence(c, att, what, label, key):
        if c is None:
            return f"Không có lớp {att} để so với Normal."
        r = c["ratio_mean"]
        s5 = stats[key + "_share_ge5"]
        verdict = (f"cao gấp {vn(r, 1)} lần, có dấu hiệu {label}" if r >= 1.5 else
                   f"chỉ bằng {vn(r, 2)} lần, không có dấu hiệu {label} ở mức node" if r <= 0.67 else
                   f"xấp xỉ nhau (gấp {vn(r, 2)} lần), bậc này không phân biệt được lớp")
        return (f"Bậc {what} trung bình của {att} là {vn(c['mean_attack'], 2)} (lớn nhất {vn(c['max_attack'])}; "
                f"{pct(s5[att], 2)} node có bậc ≥ 5) so với {vn(c['mean_normal'], 2)} của Normal ({pct(s5['Normal'], 2)} node có bậc ≥ 5): {verdict}.")

    comment = (f"{sentence(fan_in, 'DoS_DDoS', 'vào', 'fan-in', 'in')} {sentence(fan_out, 'Truy_cap_trai_phep', 'ra', 'fan-out', 'out')}")
    rec.save(fig, "degree_by_class", f"{ds_title}: bậc vào và bậc ra của node theo lớp", "graph",
             {"in_degree_by_class": stats["in"], "out_degree_by_class": stats["out"],
              "share_in_degree_ge5": stats["in_share_ge5"], "share_out_degree_ge5": stats["out_share_ge5"],
              "fan_in_ddos_vs_normal": fan_in, "fan_out_scan_vs_normal": fan_out,
              "degree_definition": "số cạnh (cặp src→dst) trong cửa sổ 5 s, mỗi cặp tính một lần"}, comment)


def fig_homophily(rec, ds_title, A, classes):
    C = len(classes)
    node_cnt = np.bincount(A["y"], minlength=C)
    mix_all = A["mix"].sum(0)
    h = homophily_from_mix(mix_all, node_cnt)
    per_split = {}
    for pi, p in enumerate(PARTS):
        hp = homophily_from_mix(A["mix"][pi], np.bincount(A["y"][A["node_part"] == pi], minlength=C))
        per_split[p] = {k: hp[k] for k in ("edge_homophily", "class_insensitive_homophily", "adjusted_homophily")}
    present = [k for k in range(C) if node_cnt[k] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained", gridspec_kw={"width_ratios": [1.1, 1]})
    sub = mix_all[np.ix_(present, present)].astype(float)
    rown = sub / np.maximum(sub.sum(1, keepdims=True), 1)
    ax = axes[0]
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    names = [CLASS_WRAP[classes[k]] for k in present]
    ax.set_xticks(range(len(present)), names)
    ax.set_yticks(range(len(present)), names)
    for i in range(len(present)):
        for j in range(len(present)):
            ax.text(j, i, f"{vn(100 * rown[i, j], 1)} %\n({vn(sub[i, j])})", ha="center", va="center", fontsize=8,
                    color="white" if rown[i, j] > 0.6 else "black")
    ax.set_xlabel("Lớp của node đích (dst)")
    ax.set_ylabel("Lớp của node nguồn (src)")
    ax.set_title("(a) Cạnh nối các lớp (hàng = 100 %)")
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Tỉ lệ cạnh theo hàng (không đơn vị)")
    ax = axes[1]
    xs = np.arange(len(present))
    ax.bar(xs - 0.2, [h["class_wise_homophily"][k] for k in present], 0.4, label="h_k: tỉ lệ đầu cạnh cùng lớp",
           color=[CLASS_COLORS[classes[k]] for k in present])
    ax.bar(xs + 0.2, [h["node_fraction"][k] for k in present], 0.4, label="n_k / n: tỉ lệ node của lớp (mức ngẫu nhiên)",
           color="#BBBBBB")
    ax.set_xticks(xs, names)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("Lớp của node")
    ax.set_ylabel("Tỉ lệ (không đơn vị)")
    ax.set_title(f"(b) Homophily theo lớp so với mức ngẫu nhiên\nEdge = {vn(h['edge_homophily'], 3)}; "
                 f"class-insensitive = {vn(h['class_insensitive_homophily'], 3)}; adjusted = {vn(h['adjusted_homophily'], 3)}")
    ax.legend(loc="upper center", fontsize=8, ncol=1)
    he, hci, hr = h["edge_homophily"], h["class_insensitive_homophily"], h["random_edge_homophily"]
    level = "cao" if hci >= 0.7 else "khá cao" if hci >= 0.5 else "trung bình" if hci >= 0.2 else "thấp"
    if hci >= 0.5:
        meaning = (f"Phần lớn cạnh nối các node cùng lớp (cao hơn rõ rệt mức ngẫu nhiên), nên gom thông điệp từ láng giềng "
                   f"có thể giúp GNN làm mượt nhãn; tuy vậy {pct(1 - he)} cạnh vẫn nối hai lớp khác nhau (chủ yếu tấn công ↔ Normal), "
                   f"nên láng giềng không thay thế được đặc trưng riêng của từng node.")
    elif hci >= 0.2:
        meaning = ("Một phần đáng kể cạnh nối hai lớp khác nhau, nên gom láng giềng chỉ giúp vừa phải; GNN cần attention và "
                   "đặc trưng cạnh để lọc láng giềng, và có thể chỉ hơn baseline một chút.")
    else:
        meaning = ("Cạnh gần như không ưu tiên nối node cùng lớp (kiểu tấn công ↔ nạn nhân), nên trung bình hóa láng giềng dễ "
                   "trộn lẫn lớp; GNN khó vượt baseline chỉ dùng đặc trưng node, và hybrid GAT + LightGBM (giữ đặc trưng thô "
                   "của chính node) có lý do tồn tại.")
    comment = (f"Edge homophily = {vn(he, 3)} (mức ngẫu nhiên {vn(hr, 3)}) và class-insensitive homophily = {vn(hci, 3)}, "
               f"tức mức {level}. {meaning}")
    rec.save(fig, "homophily", f"{ds_title}: edge homophily và class-insensitive homophily", "graph",
             {"edge_homophily": he, "class_insensitive_homophily": hci, "adjusted_homophily": h["adjusted_homophily"],
              "random_edge_homophily": hr, "level_by_class_insensitive": level,
              "class_wise_homophily": {classes[k]: float(h["class_wise_homophily"][k]) for k in present},
              "node_fraction": {classes[k]: float(h["node_fraction"][k]) for k in present},
              "mixing_matrix_counts": {classes[a]: {classes[b]: int(mix_all[a, b]) for b in present} for a in present},
              "per_split": per_split, "n_edges": int(mix_all.sum()), "n_nodes": int(node_cnt.sum()),
              "direction": "edge_homophily trên cạnh có hướng; h_k theo cạnh vô hướng (M + M^T)"}, comment)
    return {"edge_homophily": he, "class_insensitive_homophily": hci, "adjusted_homophily": h["adjusted_homophily"],
            "random_edge_homophily": hr, "level": level, "meaning_for_gnn": meaning, "per_split": per_split,
            "class_wise_homophily": {classes[k]: float(h["class_wise_homophily"][k]) for k in present}}


# ---------------------------------------------------------------- vẽ cửa sổ mẫu
def pick_windows(A, classes):
    """Chọn cửa sổ mẫu trong tập train, ≤ MAX_NODES_DRAW node: (khóa, chỉ số, tiêu chí, hub_mode, thứ hạng)."""
    cid = {c: i for i, c in enumerate(classes)}
    n, cnt = A["n_nodes"], A["cls_cnt"]
    train = A["part"] == 0
    small = train & (n <= MAX_NODES_DRAW)
    out = []

    def best(mask, score, tiebreak):
        idx = np.flatnonzero(mask)
        order = np.lexsort((idx, -tiebreak[idx], -score[idx]))
        return int(idx[order[0]])

    pure = small & (cnt[:, 0] == n)
    if pure.any():
        target = np.percentile(n[pure], 50)
        i = best(pure, -np.abs(n - target), A["n_edges"])
        out.append(("normal", i, f"cửa sổ chỉ có node Normal, kích thước gần median ({vn(target)} node) của các cửa sổ Normal", "total", None))
    for key, cname, sc_arr, mode, text in (
            ("ddos", "DoS_DDoS", "max_in", "in", "bậc vào lớn nhất của node DoS_DDoS"),
            ("scan", "Truy_cap_trai_phep", "max_out", "out", "bậc ra lớn nhất của node Truy_cap_trai_phep")):
        if cname not in cid:
            continue
        k = cid[cname]
        maj = train & (cnt[:, k] * 2 >= n)
        sel = maj & (n <= MAX_NODES_DRAW)
        if not sel.any():
            continue
        score = A[sc_arr][:, k]
        i = best(sel, score, n)
        rank = float((score[maj] < score[i]).mean())
        out.append((key, i, f"cửa sổ ≤ {MAX_NODES_DRAW} node, ≥ 50 % node {cname}, có {text} (cao hơn {pct(rank, 1)} cửa sổ {cname} khác trong train)", mode, rank))
    if "Botnet_Backdoor" in cid:
        k = cid["Botnet_Backdoor"]
        maj = train & (cnt[:, k] * 2 >= n)
        sel = maj & (n <= MAX_NODES_DRAW)
        if sel.any():
            i = best(sel, cnt[:, k], A["n_edges"])
            out.append(("backdoor", i, f"cửa sổ ≤ {MAX_NODES_DRAW} node, ≥ 50 % node Botnet_Backdoor, nhiều node Botnet_Backdoor nhất", "total", None))
    return out


def component_layout(G):
    """Layout spring (seed cố định) cho từng thành phần liên thông; thành phần lớn nhất ở giữa, các thành phần còn lại
    xếp quanh trên một vòng, nhỏ hơn theo √số node. Tránh để spring_layout đẩy các thành phần rời ra xa làm cụm chính co nhỏ."""
    comps = sorted(nx.connected_components(G), key=lambda c: (-len(c), min(c)))
    n = G.number_of_nodes()
    pos = {}
    scales = [max(np.sqrt(len(c) / n), 0.12) for c in comps]
    ring = scales[0] + 0.4 + (max(scales[1:]) if len(comps) > 1 else 0.0)
    for ci, comp in enumerate(comps):
        sub = G.subgraph(sorted(comp))
        p = nx.spring_layout(sub, seed=SEED, k=1.6 / np.sqrt(max(len(comp), 2)), iterations=200) if len(comp) > 1 \
            else {next(iter(comp)): np.zeros(2)}
        s = scales[ci]
        if ci == 0:
            center = np.zeros(2)
        else:
            ang = 2 * np.pi * (ci - 1) / max(len(comps) - 1, 1)
            center = ring * np.array([np.cos(ang), np.sin(ang)])
        for v, xy in p.items():
            pos[v] = center + s * np.asarray(xy)
    return pos


def draw_window(ax, g, classes, hub_mode, title):
    n = g.num_nodes
    ei, y = g.edge_index.numpy(), g.y.numpy()
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    pos = component_layout(G.to_undirected())
    o, i = np.bincount(ei[0], minlength=n), np.bincount(ei[1], minlength=n)
    deg = o + i
    sizes = 45 + 60 * np.sqrt(deg)
    colors = [CLASS_COLORS[classes[c]] for c in y]
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.7, alpha=0.55, edge_color="#666666", arrows=True, arrowsize=7,
                           arrowstyle="-|>", node_size=sizes)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors, edgecolors="white", linewidths=0.7)
    ax.set_axis_off()
    ax.margins(0.12)
    ax.set_title(title, fontsize=10)
    if hub_mode == "in":
        h = int(np.argmax(np.where(y == (list(classes).index("DoS_DDoS")), i, -1)))
        note = f"fan-in: {int(i[h])} IP cùng gửi tới 1 IP"
    elif hub_mode == "out":
        h = int(np.argmax(np.where(y == (list(classes).index("Truy_cap_trai_phep")), o, -1)))
        note = f"fan-out: 1 IP gửi tới {int(o[h])} IP"
    else:
        h = int(np.argmax(deg))
        note = f"node bậc lớn nhất: {int(deg[h])}"
    # đặt chú thích ở góc xa node trung tâm nhất để mũi tên ngắn và không che đồ thị
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    hx, hy = (pos[h][0] - x0) / (x1 - x0), (pos[h][1] - y0) / (y1 - y0)
    corner = min(((0.02, 0.98), (0.98, 0.98), (0.02, 0.02), (0.98, 0.02)), key=lambda c: (c[0] - hx) ** 2 + (c[1] - hy) ** 2)
    ax.annotate(note, xy=pos[h], xycoords="data", xytext=corner, textcoords="axes fraction",
                va="top" if corner[1] > 0.5 else "bottom", ha="left" if corner[0] < 0.5 else "right",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#888888", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color="#222222", lw=1.3, shrinkA=0, shrinkB=4))
    return {"hub_node": h, "note": note, "in_degree_hub": int(i[h]), "out_degree_hub": int(o[h])}


def _legend(fig, classes, present):
    fig.legend([Line2D([], [], marker="o", ls="", color=CLASS_COLORS[classes[k]], ms=9) for k in present],
               [classes[k] for k in present], loc="outside lower center", ncol=len(present),
               title="Lớp của node (màu). Kích thước node ∝ √(bậc vào + bậc ra); mũi tên = flow src → dst")


KIND_TITLES = {"normal": "bình thường", "ddos": "DDoS/DoS", "scan": "dò quét/truy cập trái phép", "backdoor": "backdoor"}


def fig_windows(rec, ds_title, A, classes):
    picks = pick_windows(A, classes)
    pos_ok = picks
    infos = {}
    for key, gi, why, mode, rank in pos_ok:
        g = A["graphs"][gi]
        fig, ax = plt.subplots(figsize=(7.8, 7.4), layout="constrained")
        ttl = f"{ds_title}: cửa sổ {KIND_TITLES[key]} (win {g.win}, {g.num_nodes} node, {g.num_edges} cạnh)"
        draw_window(ax, g, classes, mode, "")
        present = sorted(set(g.y.tolist()))
        _legend(fig, classes, present)
        cls_cnt = {classes[k]: int((g.y == k).sum()) for k in present}
        t0 = pd.to_datetime(g.t_start, unit="s")
        info = {"win": int(g.win), "t_start_utc": str(t0), "n_nodes": int(g.num_nodes), "n_edges": int(g.num_edges),
                "class_counts": cls_cnt, "criterion": why, "split": "train", "layout_seed": SEED}
        hub = {"in": ("fan-in", "bậc vào"), "out": ("fan-out", "bậc ra")}.get(mode)
        # số liệu hub tính lại từ đồ thị để chú thích khớp hình
        ei, y = g.edge_index.numpy(), g.y.numpy()
        o, i = np.bincount(ei[0], minlength=g.num_nodes), np.bincount(ei[1], minlength=g.num_nodes)
        if mode == "in":
            hub_deg, mean_other = int(i[y == classes.index("DoS_DDoS")].max()), float(np.mean(i))
        elif mode == "out":
            hub_deg, mean_other = int(o[y == classes.index("Truy_cap_trai_phep")].max()), float(np.mean(o))
        else:
            hub_deg, mean_other = int((o + i).max()), float(np.mean(o + i))
        reach = min(1.0, hub_deg / max(g.num_nodes - 1, 1))      # tỉ lệ node còn lại mà node trung tâm nối tới
        info.update({"hub_degree": hub_deg, "mean_degree_in_window": mean_other, "hub_reach": reach})
        if hub:
            ctext = (f"Cửa sổ {KIND_TITLES[key]} ({g.num_nodes} node, {g.num_edges} cạnh) có một node {hub[0]} với {hub[1]} {hub_deg}, "
                     f"trong khi bậc trung bình toàn cửa sổ chỉ {vn(mean_other, 2)}"
                     + (f"; node này nằm trên mức cao hơn {pct(rank, 1)} cửa sổ cùng loại trong train." if rank is not None else "."))
        else:
            ctext = (f"Cửa sổ {KIND_TITLES[key]} gồm {g.num_nodes} node và {g.num_edges} cạnh; node bậc lớn nhất có bậc {hub_deg} "
                     f"(nối tới {pct(reach, 0)} số node còn lại) so với bậc trung bình {vn(mean_other, 2)} của cửa sổ.")
        infos[key] = info
        rec.save(fig, f"window_{key}", ttl, "graph_viz", info, ctext)
    # hình tổng hợp: cùng layout seed, cạnh nhau để so sánh
    ncol = len(pos_ok) if len(pos_ok) <= 3 else 2
    nrow = int(np.ceil(len(pos_ok) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6 * ncol, 6.8 * nrow + 0.5), layout="constrained")
    present = set()
    for ax in np.ravel(axes)[len(pos_ok):]:
        ax.set_axis_off()
    for ax, (key, gi, why, mode, rank) in zip(np.ravel(axes), pos_ok):
        g = A["graphs"][gi]
        draw_window(ax, g, classes, mode, f"{KIND_TITLES[key]}: win {g.win}, {g.num_nodes} node, {g.num_edges} cạnh")
        present |= set(g.y.tolist())
    _legend(fig, classes, sorted(present))
    parts = "; ".join(f"{KIND_TITLES[k][0].upper() + KIND_TITLES[k][1:]}: {v['n_nodes']} node, node trung tâm bậc {v['hub_degree']} (nối {pct(v['hub_reach'], 0)} node còn lại)"
                      for k, v in infos.items())
    rec.save(fig, "windows_overview", f"{ds_title}: so sánh các cửa sổ mẫu (cùng layout seed {SEED})", "graph_viz",
             {"windows": infos}, f"Cùng seed layout và cùng cách tô màu/kích thước cho mọi cửa sổ. {parts}.")


def run_graph(ds, graph_dir, rec):
    from src.data import load_split
    title = DATASET_TITLES[ds]
    t0 = time.time()
    parts, meta = load_split(graph_dir)
    classes = list(meta["classes"])
    print(f"[{ds}] đã nạp đồ thị: " + ", ".join(f"{p} {len(parts[p])}" for p in PARTS) + f"; lớp: {classes}")
    A = collect_graph_arrays(parts, len(classes))
    print(f"[{ds}] đã duyệt {len(A['graphs']):,} đồ thị trong {time.time() - t0:.0f} s")
    fig_graph_size(rec, title, A)
    fig_degree(rec, title, A, classes)
    hom = fig_homophily(rec, title, A, classes)
    fig_windows(rec, title, A, classes)
    summ = json.loads((Path(graph_dir) / "summary.json").read_text(encoding="utf-8"))
    return {"classes": classes, "n_graphs": len(A["graphs"]), "homophily": hom,
            "pipeline": {"flows_raw": summ["clean_log"][0]["so_dong"], "flows_clean": summ["clean_log"][-1]["so_dong"],
                         "n_graphs": summ["n_graphs"], "mean_nodes": float(A["n_nodes"].mean()),
                         "mean_edges": float(A["n_edges"].mean()), "window_sec": meta["window_sec"]},
            "seconds": round(time.time() - t0, 1)}


# ================================================================ SƠ ĐỒ QUY TRÌNH
def fig_workflow(outdir, pipes):
    fig, ax = plt.subplots(figsize=(15, 6.6), layout="constrained")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.set_axis_off()
    blue, red, green = CLASS_COLORS["Normal"], CLASS_COLORS["DoS_DDoS"], CLASS_COLORS["Truy_cap_trai_phep"]

    def box(x, w, head):
        ax.add_patch(FancyBboxPatch((x, 9), w, 33, boxstyle="round,pad=0.4,rounding_size=1.2", fc="#F6F8FA", ec="#555555", lw=1.2))
        ax.text(x + w / 2, 40, head, ha="center", va="center", fontsize=11.5, fontweight="bold")

    def arrow(x0, x1, label):
        ax.add_patch(FancyArrowPatch((x0, 25), (x1, 25), arrowstyle="simple,head_width=14,head_length=12,tail_width=6",
                                     fc="#555555", ec="none"))
        ax.text((x0 + x1) / 2, 31, label, ha="center", va="bottom", fontsize=8.5)

    box(1, 30, "① Flow NetFlow v3")
    box(37, 30, "② Cửa sổ 5 giây (W = S)")
    box(73, 26, "③ Đồ thị mỗi cửa sổ")
    arrow(31.7, 36.3, "gán theo\nthời điểm")
    arrow(67.7, 72.3, "gộp theo\n(cửa sổ, IP)")
    flows = [(0.4, "A", "V", "Normal", blue), (1.7, "B", "V", "DDoS", red), (2.9, "C", "V", "DDoS", red),
             (5.6, "A", "V", "Normal", blue), (7.3, "B", "V", "DDoS", red), (8.8, "B", "C", "DDoS", red),
             (11.2, "A", "V", "Normal", blue), (13.6, "C", "A", "Normal", blue)]
    ax.text(2.2, 36.3, "t (s)  nguồn → đích   nhãn", fontsize=8.5, family="monospace", color="#333333")
    for r, (t, s, d, lab, col) in enumerate(flows):
        yy = 33.6 - r * 2.7
        ax.plot([2.6], [yy], "o", color=col, ms=6)
        ax.text(3.6, yy, f"{t:4.1f}    {s} → {d}      {lab}", fontsize=8.5, family="monospace", va="center")
    ax.text(16, 10.9, "(ví dụ minh họa; mỗi dòng còn có\n~49 đặc trưng: byte, gói, cờ TCP, ...)", fontsize=7.5, ha="center",
            va="bottom", color="#555555")
    # trục thời gian
    x0, x1 = 39.5, 64.5
    ax.plot([x0, x1], [20, 20], color="#333333", lw=1.5)
    tx = lambda t: x0 + (x1 - x0) * t / 15.0
    for k, (a, b) in enumerate(((0, 5), (5, 10), (10, 15))):
        ax.add_patch(plt.Rectangle((tx(a), 20.7), tx(b) - tx(a), 14.5, fc=("#DDE8F3" if k % 2 == 0 else "#EEF3F8"), ec="#7A8DA3", lw=1))
        ax.text((tx(a) + tx(b)) / 2, 36.1, f"cửa sổ {k}", ha="center", fontsize=9)
        ax.text(tx(a), 18.2, f"{a} s", ha="center", fontsize=8.5)
    ax.text(tx(15), 18.2, "15 s", ha="center", fontsize=8.5)
    ax.text((x0 + x1) / 2, 16.2, "thời gian →", ha="center", fontsize=8.5, color="#333333")
    for r, (t, s, d, lab, col) in enumerate(flows):
        w = int(t // 5)
        row = sum(1 for q in flows[:r] if int(q[0] // 5) == w)
        ax.plot([tx(t)], [33.0 - row * 3.5], "o", color=col, ms=8)
        ax.text(tx(t), 33.0 - row * 3.5 - 1.5, f"{s}→{d}", ha="center", fontsize=7.5)
    ax.text((x0 + x1) / 2, 11.0, "mỗi flow vào đúng một cửa sổ; chia theo khối thời gian,\nchừa 1 cửa sổ trống giữa train / val / test",
            ha="center", va="bottom", fontsize=8, color="#555555")
    # đồ thị mẫu của cửa sổ 0
    nodes = {"A": (79, 24), "B": (87, 33), "C": (93, 26), "V": (86, 17.5)}
    ncol = {"A": blue, "B": red, "C": red, "V": red}
    for s, d in (("A", "V"), ("B", "V"), ("C", "V")):
        ax.add_patch(FancyArrowPatch(nodes[s], nodes[d], arrowstyle="-|>", mutation_scale=12, color="#555555", lw=1.4,
                                     shrinkA=9, shrinkB=11))
    for k, (x, y) in nodes.items():
        ax.plot([x], [y], "o", ms=22 if k == "V" else 15, color=ncol[k], mec="white", mew=1.2)
        ax.text(x, y, k, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(86, 37.0, "cửa sổ 0", ha="center", fontsize=9)
    ax.text(86, 14.8, "node = IP: 18 đặc trưng (số flow, byte,\nsố IP đối tác, tỉ lệ RST, ...)\ncạnh = flow gộp theo src → dst: 3 đặc trưng",
            ha="center", va="top", fontsize=8)
    # chân trang: số liệu thật
    lines = []
    for ds, p in pipes.items():
        lines.append(f"{DATASET_TITLES[ds]}: {vn(p['flows_raw'])} flow gốc, {vn(p['flows_clean'])} flow sau làm sạch  →  "
                     f"{vn(p['n_graphs'])} đồ thị (trung bình {vn(p['mean_nodes'], 1)} node và {vn(p['mean_edges'], 1)} cạnh mỗi cửa sổ)")
    ax.text(50, 5.3, "\n".join(lines), ha="center", va="center", fontsize=9.5)
    ax.text(50, 0.4, "Nhãn node = nhãn chiếm đa số trong các flow node đó khởi tạo; mô hình phân loại từng node vào 5 lớp.  "
                     "Màu: xanh = Normal, cam đỏ = DoS_DDoS.", ha="center", va="center", fontsize=8.5, color="#444444")
    fig.suptitle("Quy trình dựng đồ thị: flow → cửa sổ 5 giây → đồ thị (node = IP, cạnh = flow)")
    f = Path(outdir) / "gd2_chung_quy_trinh.png"
    fig.savefig(f, dpi=DPI)
    plt.close(fig)
    return jc({"id": "chung/quy_trinh", "dataset": "chung", "level": "diagram", "file": f.name,
               "title": "Quy trình dựng đồ thị: flow → cửa sổ 5 giây → đồ thị", "dpi": DPI, "axes": [], "axis_label_problems": [],
               "numbers": pipes,
               "comment": " ".join(lines[:2]) + " Mỗi cửa sổ 5 giây là một đồ thị, node là IP và cạnh là flow."})


# ================================================================ ĐIỂM VÀO
def cmd_run(argv):
    smoke = "--smoke" in argv
    ds, csv, gdir, outdir = [a for a in argv if not a.startswith("--")][:4]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rec = Recorder(out, ds)
    t0 = time.time()
    flow = run_flow(ds, csv, rec, out, smoke)
    graph = run_graph(ds, gdir, rec)
    res = {"dataset": ds, "title": DATASET_TITLES[ds], "smoke": smoke, "flow": flow, "graph": graph,
           "figures": rec.figures, "seconds": round(time.time() - t0, 1)}
    (out / f"eda_summary_{ds}.json").write_text(json.dumps(jc(res), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{ds}] xong {len(rec.figures)} hình trong {time.time() - t0:.0f} s")
    return 0


def cmd_merge(argv):
    out = Path(argv[0])
    dss = argv[1:]
    parts = {ds: json.loads((out / f"eda_summary_{ds}.json").read_text(encoding="utf-8")) for ds in dss}
    wf = fig_workflow(out, {ds: p["graph"]["pipeline"] for ds, p in parts.items()})
    figs = [f for p in parts.values() for f in p["figures"]] + [wf]
    hom = {ds: p["graph"]["homophily"] for ds, p in parts.items()}
    rows = [{"dataset": DATASET_TITLES[ds], "edge_homophily": h["edge_homophily"],
             "class_insensitive_homophily": h["class_insensitive_homophily"], "adjusted_homophily": h["adjusted_homophily"],
             "random_edge_homophily": h["random_edge_homophily"], "level": h["level"]} for ds, h in hom.items()]
    pd.DataFrame(rows).to_csv(out / "gd2_homophily.csv", index=False, float_format="%.6g")
    summary = {"description": "EDA mức flow, mức đồ thị và đồ thị mẫu cho NF-ToN-IoT-v3 và NF-BoT-IoT-v3 (task 2.1–2.3)",
               "sampling": {"uniform_fraction": UNIFORM_FRAC, "balanced_cap_per_raw_label": BALANCED_CAP, "mi_cap_per_raw_label": MI_CAP,
                            "corr_rows": CORR_ROWS, "seed": SEED},
               "n_figures": len(figs), "n_figures_per_dataset": {ds: len(p["figures"]) for ds, p in parts.items()},
               "figures": figs, "homophily": hom,
               "flow_tables": {ds: p["flow"]["flow_stats_table"] for ds, p in parts.items()},
               "datasets": {ds: {"n_flows": p["flow"]["n_flows"], "n_graphs": p["graph"]["n_graphs"], "classes": p["graph"]["classes"],
                                 "smoke": p["smoke"]} for ds, p in parts.items()}}
    (out / "eda_summary.json").write_text(json.dumps(jc(summary), ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [f["id"] for f in figs if f["axis_label_problems"] or not f["comment"]]
    print(f"Đã gộp {len(figs)} hình; hình thiếu nhãn trục hoặc nhận xét: {bad or 'không'}")
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cmd, argv = sys.argv[1], sys.argv[2:]
    return {"run": cmd_run, "merge": cmd_merge}[cmd](argv)


if __name__ == "__main__":
    sys.exit(main())
