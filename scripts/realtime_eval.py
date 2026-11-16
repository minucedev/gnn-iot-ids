"""Đánh giá kết quả chạy của scripts/detector.py (task 8.3, 8.4): độ trễ, khớp online/offline, cảnh báo giả, thời gian phát hiện, RAM, hình.

Chạy:  python -I scripts/realtime_eval.py <thư mục kết quả detector> <thư mục ghi bảng> <thư mục ghi hình> [--episode-gap 60]
Đọc:   windows.csv, detector_summary.json, alerts.csv (trong thư mục kết quả detector)
Ghi:   latency.json, realtime_eval.json, alerts_sample.csv (≤ 1.000 dòng) vào thư mục bảng; gd8_latency_hist.png, gd8_latency_vs_nodes.png vào thư mục hình.

Quy ước:
  * Độ trễ (build_ms + infer_ms = total_ms) tính trên các cửa sổ sau warm-up (cột warmup = 0).
  * Đoạn chỉ có Normal = cửa sổ không có node tấn công nào (nhãn thật của node, từ đồ thị offline); cảnh báo giả = node bị báo tấn công trong các cửa sổ đó.
    Số cảnh báo giả mỗi giờ = số cảnh báo giả / (số cửa sổ Normal-only × độ dài cửa sổ) theo thời gian dữ liệu.
  * Đợt tấn công của lớp k = các cửa sổ có node nhãn k, hai cửa sổ cách nhau không quá `episode-gap` giây thì cùng một đợt. Thời điểm bắt đầu =
    flow đầu tiên nhãn k của cửa sổ đầu đợt; được phát hiện khi cửa sổ có node nhãn k bị báo tấn công. Thời gian phát hiện = thời điểm đóng
    cửa sổ đó - thời điểm bắt đầu + thời gian xử lý của cửa sổ đó.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

P95_LIMIT_MS = 1000.0
AGREE_MIN = 0.999
FEAT_TOL = 1e-6
RAM_TOL = 0.10
FPR_LIMIT, RECALL_MIN = 0.0408, 0.95
MAX_PARAMS, MAX_SIZE_KB = 100_000, 1024
NODE_BINS = (0, 10, 20, 50, 100, 200, 500, 1000, 10**9)


def pct(a):
    a = np.asarray(a, dtype=np.float64)
    if len(a) == 0:
        return {k: None for k in ("mean", "p50", "p95", "p99", "max")}
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def measured(win):
    """Các cửa sổ tính vào thống kê (bỏ warm-up)."""
    return win[win["warmup"] == 0]


def latency_report(win, summary):
    m = measured(win)
    flows, secs = float(m["n_flows"].sum()), float(m["total_ms"].sum()) / 1000
    by_nodes = []
    for lo, hi in zip(NODE_BINS[:-1], NODE_BINS[1:]):
        s = m[(m["n_nodes"] >= lo) & (m["n_nodes"] < hi)]
        if len(s):
            by_nodes.append({"n_nodes_from": lo, "n_nodes_to": hi if hi < 10**9 else None, "n_windows": int(len(s)),
                             "total_ms": pct(s["total_ms"]), "build_ms": pct(s["build_ms"]), "infer_ms": pct(s["infer_ms"])})
    p95 = float(np.percentile(m["total_ms"], 95)) if len(m) else float("nan")
    return {
        "model": summary["model"], "seed": summary["seed"], "dataset": summary["dataset"], "split": summary["split"],
        "n_windows_total": int(len(win)), "n_warmup": int(win["warmup"].sum()), "n_windows_measured": int(len(m)),
        "threads": 1, "definition": "từ lúc đóng cửa sổ (lấy flow ra khỏi bộ gom) đến lúc có dự đoán; total = build + infer",
        "build_ms": pct(m["build_ms"]), "infer_ms": pct(m["infer_ms"]), "total_ms": pct(m["total_ms"]),
        "p95_limit_ms": P95_LIMIT_MS, "p95_ok": bool(p95 <= P95_LIMIT_MS), "n_windows_over_limit": int((m["total_ms"] > P95_LIMIT_MS).sum()),
        "throughput": {"flows_per_s": flows / secs if secs else None, "windows_per_s": len(m) / secs if secs else None,
                       "processed_flows": int(flows), "processing_seconds": secs},
        "frozen_style_p95_ms": summary.get("frozen_style_p95_ms"),
        "frozen_style_note": "p95 của 200 lần suy luận một cửa sổ trung vị (cách đo của GĐ H), không gồm dựng đồ thị",
        "by_nodes": by_nodes, "env": summary.get("env"),
    }


def ram_report(win, early_frac=0.10):
    r = win["rss_mb"].to_numpy(dtype=np.float64)
    n = len(r)
    w = int(win["warmup"].sum())
    k = max(1, int(n * early_frac))
    early = float(np.median(r[w:w + k])) if n > w else float("nan")
    late = float(np.median(r[-k:]))
    slope = float(np.polyfit(np.arange(n - n // 2) / 1000.0, r[n // 2:], 1)[0]) if n >= 20 else float("nan")
    growth = late / early - 1 if early == early and early > 0 else float("nan")
    return {"ram_first_window_mb": float(r[0]), "ram_last_window_mb": float(r[-1]), "ram_max_mb": float(np.nanmax(r)),
            "ram_early_median_mb": early, "ram_late_median_mb": late, "growth_late_vs_early": growth,
            "slope_mb_per_1000_windows_second_half": slope, "tolerance": RAM_TOL, "stable": bool(growth == growth and abs(growth) <= RAM_TOL),
            "definition": f"trung vị RSS {k} cửa sổ ngay sau warm-up so với {k} cửa sổ cuối"}


def agreement_report(win):
    r = win[win["ref_found"] == 1]
    if len(r) == 0:
        return None
    n_nodes = int(r["n_nodes"].sum())
    mis_thr, mis_arg = int(r["n_mis_thr"].sum()), int(r["n_mis_arg"].sum())
    dx, de = float(r["max_dx"].max()), float(r["max_de"].max())
    return {"n_windows_compared": int(len(r)), "n_windows_without_reference": int((win["ref_found"] == 0).sum()), "n_nodes": n_nodes,
            "feature_max_abs_diff_x": dx, "feature_max_abs_diff_edge_attr": de, "feature_tol": FEAT_TOL,
            "windows_structure_mismatch": int((r["struct_ok"] == 0).sum()),
            "pred_agreement_threshold": 1 - mis_thr / n_nodes, "pred_agreement_argmax": 1 - mis_arg / n_nodes,
            "n_node_mismatch_threshold": mis_thr, "n_node_mismatch_argmax": mis_arg,
            "windows_with_mismatch_threshold": int((r["n_mis_thr"] > 0).sum()), "max_abs_diff_proba": float(r["max_dproba"].max()),
            "features_ok": bool(max(dx, de) < FEAT_TOL and (r["struct_ok"] == 1).all()),
            "pred_ok": bool(1 - mis_thr / n_nodes >= AGREE_MIN)}


def false_alarm_report(win, window_sec=5):
    r = win[win["ref_found"] == 1]
    only = r[r["n_true_attack"] == 0]
    hours = len(only) * window_sec / 3600
    fa = int(only["n_alert"].sum())
    return {"n_windows_normal_only": int(len(only)), "hours_normal_only": hours, "n_false_alerts": fa,
            "false_alerts_per_hour": fa / hours if hours > 0 else None,
            "windows_with_false_alert": int((only["n_alert"] > 0).sum()),
            "windows_with_false_alert_per_hour": int((only["n_alert"] > 0).sum()) / hours if hours > 0 else None,
            "node_fpr_in_normal_only_windows": fa / max(1, int(only["n_nodes"].sum())),
            "note": "cảnh báo = node bị báo tấn công; đoạn chỉ có Normal = cửa sổ không có node tấn công nào"}


def episodes(win, k, gap_s=60.0, window_sec=5):
    """Các đợt tấn công của lớp k (mã lớp) và thời gian phát hiện của từng đợt."""
    col_true, col_det, col_first = f"n_true_{k}", f"n_det_{k}", f"first_ts_{k}"
    r = win[(win["ref_found"] == 1) & (win[col_true] > 0)].sort_values("win")
    out = []
    cur = []
    prev_close = None
    for _, row in r.iterrows():
        if prev_close is not None and row["t_start"] - prev_close > gap_s:
            out.append(cur)
            cur = []
        cur.append(row)
        prev_close = row["t_close"]
    if cur:
        out.append(cur)
    res = []
    for ep in out:
        first = ep[0]
        start = float(first[col_first]) if col_first in first and first[col_first] == first[col_first] else float(first["t_start"])
        det = next((w for w in ep if w[col_det] > 0), None)
        res.append({"class_code": k, "start_ts": start, "end_ts": float(ep[-1]["t_close"]), "n_windows": len(ep),
                    "n_true_nodes": int(sum(w[col_true] for w in ep)), "detected": det is not None,
                    "first_alert_window": None if det is None else int(det["win"]),
                    "detection_s": None if det is None else float(det["t_close"] - start + det["total_ms"] / 1000)})
    return res


def detection_report(win, classes, gap_s=60.0, window_sec=5):
    eps = []
    for k in range(1, len(classes)):
        if f"n_true_{k}" in win:
            for e in episodes(win, k, gap_s, window_sec):
                e["class"] = classes[k]
                eps.append(e)
    d = np.array([e["detection_s"] for e in eps if e["detected"]], dtype=np.float64)
    per_class = {}
    for k in range(1, len(classes)):
        es = [e for e in eps if e["class_code"] == k]
        dd = np.array([e["detection_s"] for e in es if e["detected"]], dtype=np.float64)
        per_class[classes[k]] = {"n_episodes": len(es), "n_detected": int(len(dd)), "median_s": float(np.median(dd)) if len(dd) else None,
                                 "max_s": float(dd.max()) if len(dd) else None}
    return {"episode_gap_s": gap_s, "n_episodes": len(eps), "n_detected": int(len(d)), "n_undetected": len(eps) - int(len(d)),
            "detection_median_s": float(np.median(d)) if len(d) else None, "detection_p95_s": float(np.percentile(d, 95)) if len(d) else None,
            "detection_max_s": float(d.max()) if len(d) else None, "per_class": per_class, "episodes": eps}


def node_report(win):
    """FPR và recall tấn công mức node tính từ chính các cảnh báo online (đối chiếu với đo offline của GĐ H)."""
    r = win[win["ref_found"] == 1]
    n_norm, n_att = int(r["n_normal"].sum()), int(r["n_true_attack"].sum())
    return {"n_nodes": int(r["n_nodes"].sum()), "n_normal": n_norm, "n_attack": n_att, "fpr": int(r["n_fp"].sum()) / max(1, n_norm),
            "recall_attack": int(r["n_tp"].sum()) / max(1, n_att)}


def targets_table(lat, ram, agree, fa, det, node, summary):
    t = []

    def add(name, limit, value, ok, note=""):
        t.append({"chi_tieu": name, "nguong": limit, "gia_tri": value, "dat": ok, "ghi_chu": note})

    add("Độ trễ p95 end-to-end ≤ 1 s mỗi cửa sổ", f"≤ {P95_LIMIT_MS:.0f} ms", lat["total_ms"]["p95"], lat["p95_ok"],
        f"{lat['n_windows_measured']} cửa sổ sau warm-up, 1 luồng; nguồn: task 8.3")
    if agree:
        add("Đặc trưng online khớp offline", f"max |sai khác| < {FEAT_TOL:g}", max(agree["feature_max_abs_diff_x"], agree["feature_max_abs_diff_edge_attr"]),
            agree["features_ok"], f"{agree['n_windows_compared']} cửa sổ; nguồn: task 8.1")
        add("Dự đoán online khớp offline", f"≥ {AGREE_MIN:.1%} node", agree["pred_agreement_threshold"], agree["pred_ok"], "nguồn: task 8.2")
    add("RAM không tăng theo thời gian chạy", f"±{RAM_TOL:.0%}", ram["growth_late_vs_early"], ram["stable"], ram["definition"] + "; nguồn: task 8.1")
    add("Tổng tham số ≤ 100.000", f"≤ {MAX_PARAMS:,}".replace(",", "."), summary["n_params"], bool(summary["n_params"] <= MAX_PARAMS), "KE_HOACH mục 3")
    add("Kích thước file mô hình ≤ 1 MB", f"≤ {MAX_SIZE_KB} KB", summary["size_kb"], bool(summary["size_kb"] <= MAX_SIZE_KB), "KE_HOACH mục 3")
    if node:
        add("FPR nhị phân mức node (online)", f"≤ {FPR_LIMIT:.2%}", node["fpr"], bool(node["fpr"] <= FPR_LIMIT), "ngưỡng đã đóng băng; KE_HOACH mục 3")
        add("Recall tấn công mức node (online)", f"≥ {RECALL_MIN:.0%}", node["recall_attack"], bool(node["recall_attack"] >= RECALL_MIN), "KE_HOACH mục 3")
    if fa:
        add("Cảnh báo giả mỗi giờ (đoạn chỉ có Normal)", "chưa có chỉ tiêu trong KE_HOACH", fa["false_alerts_per_hour"], None,
            f"{fa['n_windows_normal_only']} cửa sổ Normal-only ({fa['hours_normal_only']:.3f} giờ dữ liệu); Phúc chốt ngưỡng khi viết báo cáo")
    if det:
        add("Thời gian phát hiện mỗi đợt tấn công (median, max)", "chưa có chỉ tiêu trong KE_HOACH",
            {"median_s": det["detection_median_s"], "max_s": det["detection_max_s"], "n_undetected": det["n_undetected"]}, None,
            f"{det['n_episodes']} đợt (gộp cách nhau ≤ {det['episode_gap_s']:.0f} s); Phúc chốt ngưỡng khi viết báo cáo")
    return t


# ---------------------------------------------------------------- hình
def plot_hist(win, lat, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = measured(win)
    fig, ax = plt.subplots(figsize=(9, 5))
    top = max(float(m["total_ms"].max()), 1.0)
    bins = np.logspace(np.log10(max(float(min(m["build_ms"].min(), m["infer_ms"].min())), 0.01)), np.log10(max(top, P95_LIMIT_MS * 1.1)), 60)
    ax.hist(m["total_ms"], bins=bins, alpha=0.55, color="#4C78A8", label="Tổng (dựng đồ thị + suy luận)")
    ax.hist(m["build_ms"], bins=bins, alpha=0.45, color="#F58518", label="Dựng đồ thị")
    ax.hist(m["infer_ms"], bins=bins, alpha=0.45, color="#54A24B", label="Suy luận")
    for key, ls in (("p50", "-"), ("p95", "--"), ("p99", ":")):
        ax.axvline(lat["total_ms"][key], color="black", ls=ls, lw=1.2, label=f"Tổng {key} = {lat['total_ms'][key]:.1f} ms")
    ax.axvline(P95_LIMIT_MS, color="red", lw=1.5, label="Chỉ tiêu p95 = 1.000 ms")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Độ trễ mỗi cửa sổ (ms, thang log)")
    ax.set_ylabel("Số cửa sổ (thang log)")
    ax.set_title(f"Phân bố độ trễ, mô hình {lat['model']}, {lat['n_windows_measured']:,} cửa sổ, CPU 1 luồng")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_vs_nodes(win, lat, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = measured(win)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, col, color, title in ((axes[0], "total_ms", "#4C78A8", "Tổng"), (axes[1], None, None, "Tách dựng đồ thị và suy luận")):
        if col:
            ax.scatter(m["n_nodes"], m[col], s=5, alpha=0.3, color=color)
        else:
            ax.scatter(m["n_nodes"], m["build_ms"], s=5, alpha=0.3, color="#F58518", label="Dựng đồ thị")
            ax.scatter(m["n_nodes"], m["infer_ms"], s=5, alpha=0.3, color="#54A24B", label="Suy luận")
            ax.legend(markerscale=3)
        ax.axhline(P95_LIMIT_MS, color="red", lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Số node của cửa sổ (thang log)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Độ trễ (ms, thang log)")
    for b in lat["by_nodes"]:
        x = (b["n_nodes_from"] + (b["n_nodes_to"] or b["n_nodes_from"] * 2)) / 2 if b["n_nodes_from"] else (b["n_nodes_to"] or 1) / 2
        axes[0].plot(max(x, 1), b["total_ms"]["p95"], "k_", ms=14)
    axes[0].plot([], [], "k_", ms=14, label="p95 theo nhóm số node")
    axes[0].axhline(P95_LIMIT_MS, color="red", lw=1.2, label="Chỉ tiêu 1.000 ms")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Độ trễ theo số node của cửa sổ, mô hình {lat['model']}")
    return _save(fig, path)


def _save(fig, path):
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def alerts_sample(alerts_csv, out_csv, n=1000):
    """≤ n dòng cảnh báo lấy cách đều trên toàn bộ file (giữ thứ tự thời gian)."""
    df = pd.read_csv(alerts_csv)
    if len(df) > n:
        df = df.iloc[np.linspace(0, len(df) - 1, n).astype(int)]
    df.to_csv(out_csv, index=False)
    return len(df)


def evaluate(det_dir, tables_dir, figures_dir, episode_gap=60.0):
    det_dir, tables_dir = Path(det_dir), Path(tables_dir)
    win = pd.read_csv(det_dir / "windows.csv")
    summary = json.loads((det_dir / "detector_summary.json").read_text(encoding="utf-8"))
    ws = summary["window_sec"]
    lat = latency_report(win, summary)
    ram = ram_report(win)
    agree = agreement_report(win)
    fa = false_alarm_report(win, ws) if agree else None
    det = detection_report(win, summary["classes"], episode_gap, ws) if agree else None
    node = node_report(win) if agree else None
    tables_dir.mkdir(parents=True, exist_ok=True)
    lat["ram"] = ram
    (tables_dir / "latency.json").write_text(json.dumps(lat, ensure_ascii=False, indent=1), encoding="utf-8")
    ev = {"model": summary["model"], "seed": summary["seed"], "dataset": summary["dataset"], "split": summary["split"], "threshold": summary["threshold"],
          "n_params": summary["n_params"], "size_kb": summary["size_kb"], "n_windows": int(len(win)), "n_flows": summary["n_flows"],
          "n_ref_windows": summary.get("n_ref_windows"), "n_ref_windows_not_streamed": summary.get("n_ref_windows_not_streamed"),
          "online_vs_offline": agree, "false_alarms": fa, "detection": det, "node_level_online": node, "ram": ram,
          "targets": targets_table(lat, ram, agree, fa, det, node, summary)}
    (tables_dir / "realtime_eval.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    plot_hist(win, lat, Path(figures_dir) / "gd8_latency_hist.png")
    plot_vs_nodes(win, lat, Path(figures_dir) / "gd8_latency_vs_nodes.png")
    if (det_dir / "alerts.csv").exists():
        ev["alerts_sample_rows"] = alerts_sample(det_dir / "alerts.csv", tables_dir / "alerts_sample.csv")
    return lat, ev


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gap = float(sys.argv[sys.argv.index("--episode-gap") + 1]) if "--episode-gap" in sys.argv else 60.0
    if "--episode-gap" in sys.argv:
        args.remove(sys.argv[sys.argv.index("--episode-gap") + 1])
    lat, ev = evaluate(args[0], args[1], args[2], gap)
    print(json.dumps({"latency_total_ms": lat["total_ms"], "build_ms": lat["build_ms"], "infer_ms": lat["infer_ms"]}, indent=1))
    for r in ev["targets"]:
        print(f"{'ĐẠT' if r['dat'] else ('KHÔNG ĐẠT' if r['dat'] is False else 'báo cáo')}: {r['chi_tieu']} | ngưỡng {r['nguong']} | {r['gia_tri']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
