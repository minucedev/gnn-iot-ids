"""Tổng hợp GĐ H (task 6.1, 6.2, 6.3, 5.6): bảng, kiểm định McNemar, chỉ tiêu, phân tích lỗi và hình `gd6_*.png`.

Chạy:  python -I scripts/aggregate_h.py <config.json>

config.json:
  eval_split   "test" (thật) hoặc "val" (chạy thử trên đầu ra của notebook 17a; khi đó phần "test" trong mọi bảng thực chất là val)
  eval_dir     thư mục output của notebook 17 (hoặc 17a): results_<split>_<bộ>.csv, tiers_*, pred_*, nodes_*, val_check_*, ledger_*
  val_dir      thư mục output của notebook 17a (dòng val của mọi mô hình cuối); bỏ trống khi eval_split = "val"
  repo         thư mục repo (đọc đường học, attention, ngưỡng đã đóng băng trong reports/kaggle/)
  out          thư mục ghi kết quả

Sinh ra: results.csv (20 cột, val + test), summary_val.csv, summary_test.csv, summary_test_tiers.csv, mcnemar.csv, conclusions.csv,
targets.csv, error_analysis.json, aggregate_h_summary.json, gd6_*.png.

Quy ước: `model` có hậu tố `_thr` là chế độ ngưỡng đã hiệu chỉnh, không có hậu tố là argmax (README của notebook 17). Mean ± std của 3 seed,
std với ddof = 1. "Hybrid vượt baseline" theo KE_HOACH mục 3: chênh F1-macro trung bình > 2 · max(std hai bên) VÀ McNemar p < 0,05 ở mọi seed.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import evaluate as ev  # noqa: E402

DATASETS = ["nf_ton_iot_v3", "nf_bot_iot_v3"]
MAIN_DATASET = "nf_ton_iot_v3"
ML = ["logreg", "random_forest", "xgboost", "lightgbm"]
DL = ["mlp"]
GNNS = ["gcn", "sage", "gat"]
HYBRID, LIGHT, NBR = "hybrid_gat_lgbm_oofavg_il", "hybrid_light", "lgbm_neighbor_mean"
SINGLE = [*ML, *DL, NBR, *GNNS]                       # mọi mô hình không phải hybrid
ALL_MODELS = [*SINGLE, HYBRID, LIGHT]
GROUP = {**{m: "ML" for m in ML}, "mlp": "MLP", NBR: "c", **{m: "GNN" for m in GNNS}}
GROUP_DESC = {"ML": "ML tốt nhất", "MLP": "MLP", "GNN": "GNN đơn tốt nhất", "c": "(c) lgbm_neighbor_mean"}
LABEL = {"logreg": "LogReg", "random_forest": "RandomForest", "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP",
         NBR: "(c) LightGBM+hàng xóm", "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT", HYBRID: "Hybrid chính", LIGHT: "Hybrid nhẹ"}
METRICS = ["accuracy", "balanced_acc", "precision_macro", "recall_macro", "f1_macro", "fpr", "recall_attack", "roc_auc",
           "n_params", "size_kb", "train_time_s", "infer_ms_per_window"]
TIER_METRICS = ["accuracy", "f1_macro", "fpr", "recall_attack", "roc_auc"]
LIMIT_FPR, LIMIT_RECALL, LIMIT_ACC, LIMIT_F1, LIMIT_STD = 0.0408, 0.95, 0.91, 0.80, 0.015
LIMIT_PARAMS, LIMIT_KB, LIMIT_MS = 100_000, 1024.0, 1000.0
BLUE, GRAY, RED, ORANGE, GREEN = "#1f77b4", "#9aa0a6", "#d62728", "#ff7f0e", "#2ca02c"
COMMON = ["Normal", "DoS_DDoS", "Truy_cap_trai_phep"]


# ---------------------------------------------------------------- tiện ích
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def split_mode(model):
    """Tên mô hình trong results → (tên gốc, chế độ): hậu tố `_thr` là chế độ ngưỡng hiệu chỉnh, còn lại là argmax."""
    return (model[:-4], "thr") if model.endswith("_thr") else (model, "argmax")


def ms(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return float(v.mean()), (0.0 if sd < 1e-12 else sd)


def fmt(mean, std, digits=4):
    return "" if math.isnan(mean) else f"{mean:.{digits}f} ± {std:.{digits}f}"


def beats(diff, *stds):
    """diff > 2 · max(std) (cùng quy ước aggregate_g.beats); std = 0 thì chỉ cần diff > 0."""
    spread = max(stds)
    return bool(diff > 2 * spread) if spread > 0 else bool(diff > 0)


def mcnemar_test(correct_a, correct_b):
    """McNemar trên hai mảng đúng/sai của cùng các node. b = A đúng, B sai; c = A sai, B đúng.
    b + c < 25: kiểm định nhị thức chính xác hai phía; ngược lại thống kê χ² có hiệu chỉnh liên tục (1 bậc tự do)."""
    from scipy import stats

    a, b_ = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n10, n01 = int((a & ~b_).sum()), int((~a & b_).sum())
    n = n10 + n01
    if n == 0:
        return {"b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0, "method": "no_discordant"}
    if n < 25:
        return {"b": n10, "c": n01, "statistic": float(min(n10, n01)), "p_value": float(stats.binomtest(min(n10, n01), n, 0.5).pvalue), "method": "exact"}
    chi2 = (abs(n10 - n01) - 1) ** 2 / n
    return {"b": n10, "c": n01, "statistic": float(chi2), "p_value": float(stats.chi2.sf(chi2, 1)), "method": "chi2_cc"}


# ---------------------------------------------------------------- đọc dữ liệu
def read_results(d, split):
    parts = []
    for ds in DATASETS:
        p = Path(d) / f"results_{split}_{ds}.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
    df = pd.concat(parts, ignore_index=True)
    assert list(df.columns) == ev.RESULT_COLUMNS, list(df.columns)
    return df


def read_tiers(d, split):
    return pd.concat([pd.read_csv(Path(d) / f"tiers_{split}_{ds}.csv") for ds in DATASETS if (Path(d) / f"tiers_{split}_{ds}.csv").exists()], ignore_index=True)


def add_keys(df):
    df = df.copy()
    base_mode = df["model"].map(split_mode)
    df["base"], df["mode"] = base_mode.map(lambda t: t[0]), base_mode.map(lambda t: t[1])
    return df


def summarize(df, keys=("dataset", "base", "mode"), metrics=METRICS):
    """Mean ± std theo `keys` (3 seed). Cột `<chỉ số>_mean`, `<chỉ số>_std`, `n_seeds`, `<chỉ số>_fmt` cho F1-macro, FPR, recall, accuracy."""
    rows = []
    for k, g in df.groupby(list(keys), sort=False):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r["n_seeds"] = int(g["seed"].nunique())
        for m in metrics:
            if m in g:
                r[f"{m}_mean"], r[f"{m}_std"] = ms(g[m])
        for m in ("accuracy", "f1_macro", "fpr", "recall_attack"):
            if f"{m}_mean" in r:
                r[f"{m}_fmt"] = fmt(r[f"{m}_mean"], r[f"{m}_std"])
        rows.append(r)
    return pd.DataFrame(rows)


def get(summary, ds, base, mode, col):
    s = summary[(summary["dataset"] == ds) & (summary["base"] == base) & (summary["mode"] == mode)]
    return float(s[col].iloc[0]) if len(s) else float("nan")


def models_of(ds):
    return [m for m in ALL_MODELS if not (m == LIGHT and ds != MAIN_DATASET)]


# ---------------------------------------------------------------- McNemar
def best_by_val(summary_val, ds):
    """Mô hình tốt nhất của từng nhóm theo F1-macro val trung bình (chế độ argmax); chọn trên val nên không nhìn test."""
    best = {}
    for g, members in (("ML", ML), ("MLP", DL), ("GNN", GNNS), ("c", [NBR])):
        scores = {m: get(summary_val, ds, m, "argmax", "f1_macro_mean") for m in members}
        best[g] = max(scores, key=lambda m: (scores[m] if not math.isnan(scores[m]) else -1))
    return best


def mcnemar_table(pred_dir, summary_val, seeds=(42, 43, 44)):
    rows = []
    for ds in DATASETS:
        best = best_by_val(summary_val, ds)
        for hy in (HYBRID, LIGHT):
            if hy not in models_of(ds):
                continue
            for seed in seeds:
                a = pd.read_parquet(Path(pred_dir) / f"pred_{ds}_{hy}_s{seed}.parquet")
                y = a["y_true"].to_numpy()
                for other in SINGLE:
                    b = pd.read_parquet(Path(pred_dir) / f"pred_{ds}_{other}_s{seed}.parquet")
                    assert np.array_equal(y, b["y_true"].to_numpy()), "thứ tự node khác nhau giữa hai mô hình"
                    for mode, col in (("argmax", "pred"), ("thr", "pred_thr")):
                        ca, cb = a[col].to_numpy() == y, b[col].to_numpy() == y
                        t = mcnemar_test(ca, cb)
                        rows.append({"dataset": ds, "hybrid": hy, "baseline": other, "group": GROUP[other], "selected_best_by_val": best[GROUP[other]] == other,
                                     "seed": seed, "mode": mode, "n_nodes": int(len(y)), "acc_hybrid": float(ca.mean()), "acc_baseline": float(cb.mean()),
                                     **t, "significant": bool(t["p_value"] < 0.05),
                                     "better": "hybrid" if t["b"] > t["c"] else ("baseline" if t["c"] > t["b"] else "tie")})
    return pd.DataFrame(rows)


def conclusions_table(summary_test, mcn):
    """Với mỗi (bộ, hybrid, chế độ, nhóm baseline được chọn theo val): chênh F1-macro, 2·std, McNemar ở 3 seed và kết luận."""
    rows = []
    for (ds, hy, mode, grp), g in mcn[mcn["selected_best_by_val"]].groupby(["dataset", "hybrid", "mode", "group"], sort=False):
        base = g["baseline"].iloc[0]
        f_h, s_h = get(summary_test, ds, hy, mode, "f1_macro_mean"), get(summary_test, ds, hy, mode, "f1_macro_std")
        f_b, s_b = get(summary_test, ds, base, mode, "f1_macro_mean"), get(summary_test, ds, base, mode, "f1_macro_std")
        diff = f_h - f_b
        n_sig_better = int(((g["better"] == "hybrid") & g["significant"]).sum())
        n_sig_worse = int(((g["better"] == "baseline") & g["significant"]).sum())
        beats_std = beats(diff, s_h, s_b)
        verdict = ("hybrid VƯỢT" if beats_std and n_sig_better == len(g) else
                   "baseline VƯỢT" if (-diff > 2 * max(s_h, s_b, 1e-12)) and n_sig_worse == len(g) else
                   "KHÔNG phân biệt được" if n_sig_better == 0 and n_sig_worse == 0 else "hỗn hợp (không đủ điều kiện vượt)")
        rows.append({"dataset": ds, "hybrid": hy, "mode": mode, "group": GROUP_DESC[grp], "baseline": base, "f1_hybrid": f_h, "f1_hybrid_std": s_h,
                     "f1_baseline": f_b, "f1_baseline_std": s_b, "diff": diff, "two_std": 2 * max(s_h, s_b), "diff_gt_2std": beats_std,
                     "n_seeds": len(g), "n_seed_mcnemar_hybrid_better": n_sig_better, "n_seed_mcnemar_baseline_better": n_sig_worse,
                     "p_value_max_over_seeds": float(g["p_value"].max()), "verdict": verdict})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- chỉ tiêu (KE_HOACH mục 3)
def targets_table(summary_test, conclusions, val_check_max, eval_split):
    """Mỗi dòng: một chỉ tiêu × bộ × mô hình × chế độ ngưỡng. Bộ chính (NF-ToN-IoT-v3) là bộ nghiệm thu; bộ BoT ghi thêm để tham khảo."""
    rows = []

    def add(target, ds, model, mode, value, thr, ok, unit, note=""):
        rows.append({"target": target, "dataset": ds, "model": model, "mode": mode, "value": value, "threshold": thr, "unit": unit,
                     "passed": None if ok is None else bool(ok), "result": "đạt" if ok else ("không đạt" if ok is not None else "không đo được"),
                     "official": ds == MAIN_DATASET, "note": note, "split": eval_split})

    for ds in DATASETS:
        for hy in (HYBRID, LIGHT):
            if hy not in models_of(ds):
                continue
            for mode in ("argmax", "thr"):
                g = lambda c: get(summary_test, ds, hy, mode, c)
                add("Accuracy đa lớp ≥ 91%", ds, hy, mode, g("accuracy_mean"), LIMIT_ACC, g("accuracy_mean") >= LIMIT_ACC, "tỉ lệ")
                add("FPR nhị phân ≤ 4,08%", ds, hy, mode, g("fpr_mean"), LIMIT_FPR, g("fpr_mean") <= LIMIT_FPR, "tỉ lệ")
                add("Recall tấn công ≥ 95%", ds, hy, mode, g("recall_attack_mean"), LIMIT_RECALL, g("recall_attack_mean") >= LIMIT_RECALL, "tỉ lệ")
                add("F1-macro (lớp chính) ≥ 0,80", ds, hy, mode, g("f1_macro_mean"), LIMIT_F1, g("f1_macro_mean") >= LIMIT_F1, "điểm F1")
                add("Độ ổn định: std accuracy ≤ 1,5 điểm %", ds, hy, mode, g("accuracy_std"), LIMIT_STD, g("accuracy_std") <= LIMIT_STD, "tỉ lệ (std 3 seed)")
                add("Tổng tham số ≤ 100.000", ds, hy, mode, g("n_params_mean"), LIMIT_PARAMS, g("n_params_mean") <= LIMIT_PARAMS, "tham số",
                    "hybrid = tham số các GAT + số lá LightGBM (như GĐ F)")
                add("Kích thước file ≤ 1 MB", ds, hy, mode, g("size_kb_mean"), LIMIT_KB, g("size_kb_mean") <= LIMIT_KB, "KB")
                add("Độ trễ p95 ≤ 1 s mỗi cửa sổ", ds, hy, mode, g("infer_ms_per_window_mean"), LIMIT_MS, g("infer_ms_per_window_mean") <= LIMIT_MS, "ms",
                    "đo sơ bộ: p95 của 200 lần trên 1 cửa sổ trung vị, CPU Kaggle 1 luồng; chưa phải ≥ 1.000 cửa sổ (đo đủ ở GĐ I, task 8.x)")
        c = conclusions[conclusions["dataset"] == ds] if len(conclusions) else conclusions
        for (hy, mode), cg in (c.groupby(["hybrid", "mode"]) if len(c) else []):
            ok = bool((cg["verdict"] == "hybrid VƯỢT").all())
            detail = "; ".join(f"{r.group}: {r.baseline} {r.diff:+.4f} (2·std {r.two_std:.4f}), McNemar {r.n_seed_mcnemar_hybrid_better}/{r.n_seeds} seed, {r.verdict}" for r in cg.itertuples())
            add("Hybrid > baseline tốt nhất (ML, MLP, GNN đơn, và (c)): chênh F1-macro > 2·std và McNemar p < 0,05", ds, hy, mode,
                float((cg["verdict"] == "hybrid VƯỢT").sum()), float(len(cg)), ok, "số nhóm vượt / số nhóm", detail)
    add("Số bộ dữ liệu ≥ 2", "cả hai bộ", "—", "—", float(len(DATASETS)), 2.0, True, "bộ", "NF-ToN-IoT-v3 và NF-BoT-IoT-v3 đều có kết quả " + eval_split)
    add("Tái lập: chạy lại lệch ≤ 0,5 điểm %", "cả hai bộ", "mọi mô hình", "argmax", val_check_max, 0.005, val_check_max <= 0.005, "F1-macro val",
        "sơ bộ: mô hình dựng lại ở notebook 17 lệch F1-macro val tối đa so với lúc hiệu chỉnh ngưỡng; GĐ J chạy lại đầy đủ (task 9.x)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- phân tích lỗi
def error_analysis(eval_dir, ds, hybrid, seed, classes, features, top_ips=5, n_examples=3):
    """Các nhóm lỗi của hybrid chính: (1) FP trên node chỉ nhận; (2) FP trên node có khởi tạo; (3) nhầm giữa các lớp tấn công; (4) FN (tấn công bị coi là Normal);
    (5) lớp hiếm bị bỏ sót (nếu có). Mỗi nhóm: số node, tỉ lệ, đặc điểm, IP thường gặp, ví dụ cụ thể (IP, cửa sổ, nhãn thật, dự đoán, điểm, đặc trưng chính)."""
    p = pd.read_parquet(Path(eval_dir) / f"pred_{ds}_{hybrid}_s{seed}.parquet")
    nodes = pd.read_parquet(Path(eval_dir) / f"nodes_{ds}_{'test' if (Path(eval_dir) / f'nodes_{ds}_test.parquet').exists() else 'val'}.parquet")
    df = pd.concat([nodes.reset_index(drop=True), p.reset_index(drop=True)], axis=1)
    attack_true, attack_pred = df["y_true"] != 0, df["pred"] != 0
    wrong = df["pred"] != df["y_true"]
    key = ["n_out", "n_in", "bytes_sent", "bytes_recv", "uniq_dst", "uniq_dport", "noreply_ratio", "rst_ratio", "tcp_ratio", "udp_ratio"]
    key = [k for k in key if k in features]
    win_size = df.groupby("gi")["y_true"].transform("size")
    df["win_nodes"] = win_size
    ref = {"n_nodes": int(len(df)), "n_wrong": int(wrong.sum()), "error_rate": float(wrong.mean()), "initiator_share_all": float(df["initiator"].mean()),
           "mean_window_nodes_all": float(win_size.mean())}

    def group(name, mask, why):
        sub = df[mask]
        g = {"group": name, "n_nodes": int(len(sub)), "share_of_errors": float(len(sub) / max(int(wrong.sum()), 1)), "explanation": why}
        if len(sub) == 0:
            return {**g, "examples": []}
        g["initiator_share"] = float(sub["initiator"].mean())
        g["mean_window_nodes"] = float(sub["win_nodes"].mean())
        g["mean_score"] = float(sub["score"].mean())
        g["feature_mean_vs_all"] = {k: {"group": float(sub[k].mean()), "all": float(df[k].mean())} for k in key}
        g["top_ips"] = [{"ip": ip, "n": int(n), "share": float(n / len(sub)), "n_windows": int(sub[sub["ip"] == ip]["win"].nunique())}
                        for ip, n in sub["ip"].value_counts().head(top_ips).items()]
        order = sub.assign(_d=(sub["score"] - 0.5).abs()).sort_values("_d", ascending=False)           # ví dụ: lỗi "chắc chắn" nhất trước
        g["examples"] = [{"ip": r.ip, "window": int(r.win), "true": classes[int(r.y_true)], "pred": classes[int(r.pred)], "attack_score": float(r.score),
                          "initiator": bool(r.initiator), "window_nodes": int(r.win_nodes), "features": {k: float(getattr(r, k)) for k in key[:6]}}
                         for r in order.head(n_examples).itertuples()]
        return g

    groups = [
        group("FP trên node chỉ nhận flow", ~attack_true & attack_pred & ~df["initiator"],
              "Node Normal không khởi tạo flow nào (chỉ nhận) bị đoán là tấn công: node nạn nhân hoặc máy chủ nhận flow từ nguồn tấn công; nhãn node lấy từ flow nhận."),
        group("FP trên node có khởi tạo flow", ~attack_true & attack_pred & df["initiator"],
              "Node Normal có khởi tạo flow nhưng mẫu lưu lượng giống tấn công (số đích/cổng lớn, nhiều flow không phản hồi)."),
        group("Tấn công bị bỏ sót (FN, đoán là Normal)", attack_true & ~attack_pred,
              "Node tấn công có đặc trưng giống node Normal (ít flow, ít đích) nên không phân biệt được ở mức node."),
        group("Nhầm giữa các lớp tấn công", attack_true & attack_pred & wrong,
              "Phát hiện đúng là tấn công nhưng sai loại (ví dụ DoS_DDoS với Truy_cap_trai_phep, hoặc Botnet_Backdoor, Khac): các loại có đặc trưng cấp node gần nhau."),
    ]
    rare = [c for c in range(2, len(classes)) if classes[c] in ("Botnet_Backdoor", "Khac") and (df["y_true"] == c).any()]
    for c in rare:
        m = (df["y_true"] == c) & wrong
        groups.append(group(f"Lớp hiếm {classes[c]} bị đoán sai", m, f"Lớp {classes[c]} ít node ({int((df['y_true'] == c).sum())} trên tập đánh giá) nên mô hình học ít; recall thấp."))
    confusion = {classes[i]: {classes[j]: int(((df["y_true"] == i) & (df["pred"] == j)).sum()) for j in range(len(classes))} for i in range(len(classes))}
    return {"dataset": ds, "model": hybrid, "seed": seed, "overall": ref, "confusion_matrix_counts": confusion, "groups": groups}


# ---------------------------------------------------------------- hình
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.axisbelow": True, "figure.dpi": 100})
    return plt


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)


def color_of(base):
    return RED if base == HYBRID else (ORANGE if base == LIGHT else (GREEN if base == NBR else (BLUE if base in GNNS else GRAY)))


def fig_bars(summary_test, ds, path, plt, label_split):
    models = models_of(ds)
    y = np.arange(len(models))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 0.5 * len(models) + 2.6), sharey=True)
    h = 0.38
    for off, mode, alpha in ((-h / 2, "argmax", 0.55), (h / 2, "thr", 1.0)):
        f = [get(summary_test, ds, m, mode, "f1_macro_mean") for m in models]
        fe = [get(summary_test, ds, m, mode, "f1_macro_std") for m in models]
        a1.barh(y + off, f, xerr=fe, height=h, color=[color_of(m) for m in models], alpha=alpha, capsize=2,
                label="argmax" if mode == "argmax" else "ngưỡng hiệu chỉnh")
        fp = [get(summary_test, ds, m, mode, "fpr_mean") for m in models]
        fpe = [get(summary_test, ds, m, mode, "fpr_std") for m in models]
        a2.barh(y + off, fp, xerr=fpe, height=h, color=[color_of(m) for m in models], alpha=alpha, capsize=2)
    a1.set_xlabel(f"F1-macro trên {label_split} (3 seed ± std)")
    lo = min(get(summary_test, ds, m, md, "f1_macro_mean") for m in models for md in ("argmax", "thr"))
    a1.set_xlim(max(0, lo - 0.06), 1.0)
    a2.axvline(LIMIT_FPR, color=RED, ls="--", lw=1)
    a2.text(LIMIT_FPR, -0.8, " 4,08%", color=RED, fontsize=8, va="bottom")
    a2.set_xlabel(f"FPR trên {label_split} (3 seed ± std)")
    a1.set_yticks(y, [LABEL[m] for m in models])
    a1.invert_yaxis()
    a1.legend(fontsize=8, loc="lower left", title="cột nhạt = argmax, cột đậm = ngưỡng hiệu chỉnh", title_fontsize=7)
    fig.suptitle(f"So sánh mọi mô hình trên {label_split}, {ds}", fontsize=11)
    _save(fig, path)


def fig_confusion(eval_dir, ds, classes, path, plt, seed=42, label_split="test"):
    from sklearn.metrics import confusion_matrix

    p = pd.read_parquet(Path(eval_dir) / f"pred_{ds}_{HYBRID}_s{seed}.parquet")
    labels = list(range(len(classes)))
    fig, axes = plt.subplots(1, 2, figsize=(2 * (1.1 * len(labels) + 3.2), 1.1 * len(labels) + 2.6))
    for ax, col, title in ((axes[0], "pred", "argmax"), (axes[1], "pred_thr", "ngưỡng hiệu chỉnh")):
        cm = confusion_matrix(p["y_true"], p[col], labels=labels, normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in labels:
            for j in labels:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(labels, classes, rotation=30, ha="right")
        ax.set_yticks(labels, classes)
        ax.set_xlabel("Lớp dự đoán")
        ax.set_ylabel("Lớp thật")
        ax.set_title(title)
        ax.grid(False)
    fig.colorbar(im, ax=axes, label="tỉ lệ theo hàng (recall từng lớp)", shrink=0.85)
    fig.suptitle(f"Ma trận nhầm lẫn của hybrid chính trên {label_split}, {ds} (seed {seed})", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_roc_pr(eval_dir, ds, path, plt, seed=42, label_split="test"):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for m in models_of(ds):
        p = pd.read_parquet(Path(eval_dir) / f"pred_{ds}_{m}_s{seed}.parquet")
        attack = (p["y_true"] != 0).to_numpy()
        s = p["score"].to_numpy()
        fpr, tpr, _ = roc_curve(attack, s)
        prec, rec, _ = precision_recall_curve(attack, s)
        lw = 2.2 if m in (HYBRID, LIGHT) else 1.0
        a1.plot(fpr, tpr, color=color_of(m), lw=lw, alpha=1 if m in (HYBRID, LIGHT, NBR) else 0.6, label=f"{LABEL[m]} (AUC {roc_auc_score(attack, s):.3f})")
        a2.plot(rec, prec, color=color_of(m), lw=lw, alpha=1 if m in (HYBRID, LIGHT, NBR) else 0.6, label=f"{LABEL[m]} (AP {average_precision_score(attack, s):.3f})")
    a1.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    a1.axvline(LIMIT_FPR, color=RED, ls=":", lw=1)
    a1.set(xlabel="FPR (tỉ lệ)", ylabel="Recall tấn công (tỉ lệ)", title="ROC (tấn công so với Normal)")
    a2.set(xlabel="Recall tấn công (tỉ lệ)", ylabel="Precision (tỉ lệ)", title="Precision-Recall")
    for a in (a1, a2):
        a.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"ROC và PR trên {label_split}, {ds} (seed {seed}; vạch đỏ = FPR 4,08%)", fontsize=11)
    _save(fig, path)


def fig_f1_vs(summary_test, ds, x_col, xlabel, limit, limit_label, path, plt, log=True, label_split="test"):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for m in models_of(ds):
        for mode, mk in (("argmax", "o"), ("thr", "s")):
            x, f = get(summary_test, ds, m, mode, x_col), get(summary_test, ds, m, mode, "f1_macro_mean")
            if math.isnan(x) or math.isnan(f):
                continue
            ax.errorbar(x, f, yerr=get(summary_test, ds, m, mode, "f1_macro_std"), fmt=mk, color=color_of(m), ms=7, capsize=2, alpha=0.9 if mode == "argmax" else 0.55,
                        mfc=color_of(m) if mode == "argmax" else "white")
        x0 = get(summary_test, ds, m, "argmax", x_col)
        if not math.isnan(x0):
            ax.annotate(LABEL[m], (x0, get(summary_test, ds, m, "argmax", "f1_macro_mean")), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
    if limit is not None:
        ax.axvline(limit, color=RED, ls="--", lw=1)
        ax.text(limit, ax.get_ylim()[0], f" {limit_label}", color=RED, fontsize=8, va="bottom")
    if log:
        ax.set_xscale("log")
    ax.set(xlabel=xlabel, ylabel=f"F1-macro trên {label_split}", title=f"F1-macro theo {xlabel.split(' (')[0].lower()}, {ds} (tròn đặc = argmax, vuông rỗng = ngưỡng hiệu chỉnh)")
    ax.title.set_fontsize(9.5)
    _save(fig, path)


def fig_per_class(summary_rows, ds, classes, path, plt, label_split="test"):
    """F1 theo lớp của các mô hình chính (trung bình 3 seed, chế độ argmax) từ cột f1_per_class của results."""
    sel = [HYBRID, LIGHT, NBR, "lightgbm", "mlp", "gat"]
    sel = [m for m in sel if m in models_of(ds)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.8 / len(sel)
    for i, m in enumerate(sel):
        g = summary_rows[(summary_rows["dataset"] == ds) & (summary_rows["model"] == m)]
        vals = np.array([[json.loads(s)[c] for c in classes] for s in g["f1_per_class"]])
        ax.bar(np.arange(len(classes)) + i * w, vals.mean(0), w, yerr=vals.std(0, ddof=1) if len(vals) > 1 else None, color=color_of(m), alpha=0.5 + 0.5 * (m in (HYBRID, LIGHT)),
               capsize=2, label=LABEL[m])
    ax.set_xticks(np.arange(len(classes)) + w * (len(sel) - 1) / 2, classes)
    ax.set(ylabel=f"F1 theo lớp trên {label_split} (argmax, 3 seed ± std)", title=f"F1 theo lớp, {ds}")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_train_time(summary_test, ds, path, plt):
    models = [m for m in models_of(ds) if m != LIGHT]
    vals = [get(summary_test, ds, m, "argmax", "train_time_s_mean") for m in models]
    errs = [get(summary_test, ds, m, "argmax", "train_time_s_std") for m in models]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(range(len(models)), vals, yerr=errs, color=[color_of(m) for m in models], capsize=2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(models)), [LABEL[m] for m in models], rotation=30, ha="right")
    ax.set(ylabel="Thời gian huấn luyện (giây, thang log)", title=f"Thời gian huấn luyện, {ds} (GNN và hybrid lấy từ lúc huấn luyện gốc)")
    _save(fig, path)


def fig_tiers(tiers_summary, ds, path, plt, label_split="test"):
    sel = [m for m in (HYBRID, LIGHT, NBR, "lightgbm", "mlp", "gat") if m in models_of(ds)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, metric, title in ((axes[0], "fpr", "FPR"), (axes[1], "recall_attack", "Recall tấn công")):
        w = 0.8 / len(sel)
        for i, m in enumerate(sel):
            vals, errs = [], []
            for tier in ("all", "initiator", "receive_only"):
                r = tiers_summary[(tiers_summary["dataset"] == ds) & (tiers_summary["base"] == m) & (tiers_summary["mode"] == "argmax") & (tiers_summary["tier"] == tier)]
                vals.append(float(r[f"{metric}_mean"].iloc[0]) if len(r) else float("nan"))
                errs.append(float(r[f"{metric}_std"].iloc[0]) if len(r) else 0.0)
            ax.bar(np.arange(3) + i * w, vals, w, yerr=errs, color=color_of(m), alpha=0.5 + 0.5 * (m in (HYBRID, LIGHT)), capsize=2, label=LABEL[m])
        ax.set_xticks(np.arange(3) + w * (len(sel) - 1) / 2, ["toàn bộ node", "node có khởi tạo flow", "node chỉ nhận flow"])
        ax.set(ylabel=f"{title} ({label_split}, argmax)", title=title)
        if metric == "fpr":
            ax.axhline(LIMIT_FPR, color=RED, ls="--", lw=1)
            ax.legend(fontsize=7)
    fig.suptitle(f"Báo cáo theo tầng node, {ds}", fontsize=11)
    _save(fig, path)


def fig_cross(summary_test, path, plt, label_split="test"):
    pairs = [d for d in summary_test["dataset"].unique() if "→" in d]
    if not pairs:
        return False
    models = [m for m in (HYBRID, "gat", "lightgbm", NBR)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, title in ((axes[0], "f1_macro", "F1-macro (lớp chung)"), (axes[1], "fpr", "FPR")):
        w = 0.8 / max(len(pairs), 1)
        for i, d in enumerate(sorted(pairs)):
            vals = [get(summary_test, d, m, "argmax", f"{metric}_mean") for m in models]
            errs = [get(summary_test, d, m, "argmax", f"{metric}_std") for m in models]
            ax.bar(np.arange(len(models)) + i * w, vals, w, yerr=errs, capsize=2, label=d.replace("nf_", "").replace("_iot_v3", "").replace("→", " → "))
        ax.set_xticks(np.arange(len(models)) + w * (len(pairs) - 1) / 2, [LABEL[m] for m in models])
        ax.set(ylabel=f"{title} trên {label_split}", title=title)
        ax.legend(fontsize=8, title="huấn luyện → đánh giá", title_fontsize=7)
    fig.suptitle("Tổng quát hóa giữa các bộ (argmax; chỉ các lớp chung Normal, DoS_DDoS, Truy_cap_trai_phep)", fontsize=10.5)
    _save(fig, path)
    return True


def fig_curves(repo, ds, path, plt):
    """Đường huấn luyện (loss train, loss val, F1-macro val theo epoch) của GCN, GraphSAGE, GAT cấu hình mặc định, seed 42."""
    k = Path(repo) / "reports" / "kaggle"
    if ds == "nf_ton_iot_v3":
        parts = [(k / "gnn-iot-ids-16-ablation" / "curves_grid_nf_ton_iot_v3.csv", {"gat_default": "GAT"}),
                 (k / "gnn-iot-ids-16g-tuning-ton" / "curves_grid_nf_ton_iot_v3.csv", {"gcn_h16_l2": "GCN", "sage_h16_l2": "GraphSAGE"})]
    else:
        parts = [(k / "gnn-iot-ids-14-train-gnn-bot" / "curves_nf_bot_iot_v3.csv", {"gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"})]
    cur = pd.concat([pd.read_csv(p)[lambda d: d["model"].isin(names)].assign(name=lambda d: d["model"].map(names)) for p, names in parts if p.exists()], ignore_index=True)
    if cur.empty:
        return False
    cur = cur[(cur["seed"] == 42) & (cur["dataset"] == ds)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, g in cur.groupby("name"):
        c = {"GCN": GRAY, "GraphSAGE": GREEN, "GAT": BLUE}[name]
        axes[0].plot(g["epoch"], g["train_loss"], color=c, label=name)
        axes[1].plot(g["epoch"], g["val_loss"], color=c, label=name)
        axes[2].plot(g["epoch"], g["val_f1_macro"], color=c, label=name)
        b = g[g["is_best"] == 1]
        if len(b):
            axes[2].scatter(b["epoch"], b["val_f1_macro"], color=c, zorder=3)
    for a, t in zip(axes, ("Loss huấn luyện", "Loss val", "F1-macro val (chấm = epoch chọn)")):
        a.set(xlabel="Epoch", title=t)
        a.legend(fontsize=8)
    fig.suptitle(f"Đường huấn luyện của GNN đơn, {ds} (seed 42)", fontsize=11)
    _save(fig, path)
    return True


def fig_attention(repo, ds, path, plt):
    """Hình attention của GAT (seed 42) trên 2 cửa sổ val mẫu (trọng số lưu ở task 5.3): độ dày và màu cạnh = hệ số attention trung bình các head
    của thông điệp từ nguồn tới đích, ở lớp GAT thứ hai; viền node = lớp thật, tô node = lớp dự đoán."""
    import networkx as nx

    k = Path(repo) / "reports" / "kaggle"
    src = k / ("gnn-iot-ids-16d-ablation-ton/attention_samples_nf_ton_iot_v3.json" if ds == "nf_ton_iot_v3" else "gnn-iot-ids-16a-ablation-bot/attention_samples_nf_bot_iot_v3.json")
    if not src.exists():
        return False
    data = json.loads(src.read_text(encoding="utf-8"))
    samples = data["samples"]
    palette = {c: col for c, col in zip(data["classes"], [GREEN, RED, ORANGE, BLUE, GRAY])}
    fig, axes = plt.subplots(1, len(samples), figsize=(7.2 * len(samples), 6.4))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, samples):
        n, e = s["n_nodes"], s["n_edges_original"]
        layer = s["layers"][-1]
        ei, alpha = np.array(layer["edge_index"]), np.array(layer["alpha"]).mean(1)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(e):
            G.add_edge(int(ei[0][i]), int(ei[1][i]))
        for i in range(e):                                   # thông điệp ngược (bidirectional)
            G.add_edge(int(ei[0][e + i]), int(ei[1][e + i]))
        pos = nx.spring_layout(nx.Graph(G), seed=3, k=1.4 / math.sqrt(max(n, 1)))
        a_edge = {(int(ei[0][i]), int(ei[1][i])): float(alpha[i]) for i in range(2 * e)}
        edges = list(a_edge)
        widths = [0.6 + 5.0 * a_edge[ed] for ed in edges]
        cmap = plt.cm.viridis
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths, edge_color=[cmap(a_edge[ed]) for ed in edges], arrows=True, arrowsize=9,
                               connectionstyle="arc3,rad=0.14", ax=ax, node_size=420, alpha=0.9)
        nx.draw_networkx_nodes(G, pos, node_color=[palette.get(s["pred"][i], GRAY) for i in range(n)], edgecolors=[palette.get(s["y"][i], GRAY) for i in range(n)],
                               linewidths=2.6, node_size=420, ax=ax)
        nx.draw_networkx_labels(G, pos, {i: str(i) for i in range(n)}, font_size=7, font_color="white", ax=ax)
        ax.set_title(f"cửa sổ {s['window']} ({s['sample_class']}): {n} node, {e} cạnh; đoán đúng "
                     f"{sum(1 for i in range(n) if s['y'][i] == s['pred'][i])}/{n}", fontsize=9)
        ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes, label="attention (trung bình 4 head, thông điệp nguồn → đích, lớp 2)", shrink=0.7)
    handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=c, mec=c, label=n) for n, c in palette.items()]
    axes[0].legend(handles=handles, fontsize=7, title="lớp (tô = dự đoán, viền = thật)", title_fontsize=7, loc="lower left")
    fig.suptitle(f"Attention của GAT trên 2 cửa sổ val mẫu, {ds} (seed 42; lưu ở task 5.3)", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------- chính
def run(cfg):
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    ev_split = cfg["eval_split"]
    eval_dir, repo = cfg["eval_dir"], cfg["repo"]
    val_dir = cfg.get("val_dir") or eval_dir
    plt = _plt()

    test_rows = read_results(eval_dir, ev_split)
    val_rows = read_results(val_dir, "val")
    results = pd.concat([val_rows, test_rows], ignore_index=True) if ev_split == "test" else val_rows.copy()
    results.to_csv(out / "results.csv", index=False)

    sv = summarize(add_keys(val_rows[val_rows["task"] == "multiclass"]))
    st = summarize(add_keys(test_rows[test_rows["task"] == "multiclass"]))
    sv.to_csv(out / "summary_val.csv", index=False)
    st.to_csv(out / ("summary_test.csv" if ev_split == "test" else "summary_test_as_val_dryrun.csv"), index=False)
    tiers = add_keys(read_tiers(eval_dir, ev_split).rename(columns={"model": "model"}))
    tiers_sum = summarize(tiers, keys=("dataset", "base", "mode", "tier"), metrics=TIER_METRICS)
    tiers_n = tiers.groupby(["dataset", "base", "mode", "tier"], sort=False)[["n_nodes", "n_normal", "n_attack"]].first().reset_index()
    tiers_sum = tiers_sum.merge(tiers_n, on=["dataset", "base", "mode", "tier"], how="left")
    tiers_sum.to_csv(out / "summary_test_tiers.csv", index=False)

    mcn = mcnemar_table(eval_dir, sv)
    mcn.to_csv(out / "mcnemar.csv", index=False)
    conc = conclusions_table(st, mcn)
    conc.to_csv(out / "conclusions.csv", index=False)
    led = pd.read_csv(Path(eval_dir) / f"ledger_{ev_split}.csv")
    val_check_max = float(led["val_f1_diff"].abs().max())
    targets = targets_table(st, conc, val_check_max, ev_split)
    targets.to_csv(out / "targets.csv", index=False)

    lab = "test" if ev_split == "test" else "val (chạy thử)"
    made = []
    classes_of, features_of = {}, {}
    for ds in DATASETS:
        meta = json.loads((Path(cfg["data"][ds]) / "split.json").read_text(encoding="utf-8")) if cfg.get("data", {}).get(ds) else None
        classes_of[ds] = meta["classes"] if meta else (["Normal", "DoS_DDoS", "Botnet_Backdoor", "Truy_cap_trai_phep", "Khac"] if ds == MAIN_DATASET else COMMON)
        features_of[ds] = [c for c in pd.read_parquet(Path(eval_dir) / f"nodes_{ds}_{ev_split}.parquet").columns if c not in ("gi", "win", "ip", "y", "initiator")]
        t_ds = st[~st["dataset"].str.contains("→")]
        for name, fn in (
            (f"gd6_bar_f1_fpr_{ds}.png", lambda p: fig_bars(t_ds, ds, p, plt, lab)),
            (f"gd6_confusion_hybrid_{ds}.png", lambda p: fig_confusion(eval_dir, ds, classes_of[ds], p, plt, label_split=lab)),
            (f"gd6_roc_pr_{ds}.png", lambda p: fig_roc_pr(eval_dir, ds, p, plt, label_split=lab)),
            (f"gd6_f1_vs_params_{ds}.png", lambda p: fig_f1_vs(t_ds, ds, "n_params_mean", "Số tham số (thang log)", LIMIT_PARAMS, "100K", p, plt, label_split=lab)),
            (f"gd6_f1_vs_size_{ds}.png", lambda p: fig_f1_vs(t_ds, ds, "size_kb_mean", "Kích thước file mô hình (KB, thang log)", LIMIT_KB, "1 MB", p, plt, label_split=lab)),
            (f"gd6_f1_vs_latency_{ds}.png", lambda p: fig_f1_vs(t_ds, ds, "infer_ms_per_window_mean", "Độ trễ p95 mỗi cửa sổ (ms, thang log, CPU 1 luồng)", LIMIT_MS, "1 s", p, plt, label_split=lab)),
            (f"gd6_f1_per_class_{ds}.png", lambda p: fig_per_class(test_rows[~test_rows["model"].str.endswith("_thr") & ~test_rows["dataset"].str.contains("→")], ds, classes_of[ds], p, plt, label_split=lab)),
            (f"gd6_train_time_{ds}.png", lambda p: fig_train_time(t_ds, ds, p, plt)),
            (f"gd6_tiers_{ds}.png", lambda p: fig_tiers(tiers_sum, ds, p, plt, label_split=lab)),
            (f"gd6_training_curves_{ds}.png", lambda p: fig_curves(repo, ds, p, plt)),
            (f"gd6_attention_{ds}.png", lambda p: fig_attention(repo, ds, p, plt)),
        ):
            ok = fn(out / name)
            if ok is not False:
                made.append(name)
    if fig_cross(st, out / "gd6_cross_dataset.png", plt, lab):
        made.append("gd6_cross_dataset.png")

    ea = {ds: error_analysis(eval_dir, ds, HYBRID, 42, classes_of[ds], features_of[ds]) for ds in DATASETS}
    (out / "error_analysis.json").write_text(json.dumps(clean(ea), ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {"eval_split": ev_split, "n_results_rows": int(len(results)), "figures": made, "val_check_max_abs_f1_diff": val_check_max,
               "n_error_groups_with_examples": {ds: sum(1 for g in v["groups"] if g["examples"]) for ds, v in ea.items()},
               "mcnemar_rows": int(len(mcn)), "conclusions": clean(conc.to_dict("records"))}
    (out / "aggregate_h_summary.json").write_text(json.dumps(clean(summary), ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(clean({k: v for k, v in summary.items() if k != "conclusions"}), ensure_ascii=False, indent=1))
    print(conc[["dataset", "hybrid", "mode", "group", "baseline", "diff", "two_std", "n_seed_mcnemar_hybrid_better", "verdict"]].to_string())
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    return run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))


if __name__ == "__main__":
    sys.exit(main())
