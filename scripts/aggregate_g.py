"""Tổng hợp GĐ G (task 5.2, 5.3, 5.4, 5.5, 5.8) từ output của các notebook 16*: bảng, thresholds.json và hình. Chỉ val.

Chạy:  python -I scripts/aggregate_g.py <out_dir> repo=<thư mục repo> bot=<16a> ton_hybrid=<16d> ton_thr=<16e> ton_tuning=<16g>
       ton_grid=<16> windows=<16f>
Mỗi tham số là thư mục chứa output của notebook tương ứng; thiếu một nguồn thì phần dùng nguồn đó bị bỏ qua và ghi vào `missing`.
Kết quả cũ (GĐ D, E, F) đọc từ `<repo>/reports/kaggle/...`.

Sinh ra: ablation_hybrid_<bộ>.csv, pareto_hybrid_<bộ>.csv, light_choice_<bộ>.json, ablation_gat.csv, ablation_gat_conclusion.json,
tuning_gnn.csv, improved_gat.csv, thresholds.json, aggregate_summary.json, gd5_ablation_gat_<yếu tố>.png, gd5_ablation_hybrid_<bộ>.png,
gd5_pareto_<bộ>.png, gd5_compare_<bộ>.png.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import threshold as TH  # noqa: E402

DATASETS = ["nf_ton_iot_v3", "nf_bot_iot_v3"]
OLD = {
    "nf_ton_iot_v3": {"base": "gnn-iot-ids-05-baselines/results_baselines_nf_ton_iot_v3.csv",
                      "gnn": "gnn-iot-ids-07-train-gnn/results_gnn_nf_ton_iot_v3.csv",
                      "hybrid": "gnn-iot-ids-15f-hybrid-il-ton/results_hybrid_oofavg_il_nf_ton_iot_v3.csv"},
    "nf_bot_iot_v3": {"base": "gnn-iot-ids-13-baselines/results_baselines_nf_bot_iot_v3.csv",
                      "gnn": "gnn-iot-ids-14-train-gnn-bot/results_gnn_nf_bot_iot_v3.csv",
                      "hybrid": "gnn-iot-ids-15e-hybrid-il/results_hybrid_oofavg_il_nf_bot_iot_v3.csv"},
}
HYBRID_FULL = "hybrid_gat_lgbm_oofavg_il"
PARAM_LIMIT, SIZE_LIMIT_KB = 100_000, 1024.0
SEEDS = (42, 43, 44)
VARIANT_NAMES = {"a": "(a) LightGBM, 18 đặc trưng", "b": "(b) GAT đơn", "c": "(c) LightGBM + trung bình hàng xóm (tính tay)",
                 "d": "(d) LightGBM chỉ embedding GAT", "e": "(e) Hybrid đầy đủ"}
MAIN_VALUE = {"heads": 4, "layers": 2, "hidden": 16, "edge": "yes", "window": 5}
FACTORS = ["heads", "layers", "hidden", "edge", "window"]
FACTOR_TITLES = {"heads": "Số head", "layers": "Số lớp GAT", "hidden": "Độ rộng mỗi head", "edge": "Đặc trưng cạnh",
                 "window": "Kích thước cửa sổ (giây)"}

# màu: một sắc xanh cho số liệu chính, xám cho tham chiếu, cam nhấn mạnh cấu hình chính (Okabe-Ito, an toàn cho mù màu)
BLUE, ORANGE, GRAY, RED = "#0072B2", "#E69F00", "#8c8c8c", "#D55E00"
VARIANT_COLORS = {"gat5": "#0072B2", "gat3": "#E69F00", "gat2": "#009E73", "gat1": "#D55E00", "gatfull": "#CC79A7"}


def clean(o):
    """Đổi NaN, inf, numpy thành kiểu JSON an toàn."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def ms(series):
    """(mean, std ddof=1; 0 khi một giá trị hoặc các giá trị y hệt)."""
    v = np.asarray(series, dtype=float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return float(v.mean()), (0.0 if sd < 1e-12 else sd)


def read_val(path, dataset):
    df = pd.read_csv(path)
    return df[(df["dataset"] == dataset) & (df["split"] == "val") & (df["task"] == "multiclass")]


def beats(diff, *stds):
    """diff > 2 · max(std) (cùng quy ước scripts/compare_val.py)."""
    spread = max(stds)
    return bool(diff > 2 * spread) if spread > 0 else bool(diff > 0)


# ------------------------------------------------------------------ 5.8 ablation hybrid
def ablation_hybrid(ds, repo, cd_dir):
    paths = {k: Path(repo) / "reports" / "kaggle" / v for k, v in OLD[ds].items()}
    cd = read_val(Path(cd_dir) / f"ablation_cd_{ds}.csv", ds)
    src = {"a": (read_val(paths["base"], ds), "lightgbm", "GĐ D"), "b": (read_val(paths["gnn"], ds), "gat", "GĐ A/E"),
           "c": (cd, "abl_lgbm_x_nbrmean", "16a/16d"), "d": (cd, "abl_lgbm_emb_only", "16a/16d"),
           "e": (read_val(paths["hybrid"], ds), HYBRID_FULL, "GĐ F (15e/15f)")}
    rows = []
    for key, (df, model, origin) in src.items():
        d = df[df["model"] == model].sort_values("seed")
        assert sorted(d["seed"]) == list(SEEDS), f"{ds} {key} {model}: seed {sorted(d['seed'])}"
        row = {"dataset": ds, "variant": key, "description": VARIANT_NAMES[key], "source_model": model, "origin": origin, "n_seeds": len(d)}
        for m in ("f1_macro", "fpr", "recall_attack", "accuracy"):
            row[f"{m}_mean"], row[f"{m}_std"] = ms(d[m])
        for s, v in zip(d["seed"], d["f1_macro"]):
            row[f"f1_macro_s{s}"] = float(v)
        row["n_params"], row["size_kb"] = int(d["n_params"].max()), float(d["size_kb"].max())
        rows.append(row)
    out = pd.DataFrame(rows)
    e, c = out.set_index("variant").loc["e"], out.set_index("variant").loc["c"]
    diff = e["f1_macro_mean"] - c["f1_macro_mean"]
    concl = {"e_minus_c_f1": diff, "e_beats_c_2std": beats(diff, e["f1_macro_std"], c["f1_macro_std"]),
             "e_minus_c_fpr": e["fpr_mean"] - c["fpr_mean"],
             "e_minus_a_f1": e["f1_macro_mean"] - out.set_index("variant").loc["a"]["f1_macro_mean"],
             "e_minus_d_f1": e["f1_macro_mean"] - out.set_index("variant").loc["d"]["f1_macro_mean"],
             "d_minus_c_f1": out.set_index("variant").loc["d"]["f1_macro_mean"] - c["f1_macro_mean"]}
    return out, concl


# ------------------------------------------------------------------ 5.8 Pareto hybrid nhẹ
def pareto(ds, hyb_dir):
    raw = pd.read_csv(Path(hyb_dir) / f"pareto_raw_{ds}.csv")
    rows = []
    for (variant, n, l), g in raw.groupby(["variant", "lgbm_n", "lgbm_l"]):
        assert sorted(g["seed"]) == list(SEEDS), f"{ds} {variant} {n}/{l}: seed {sorted(g['seed'])}"
        row = {"dataset": ds, "variant": variant, "n_gat": int(g["n_gat"].iloc[0]), "lgbm_n": int(n), "lgbm_l": int(l),
               "is_hybrid_full": bool(g["is_hybrid_full"].any())}
        for m in ("f1_macro", "fpr", "recall_attack", "accuracy", "thr_f1_macro", "thr_fpr", "thr_recall_attack"):
            row[f"{m}_mean"], row[f"{m}_std"] = ms(g[m])
        row["n_params"], row["size_kb"], row["p95_ms"] = int(g["n_params"].max()), float(g["size_kb"].max()), float(g["p95_ms"].max())
        row["meets_params"], row["meets_size"] = row["n_params"] <= PARAM_LIMIT, row["size_kb"] <= SIZE_LIMIT_KB
        row["meets_both"] = row["meets_params"] and row["meets_size"]
        rows.append(row)
    out = pd.DataFrame(rows)
    full = out[out["is_hybrid_full"]].iloc[0]
    out["f1_macro_diff_vs_full"] = out["f1_macro_mean"] - full["f1_macro_mean"]
    out["fpr_diff_vs_full"] = out["fpr_mean"] - full["fpr_mean"]
    out["thr_recall_diff_vs_full"] = out["thr_recall_attack_mean"] - full["thr_recall_attack_mean"]
    # điểm không bị lấn át: không có điểm nào vừa nhỏ hơn (hoặc bằng) vừa F1 cao hơn (hoặc bằng) và hơn ở một trục
    front = []
    for i, r in out.iterrows():
        dom = ((out["size_kb"] <= r["size_kb"]) & (out["f1_macro_mean"] >= r["f1_macro_mean"])
               & ((out["size_kb"] < r["size_kb"]) | (out["f1_macro_mean"] > r["f1_macro_mean"])))
        front.append(not bool(dom.any()))
    out["pareto_front"] = front
    out = out.sort_values(["size_kb", "f1_macro_mean"], ascending=[True, False]).reset_index(drop=True)
    ok = out[out["meets_both"]]
    choice = {"dataset": ds, "limits": {"n_params": PARAM_LIMIT, "size_kb": SIZE_LIMIT_KB},
              "hybrid_full": full.to_dict(), "n_points": len(out), "n_points_meeting_both": int(len(ok))}
    if len(ok):
        best = ok.sort_values(["f1_macro_mean", "size_kb"], ascending=[False, True]).iloc[0]
        small = ok.sort_values(["size_kb", "f1_macro_mean"], ascending=[True, False]).iloc[0]
        choice["light_best_f1"], choice["light_smallest"] = best.to_dict(), small.to_dict()
    else:
        choice["light_best_f1"] = choice["light_smallest"] = None
    return out, raw, choice


# ------------------------------------------------------------------ 5.5 ablation GAT
def gat_ablation(grid_dir, win_dir):
    g = pd.read_csv(Path(grid_dir) / "grid_nf_ton_iot_v3.csv")
    g = g[g["part"] == "arch"].copy()
    parts = [g]
    for w in (1, 2, 10):
        p = Path(win_dir) / f"grid_nf_ton_iot_v3_w{w}.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
    allg = pd.concat(parts, ignore_index=True)
    main = allg[allg["factor"] == "default"]
    rows = []
    for factor in FACTORS:
        base = main.copy()
        base["value"] = MAIN_VALUE[factor]
        if factor == "window":
            base["window_sec"] = 5
        sub = pd.concat([base, allg[allg["factor"] == factor]], ignore_index=True)
        sub["value"] = sub["value"].astype(str)
        order = sorted(sub["value"].unique(), key=lambda v: (0, -1) if v == "yes" else ((1, 0) if v == "no" else (0, float(v))))
        for value in order:
            d = sub[sub["value"] == value]
            r = {"factor": factor, "value": value, "is_main": value == str(MAIN_VALUE[factor]), "n_seeds": d["seed"].nunique()}
            for m in ("f1_macro", "fpr", "recall_attack", "recall_botnet", "accuracy", "epochs", "train_s"):
                r[f"{m}_mean"], r[f"{m}_std"] = ms(d[m].dropna()) if d[m].notna().any() else (float("nan"), float("nan"))
            r["n_params"], r["size_kb"], r["p95_ms"] = int(d["n_params"].max()), float(d["size_kb"].max()), float(d["p95_ms"].mean())
            rows.append(r)
    out = pd.DataFrame(rows)
    concl = {}
    for factor in FACTORS:
        d = out[out["factor"] == factor]
        if d.empty:
            continue
        m = d[d["is_main"]].iloc[0]
        best = d.sort_values("f1_macro_mean", ascending=False).iloc[0]
        gap = best["f1_macro_mean"] - m["f1_macro_mean"]
        cheaper = d[(d["n_params"] < m["n_params"]) & (m["f1_macro_mean"] - d["f1_macro_mean"] <= 0.01)]
        concl[factor] = {
            "values": d["value"].tolist(), "main_value": m["value"], "main_f1": m["f1_macro_mean"], "main_f1_std": m["f1_macro_std"],
            "best_value": best["value"], "best_f1": best["f1_macro_mean"], "gap_best_minus_main": gap,
            "main_within_2std_of_best": bool(not beats(gap, m["f1_macro_std"], best["f1_macro_std"])),
            "cheaper_within_1pt": cheaper[["value", "n_params", "f1_macro_mean"]].to_dict("records"),
        }
    return out, concl


# ------------------------------------------------------------------ 5.4 bản cải tiến
def improved(grid_dir):
    g = pd.read_csv(Path(grid_dir) / "grid_nf_ton_iot_v3.csv")
    base = g[g["cfg_id"] == "gat_default"]
    rows = []
    for cfg, d in pd.concat([base, g[g["part"] == "improved"]]).groupby("cfg_id", sort=False):
        r = {"cfg_id": cfg, "description": d["value"].iloc[0], "n_seeds": d["seed"].nunique()}
        for m in ("f1_macro", "fpr", "recall_attack", "recall_botnet", "accuracy"):
            r[f"{m}_mean"], r[f"{m}_std"] = ms(d[m])
        r["n_params"], r["p95_ms"], r["size_kb"] = int(d["n_params"].max()), float(d["p95_ms"].mean()), float(d["size_kb"].max())
        rows.append(r)
    out = pd.DataFrame(rows)
    b = out[out["cfg_id"] == "gat_default"].iloc[0]
    notes, acc, c1l, c2l = [], [], [], []
    for _, r in out.iterrows():
        if r["cfg_id"] == "gat_default":
            notes.append("cấu hình chính (mốc so sánh)")
            acc.append(None), c1l.append(None), c2l.append(None)
            continue
        df1 = r["f1_macro_mean"] - b["f1_macro_mean"]
        drb = r["recall_botnet_mean"] - b["recall_botnet_mean"]
        c1_f1 = beats(df1, r["f1_macro_std"], b["f1_macro_std"])
        c1_rb = beats(drb, r["recall_botnet_std"], b["recall_botnet_std"])
        red_p = 1 - r["n_params"] / b["n_params"]
        red_l = 1 - r["p95_ms"] / b["p95_ms"]
        c2 = (red_p >= 0.30 or red_l >= 0.30) and (-df1 <= 0.01)
        c1, a = bool(c1_f1 or c1_rb), None
        a = bool(c1 or c2)
        notes.append(f"ΔF1-macro {df1:+.4f} (>2 std: {c1_f1}); Δrecall Botnet_Backdoor {drb:+.4f} (>2 std: {c1_rb}); "
                     f"tham số {red_p:+.1%} giảm, độ trễ {red_l:+.1%} giảm, F1 giảm {-df1:+.4f} (tiêu chí 2: {c2})")
        acc.append(a), c1l.append(c1), c2l.append(bool(c2))
    out["criterion1_met"], out["criterion2_met"], out["accepted"], out["note"] = c1l, c2l, acc, notes
    return out


# ------------------------------------------------------------------ thresholds.json
def merge_thresholds(repo_unused, bot_dir, ton_thr_dir, hyb_dirs, choices, grid_dir, accepted_improved):
    def flat(c):
        return {"threshold": c["threshold"], "feasible": c["feasible"], "meets_recall_target": c["meets_recall_target"],
                "n_val_normal": c["n_val_normal"], "n_val_attack": c["n_val_attack"],
                "f1_macro_before": c["before"]["f1_macro"], "fpr_before": c["before"]["fpr"],
                "recall_attack_before": c["before"]["recall_attack"], "accuracy_before": c["before"]["accuracy"],
                "f1_macro_after": c["after"]["f1_macro"], "fpr_after": c["after"]["fpr"],
                "recall_attack_after": c["after"]["recall_attack"], "accuracy_after": c["after"]["accuracy"]}

    def from_pareto_row(r):
        return {"threshold": r["thr"], "feasible": bool(r["thr_feasible"]), "meets_recall_target": bool(r["thr_meets_recall"]),
                "f1_macro_before": r["f1_macro"], "fpr_before": r["fpr"], "recall_attack_before": r["recall_attack"],
                "accuracy_before": r["accuracy"], "f1_macro_after": r["thr_f1_macro"], "fpr_after": r["thr_fpr"],
                "recall_attack_after": r["thr_recall_attack"], "accuracy_after": r["thr_accuracy"]}

    out = {"_meta": {"rule": "ngưỡng nhỏ nhất sao cho FPR val ≤ 4,08%; điểm tấn công = 1 - P(Normal); điểm > ngưỡng thì lớp tấn công có "
                             "xác suất cao nhất, ngược lại Normal; chỉ chọn trên val, giữ nguyên khi chạy test (GĐ H)",
                     "limit": TH.LIMIT, "recall_target": TH.RECALL_TARGET, "classes_note": "Normal là lớp 0"}}
    missing = []
    for ds in DATASETS:
        models = {}
        src = Path(bot_dir if ds == "nf_bot_iot_v3" else ton_thr_dir) / f"thresholds_base_gnn_{ds}.json"
        if src.exists():
            for model, by_seed in json.loads(src.read_text(encoding="utf-8"))[ds].items():
                models[model] = {s: flat(c) for s, c in by_seed.items()}
        else:
            missing.append(str(src))
        raw = pd.read_csv(Path(hyb_dirs[ds]) / f"pareto_raw_{ds}.csv")
        full = raw[raw["is_hybrid_full"]]
        models[HYBRID_FULL] = {str(r["seed"]): from_pareto_row(r) for _, r in full.iterrows()}
        light = choices[ds]["light_best_f1"]
        if light is not None:
            sel = raw[(raw["variant"] == light["variant"]) & (raw["lgbm_n"] == light["lgbm_n"]) & (raw["lgbm_l"] == light["lgbm_l"])]
            models["hybrid_light"] = {str(r["seed"]): {**from_pareto_row(r), "config": {
                "variant": light["variant"], "n_gat": int(light["n_gat"]), "lgbm_n": int(light["lgbm_n"]), "lgbm_l": int(light["lgbm_l"])}}
                for _, r in sel.iterrows()}
        out[ds] = models
    if grid_dir:
        p = Path(grid_dir) / "thresholds_grid_nf_ton_iot_v3.json"
        if p.exists():
            grid = json.loads(p.read_text(encoding="utf-8"))["nf_ton_iot_v3"]
            for cfg in ("gat_focal", "gat_resln", "gat_focal_resln"):
                if cfg in grid:
                    out["nf_ton_iot_v3"][f"{cfg}"] = {s: {**flat(c), "accepted_by_5_4": bool(accepted_improved.get(cfg))}
                                                      for s, c in grid[cfg].items()}
    return out, missing


# ------------------------------------------------------------------ hình
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.axisbelow": True, "figure.dpi": 100})
    return plt


def fig_gat_factor(factor, d, out_path, plt):
    d = d.reset_index(drop=True)
    x = np.arange(len(d))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    cols = [ORANGE if m else BLUE for m in d["is_main"]]
    a1.bar(x, d["f1_macro_mean"], yerr=d["f1_macro_std"], color=cols, capsize=4, width=0.6)
    lo = max(0.0, float((d["f1_macro_mean"] - d["f1_macro_std"]).min()) - 0.03)
    a1.set_ylim(lo, min(1.0, float((d["f1_macro_mean"] + d["f1_macro_std"]).max()) + 0.015))
    for xi, m, s in zip(x, d["f1_macro_mean"], d["f1_macro_std"]):
        a1.text(xi, m + s + 0.002, f"{m:.4f}", ha="center", va="bottom", fontsize=8)
    a1.set_ylabel("F1-macro trên val (trung bình 3 seed ± std)")
    a1.set_title(f"Ablation GAT, NF-ToN-IoT-v3: {FACTOR_TITLES[factor]} (cam = cấu hình chính)")
    if factor == "window":
        a2.bar(x, d["p95_ms"], color=GRAY, width=0.6)
        a2.set_ylabel("Độ trễ p95 (ms/cửa sổ)")
        for xi, v in zip(x, d["p95_ms"]):
            a2.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    else:
        a2.bar(x, d["n_params"], color=GRAY, width=0.6)
        a2.set_ylabel("Số tham số")
        for xi, v in zip(x, d["n_params"]):
            a2.text(xi, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    a2.set_xticks(x, d["value"].astype(str))
    a2.set_xlabel(FACTOR_TITLES[factor])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_ablation_hybrid(ds, d, out_path, plt):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(d))
    cols = [BLUE if v == "e" else GRAY for v in d["variant"]]
    for ax, m, title in ((a1, "f1_macro", "F1-macro (cao hơn tốt hơn)"), (a2, "fpr", "FPR (thấp hơn tốt hơn)")):
        ax.bar(x, d[f"{m}_mean"], yerr=d[f"{m}_std"], color=cols, capsize=4, width=0.6)
        for xi, v, s in zip(x, d[f"{m}_mean"], d[f"{m}_std"]):
            ax.text(xi, v + s, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, [VARIANT_NAMES[v].split(" ", 1)[0] for v in d["variant"]])
        ax.set_title(title)
    lo = float((d["f1_macro_mean"] - d["f1_macro_std"]).min()) - 0.03
    a1.set_ylim(max(0, lo), 1.0)
    a2.axhline(0.0408, color=RED, ls="--", lw=1)
    a2.text(len(d) - 0.5, 0.0408, " ngưỡng 4,08%", color=RED, va="bottom", ha="right", fontsize=8)
    fig.suptitle(f"Ablation hybrid trên val, {ds} (3 seed). " + "; ".join(VARIANT_NAMES[v] for v in d["variant"]), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_pareto(ds, p, out_path, plt):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, xcol, lim, xlabel in ((a1, "size_kb", SIZE_LIMIT_KB, "Kích thước file mô hình (KB, thang log)"),
                                  (a2, "n_params", PARAM_LIMIT, "Số tham số (GAT + lá LightGBM, thang log)")):
        for v, g in p.groupby("variant"):
            ax.errorbar(g[xcol], g["f1_macro_mean"], yerr=g["f1_macro_std"], fmt="o", ms=5, color=VARIANT_COLORS.get(v, GRAY),
                        ecolor=VARIANT_COLORS.get(v, GRAY), elinewidth=1, capsize=2, alpha=0.9, label=v if ax is a1 else None)
        full = p[p["is_hybrid_full"]]
        ax.scatter(full[xcol], full["f1_macro_mean"], s=150, facecolors="none", edgecolors="black", linewidths=1.4, zorder=5,
                   label="hybrid đầy đủ" if ax is a1 else None)
        fr = p[p["pareto_front"]].sort_values(xcol)
        ax.step(fr[xcol], fr["f1_macro_mean"], where="post", color="black", lw=0.8, alpha=0.5)
        ax.axvline(lim, color=RED, ls="--", lw=1)
        ax.text(lim, ax.get_ylim()[0], " ngưỡng nhẹ", color=RED, rotation=90, va="bottom", ha="left", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
    a1.set_ylabel("F1-macro trên val (3 seed ± std)")
    a1.legend(title="GAT dùng khi suy luận", fontsize=8, loc="lower right")
    fig.suptitle(f"Đường Pareto hybrid nhẹ, {ds}: mỗi điểm là một (số GAT, n_estimators, num_leaves); đường đen = biên Pareto", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_compare(ds, rows, out_path, plt):
    """Cột F1-macro và FPR (trước và sau hiệu chỉnh ngưỡng) cho mọi mô hình; vạch 4,08%."""
    rows = rows.reset_index(drop=True)
    y = np.arange(len(rows))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 0.45 * len(rows) + 2.2), sharey=True)
    a1.barh(y, rows["f1_macro_mean"], xerr=rows["f1_macro_std"], color=[BLUE if str(m).startswith("hybrid") else GRAY for m in rows["model"]],
            capsize=3, height=0.6)
    a1.set_xlabel("F1-macro trên val")
    a1.set_xlim(max(0, float(rows["f1_macro_mean"].min()) - 0.08), 1.0)
    h = 0.36
    a2.barh(y - h / 2, rows["fpr_mean"], xerr=rows["fpr_std"], height=h, color=GRAY, capsize=2, label="trước hiệu chỉnh (argmax)")
    a2.barh(y + h / 2, rows["fpr_after_mean"], xerr=rows["fpr_after_std"], height=h, color=BLUE, capsize=2, label="sau hiệu chỉnh ngưỡng")
    a2.axvline(0.0408, color=RED, ls="--", lw=1)
    a2.text(0.0408, -0.9, " 4,08%", color=RED, fontsize=8, va="bottom")
    a2.set_xlabel("FPR trên val")
    a2.legend(fontsize=8, loc="lower right")
    a1.set_yticks(y, rows["model"])
    a1.invert_yaxis()
    fig.suptitle(f"So sánh trên val, {ds} (3 seed ± std)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ chính
def run(out_dir, repo, bot, ton_hybrid, ton_thr, ton_tuning, ton_grid, windows):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    hyb_dirs = {"nf_ton_iot_v3": ton_hybrid, "nf_bot_iot_v3": bot}
    summary, missing, choices, par_tables = {}, [], {}, {}

    for ds in DATASETS:
        d = hyb_dirs[ds]
        abl, concl = ablation_hybrid(ds, repo, d)
        abl.to_csv(out / f"ablation_hybrid_{ds}.csv", index=False)
        fig_ablation_hybrid(ds, abl, out / f"gd5_ablation_hybrid_{ds}.png", plt)
        par, raw, choice = pareto(ds, d)
        par.to_csv(out / f"pareto_hybrid_{ds}.csv", index=False)
        fig_pareto(ds, par, out / f"gd5_pareto_{ds}.png", plt)
        choices[ds] = choice
        par_tables[ds] = par
        (out / f"light_choice_{ds}.json").write_text(json.dumps(clean(choice), ensure_ascii=False, indent=1), encoding="utf-8")
        summary[ds] = {"ablation_hybrid": concl}
        print(f"[{ds}] ablation hybrid: {json.dumps(clean(concl), ensure_ascii=False)}")
        lb = choice["light_best_f1"]
        if lb:
            print(f"[{ds}] hybrid nhẹ (F1 cao nhất trong ≤ 100K tham số và ≤ 1 MB): {lb['variant']} n={lb['lgbm_n']} l={lb['lgbm_l']} "
                  f"F1 {lb['f1_macro_mean']:.4f} (so với đầy đủ {lb['f1_macro_diff_vs_full']:+.4f}), FPR {lb['fpr_mean']:.4f} "
                  f"({lb['fpr_diff_vs_full']:+.4f}), {lb['n_params']:,} tham số, {lb['size_kb']:.0f} KB")
        else:
            print(f"[{ds}] KHÔNG có điểm Pareto nào đạt cả ≤ 100K tham số và ≤ 1 MB")

    # 5.5 / 5.2 / 5.4 (cần notebook 16, 16f, 16g, 16a)
    accepted = {}
    if ton_grid:
        gat, concl = gat_ablation(ton_grid, windows or "")
        gat.to_csv(out / "ablation_gat.csv", index=False)
        (out / "ablation_gat_conclusion.json").write_text(json.dumps(clean(concl), ensure_ascii=False, indent=1), encoding="utf-8")
        for f in FACTORS:
            if f in concl:
                fig_gat_factor(f, gat[gat["factor"] == f], out / f"gd5_ablation_gat_{f}.png", plt)
        summary["ablation_gat"] = concl
        imp = improved(ton_grid)
        imp.to_csv(out / "improved_gat.csv", index=False)
        accepted = {r["cfg_id"]: r["accepted"] for _, r in imp.iterrows() if r["accepted"] is not None}
        summary["improved_gat"] = imp[["cfg_id", "f1_macro_mean", "f1_macro_std", "recall_botnet_mean", "n_params", "criterion1_met",
                                       "criterion2_met", "accepted", "note"]].to_dict("records")
        print(imp[["cfg_id", "f1_macro_mean", "f1_macro_std", "recall_botnet_mean", "n_params", "accepted"]].to_string())
    else:
        missing.append("ton_grid (notebook 16): ablation_gat, improved_gat")
    frames = []
    if ton_grid:
        g = pd.read_csv(Path(ton_grid) / "grid_nf_ton_iot_v3.csv")
        frames.append(g[g["part"] == "arch"])
    if ton_tuning:
        frames.append(pd.read_csv(Path(ton_tuning) / "grid_nf_ton_iot_v3.csv"))
    else:
        missing.append("ton_tuning (notebook 16g): tuning GCN/GraphSAGE ToN")
    if bot and (Path(bot) / "grid_nf_bot_iot_v3.csv").exists():
        frames.append(pd.read_csv(Path(bot) / "grid_nf_bot_iot_v3.csv"))
    if frames:
        tun = pd.concat(frames, ignore_index=True)
        tun.to_csv(out / "tuning_gnn.csv", index=False)
        s = tun.groupby(["dataset", "kind", "cfg_id"])[["f1_macro", "fpr", "recall_attack", "n_params"]].mean().reset_index()
        summary["tuning_gnn"] = {"trials_per_model": {f"{d}|{k}": int(n) for (d, k), n in tun.groupby(["dataset", "kind"])["cfg_id"].nunique().items()},
                                 "best_per_model": {f"{d}|{k}": g.sort_values("f1_macro", ascending=False).iloc[0][["cfg_id", "f1_macro", "n_params"]].to_dict()
                                                    for (d, k), g in s.groupby(["dataset", "kind"])}}
        print(json.dumps(clean(summary["tuning_gnn"]), ensure_ascii=False, indent=1))

    thr, miss2 = merge_thresholds(None, bot, ton_thr, hyb_dirs, choices, ton_grid, accepted)
    missing += miss2
    (out / "thresholds.json").write_text(json.dumps(clean(thr), ensure_ascii=False, indent=1), encoding="utf-8")

    # so sánh với mọi mô hình (bảng cho hình gd5_compare_*): F1, FPR trước và sau hiệu chỉnh
    for ds in DATASETS:
        paths = {k: Path(repo) / "reports" / "kaggle" / v for k, v in OLD[ds].items()}
        rows = []
        for key in ("base", "gnn", "hybrid"):
            df = read_val(paths[key], ds)
            for model, d in df.groupby("model"):
                if key == "hybrid" and model != HYBRID_FULL:
                    continue
                r = {"model": model, "f1_macro_mean": ms(d["f1_macro"])[0], "f1_macro_std": ms(d["f1_macro"])[1],
                     "fpr_mean": ms(d["fpr"])[0], "fpr_std": ms(d["fpr"])[1]}
                t = thr[ds].get(model)
                aft = [v["fpr_after"] for v in t.values()] if t else [float("nan")]
                r["fpr_after_mean"], r["fpr_after_std"] = ms(aft)
                rows.append(r)
        light = thr[ds].get("hybrid_light")
        lb = choices[ds]["light_best_f1"]
        if light and lb:
            r = {"model": "hybrid_light", "f1_macro_mean": lb["f1_macro_mean"], "f1_macro_std": lb["f1_macro_std"], "fpr_mean": lb["fpr_mean"],
                 "fpr_std": lb["fpr_std"]}
            r["fpr_after_mean"], r["fpr_after_std"] = ms([v["fpr_after"] for v in light.values()])
            rows.append(r)
        tab = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
        tab.to_csv(out / f"compare_models_{ds}.csv", index=False)
        fig_compare(ds, tab, out / f"gd5_compare_{ds}.png", plt)
        ok = {}
        for model, by_seed in thr[ds].items():
            if model == "_meta":
                continue
            vals = [v["fpr_after"] for v in by_seed.values()]
            ok[model] = {"fpr_after_mean": float(np.mean(vals)), "recall_after_mean": float(np.mean([v["recall_attack_after"] for v in by_seed.values()])),
                         "feasible_all_seeds": all(v["feasible"] for v in by_seed.values()),
                         "meets_recall_target_all_seeds": all(v["meets_recall_target"] for v in by_seed.values())}
        summary[ds]["thresholds"] = ok
    summary["missing"] = missing
    (out / "aggregate_summary.json").write_text(json.dumps(clean(summary), ensure_ascii=False, indent=1), encoding="utf-8")
    print("missing:", missing)
    return summary


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pos = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    run(pos[0], opt["repo"], opt["bot"], opt["ton_hybrid"], opt["ton_thr"], opt.get("ton_tuning"), opt.get("ton_grid"), opt.get("windows"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
