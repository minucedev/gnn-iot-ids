"""Phân tích FPR trên val (task 5.3, tiêu chí 2): node Normal nào bị đoán là tấn công và vì sao.

Chạy:  python -I scripts/fpr_analysis.py <data_dir> <dataset> <flows.csv> <baseline_search.json> <gat_ckpt.pt> <out_dir>

Mô hình: LightGBM (cấu hình tốt nhất trong baseline_search, seed 42, huấn luyện lại trên train) và GAT (checkpoint
seed 42 của task 5.3). Chỉ dùng train và val, không đọc test. Trả lời 4 câu: (1) IP nào, có xuất hiện như node tấn
công ở cửa sổ khác không; (2) độ thuần nhãn, nhãn lấy từ flow ra hay flow nhận; (3) đặc trưng gần tâm lớp nào;
(4) cách gán nhãn khác (phản thực tế trên dự đoán đã có, KHÔNG huấn luyện lại, KHÔNG đổi nhãn của dữ liệu).
Ghi <out_dir>/fpr_analysis_<dataset>.json và gd5_fpr_<dataset>_{ip,purity,centroid,relabel}.png.
Cần dựng lại nhãn flow từ CSV gốc (đồ thị không lưu nhãn flow) nên đọc CSV gốc của bộ dữ liệu.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build_graph as bg  # noqa: E402
from src import data as D  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import fpr as F  # noqa: E402
from src import models as M  # noqa: E402
from src import train as T  # noqa: E402

SEED = 42


def clean(o):
    """Đổi kiểu numpy sang kiểu Python, NaN/inf thành None để JSON hợp lệ."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def lightgbm_cfg(search_json):
    log = json.loads(Path(search_json).read_text(encoding="utf-8"))["lightgbm"]
    best = next(e["best"] for e in log if "best" in e)
    return int(best["n"]), int(best["l"])


def fit_lightgbm(Xtr, ytr, n_classes, n_est, leaves):
    w = D.class_weights(ytr, n_classes)
    m = LGBMClassifier(n_estimators=n_est, num_leaves=leaves, learning_rate=0.1, n_jobs=-1, random_state=SEED,
                       verbose=-1)
    return m.fit(Xtr, ytr, sample_weight=w[ytr])


def group_purity(df, n_bins=10):
    """Thống kê độ thuần cho một nhóm node (bảng `purity_table` đã lọc)."""
    if len(df) == 0:
        return {"n": 0}
    via_out = df[~df["from_in_only"] & (df["n_in"] > 0)]
    hist = np.histogram(df["all_normal_frac"], bins=n_bins, range=(0, 1))[0]
    return {
        "n": int(len(df)),
        "share_label_from_in_only": float(df["from_in_only"].mean()),
        "mean_n_out": float(df["n_out"].mean()), "median_n_out": float(df["n_out"].median()),
        "mean_n_in": float(df["n_in"].mean()), "median_n_in": float(df["n_in"].median()),
        "mean_all_normal_frac": float(df["all_normal_frac"].mean()),
        "median_all_normal_frac": float(df["all_normal_frac"].median()),
        "share_all_flows_normal": float((df["all_normal_frac"] == 1.0).mean()),
        "share_less_than_half_flows_normal": float((df["all_normal_frac"] < 0.5).mean()),
        "mean_label_purity": float(df["purity"].mean()),
        "share_label_via_out_but_in_flows_mostly_attack":
            float((via_out["in_normal_frac"] < 0.5).mean()) if len(via_out) else None,
        "hist_all_normal_frac": [int(v) for v in hist],
    }


def group_centroid(dist, mask, classes):
    """Lớp có tâm gần nhất của các node trong nhóm, và khoảng cách trung bình tới tâm Normal / tâm tấn công gần nhất."""
    d = dist[mask]
    if len(d) == 0:
        return {"n": 0}
    near = d.argmin(1)
    att = d[:, 1:].min(1)
    return {
        "n": int(len(d)),
        "nearest_share": {c: float((near == k).mean()) for k, c in enumerate(classes)},
        "mean_dist_to_normal": float(d[:, 0].mean()),
        "mean_dist_to_nearest_attack": float(att.mean()),
        "share_closer_to_normal_than_any_attack": float((d[:, 0] < att).mean()),
    }


def analyze_model(pred, y, nodes_val, ref_nodes, pur, dist, rel, classes):
    K = len(classes)
    fp, tn = (y == 0) & (pred != 0), (y == 0) & (pred == 0)
    res = {"rates": F.binary_rates(y, pred),
           "fp_predicted_as": {classes[k]: int(((pred == k) & fp).sum()) for k in range(1, K)},
           "ip": F.ip_report(nodes_val, fp, ref_nodes),
           "purity": {"fp": group_purity(pur[fp]), "tn": group_purity(pur[tn])},
           "centroid": {"fp": group_centroid(dist, fp, classes), "tn": group_centroid(dist, tn, classes)}}
    rl = {}
    for rule, lab in rel.items():
        keep = lab >= 0
        rates = F.binary_rates(lab, pred, keep)
        rl[rule] = {"desc": F.RULE_DESC[rule], **rates,
                    "n_nodes_kept": int(keep.sum()), "n_dropped": int((~keep).sum()),
                    "n_label_changed": int((keep & (lab != y)).sum()),
                    "share_label_changed": float((keep & (lab != y)).sum() / len(y)),
                    "label_counts": {classes[k]: int((lab == k).sum()) for k in range(K)}}
    res["relabel"] = rl
    return res


# ---------------------------------------------------------------- hình
def plot_all(out_dir, ds, results, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    col = {"fp": "#d1495b", "tn": "#2e86ab"}

    # 1. FP theo IP
    fig, axes = plt.subplots(1, len(names), figsize=(6.2 * len(names), 4.6), squeeze=False)
    for ax, n in zip(axes[0], names):
        top = results[n]["ip"]["top"][:10][::-1]
        labels = [f"{r['ip']}" for r in top]
        ax.barh(labels, [r["fp"] for r in top], color=col["fp"])
        for i, r in enumerate(top):
            ax.text(r["fp"], i, f" {r['fp']}/{r['val_normal_nodes']} node Normal; tấn công ở cửa sổ khác: {r['ref_attack_nodes']}",
                    va="center", fontsize=7)
        ax.set(title=f"{n}: 10 IP có nhiều FP nhất (tổng {results[n]['ip']['total_fp']} FP)", xlabel="số node Normal bị đoán là tấn công")
        ax.margins(x=0.75)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"{ds}: node Normal bị báo nhầm theo IP (val)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"gd5_fpr_{ds}_ip.png", dpi=130)
    plt.close(fig)

    # 2. độ thuần: tỉ lệ flow Normal của node
    fig, axes = plt.subplots(1, len(names), figsize=(5.6 * len(names), 4.2), squeeze=False)
    xs = np.arange(10)
    for ax, n in zip(axes[0], names):
        for j, (g, lab) in enumerate((("fp", "Normal bị đoán tấn công (FP)"), ("tn", "Normal đoán đúng (TN)"))):
            h = np.array(results[n]["purity"][g].get("hist_all_normal_frac", [0] * 10), float)
            ax.bar(xs + (j - 0.5) * 0.4, h / max(h.sum(), 1), width=0.4, color=col[g], label=f"{lab}, n = {int(h.sum())}")
        ax.set_xticks(xs, [f"{i * 10}–{i * 10 + 10}" for i in xs], rotation=40, fontsize=8)
        ax.set(title=n, xlabel="% flow của node mang nhãn Normal (ra + nhận)", ylabel="tỉ lệ node trong nhóm")
        ax.legend(fontsize=8)
    fig.suptitle(f"{ds}: độ thuần nhãn của node Normal (val)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"gd5_fpr_{ds}_purity.png", dpi=130)
    plt.close(fig)

    # 3. tâm lớp gần nhất
    fig, axes = plt.subplots(1, len(names), figsize=(5.6 * len(names), 4.2), squeeze=False)
    xs = np.arange(len(classes))
    for ax, n in zip(axes[0], names):
        for j, (g, lab) in enumerate((("fp", "FP"), ("tn", "TN"))):
            c = results[n]["centroid"][g]
            v = [c.get("nearest_share", {}).get(k, 0.0) for k in classes]
            ax.bar(xs + (j - 0.5) * 0.4, v, width=0.4, color=col[g], label=f"{lab}, n = {c['n']}")
        ax.set_xticks(xs, classes, rotation=25, ha="right", fontsize=8)
        ax.set(title=n, ylabel="tỉ lệ node có tâm lớp này gần nhất", ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.suptitle(f"{ds}: tâm lớp (trung bình đặc trưng train) gần nhất với node Normal (val)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"gd5_fpr_{ds}_centroid.png", dpi=130)
    plt.close(fig)

    # 4. cách gán nhãn khác
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    rules = list(F.RULES)
    xs = np.arange(len(rules))
    for j, n in enumerate(names):
        v = [results[n]["relabel"][r]["fpr"] * 100 for r in rules]
        b = ax.bar(xs + (j - (len(names) - 1) / 2) * 0.38, v, width=0.38, label=n)
        for rect, val in zip(b, v):
            ax.text(rect.get_x() + rect.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, [f"{r}\n(đổi nhãn {results[names[0]]['relabel'][r]['share_label_changed'] * 100:.1f}% node)" for r in rules], fontsize=8)
    ax.set(ylabel="FPR (%) của dự đoán đã có", title=f"{ds}: FPR nếu gán nhãn node theo cách khác (phản thực tế, không huấn luyện lại)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"gd5_fpr_{ds}_relabel.png", dpi=130)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data_dir, ds, flows_csv, search_json, gat_ckpt, out_dir = sys.argv[1:7]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    graphs, meta = D.load_split(data_dir)
    classes, K = meta["classes"], len(meta["classes"])
    assert meta["window_sec"] == meta["stride_sec"], "Phân tích này giả định cửa sổ không chồng (W = S)"
    train_g, val_g = graphs["train"], graphs["val"]
    Xtr, ytr = D.node_arrays(train_g)
    Xva, yva = D.node_arrays(val_g)
    nodes_val = F.node_table(val_g)
    ref_nodes = F.node_table(train_g + val_g)
    assert (nodes_val["y"].to_numpy() == yva).all()
    print(f"{ds}: train {len(ytr):,} node, val {len(yva):,} node, lớp {classes}")

    # ---- dự đoán val của 2 mô hình
    t0 = time.time()
    n_est, leaves = lightgbm_cfg(search_json)
    lgbm = fit_lightgbm(Xtr, ytr, K, n_est, leaves)
    p_lgb = lgbm.predict_proba(Xva)
    pred_lgb = p_lgb.argmax(1)
    print(f"LightGBM n={n_est} leaves={leaves}: {time.time() - t0:.0f} s")
    gat = M.GNN("gat", in_dim=len(meta["features"]), n_classes=K)
    gat.load_state_dict(torch.load(gat_ckpt, map_location="cpu"))
    y_chk, p_gat = T.predict(gat.eval(), T.strip(val_g), torch.device("cpu"))
    assert (y_chk == yva).all()
    pred_gat = p_gat.argmax(1)
    preds = {"lightgbm": pred_lgb, "gat": pred_gat}
    overview = {}
    for n, (p, pr) in {"lightgbm": (p_lgb, pred_lgb), "gat": (p_gat, pred_gat)}.items():
        m = ev.compute_metrics(yva, pr, p, class_names=classes, macro_labels=T.MACRO_CODES)
        overview[n] = {k: m[k] for k in ("accuracy", "f1_macro", "fpr", "recall_attack", "roc_auc")}
        print(f"  {n}: F1-macro {m['f1_macro']:.4f} FPR {m['fpr']:.4f} recall_tc {m['recall_attack']:.4f}")

    # ---- dựng lại số flow theo nhãn của từng node val từ CSV gốc
    t0 = time.time()
    flows, classes2, _ = bg.load_clean(ds, flows_csv)
    assert classes2 == classes, f"lớp từ CSV {classes2} khác split.json {classes}"
    flows["win"] = np.floor(flows["ts"].to_numpy() / meta["stride_sec"]).astype(np.int64)
    flows = flows[flows["win"].isin(np.asarray(meta["val"]["windows"], dtype=np.int64))][["win", "src", "dst", "y"]]
    print(f"Flow trong cửa sổ val: {len(flows):,} ({time.time() - t0:.0f} s)")
    counts = F.flow_label_counts(flows, K)
    del flows
    out_c, in_c = F.attach_counts(nodes_val, counts, K)
    rel = {r: F.relabel(out_c, in_c, r) for r in F.RULES}
    match = float((rel["hien_tai"] == yva).mean())
    print(f"Nhãn dựng lại (quy tắc hiện tại) khớp nhãn đồ thị: {match:.6f}")
    pur = F.purity_table(out_c, in_c)

    C = F.class_centroids(Xtr, ytr, K)
    dist = F.centroid_distances(Xva, C)

    results = {n: analyze_model(pr, yva, nodes_val, ref_nodes, pur, dist, rel, classes) for n, pr in preds.items()}
    summary = {
        "dataset": ds, "classes": classes, "n_val_nodes": int(len(yva)), "n_val_normal": int((yva == 0).sum()),
        "model_config": {"lightgbm": {"n_estimators": n_est, "num_leaves": leaves, "seed": SEED, "train": "train"},
                         "gat": {"ckpt": Path(gat_ckpt).name}},
        "val_overview": overview,
        "check_recomputed_label_matches_graph": match,
        "n_nodes_without_out_flow": int(pur["from_in_only"].sum()),
        "val_label_counts_by_rule": {r: {classes[k]: int((rel[r] == k).sum()) for k in range(K)} for r in F.RULES},
        "ip_overall": {"n_ip_train_val": int(ref_nodes["ip"].nunique()),
                       "n_ip_attack_node_train_val": int(ref_nodes.loc[ref_nodes["y"] != 0, "ip"].nunique())},
        "models": results,
        "seconds": round(time.time() - t_all, 1),
    }
    (out / f"fpr_analysis_{ds}.json").write_text(json.dumps(clean(summary), ensure_ascii=False, indent=1), encoding="utf-8")
    plot_all(out, ds, results, classes)
    for n, r in results.items():
        a = r["rates"]
        print(f"[{n}] FPR {a['fpr']:.4f} ({a['n_fp']}/{a['n_normal']}); top-1 IP {r['ip']['share_fp_top1']:.2f} FP, "
              f"top-3 {r['ip']['share_fp_top3']:.2f}; FP có IP cũng là node tấn công {r['ip']['share_fp_ip_also_attack_node']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
