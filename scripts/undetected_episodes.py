"""Vì sao các đợt tấn công không có cảnh báo nào (GĐ J, bổ sung task 8.4).

Chạy (Kaggle CPU, cùng detector_config.json của notebook 20; đầu vào gắn từ output notebook 20):
    python -I scripts/undetected_episodes.py --config detector_config.json --windows-csv <20>/windows.csv \
        --summary-json <20>/detector_summary.json --out undetected_episodes.json [--gap 60] [--flows-cache /tmp/flows_test.parquet]

Đợt tấn công = đúng định nghĩa của scripts/realtime_eval.py (các cửa sổ có node nhãn lớp k, cách nhau ≤ `gap` giây gộp một đợt; phát hiện =
có node nhãn k bị báo tấn công). Danh sách đợt và cửa sổ lấy từ windows.csv của lần chạy thời gian thực; điểm tấn công của từng node tính lại
bằng mô hình đã đóng băng (đường offline, cùng kết quả với đường online: khớp 100% ở notebook 20). Không đổi mô hình, ngưỡng, dự đoán.

Với mỗi đợt: lớp, số flow (của lớp đó và của mọi lớp trong các cửa sổ của đợt), số node nhãn k, số IP, độ dài, tỉ lệ node chỉ nhận flow,
thống kê điểm tấn công 1 - P(Normal) so với ngưỡng, phân bố lớp argmax, và lý do không có cảnh báo:
  bi_gan_normal   mọi node nhãn k có điểm tấn công < 0,5 (P(Normal) > 0,5: mô hình tin đó là Normal)
  duoi_nguong     có node điểm ≥ 0,5 nhưng không node nào vượt ngưỡng đã đóng băng
  qua_ngan        đợt chỉ kéo dài 1 cửa sổ (5 giây): mô hình chỉ có một lần quyết định (nhãn phụ, không thay lý do chính)
Lý do chính = bi_gan_normal nếu điểm lớn nhất < 0,5, ngược lại duoi_nguong.
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

HALF = 0.5


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def classify_reason(max_score, n_windows, threshold):
    """(lý do chính, danh sách nhãn) của một đợt không phát hiện, từ điểm tấn công lớn nhất của các node nhãn k."""
    tags = []
    if max_score < HALF:
        tags.append("bi_gan_normal")
    elif max_score <= threshold:
        tags.append("duoi_nguong")
    if n_windows == 1:
        tags.append("qua_ngan")
    primary = "bi_gan_normal" if max_score < HALF else "duoi_nguong"
    return primary, tags


def describe_episode(ep, classes, nodes, flow_counts, threshold, ws):
    """Mô tả một đợt. `nodes`: bảng node test (win, y, score, pred, argmax, receive_only); `flow_counts`: Series (win, label) → số flow."""
    k, name = ep["class_code"], classes[ep["class_code"]]
    sel = nodes[(nodes["y"] == k) & nodes["win"].between(ep["first_window"], ep["last_window"])]
    wins = np.unique(sel["win"].to_numpy())
    sc = sel["score"].to_numpy()
    fl = flow_counts[flow_counts.index.get_level_values("win").isin(wins)]
    argmax_dist = {classes[c]: int(n) for c, n in sel["argmax"].value_counts().sort_index().items()}
    out = {"class": name, "class_code": k, "detected": bool(ep["detected"]),
           "start_ts": ep["start_ts"], "start_utc": datetime.fromtimestamp(ep["start_ts"], tz=timezone.utc).isoformat(timespec="seconds"),
           "end_ts": ep["end_ts"], "duration_s": ep["end_ts"] - ep["start_ts"], "n_windows": int(len(wins)),
           "first_window": ep["first_window"], "last_window": ep["last_window"],
           "n_flows_class": int(fl[fl.index.get_level_values("label") == name].sum()), "n_flows_all": int(fl.sum()),
           "n_nodes": int(len(sel)), "n_ips": int(sel["ip"].nunique()), "share_receive_only": float(sel["receive_only"].mean()),
           "score_max": float(sc.max()), "score_median": float(np.median(sc)), "score_mean": float(sc.mean()),
           "n_nodes_score_ge_half": int((sc >= HALF).sum()), "n_windows_score_ge_half": int(sel[sel["score"] >= HALF]["win"].nunique()),
           "n_nodes_pred_attack": int((sel["pred"] != 0).sum()), "share_argmax_normal": float((sel["argmax"] == 0).mean()),
           "argmax_distribution": argmax_dist, "threshold": threshold}
    if out["detected"]:
        out.update(detection_s=ep["detection_s"], first_alert_window=ep["first_alert_window"])
    else:
        out["reason_primary"], out["reason_tags"] = classify_reason(out["score_max"], out["n_windows"], threshold)
    return out


def summarize(eps):
    und = [e for e in eps if not e["detected"]]
    by_class, reasons = {}, {}
    for e in und:
        c = by_class.setdefault(e["class"], {"n_undetected": 0, "reasons": {}, "score_max_median": []})
        c["n_undetected"] += 1
        c["reasons"][e["reason_primary"]] = c["reasons"].get(e["reason_primary"], 0) + 1
        c["score_max_median"].append(e["score_max"])
        reasons[e["reason_primary"]] = reasons.get(e["reason_primary"], 0) + 1
    for c in by_class.values():
        c["score_max_median"] = float(np.median(c["score_max_median"]))
    return {"n_undetected": len(und), "by_class": by_class, "reasons_primary": reasons,
            "n_single_window": sum(1 for e in und if e["n_windows"] == 1),
            "n_receive_only_majority": sum(1 for e in und if e["share_receive_only"] > 0.5)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="hybrid_gat_lgbm_oofavg_il")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--windows-csv", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap", type=float, default=60.0)
    ap.add_argument("--flows-cache", default=None)
    a = ap.parse_args()

    D = _load(ROOT / "scripts" / "detector.py", "detector_mod")
    RE = _load(ROOT / "scripts" / "realtime_eval.py", "realtime_eval_mod")
    from src import fpr as F
    from src import threshold as TH

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    ds = cfg["dataset"]
    summary = json.loads(Path(a.summary_json).read_text(encoding="utf-8"))
    win = pd.read_csv(a.windows_csv)
    ctx = D.FT.Context(cfg, ds, cfg["datasets"][ds], cfg.get("split", "test"))
    model = D.OnlineModel(ctx, a.model, a.seed)
    classes, ws = ctx.classes, ctx.meta["window_sec"]
    assert abs(model.threshold - summary["threshold"]) < 1e-12, "ngưỡng khác lần chạy thời gian thực"
    zero = float(ctx.Xtr[:, 0].min())
    graphs = ctx.target_raw
    proba = np.asarray(model.proba_batch(ctx.target))
    X = np.concatenate([g.x.numpy() for g in graphs])
    nodes = pd.DataFrame({"win": np.concatenate([np.full(g.num_nodes, int(g.win), dtype=np.int64) for g in graphs]),
                          "ip": [str(v) for g in graphs for v in g.ips], "y": np.concatenate([g.y.numpy() for g in graphs]),
                          "score": TH.attack_score(proba), "pred": TH.apply_threshold(proba, model.threshold), "argmax": proba.argmax(1),
                          "receive_only": ~F.initiator_mask(X, zero)})
    assert len(nodes) == len(proba) == int(win.loc[win["ref_found"] == 1, "n_nodes"].sum())
    del proba, X
    for attr in ("train", "val", "Xtr", "ytr", "Xva", "yva", "Ftr_nbr", "window", "target", "target_raw"):
        setattr(ctx, attr, None)
    cache = a.flows_cache
    if cache and Path(cache).exists():
        flows = pd.read_parquet(cache, columns=["ts", "label"])
    else:
        flows = D.load_stream_flows(cfg, ctx, "test")
        if cache:
            flows.to_parquet(cache)
        flows = flows[["ts", "label"]]
    flow_counts = pd.DataFrame({"win": SF_window_ids(flows["ts"].to_numpy(), ws), "label": flows["label"].to_numpy()}).groupby(["win", "label"]).size()
    del flows

    det = RE.detection_report(win, classes, a.gap, ws)
    eps = []
    for e in det["episodes"]:
        d = describe_episode(e, classes, nodes, flow_counts, model.threshold, ws)
        # kiểm chéo: kết luận phát hiện theo xác suất offline phải trùng kết luận theo cảnh báo online của windows.csv
        assert (d["n_nodes_pred_attack"] > 0) == d["detected"], f"đợt {d['class']} {d['start_utc']}: offline và online lệch"
        eps.append(d)
    und = [e for e in eps if not e["detected"]]
    res = {"model": a.model, "seed": a.seed, "dataset": ds, "threshold": model.threshold, "episode_gap_s": a.gap, "window_sec": ws,
           "source_windows_csv": Path(a.windows_csv).name, "n_episodes": len(eps), "n_detected": len(eps) - len(und), "n_undetected": len(und),
           "consistent_with_realtime_eval": {"n_episodes": det["n_episodes"], "n_undetected": det["n_undetected"],
                                             "same": det["n_episodes"] == len(eps) and det["n_undetected"] == len(und)},
           "definitions": __doc__.split("Với mỗi đợt:")[1].strip(),
           "summary": summarize(eps), "episodes_undetected": und, "episodes_detected": [e for e in eps if e["detected"]]}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"n_episodes": res["n_episodes"], "n_undetected": res["n_undetected"], "summary": res["summary"]}, ensure_ascii=False, indent=1))
    for e in und:
        print(f"{e['class']:20s} {e['start_utc']} win={e['n_windows']:3d} flows={e['n_flows_class']:7d}/{e['n_flows_all']:7d} nodes={e['n_nodes']:4d} "
              f"recv={e['share_receive_only']:.0%} max={e['score_max']:.3f} med={e['score_median']:.3f} → {e['reason_primary']} {e['reason_tags']}")
    return 0


def SF_window_ids(ts, window_sec):
    return np.floor(np.asarray(ts, dtype=np.float64) / window_sec).astype(np.int64)


if __name__ == "__main__":
    sys.exit(main())
