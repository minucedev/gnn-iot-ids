"""Hiệu chỉnh ngưỡng "tấn công" trên val cho mọi baseline và GNN đơn (task 5.8, GĐ G). Chỉ val, không đọc test.

Chạy:  python -I scripts/run_thresholds.py <data_dir> <dataset> <out.json> baseline_search=<json> baseline_csv=<csv> gnn_csv=<csv>
       gnn_ckpt=<thư mục ckpt>

- Baseline (logreg, random_forest, xgboost, lightgbm, mlp): huấn luyện lại trên train với cấu hình tốt nhất đã chọn ở GĐ D
  (`baseline_search_<bộ>.json`), 3 seed, rồi chọn ngưỡng trên val. Kết quả val của baseline cũ không lưu xác suất nên phải dựng lại;
  F1-macro dựng lại được đối chiếu với `results_baselines_<bộ>.csv` (cột `repro_f1_diff`, mong đợi ≈ 0).
- GNN đơn (gcn, sage, gat): nạp checkpoint của GĐ A/E (`<ckpt>/<bộ>_<mô hình>_s<seed>.pt`, cấu hình mặc định), dự đoán val, chọn ngưỡng;
  đối chiếu F1-macro với `results_gnn_<bộ>.csv`.
Ngưỡng: nhỏ nhất sao cho FPR val ≤ 4,08% (src/threshold.py). JSON: {dataset: {model: {seed: {...calibrate..., repro_f1_diff}}}}.
"""

import ast
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data as D  # noqa: E402
from src import models as M  # noqa: E402
from src import threshold as TH  # noqa: E402
from src import train as T  # noqa: E402

SEEDS = (42, 43, 44)
BASELINES = ("logreg", "random_forest", "xgboost", "lightgbm", "mlp")


def _load_baselines_module():
    spec = importlib.util.spec_from_file_location("run_baselines", ROOT / "scripts" / "run_baselines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_cfg(best):
    """`best` trong baseline_search_*.json có giá trị là chuỗi (vd. "100", "None", "(64, 64)"): đổi về kiểu gốc."""
    return {k: ast.literal_eval(v) if isinstance(v, str) else v for k, v in best.items()}


def full_proba(model, X, n_classes):
    """predict_proba đủ n_classes cột kể cả khi train thiếu một lớp."""
    p = model.predict_proba(X)
    if p.shape[1] == n_classes:
        return p
    out = np.zeros((len(X), n_classes))
    out[:, model.classes_] = p
    return out


def read_f1(path, dataset):
    """(model, seed) -> f1_macro từ một results.csv (chỉ split=val, task=multiclass, đúng bộ)."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset and r["split"] == "val" and r["task"] == "multiclass":
                out[(r["model"], int(r["seed"]))] = float(r["f1_macro"])
    return out


def run(data_dir, dataset, out_json, search_json, baseline_csv, gnn_csv, ckpt_dir, seeds=SEEDS, only=None):
    graphs, meta = D.load_split(data_dir)
    classes, K = meta["classes"], len(meta["classes"])
    Xtr, ytr = D.node_arrays(graphs["train"])
    Xva, yva = D.node_arrays(graphs["val"])
    val = T.strip(graphs["val"])
    w = D.class_weights(ytr, K)
    RB = _load_baselines_module()
    best = {n: parse_cfg(json.loads(Path(search_json).read_text(encoding="utf-8"))[n][-1]["best"]) for n in BASELINES}
    old_base, old_gnn = read_f1(baseline_csv, dataset), read_f1(gnn_csv, dataset)
    res = {}
    out = Path(out_json)

    def save():
        out.write_text(json.dumps({dataset: res}, ensure_ascii=False, indent=1), encoding="utf-8")

    def record(name, seed, proba, old):
        cal = TH.calibrate(yva, proba, classes, T.MACRO_CODES)
        cal["repro_f1_diff"] = cal["before"]["f1_macro"] - old if old is not None else None
        res.setdefault(name, {})[str(seed)] = cal
        b, a = cal["before"], cal["after"]
        print(f"[{dataset} {name} seed {seed}] trước: F1 {b['f1_macro']:.4f} FPR {b['fpr']:.4f} recall {b['recall_attack']:.4f} | "
              f"ngưỡng {cal['threshold']:.6f} → F1 {a['f1_macro']:.4f} FPR {a['fpr']:.4f} recall {a['recall_attack']:.4f}"
              f" | khả thi {cal['feasible']} | lệch F1 so với kết quả cũ {cal['repro_f1_diff']}")

    for kind in ("gcn", "sage", "gat"):
        if only and kind not in only:
            continue
        for seed in seeds:
            model = M.GNN(kind, in_dim=len(meta["features"]), n_classes=K)
            model.load_state_dict(torch.load(Path(ckpt_dir) / f"{dataset}_{kind}_s{seed}.pt", map_location="cpu", weights_only=True))
            _, proba = T.predict(model.eval(), val, torch.device("cpu"))
            record(kind, seed, proba, old_gnn.get((kind, seed)))
        save()

    for name in BASELINES:
        if only and name not in only:
            continue
        for seed in seeds:
            t0 = time.time()
            m = RB.fit(name, RB.make_models(seed)[name][0](best[name]), Xtr, ytr, w)
            record(name, seed, full_proba(m, Xva, K), old_base.get((name, seed)))
            print(f"    ({name} seed {seed}: {time.time() - t0:.0f} s, cấu hình {best[name]})")
            del m
        save()
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pos = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    data_dir, dataset, out_json = pos[:3]
    run(data_dir, dataset, out_json, opt["baseline_search"], opt["baseline_csv"], opt["gnn_csv"], opt["gnn_ckpt"],
        only=opt["only"].split(",") if "only" in opt else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
