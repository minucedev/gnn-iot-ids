"""Ngưỡng hiệu chỉnh cho phương án (c) `lgbm_neighbor_mean` (freeze_list.md mục 5.1, GĐ H, trước tag freeze-v1). Chỉ val.

Chạy:  python -I scripts/run_threshold_c.py <data_dir> <dataset> <out_dir> search_json=<ablation_search_*.json> ablation_csv=<ablation_cd_*.csv>
       [seeds=42,43,44]

(c) = LightGBM trên 18 đặc trưng node + trung bình 18 đặc trưng của các node kề (36 cột, tính tay bằng `hybrid.neighbor_mean`),
đúng như bản `abl_lgbm_x_nbrmean` của ablation GĐ G (`run_hybrid_ablation.py`): cùng đặc trưng, cùng `hybrid.fit_lgbm`, cùng seed,
cấu hình LightGBM lấy từ `ablation_search_<bộ>.json` (khóa `abl_lgbm_x_nbrmean`, đã chọn trên val ở GĐ G).

Với mỗi seed: huấn luyện trên train, đối chiếu F1-macro val với `ablation_cd_<bộ>.csv` (dòng `abl_lgbm_x_nbrmean`, split val), chọn ngưỡng
theo đúng quy tắc của `src/threshold.py` (nhỏ nhất sao cho FPR val ≤ 4,08%), lưu mô hình. KHÔNG đọc tập test.

Ghi vào <out_dir>:
- thresholds_c_<dataset>.json: {dataset: {"lgbm_neighbor_mean": {seed: {...calibrate..., repro_f1_diff, lgbm_cfg, n_params, size_kb, train_s}}}}
  (cùng khuôn với thresholds_base_gnn_<dataset>.json).
- models_c/<dataset>_s<seed>/lgbm_neighbor_mean.pkl: mô hình đã huấn luyện (cho GĐ H).
"""

import csv
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import hybrid as H  # noqa: E402
from src import threshold as TH  # noqa: E402
from src import train as T  # noqa: E402
import numpy as np  # noqa: E402

SEEDS = (42, 43, 44)
MODEL = "lgbm_neighbor_mean"
ABLATION_MODEL = "abl_lgbm_x_nbrmean"      # tên của (c) trong ablation GĐ G
TOLERANCE = 5e-4                           # lệch F1 val tối đa so với ablation (xgboost từng lệch 3e-4; LightGBM mong đợi 0)


def neighbor_features(graphs):
    """Đặc trưng của (c): [x (18) | trung bình x của node kề (18)], theo thứ tự node của `data.node_arrays`."""
    X, y = D.node_arrays(graphs)
    return np.hstack([X, H.neighbor_mean(graphs)]), y


def read_ablation_f1(path, dataset):
    """seed -> F1-macro val của (c) trong ablation_cd_<bộ>.csv."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset and r["model"] == ABLATION_MODEL and r["split"] == "val" and r["task"] == "multiclass":
                out[int(r["seed"])] = float(r["f1_macro"])
    return out


def run(data_dir, dataset, out_dir, search_json, ablation_csv=None, seeds=SEEDS):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graphs, meta = D.load_split(data_dir)
    classes, K = meta["classes"], len(meta["classes"])
    train, val = T.strip(graphs["train"]), T.strip(graphs["val"])
    del graphs                                       # không giữ tập test trong bộ nhớ
    cfg = json.loads(Path(search_json).read_text(encoding="utf-8"))[ABLATION_MODEL]["best"]
    old = read_ablation_f1(ablation_csv, dataset) if ablation_csv else {}
    Ftr, ytr = neighbor_features(train)
    Fva, yva = neighbor_features(val)
    print(f"{dataset}: train {Ftr.shape}, val {Fva.shape}, lớp {classes}, cấu hình LightGBM {cfg}")

    res = {}
    for seed in seeds:
        t0 = time.time()
        lgbm = H.fit_lgbm(Ftr, ytr, cfg, seed, K)
        train_s = time.time() - t0
        proba = H.predict_proba(lgbm, Fva, K)
        cal = TH.calibrate(yva, proba, classes, T.MACRO_CODES)
        cal["repro_f1_diff"] = cal["before"]["f1_macro"] - old[seed] if seed in old else None
        cal.update(lgbm_cfg=cfg, n_params=H.lgbm_n_leaves(lgbm), size_kb=round(ev.model_size_kb(lgbm), 1), train_s=round(train_s, 1))
        res[str(seed)] = cal
        mdir = out / "models_c" / f"{dataset}_s{seed}"
        mdir.mkdir(parents=True, exist_ok=True)
        with open(mdir / f"{MODEL}.pkl", "wb") as fh:
            pickle.dump(lgbm, fh)
        b, a = cal["before"], cal["after"]
        print(f"[{dataset} {MODEL} seed {seed}] trước: F1 {b['f1_macro']:.4f} FPR {b['fpr']:.4f} recall {b['recall_attack']:.4f} | "
              f"ngưỡng {cal['threshold']:.6f} → F1 {a['f1_macro']:.4f} FPR {a['fpr']:.4f} recall {a['recall_attack']:.4f}"
              f" | khả thi {cal['feasible']} | lệch F1 so với ablation {cal['repro_f1_diff']} ({train_s:.0f} s)")
        del lgbm
    (out / f"thresholds_c_{dataset}.json").write_text(json.dumps({dataset: {MODEL: res}}, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = {s: c["repro_f1_diff"] for s, c in res.items() if c["repro_f1_diff"] is not None and abs(c["repro_f1_diff"]) > TOLERANCE}
    if bad:
        raise RuntimeError(f"F1 val không khớp ablation (> {TOLERANCE}): {bad}")
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pos = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    data_dir, dataset, out_dir = pos[:3]
    run(data_dir, dataset, out_dir, opt["search_json"], opt.get("ablation_csv"),
        seeds=tuple(int(s) for s in opt.get("seeds", "42,43,44").split(",")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
