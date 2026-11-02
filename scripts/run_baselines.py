"""Baseline ML trên đặc trưng node (task 4.1). Chỉ chạy trên val (quy ước 6.2: test chỉ dùng 1 lần ở GĐ6).

Chạy:  python -I scripts/run_baselines.py <data_dir> <dataset_name> <results.csv> [<log_search.json>]
Với mỗi mô hình: thử vài cấu hình nhỏ trên val (seed 42), chọn theo F1-macro 4 lớp chính, rồi chạy 3 seed
42, 43, 44 với cấu hình tốt nhất và ghi vào results.csv qua evaluate.log_result.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D  # noqa: E402
from src import evaluate as ev  # noqa: E402

SEEDS = (42, 43, 44)
MAIN4 = [0, 1, 2, 3]          # F1-macro tính trên 4 lớp chính (D5), lớp Khac báo riêng


def make_models(seed):
    """tên -> (hàm tạo theo cấu hình, danh sách cấu hình)."""
    return {
        "logreg": (lambda c: LogisticRegression(C=c["C"], max_iter=500, random_state=seed),
                   [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]),
        "random_forest": (lambda c: RandomForestClassifier(n_estimators=c["n"], max_depth=c["d"], n_jobs=-1,
                                                           random_state=seed),
                          [{"n": 100, "d": 12}, {"n": 200, "d": 20}, {"n": 200, "d": None}]),
        "xgboost": (lambda c: XGBClassifier(n_estimators=c["n"], max_depth=c["d"], learning_rate=0.1,
                                            tree_method="hist", n_jobs=-1, random_state=seed),
                    [{"n": 100, "d": 4}, {"n": 200, "d": 6}, {"n": 300, "d": 8}]),
        "lightgbm": (lambda c: LGBMClassifier(n_estimators=c["n"], num_leaves=c["l"], learning_rate=0.1,
                                              n_jobs=-1, random_state=seed, verbose=-1),
                     [{"n": 100, "l": 15}, {"n": 200, "l": 31}, {"n": 300, "l": 63}]),
        "mlp": (lambda c: MLPClassifier(hidden_layer_sizes=c["h"], max_iter=60, early_stopping=True,
                                        random_state=seed),
                [{"h": (32,)}, {"h": (64, 64)}]),
    }


def fit(name, model, X, y, w):
    if name in ("random_forest", "logreg"):
        model.set_params(class_weight="balanced")
        model.fit(X, y)
    elif name == "mlp":
        model.fit(X, y)                       # sklearn MLP không hỗ trợ trọng số mẫu
    else:
        model.fit(X, y, sample_weight=w[y])
    return model


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data_dir, dataset, results_csv = sys.argv[1:4]
    search_log = sys.argv[4] if len(sys.argv) > 4 else None

    graphs, meta = D.load_split(data_dir)
    classes = meta["classes"]
    n_cls = len(classes)
    Xtr, ytr = D.node_arrays(graphs["train"])
    Xva, yva = D.node_arrays(graphs["val"])
    w = D.class_weights(ytr, n_cls)
    window_x = D.sample_window(graphs["val"]).x.numpy()
    print(f"train {Xtr.shape}, val {Xva.shape}, lớp {classes}")

    def score(model):
        pred = model.predict(Xva)
        return ev.compute_metrics(yva, pred, class_names=classes, macro_labels=MAIN4)["f1_macro"]

    log = {}
    for name, (make, grid) in make_models(42).items():
        log[name] = []
        for cfg in grid:
            t0 = time.time()
            m = fit(name, make(cfg), Xtr, ytr, w)
            f1 = score(m)
            log[name].append({"cfg": {k: str(v) for k, v in cfg.items()}, "val_f1_macro4": f1,
                              "seconds": round(time.time() - t0, 1)})
            print(f"[chọn cấu hình] {name} {cfg}: val F1-macro(4 lớp) = {f1:.4f} ({time.time() - t0:.0f} s)")
        best = max(range(len(grid)), key=lambda i: log[name][i]["val_f1_macro4"])
        log[name].append({"best": {k: str(v) for k, v in grid[best].items()}})

        for seed in SEEDS:
            make_s = make_models(seed)[name][0]
            t0 = time.time()
            m = fit(name, make_s(grid[best]), Xtr, ytr, w)
            train_s = time.time() - t0
            proba = m.predict_proba(Xva)
            pred = proba.argmax(1)
            metrics = ev.compute_metrics(yva, pred, y_score=proba, class_names=classes, macro_labels=MAIN4)
            if "n_jobs" in m.get_params():
                m.set_params(n_jobs=1)             # D7: đo độ trễ trên 1 luồng
            lat = ev.measure_latency(lambda: m.predict(window_x))
            row = ev.log_result(
                metrics, dataset=dataset, model=name, task="multiclass", seed=seed, split="val",
                n_params=ev.count_params(m), size_kb=round(ev.model_size_kb(m), 1),
                train_time_s=round(train_s, 1), infer_ms_per_window=round(lat["p95_ms"], 3), path=results_csv)
            print(f"[{name} seed {seed}] acc {metrics['accuracy']:.4f} F1-macro4 {metrics['f1_macro']:.4f} "
                  f"FPR {metrics['fpr']:.4f} recall_tc {metrics['recall_attack']:.4f} "
                  f"p95 {lat['p95_ms']:.2f} ms ({row['commit']})")

    if search_log:
        Path(search_log).write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
