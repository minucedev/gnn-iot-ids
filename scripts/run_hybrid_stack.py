"""Hybrid GAT + LightGBM dạng stacking với xác suất lớp out-of-fold (task 5.7, thay embedding 64 chiều).

Chạy:  python -I scripts/run_hybrid_stack.py <data_dir> <dataset> <results.csv> <search.json> <folds_dir>
       [full_dir=<dir>] [fold_search=<json>] [models=<dir>] [k=5]

Vì sao: embedding out-of-fold (run_hybrid.py oof=1, run_hybrid_oofavg.py) thất bại vì mỗi GAT nhóm có một không gian embedding
riêng và LightGBM học được "embedding này đến từ GAT nào" (xem TASKS.md 5.7). Xác suất 5 lớp có ngữ nghĩa cố định giữa
các GAT nên không có vấn đề đó; đây là stacking chuẩn: đặc trưng LightGBM = 18 đặc trưng node + K xác suất lớp.
- train: xác suất out-of-fold (nhóm f lấy từ GAT nhóm f, chưa thấy nhóm f);
- val: trung bình xác suất của k GAT nhóm (cùng phân phối với out-of-fold: mỗi GAT thấy 80% train).
KHÔNG huấn luyện lại GAT: nạp checkpoint `gat_fold<f>.pt` của notebook 15b (`<folds_dir>/<dataset>_s<seed>/`), đã huấn luyện chỉ
trên train (val chỉ chọn epoch). CPU là đủ. Ghi model=hybrid_gat_lgbm_stack, split=val, 3 seed.
n_params = k GAT + số lá LightGBM; size_kb = k file GAT + file LightGBM; độ trễ = k lượt GAT + một lần LightGBM, CPU 1 luồng;
train_time_s = thời gian huấn luyện k GAT nhóm (đọc từ fold_search nếu có) + LightGBM cuối.
Chẩn đoán trong search.json: F1 val khi dùng xác suất của từng GAT nhóm riêng lẻ và (nếu có full_dir) của GAT huấn luyện trên toàn train.
"""

import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import hybrid as H  # noqa: E402
from src import models as M  # noqa: E402
from src import train as T  # noqa: E402

SEEDS = (42, 43, 44)
MODEL = "hybrid_gat_lgbm_stack"


def f1_of(y, proba, classes):
    return ev.compute_metrics(y, proba.argmax(1), proba, class_names=classes, macro_labels=T.MACRO_CODES)["f1_macro"]


def load_gat(path, in_dim, n_classes, hp):
    m = M.GNN("gat", in_dim=in_dim, n_classes=n_classes, **hp["model"])
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def run(data_dir, dataset, results_csv, search_json, folds_dir, full_dir=None, fold_search=None, models_dir=None, k=5,
        seeds=SEEDS, hp=None, grid=None):
    hp = {"model": {}, "train": {}, **(hp or {})}
    models_dir = Path(models_dir) if models_dir else Path(results_csv).parent / "models_hybrid"
    cpu = torch.device("cpu")
    graphs, meta = D.load_split(data_dir)
    classes, K = meta["classes"], len(meta["classes"])
    train, val = T.strip(graphs["train"]), T.strip(graphs["val"])
    Xtr, ytr = D.node_arrays(train)
    Xva, yva = D.node_arrays(val)
    window = T.strip([D.sample_window(graphs["val"])])[0]
    in_dim = Xtr.shape[1]
    fold_json = json.loads(Path(fold_search).read_text(encoding="utf-8")) if fold_search else None
    print(f"{dataset} [{MODEL}]: train {len(train):,} đồ thị ({len(ytr):,} node), val {len(val):,}, lớp {classes}")

    search_path = Path(search_json)
    search = json.loads(search_path.read_text(encoding="utf-8")) if search_path.exists() else {}
    log = search[MODEL] = {"dataset": dataset, "k": k, "seeds": {}}
    cfg, rows = None, []
    for seed in seeds:
        t_seed = time.time()
        src = Path(folds_dir) / f"{dataset}_s{seed}"
        out_dir = models_dir / f"{dataset}_s{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        gats = [load_gat(src / f"gat_fold{f}.pt", in_dim, K, hp) for f in range(k)]
        p_tr = H.oof_class_proba(train, gats, k, cpu)
        val_ps = [H.class_proba(g, val, cpu) for g in gats]
        p_va = np.mean(val_ps, axis=0)
        Ftr, Fva = H.build_features(p_tr, Xtr), H.build_features(p_va, Xva)
        assert Ftr.shape[1] == Xtr.shape[1] + K
        entry = {}
        if cfg is None:                                              # chọn cấu hình một lần, ở seed đầu, chỉ trên val
            cfg, trials = H.search_lgbm(Ftr, ytr, Fva, yva, classes, seed=seed, grid=grid)
            entry["search"] = trials
            log["best"] = cfg
            print(f"  [chọn cấu hình] {cfg}: " + ", ".join(f"{t['cfg']} → {t['val_f1_macro']:.4f}" for t in trials))

        t0 = time.time()
        lgbm = H.fit_lgbm(Ftr, ytr, cfg, seed, K)
        lgbm_s = time.time() - t0
        proba = H.predict_proba(lgbm, Fva, K)
        metrics = ev.compute_metrics(yva, proba.argmax(1), proba, class_names=classes, macro_labels=T.MACRO_CODES)
        diag = {"train_f1_macro": f1_of(ytr, H.predict_proba(lgbm, Ftr, K), classes),
                "val_f1_macro": metrics["f1_macro"],
                "val_f1_macro_each_gat": [f1_of(yva, H.predict_proba(lgbm, H.build_features(p, Xva), K), classes)
                                          for p in val_ps],
                "gat_alone_val_f1_macro_each_gat": [f1_of(yva, p, classes) for p in val_ps],
                "gat_alone_val_f1_macro_avg": f1_of(yva, p_va, classes)}
        full = Path(full_dir) / f"{dataset}_s{seed}" / "gat.pt" if full_dir else None
        if full is not None and full.exists():
            p_full = H.class_proba(load_gat(full, in_dim, K, hp), val, cpu)
            diag["val_f1_macro_full_gat_proba"] = f1_of(yva, H.predict_proba(lgbm, H.build_features(p_full, Xva), K), classes)
        entry["diag"] = diag
        train_s = lgbm_s + (sum(f["train_s"] for f in fold_json["hybrid_gat_lgbm_oofavg"]["seeds"][str(seed)]["folds"])
                            if fold_json else 0.0)

        torch.set_num_threads(1)
        lgbm.set_params(n_jobs=1)
        lat = ev.measure_latency(lambda: H.predict_window_stack(gats, lgbm, window, K))
        n_params = sum(ev.count_params(g) for g in gats) + H.lgbm_n_leaves(lgbm)
        size_kb = round(sum(ev.model_size_kb(g) for g in gats) + ev.model_size_kb(lgbm), 1)
        row = ev.log_result(
            metrics, dataset=dataset, model=MODEL, task="multiclass", seed=seed, split="val",
            n_params=n_params, size_kb=size_kb, train_time_s=round(train_s, 1),
            infer_ms_per_window=round(lat["p95_ms"], 3), path=results_csv)
        rows.append(row)
        for f in range(k):                                           # gói đủ mô hình trong output cho GĐ G, H
            shutil.copy(src / f"gat_fold{f}.pt", out_dir / f"gat_fold{f}.pt")
        with open(out_dir / "lgbm_stack.pkl", "wb") as fh:
            pickle.dump(lgbm, fh)
        (out_dir / "meta_stack.json").write_text(json.dumps({
            "model": MODEL, "dataset": dataset, "seed": seed, "lgbm_cfg": cfg, "k": k, "n_cols": int(Ftr.shape[1]),
            "n_params": n_params, "size_kb": size_kb, "classes": classes}), encoding="utf-8")
        log["seeds"][str(seed)] = entry
        print(f"[{MODEL} seed {seed}] acc {metrics['accuracy']:.4f} F1-macro {metrics['f1_macro']:.4f} "
              f"(train {diag['train_f1_macro']:.4f}; từng GAT {np.round(diag['val_f1_macro_each_gat'], 3).tolist()}; "
              f"GAT đơn trung bình {diag['gat_alone_val_f1_macro_avg']:.4f}) FPR {metrics['fpr']:.4f} "
              f"recall_tc {metrics['recall_attack']:.4f} AUC {metrics['roc_auc']:.4f} p95 {lat['p95_ms']:.2f} ms, "
              f"{n_params:,} tham số, {size_kb:.0f} KB, {time.time() - t_seed:.0f} s ({row['commit']})")
        search_path.write_text(json.dumps(search, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pos = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    data_dir, dataset, results_csv, search_json, folds_dir = pos[:5]
    run(data_dir, dataset, results_csv, search_json, folds_dir, full_dir=opt.get("full_dir"),
        fold_search=opt.get("fold_search"), models_dir=opt.get("models"), k=int(opt.get("k", 5)),
        hp=json.loads(opt["hp"]) if "hp" in opt else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
