"""Hybrid GAT + LightGBM với out-of-fold dạng ensemble (task 5.7, sửa lỗi lệch không gian embedding).

Chạy:  python -I scripts/run_hybrid_oofavg.py <data_dir> <dataset> <results.csv> <search.json> [models=<dir>] [k=5]
       [hp=<json>]

Vấn đề của `run_hybrid.py oof=1` (kết quả 14/11): embedding train của mỗi nhóm out-of-fold đến từ một GAT khác nhau
(khởi tạo và dữ liệu khác nhau), nhưng embedding val lại đến từ MỘT GAT thứ sáu (huấn luyện trên toàn bộ train), nên
LightGBM bị đo trên một không gian embedding mà nó chưa từng thấy. Ở đây val được tính như trung bình ensemble:
mỗi GAT nhóm cho một embedding val, ghép với 18 đặc trưng, LightGBM dự đoán từng góc rồi lấy trung bình xác suất
(cách chuẩn khi dùng dự đoán out-of-fold). Mọi chuyển đổi chỉ dùng train; val chỉ chọn epoch GAT và cấu hình LightGBM.

Ghi model=hybrid_gat_lgbm_oofavg (split=val, 3 seed 42, 43, 44). n_params = tổng tham số k GAT + số lá LightGBM;
size_kb = tổng k file GAT + file LightGBM; độ trễ = k lượt GAT + một lần LightGBM trên k·n dòng, CPU 1 luồng
(mô hình này tốn hơn bản dùng một GAT).

Chẩn đoán (trong search.json, cùng các GAT nhóm và cùng LightGBM): F1 val khi dùng (a) trung bình k góc, (b) từng góc
riêng lẻ, (c) embedding của GAT huấn luyện trên toàn bộ train (cách cũ). Nếu (a) ≫ (c) thì lệch không gian là nguyên nhân.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import hybrid as H  # noqa: E402
from src import train as T  # noqa: E402

SEEDS = (42, 43, 44)
MODEL = "hybrid_gat_lgbm_oofavg"          # thêm hậu tố _il khi interleave > 0


def f1_of(y, proba, classes):
    return ev.compute_metrics(y, proba.argmax(1), proba, class_names=classes, macro_labels=T.MACRO_CODES)["f1_macro"]


def run(data_dir, dataset, results_csv, search_json, models_dir=None, k=5, seeds=SEEDS, hp=None, grid=None,
        verbose=False, interleave=0, full_diag=True):
    model_name = MODEL + ("_il" if interleave > 0 else "")
    hp = {"model": {}, "train": {}, **(hp or {})}
    models_dir = Path(models_dir) if models_dir else Path(results_csv).parent / "models_hybrid"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graphs, meta = D.load_split(data_dir)
    classes, K = meta["classes"], len(meta["classes"])
    train, val = T.strip(graphs["train"]), T.strip(graphs["val"])
    Xtr, ytr = D.node_arrays(train)
    Xva, yva = D.node_arrays(val)
    window = T.strip([D.sample_window(graphs["val"])])[0]
    print(f"{dataset} [{model_name}]: train {len(train):,} đồ thị ({len(ytr):,} node), val {len(val):,}, lớp {classes}, "
          f"thiết bị {device}")

    search_path = Path(search_json)
    search = json.loads(search_path.read_text(encoding="utf-8")) if search_path.exists() else {}
    log = search[model_name] = {"dataset": dataset, "k": k, "interleave": interleave, "seeds": {}}
    cfg, rows = None, []
    for seed in seeds:
        t_seed = time.time()
        out_dir = models_dir / f"{dataset}_s{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_tr, fold_gats, val_embs, folds = H.oof_ensemble(train, val, classes, device, seed, k, hp["model"],
                                                            hp["train"], verbose, interleave=interleave)
        train_s = sum(f["train_s"] for f in folds)
        emb_full, entry_full = None, {}
        if full_diag:                                                # GAT toàn train chỉ để chẩn đoán, không dùng trong mô hình
            full, hist, secs = H.train_gat(train, val, classes, device, seed, hp["model"], hp["train"], verbose)
            emb_full, entry_full = H.extract_embeddings(full, val, device), {"epochs": len(hist), "train_s": round(secs, 1)}
        Ftr = H.build_features(emb_tr, Xtr)
        views = [H.build_features(e, Xva) for e in val_embs]
        entry = {"folds": folds, "gat_full": entry_full}
        if cfg is None:                                              # chọn cấu hình một lần, ở seed đầu, chỉ trên val
            cfg, trials = H.search_lgbm(Ftr, ytr, None, yva, classes, seed=seed, grid=grid, val_views=views)
            entry["search"] = trials
            log["best"] = cfg
            print(f"  [chọn cấu hình] {cfg}: " + ", ".join(f"{t['cfg']} → {t['val_f1_macro']:.4f}" for t in trials))

        t0 = time.time()
        lgbm = H.fit_lgbm(Ftr, ytr, cfg, seed, K)
        train_s += time.time() - t0
        proba = H.predict_proba_views(lgbm, views, K)
        metrics = ev.compute_metrics(yva, proba.argmax(1), proba, class_names=classes, macro_labels=T.MACRO_CODES)
        tr_proba = H.predict_proba(lgbm, Ftr, K)
        entry["diag"] = {
            "train_f1_macro": f1_of(ytr, tr_proba, classes),
            "val_f1_macro_avg_views": metrics["f1_macro"],
            "val_f1_macro_each_view": [f1_of(yva, H.predict_proba(lgbm, v, K), classes) for v in views]}
        if emb_full is not None:
            entry["diag"]["val_f1_macro_full_gat_embedding"] = f1_of(
                yva, H.predict_proba(lgbm, H.build_features(emb_full, Xva), K), classes)

        cpus = [g.to("cpu").eval() for g in fold_gats]
        torch.set_num_threads(1)
        lgbm.set_params(n_jobs=1)
        lat = ev.measure_latency(lambda: H.predict_window_ensemble(cpus, lgbm, window, K))
        n_params = sum(ev.count_params(g) for g in cpus) + H.lgbm_n_leaves(lgbm)
        size_kb = round(sum(ev.model_size_kb(g) for g in cpus) + ev.model_size_kb(lgbm), 1)
        row = ev.log_result(
            metrics, dataset=dataset, model=model_name, task="multiclass", seed=seed, split="val",
            n_params=n_params, size_kb=size_kb, train_time_s=round(train_s, 1),
            infer_ms_per_window=round(lat["p95_ms"], 3), path=results_csv)
        rows.append(row)
        for f, g in enumerate(cpus):
            torch.save(g.state_dict(), out_dir / f"gat_fold{f}.pt")
        with open(out_dir / ("lgbm_oofavg_il.pkl" if interleave > 0 else "lgbm_oofavg.pkl"), "wb") as fh:
            pickle.dump(lgbm, fh)
        (out_dir / ("meta_oofavg_il.json" if interleave > 0 else "meta_oofavg.json")).write_text(json.dumps({
            "model": model_name, "dataset": dataset, "seed": seed, "lgbm_cfg": cfg, "k": k, "n_cols": int(Ftr.shape[1]),
            "n_params": n_params, "size_kb": size_kb, "classes": classes}), encoding="utf-8")
        log["seeds"][str(seed)] = entry
        d = entry["diag"]
        print(f"[{model_name} seed {seed}] acc {metrics['accuracy']:.4f} F1-macro {metrics['f1_macro']:.4f} "
              f"(train {d['train_f1_macro']:.4f}; từng góc {np.round(d['val_f1_macro_each_view'], 3).tolist()}; "
              f"embedding GAT toàn train {d.get('val_f1_macro_full_gat_embedding', float('nan')):.4f}) FPR {metrics['fpr']:.4f} "
              f"recall_tc {metrics['recall_attack']:.4f} AUC {metrics['roc_auc']:.4f} p95 {lat['p95_ms']:.2f} ms, "
              f"{n_params:,} tham số, {size_kb:.0f} KB, {time.time() - t_seed:.0f} s ({row['commit']})")
        search_path.write_text(json.dumps(search, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pos = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    data_dir, dataset, results_csv, search_json = pos[:4]
    run(data_dir, dataset, results_csv, search_json, models_dir=opt.get("models"), k=int(opt.get("k", 5)),
        interleave=int(opt.get("interleave", 0)), full_diag=opt.get("fulldiag", "1") == "1", hp=json.loads(opt["hp"]) if "hp" in opt else None, verbose=opt.get("verbose") == "1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
