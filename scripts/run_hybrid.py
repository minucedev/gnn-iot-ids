"""Hybrid GAT + LightGBM (task 5.7): huấn luyện 3 seed và ghi kết quả val vào results.csv.

Chạy:  python -I scripts/run_hybrid.py <data_dir> <dataset> <results.csv> <search.json> [oof=1] [models=<dir>] [k=5]
       [hp=<json>]
oof=1 (mặc định): embedding train out-of-fold, ghi model=hybrid_gat_lgbm.
oof=0: embedding train lấy trực tiếp từ GAT huấn luyện trên toàn bộ train, ghi model=hybrid_gat_lgbm_nooof (để so sánh).
Mỗi seed 42, 43, 44: GAT toàn train (dùng cho val, và cho train khi oof=0), LightGBM trên 82 cột. Cấu hình LightGBM chọn
MỘT lần trên val ở seed đầu (42) trong cùng không gian với baseline (3 cấu hình), các seed sau dùng lại. Chỉ ghi split=val.

Mô hình lưu ở <models>/<dataset>_s<seed>/: gat.pt (dùng chung hai chế độ; chạy oof=0 sau oof=1 thì nạp lại, không
huấn luyện lại), lgbm_<oof|nooof>.pkl, meta_<oof|nooof>.json. Mặc định <models> = models_hybrid cạnh results.csv.
search.json gộp cả hai chế độ (khóa theo tên mô hình): log tìm cấu hình, thông tin các nhóm out-of-fold, F1 train và val.

Độ trễ: p95 một lượt GAT (embedding) + ghép đặc trưng + LightGBM cho 1 cửa sổ trung vị, CPU 1 luồng.
n_params = tham số GAT + số lá LightGBM; size_kb = file GAT + file LightGBM; train_time_s gồm mọi lần huấn luyện GAT
(cả các nhóm out-of-fold) và LightGBM cuối.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import hybrid as H  # noqa: E402
from src import models as M  # noqa: E402
from src import train as T  # noqa: E402

SEEDS = (42, 43, 44)


def parse_args(argv):
    pos = [a for a in argv if "=" not in a]
    opt = dict(a.split("=", 1) for a in argv if "=" in a)
    if len(pos) > 4:                                   # chấp nhận cả `1` lẫn `oof=1` ở vị trí 5
        opt.setdefault("oof", pos[4])
    return pos[:4], opt


def load_or_train_gat(gat_dir, train, val, classes, device, seed, hp, verbose):
    """GAT toàn train của một (bộ, seed); nạp từ gat.pt nếu đã có (cùng cấu hình), ngược lại huấn luyện và lưu."""
    path, meta_path = gat_dir / "gat.pt", gat_dir / "gat_meta.json"
    gat_dir.mkdir(parents=True, exist_ok=True)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        gat = M.GNN("gat", in_dim=train[0].x.shape[1], n_classes=len(classes), **hp["model"])
        gat.load_state_dict(torch.load(path, map_location=device))
        return gat.to(device), meta
    gat, hist, secs = H.train_gat(train, val, classes, device, seed, hp["model"], hp["train"], verbose)
    torch.save({k: v.cpu() for k, v in gat.state_dict().items()}, path)
    meta = {"train_s": round(secs, 1), "epochs": len(hist), "best_val_f1_macro": max(h["val_f1_macro"] for h in hist)}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return gat, meta


def run(data_dir, dataset, results_csv, search_json, oof=True, models_dir=None, k=5, seeds=SEEDS, hp=None,
        grid=None, verbose=False):
    hp = {"model": {}, "train": {}, **(hp or {})}
    model_name = "hybrid_gat_lgbm" if oof else "hybrid_gat_lgbm_nooof"
    mode = "oof" if oof else "nooof"
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
    log = search[model_name] = {"dataset": dataset, "k": k if oof else None, "seeds": {}}
    cfg = None
    rows = []
    for seed in seeds:
        t_seed = time.time()
        gat_dir = models_dir / f"{dataset}_s{seed}"
        gat, gat_meta = load_or_train_gat(gat_dir, train, val, classes, device, seed, hp, verbose)
        train_s = gat_meta["train_s"]
        emb_va = H.extract_embeddings(gat, val, device)
        if oof:
            emb_tr, folds = H.oof_embeddings(train, val, classes, device, seed, k, hp["model"], hp["train"], verbose)
            train_s += sum(f["train_s"] for f in folds)
        else:
            emb_tr, folds = H.extract_embeddings(gat, train, device), []
        Ftr, Fva = H.build_features(emb_tr, Xtr), H.build_features(emb_va, Xva)
        assert Ftr.shape[1] == Xtr.shape[1] + emb_tr.shape[1]
        entry = {"gat": gat_meta, "folds": folds}
        if cfg is None:                                    # chọn cấu hình một lần, ở seed đầu, chỉ trên val
            cfg, trials = H.search_lgbm(Ftr, ytr, Fva, yva, classes, seed=seed, grid=grid)
            entry["search"] = trials
            log["best"] = cfg
            print(f"  [chọn cấu hình] {cfg}: " + ", ".join(f"{t['cfg']} → {t['val_f1_macro']:.4f}" for t in trials))

        t0 = time.time()
        lgbm = H.fit_lgbm(Ftr, ytr, cfg, seed, K)
        train_s += time.time() - t0
        proba = H.predict_proba(lgbm, Fva, K)
        pred = proba.argmax(1)
        metrics = ev.compute_metrics(yva, pred, proba, class_names=classes, macro_labels=T.MACRO_CODES)
        tr_pred = H.predict_proba(lgbm, Ftr, K).argmax(1)       # chẩn đoán học vẹt: train gần 100% mà val tụt
        entry["diag"] = {"train_f1_macro": ev.compute_metrics(ytr, tr_pred, class_names=classes,
                                                              macro_labels=T.MACRO_CODES)["f1_macro"],
                         "val_f1_macro": metrics["f1_macro"]}

        cpu = gat.to("cpu").eval()
        torch.set_num_threads(1)
        lgbm.set_params(n_jobs=1)
        lat = ev.measure_latency(lambda: H.predict_window(cpu, lgbm, window, K))
        n_params = ev.count_params(cpu) + H.lgbm_n_leaves(lgbm)
        size_kb = round(ev.model_size_kb(cpu) + ev.model_size_kb(lgbm), 1)
        row = ev.log_result(
            metrics, dataset=dataset, model=model_name, task="multiclass", seed=seed, split="val",
            n_params=n_params, size_kb=size_kb, train_time_s=round(train_s, 1),
            infer_ms_per_window=round(lat["p95_ms"], 3), path=results_csv)
        rows.append(row)
        with open(gat_dir / f"lgbm_{mode}.pkl", "wb") as f:
            pickle.dump(lgbm, f)
        (gat_dir / f"meta_{mode}.json").write_text(json.dumps({
            "model": model_name, "dataset": dataset, "seed": seed, "lgbm_cfg": cfg, "n_cols": int(Ftr.shape[1]),
            "n_params": n_params, "size_kb": size_kb, "classes": classes}), encoding="utf-8")
        log["seeds"][str(seed)] = entry
        print(f"[{model_name} seed {seed}] acc {metrics['accuracy']:.4f} F1-macro {metrics['f1_macro']:.4f} "
              f"(train {entry['diag']['train_f1_macro']:.4f}) FPR {metrics['fpr']:.4f} "
              f"recall_tc {metrics['recall_attack']:.4f} AUC {metrics['roc_auc']:.4f} p95 {lat['p95_ms']:.2f} ms, "
              f"{n_params:,} tham số, {size_kb:.0f} KB, {time.time() - t_seed:.0f} s ({row['commit']})")
        search_path.write_text(json.dumps(search, ensure_ascii=False, indent=1), encoding="utf-8")   # ghi sau mỗi seed
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    (data_dir, dataset, results_csv, search_json), opt = parse_args(sys.argv[1:])
    run(data_dir, dataset, results_csv, search_json, oof=opt.get("oof", "1") == "1", models_dir=opt.get("models"),
        k=int(opt.get("k", 5)), hp=json.loads(opt["hp"]) if "hp" in opt else None, verbose=opt.get("verbose") == "1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
