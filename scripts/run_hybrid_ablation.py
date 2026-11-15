"""Ablation hybrid (c, d), đường Pareto hybrid nhẹ và ngưỡng của hybrid đầy đủ (task 5.8, GĐ G). Chỉ val, 3 seed.

Chạy:  python -I scripts/run_hybrid_ablation.py <data_dir> <dataset> <out_dir> folds=<models_hybrid> full=<ckpt GAT toàn train>
       [k=5] [interleave=20] [seeds=42,43,44]

Không huấn luyện lại GAT: dùng 5 GAT nhóm đã lưu của hybrid xen kẽ (`<folds>/<dataset>_s<seed>/gat_fold<f>.pt`, `meta_oofavg_il.json`,
`lgbm_oofavg_il.pkl`) và GAT huấn luyện trên toàn bộ train của GĐ E (`<full>/<dataset>_gat_s<seed>.pt`). Embedding out-of-fold của train
dựng lại bằng đúng cách chia của lúc huấn luyện (`H.oof_embeddings_from_models`).

Ghi vào <out_dir>:
- ablation_cd_<dataset>.csv: results.csv 20 cột cho (c) `abl_lgbm_x_nbrmean` (LightGBM trên 18 đặc trưng + trung bình 18 đặc trưng của
  node kề, tính tay) và (d) `abl_lgbm_emb_only` (LightGBM chỉ trên 64 chiều embedding out-of-fold, val = trung bình 5 góc nhìn). Cấu hình
  LightGBM của mỗi phương án chọn trên val ở seed đầu trong cùng lưới 3 cấu hình với hybrid.
- pareto_raw_<dataset>.csv: mỗi dòng một (biến thể suy luận, cấu hình LightGBM, seed): F1-macro, FPR, recall, tham số, kích thước, độ trễ,
  ngưỡng đã hiệu chỉnh (FPR val ≤ 4,08%). Biến thể gat{5,3,2,1}: trung bình xác suất qua m GAT nhóm đầu tiên (gat5 = hybrid đầy đủ);
  gatfull: MỘT GAT huấn luyện trên toàn train (LightGBM vẫn học trên embedding out-of-fold). Lưới LightGBM nhỏ: n_estimators × num_leaves,
  cộng cấu hình của hybrid đầy đủ (dùng lại file đã lưu).
- models_light/<dataset>_s<seed>/lgbm_n<n>_l<l>.pkl: mọi LightGBM nhỏ đã huấn luyện (cho GĐ H).
- ablation_search_<dataset>.json, ref_check_<dataset>.json: log chọn cấu hình, và kiểm tra dựng lại hybrid đầy đủ từ file đã lưu có khớp
  kết quả cũ không.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import hybrid as H  # noqa: E402
from src import threshold as TH  # noqa: E402
from src import train as T  # noqa: E402

SEEDS = (42, 43, 44)
REF_MODEL = "hybrid_gat_lgbm_oofavg_il"
ABL_NBR, ABL_EMB = "abl_lgbm_x_nbrmean", "abl_lgbm_emb_only"
SMALL_GRID = [{"n": n, "l": l} for n in (25, 50, 100) for l in (7, 15, 31)]
FOLD_VARIANTS = (5, 3, 2, 1)                       # số GAT nhóm dùng khi suy luận (các GAT đầu tiên)


def _kb(model):
    return ev.model_size_kb(model)


def _cell_name(cell):
    return f"n{cell['n']}_l{cell['l']}"


def _light_row(dataset, variant, n_gat, cell, seed, ref, proba, y, classes, n_params, size_kb, p95):
    cal = TH.calibrate(y, proba, classes, T.MACRO_CODES)
    b, a = cal["before"], cal["after"]
    return {"dataset": dataset, "variant": variant, "n_gat": n_gat, "lgbm_n": cell["n"], "lgbm_l": cell["l"], "seed": seed,
            "is_hybrid_full": bool(ref), "accuracy": b["accuracy"], "f1_macro": b["f1_macro"], "fpr": b["fpr"],
            "recall_attack": b["recall_attack"], "n_params": int(n_params), "size_kb": round(size_kb, 1), "p95_ms": p95,
            "thr": cal["threshold"], "thr_feasible": cal["feasible"], "thr_meets_recall": cal["meets_recall_target"],
            "thr_accuracy": a["accuracy"], "thr_f1_macro": a["f1_macro"], "thr_fpr": a["fpr"], "thr_recall_attack": a["recall_attack"]}


def run(data_dir, dataset, out_dir, folds_dir, full_dir, k=5, interleave=20, seeds=SEEDS, grid=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cd_csv = out / f"ablation_cd_{dataset}.csv"
    assert not cd_csv.exists(), f"{cd_csv} đã có từ trước"
    graphs, meta = D.load_split(data_dir)
    classes, K = meta["classes"], len(meta["classes"])
    train, val = T.strip(graphs["train"]), T.strip(graphs["val"])
    Xtr, ytr = D.node_arrays(train)
    Xva, yva = D.node_arrays(val)
    window = T.strip([D.sample_window(graphs["val"])])[0]
    del graphs
    print(f"{dataset}: train {len(ytr):,} node, val {len(yva):,} node, lớp {classes}")
    t0 = time.time()
    Ftr_nbr = np.hstack([Xtr, H.neighbor_mean(train)])                 # (c): 18 + 18 = 36 cột
    Fva_nbr = np.hstack([Xva, H.neighbor_mean(val)])
    print(f"đặc trưng hàng xóm: {Ftr_nbr.shape} / {Fva_nbr.shape} ({time.time() - t0:.0f} s)")

    n_threads = torch.get_num_threads()
    search, ref_check, pareto, cfgs = {"dataset": dataset}, {}, [], {}
    latency = {}                                                       # (biến thể, ô) -> p95 ms, đo ở seed đầu
    for si, seed in enumerate(seeds):
        t_seed = time.time()
        d = Path(folds_dir) / f"{dataset}_s{seed}"
        ref_cfg = json.loads((d / "meta_oofavg_il.json").read_text(encoding="utf-8"))["lgbm_cfg"]
        gats = [H.load_gat(d / f"gat_fold{f}.pt", K) for f in range(k)]
        emb_tr = H.oof_embeddings_from_models(train, gats, k, interleave)
        views_emb = [H.extract_embeddings(g, val) for g in gats]
        Ftr = H.build_features(emb_tr, Xtr)                            # 82 cột, như hybrid đầy đủ
        views = [H.build_features(e, Xva) for e in views_emb]
        print(f"[seed {seed}] embedding out-of-fold {emb_tr.shape}, {len(views)} góc nhìn val ({time.time() - t_seed:.0f} s)")

        # ---- kiểm tra: dựng lại hybrid đầy đủ từ file đã lưu có khớp kết quả cũ
        lgbm_ref = pickle.load(open(d / "lgbm_oofavg_il.pkl", "rb"))
        lgbm_ref.set_params(n_jobs=-1)
        p_ref = H.predict_proba_views(lgbm_ref, views, K)
        m_ref = ev.compute_metrics(yva, p_ref.argmax(1), p_ref, class_names=classes, macro_labels=T.MACRO_CODES)
        ref_check[str(seed)] = {"lgbm_cfg": ref_cfg, "f1_macro": m_ref["f1_macro"], "fpr": m_ref["fpr"],
                                "recall_attack": m_ref["recall_attack"]}
        print(f"  [hybrid đầy đủ dựng lại] F1-macro {m_ref['f1_macro']:.4f} FPR {m_ref['fpr']:.4f}")

        # ---- (c) và (d)
        for name, Ftrain, vviews in ((ABL_NBR, Ftr_nbr, [Fva_nbr]), (ABL_EMB, emb_tr, views_emb)):
            if name not in cfgs:
                cfgs[name], trials = H.search_lgbm(Ftrain, ytr, None, yva, classes, seed=seed, grid=grid, val_views=vviews)
                search[name] = {"best": cfgs[name], "trials": trials}
                print(f"  [chọn cấu hình {name}] {cfgs[name]}: " + ", ".join(f"{t['cfg']} → {t['val_f1_macro']:.4f}" for t in trials))
            t1 = time.time()
            lgbm = H.fit_lgbm(Ftrain, ytr, cfgs[name], seed, K)
            train_s = time.time() - t1
            proba = H.predict_proba_views(lgbm, vviews, K)
            m = ev.compute_metrics(yva, proba.argmax(1), proba, class_names=classes, macro_labels=T.MACRO_CODES)
            n_params = H.lgbm_n_leaves(lgbm) + (sum(ev.count_params(g) for g in gats) if name == ABL_EMB else 0)
            size_kb = _kb(lgbm) + (sum(_kb(g) for g in gats) if name == ABL_EMB else 0.0)
            ev.log_result(m, dataset=dataset, model=name, task="multiclass", seed=seed, split="val", n_params=n_params,
                          size_kb=round(size_kb, 1), train_time_s=round(train_s, 1), infer_ms_per_window=float("nan"), path=cd_csv)
            print(f"  [{name} seed {seed}] F1-macro {m['f1_macro']:.4f} FPR {m['fpr']:.4f} recall {m['recall_attack']:.4f} ({train_s:.0f} s)")
            del lgbm

        # ---- Pareto: GAT dùng khi suy luận × LightGBM nhỏ
        variants = {f"gat{m}": (m, views[:m]) for m in FOLD_VARIANTS if m <= k}
        full_path = Path(full_dir) / f"{dataset}_gat_s{seed}.pt"
        gat_full = None
        if full_path.exists():
            gat_full = H.load_gat(full_path, K)
            variants["gatfull"] = (1, [H.build_features(H.extract_embeddings(gat_full, val), Xva)])
        else:
            print(f"  CẢNH BÁO: không có {full_path}, bỏ biến thể gatfull")
        gat_params, gat_kb = ev.count_params(gats[0]), [_kb(g) for g in gats]
        cells = list(grid or SMALL_GRID)
        if ref_cfg not in cells:
            cells.append(ref_cfg)
        light_dir = out / "models_light" / f"{dataset}_s{seed}"
        light_dir.mkdir(parents=True, exist_ok=True)
        for cell in cells:
            is_ref_cell = cell == ref_cfg
            if is_ref_cell:
                lgbm = lgbm_ref
            else:
                lgbm = H.fit_lgbm(Ftr, ytr, cell, seed, K)
                with open(light_dir / f"lgbm_{_cell_name(cell)}.pkl", "wb") as fh:
                    pickle.dump(lgbm, fh)
            leaves, lg_kb = H.lgbm_n_leaves(lgbm), _kb(lgbm)
            for variant, (n_gat, vv) in variants.items():
                proba = H.predict_proba_views(lgbm, vv, K)
                key = (variant, _cell_name(cell))
                if si == 0:                                            # độ trễ không phụ thuộc seed: đo một lần
                    torch.set_num_threads(1)
                    lgbm.set_params(n_jobs=1)
                    fn = ((lambda: H.predict_window(gat_full, lgbm, window, K)) if variant == "gatfull"
                          else (lambda g=gats[:n_gat]: H.predict_window_ensemble(g, lgbm, window, K)))
                    latency[key] = round(ev.measure_latency(fn)["p95_ms"], 3)
                    lgbm.set_params(n_jobs=-1)
                    torch.set_num_threads(n_threads)
                size = (gat_kb[:n_gat] if variant != "gatfull" else [_kb(gat_full)])
                pareto.append(_light_row(dataset, variant, n_gat, cell, seed, variant == f"gat{k}" and is_ref_cell, proba, yva,
                                         classes, n_gat * gat_params + leaves, sum(size) + lg_kb, latency.get(key, float("nan"))))
            if not is_ref_cell:
                del lgbm
            r = [p for p in pareto if p["seed"] == seed and p["lgbm_n"] == cell["n"] and p["lgbm_l"] == cell["l"]]
            print(f"  [Pareto seed {seed} LightGBM {cell}] " + ", ".join(f"{p['variant']}: F1 {p['f1_macro']:.4f} {p['size_kb']:.0f} KB" for p in r))
        pd.DataFrame(pareto).to_csv(out / f"pareto_raw_{dataset}.csv", index=False)
        (out / f"ablation_search_{dataset}.json").write_text(json.dumps(search, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / f"ref_check_{dataset}.json").write_text(json.dumps(ref_check, ensure_ascii=False, indent=1), encoding="utf-8")
        (light_dir / "meta.json").write_text(json.dumps({
            "dataset": dataset, "seed": seed, "k": k, "interleave": interleave, "ref_lgbm_cfg": ref_cfg,
            "fold_gats": [str(d / f"gat_fold{f}.pt") for f in range(k)], "full_gat": str(full_path), "classes": classes,
            "variants": {v: n for v, (n, _) in variants.items()}}, ensure_ascii=False), encoding="utf-8")
        print(f"[seed {seed}] xong sau {time.time() - t_seed:.0f} s")
        del emb_tr, Ftr, views, views_emb, lgbm_ref, gats
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pos = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    data_dir, dataset, out_dir = pos[:3]
    return run(data_dir, dataset, out_dir, opt["folds"], opt["full"], k=int(opt.get("k", 5)),
               interleave=int(opt.get("interleave", 20)), seeds=tuple(int(s) for s in opt.get("seeds", "42,43,44").split(",")))


if __name__ == "__main__":
    sys.exit(main())
