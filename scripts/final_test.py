"""Chạy tập test MỘT LẦN cho mọi mô hình đã đóng băng (task 6.1, 5.6, GĐ H), hoặc chạy thử trên val (`split=val`) trước khi đóng băng.

Chạy:  python -I scripts/final_test.py <config.json>

config.json (xem notebooks/17_final_test.ipynb):
  split            "test" hoặc "val" ("val" = chạy thử toàn bộ quy trình, không đụng test)
  out              thư mục ghi kết quả
  expect_commit    (tùy chọn) mã commit rút gọn bắt buộc khớp `git rev-parse --short HEAD` của repo đang chạy, không "-dirty"
  seeds, models    (tùy chọn) mặc định 42, 43, 44 và mọi mô hình
  cross            (tùy chọn) true: thử chéo giữa các bộ (task 5.6)
  thresholds       thresholds.json của GĐ G (ngưỡng đã đóng băng, mọi mô hình trừ lgbm_neighbor_mean)
  thresholds_c     {bộ: thresholds_c_<bộ>.json} của notebook 16h (ngưỡng của lgbm_neighbor_mean)
  datasets         {bộ: {data, ckpt, folds, light, baseline_search, gnn_csv, hybrid_csv, ablation_search, cross_to}}

Mô hình (cả 2 bộ, 3 seed): logreg, random_forest, xgboost, lightgbm, mlp, lgbm_neighbor_mean (huấn luyện lại bằng đúng cấu hình đã
đóng băng); gcn, sage, gat (nạp checkpoint); hybrid_gat_lgbm_oofavg_il (hybrid chính, 5 GAT nhóm) và hybrid_light (chỉ NF-ToN-IoT-v3, 2 GAT
nhóm + LightGBM n=100, l=15; ở NF-BoT-IoT-v3 hybrid nhẹ trùng hybrid chính nên không chạy riêng).

Với mỗi (bộ, mô hình, seed): (1) dựng mô hình; (2) dự đoán val và KIỂM F1-macro val khớp F1 đã ghi lúc hiệu chỉnh ngưỡng (lệch > 5e-4 thì
dừng, chưa đọc test); (3) dự đoán tập đánh giá MỘT lần; (4) từ cùng xác suất đó ghi HAI chế độ: argmax (tên mô hình giữ nguyên) và ngưỡng
đã đóng băng (tên mô hình thêm hậu tố `_thr`), mỗi chế độ một dòng 20 cột qua `evaluate.log_result`; (5) chỉ số theo tầng; (6) dự đoán từng node.
Sổ `ledger_<split>.csv` ghi thời điểm chạy; chạy lại cùng (bộ, mô hình, seed, split) trong cùng thư mục out bị từ chối ("đúng một lần").

Thử chéo (task 5.6): mô hình hybrid chính, gat, lightgbm, lgbm_neighbor_mean huấn luyện trên bộ A, đánh giá trên bộ B (đặc trưng node của B được
chuẩn hóa lại theo scaler của A), chỉ giữ node B thuộc lớp chung (Normal, DoS_DDoS, Truy_cap_trai_phep) và chỉ xét các lớp chung khi dự đoán:
argmax là argmax trên 3 cột lớp chung; "_thr" dùng ngưỡng của mô hình A trên điểm 1 - P(Normal) rồi chọn lớp tấn công chung có xác suất cao nhất.
Cột dataset là "A→B".

Ghi vào <out>:
  results_<split>_<bộ>.csv      20 cột của evaluate.py (kể cả dòng thử chéo "A→B")
  tiers_<split>_<bộ>.csv        chỉ số theo tầng: all, initiator (n_out > 0), receive_only
  pred_<bộ>_<mô hình>_s<seed>.parquet   y_true, pred (argmax), pred_thr, score (= 1 - P(Normal)), theo thứ tự node của nodes_<bộ>_<split>.parquet
  nodes_<bộ>_<split>.parquet    gi, win, ip, y, initiator, 18 đặc trưng
  val_check_<split>_<bộ>.json   F1 val dựng lại so với F1 val tham chiếu
  ledger_<split>.csv            thời điểm chạy từng mục
"""

import csv
import importlib.util
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data as D  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import fpr as F  # noqa: E402
from src import hybrid as H  # noqa: E402
from src import models as M  # noqa: E402
from src import threshold as TH  # noqa: E402
from src import train as T  # noqa: E402

SEEDS = (42, 43, 44)
BASELINES = ("logreg", "random_forest", "xgboost", "lightgbm", "mlp")
GNNS = ("gcn", "sage", "gat")
HYBRID, LIGHT, NBR = "hybrid_gat_lgbm_oofavg_il", "hybrid_light", "lgbm_neighbor_mean"
MODELS = (*BASELINES, NBR, *GNNS, HYBRID, LIGHT)
CROSS_MODELS = (HYBRID, "gat", "lightgbm", NBR)
COMMON = ["Normal", "DoS_DDoS", "Truy_cap_trai_phep"]       # các lớp có ở cả hai bộ
VAL_TOL = 5e-4                                               # lệch F1 val tối đa so với F1 val lúc hiệu chỉnh ngưỡng
LIGHT_FOLDS, LIGHT_CELL = 2, "lgbm_n100_l15.pkl"             # hybrid nhẹ ToN: 2 GAT nhóm đầu + LightGBM n=100, l=15
TIERS = ("all", "initiator", "receive_only")


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RB, RT, RC = _load_script("run_baselines"), _load_script("run_thresholds"), _load_script("run_threshold_c")


# ---------------------------------------------------------------- tiện ích
def x_of(graphs):
    return torch.cat([g.x for g in graphs]).numpy()


def y_of(graphs):
    return torch.cat([g.y for g in graphs]).numpy()


def rescale(graphs, src_scaler, dst_scaler):
    """Đưa đặc trưng node từ không gian chuẩn hóa của `src_scaler` sang của `dst_scaler` (cùng 18 đặc trưng thô sau log1p);
    đặc trưng cạnh (log1p, không chuẩn hóa) giữ nguyên."""
    out = []
    for g in graphs:
        raw = g.x.numpy().astype(np.float64) * src_scaler.scale_ + src_scaler.mean_
        x = ((raw - dst_scaler.mean_) / dst_scaler.scale_).astype(np.float32)
        out.append(Data(x=torch.from_numpy(x), y=g.y, edge_index=g.edge_index, edge_attr=g.edge_attr))
    return out


def common_codes(classes):
    """(chỉ số cột của các lớp chung trong `classes`, bảng đổi mã lớp của `classes` sang mã lớp chung hoặc -1)."""
    cols = [classes.index(c) for c in COMMON]
    lut = np.full(len(classes), -1, dtype=np.int64)
    lut[cols] = np.arange(len(COMMON))
    return cols, lut


def cross_predictions(proba, cols, threshold):
    """Dự đoán trong không gian lớp chung từ xác suất đầy đủ của mô hình nguồn. Trả về (argmax, theo ngưỡng, điểm tấn công)."""
    proba = np.asarray(proba, dtype=np.float64)
    score = 1.0 - proba[:, 0]
    sub = proba[:, cols]
    return sub.argmax(1), np.where(score > threshold, 1 + sub[:, 1:].argmax(1), 0), score


def train_times(path, dataset, model_names):
    """(mô hình, seed) -> thời gian huấn luyện (giây) từ CSV kết quả val cũ; thiếu thì nan."""
    out = {}
    if path and Path(path).exists():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["dataset"] == dataset and r["model"] in model_names and r["split"] == "val":
                    out[(r["model"], int(r["seed"]))] = float(r["train_time_s"])
    return out


def threshold_entry(entry):
    """Ngưỡng và F1 val tham chiếu từ một mục của thresholds.json (phẳng) hoặc của thresholds_c_*.json (lồng `before`)."""
    ref = entry["f1_macro_before"] if "f1_macro_before" in entry else entry["before"]["f1_macro"]
    return float(entry["threshold"]), float(ref)


class Predictor:
    """Một mô hình đã sẵn sàng: `proba(graphs)` → (số node, K); `latency()` đo p95 ms/cửa sổ trên CPU 1 luồng (gọi sau khi dự đoán xong)."""

    def __init__(self, proba, n_params, size_kb, train_time_s, latency):
        self.proba, self.n_params, self.size_kb, self.train_time_s, self.latency = proba, n_params, size_kb, train_time_s, latency


class Context:
    """Dữ liệu và đường dẫn của một bộ dữ liệu."""

    def __init__(self, cfg, dataset, spec, split):
        self.cfg, self.ds, self.spec, self.split = cfg, dataset, spec, split
        graphs, meta = D.load_split(spec["data"])
        self.meta, self.classes = meta, meta["classes"]
        self.K = len(self.classes)
        self.train, self.val = T.strip(graphs["train"]), T.strip(graphs["val"])
        self.target_raw = graphs[split]                                  # có ips, win (cho bảng node)
        self.target = T.strip(self.target_raw)
        self.window = T.strip([D.sample_window(graphs["val"])])[0]
        del graphs
        self.Xtr, self.ytr = D.node_arrays(self.train)
        self.Xva, self.yva = D.node_arrays(self.val)
        self.w = D.class_weights(self.ytr, self.K)
        self.scaler = pickle.load(open(Path(spec["data"]) / "scaler.pkl", "rb"))
        thr = json.loads(Path(cfg["thresholds"]).read_text(encoding="utf-8"))
        self.thr = thr[dataset]
        self.thr_c = json.loads(Path(cfg["thresholds_c"][dataset]).read_text(encoding="utf-8"))[dataset][NBR]
        search = json.loads(Path(spec["baseline_search"]).read_text(encoding="utf-8"))
        self.best = {n: RT.parse_cfg(search[n][-1]["best"]) for n in BASELINES}
        self.c_cfg = json.loads(Path(spec["ablation_search"]).read_text(encoding="utf-8"))[RC.ABLATION_MODEL]["best"]
        self.times = {**train_times(spec.get("gnn_csv"), dataset, GNNS), **train_times(spec.get("hybrid_csv"), dataset, (HYBRID,))}
        self.Ftr_nbr = None

    def entry(self, name, seed):
        e = (self.thr_c if name == NBR else self.thr[name])[str(seed)]
        return threshold_entry(e)

    def nbr_train(self):
        if self.Ftr_nbr is None:
            self.Ftr_nbr = RC.neighbor_features(self.train)[0]
        return self.Ftr_nbr


# ---------------------------------------------------------------- dựng mô hình
def build_baseline(ctx, name, seed):
    t0 = time.time()
    m = RB.fit(name, RB.make_models(seed)[name][0](ctx.best[name]), ctx.Xtr, ctx.ytr, ctx.w)
    train_s = time.time() - t0
    wx = ctx.window.x.numpy()

    def latency():
        if "n_jobs" in m.get_params():
            m.set_params(n_jobs=1)
        return ev.measure_latency(lambda: m.predict(wx))["p95_ms"]

    return Predictor(lambda graphs: RT.full_proba(m, x_of(graphs), ctx.K), ev.count_params(m), round(ev.model_size_kb(m), 1),
                     round(train_s, 1), latency)


def build_neighbor_mean(ctx, seed):
    t0 = time.time()
    m = H.fit_lgbm(ctx.nbr_train(), ctx.ytr, ctx.c_cfg, seed, ctx.K)
    train_s = time.time() - t0
    window = ctx.window

    def latency():
        m.set_params(n_jobs=1)
        return ev.measure_latency(lambda: m.predict(RC.neighbor_features([window])[0]))["p95_ms"]

    return Predictor(lambda graphs: H.predict_proba(m, RC.neighbor_features(graphs)[0], ctx.K), ev.count_params(m),
                     round(ev.model_size_kb(m), 1), round(train_s, 1), latency)


def build_gnn(ctx, name, seed):
    path = Path(ctx.spec["ckpt"]) / f"{ctx.ds}_{name}_s{seed}.pt"
    model = M.GNN(name, in_dim=len(ctx.meta["features"]), n_classes=ctx.K)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    window = ctx.window

    def latency():
        with torch.no_grad():
            return ev.measure_latency(lambda: model(window.x, window.edge_index, window.edge_attr))["p95_ms"]

    return Predictor(lambda graphs: T.predict(model, graphs, torch.device("cpu"))[1], ev.count_params(model),
                     round(ev.model_size_kb(model), 1), ctx.times.get((name, seed), float("nan")), latency)


def build_hybrid(ctx, name, seed):
    """Hybrid chính (5 GAT nhóm + LightGBM của 15e/15f) hoặc hybrid nhẹ (2 GAT nhóm đầu + LightGBM nhỏ của 16d)."""
    d = Path(ctx.spec["folds"]) / f"{ctx.ds}_s{seed}"
    light = name == LIGHT
    n_gat = LIGHT_FOLDS if light else 5
    gats = [H.load_gat(d / f"gat_fold{f}.pt", ctx.K) for f in range(n_gat)]
    lgbm_path = (Path(ctx.spec["light"]) / f"{ctx.ds}_s{seed}" / LIGHT_CELL) if light else (d / "lgbm_oofavg_il.pkl")
    lgbm = pickle.load(open(lgbm_path, "rb"))
    lgbm.set_params(n_jobs=-1)
    n_params = sum(ev.count_params(g) for g in gats) + H.lgbm_n_leaves(lgbm)
    if not light:
        meta = json.loads((d / "meta_oofavg_il.json").read_text(encoding="utf-8"))
        if meta["n_params"] != n_params:
            raise RuntimeError(f"{ctx.ds} seed {seed}: số tham số hybrid {n_params} khác meta {meta['n_params']}")
    size_kb = round(sum(ev.model_size_kb(g) for g in gats) + ev.model_size_kb(lgbm), 1)
    window = ctx.window

    def proba(graphs):
        x = x_of(graphs)
        views = [H.build_features(H.extract_embeddings(g, graphs), x) for g in gats]
        return H.predict_proba_views(lgbm, views, ctx.K)

    def latency():
        lgbm.set_params(n_jobs=1)
        return ev.measure_latency(lambda: H.predict_window_ensemble(gats, lgbm, window, ctx.K))["p95_ms"]

    return Predictor(proba, n_params, size_kb, float("nan") if light else ctx.times.get((HYBRID, seed), float("nan")), latency)


def build(ctx, name, seed):
    if name in BASELINES:
        return build_baseline(ctx, name, seed)
    if name == NBR:
        return build_neighbor_mean(ctx, seed)
    if name in GNNS:
        return build_gnn(ctx, name, seed)
    return build_hybrid(ctx, name, seed)


# ---------------------------------------------------------------- ghi kết quả
def _metrics(y, pred, score, classes, macro):
    return ev.compute_metrics(y, pred, score, class_names=classes, macro_labels=macro)


def tier_rows(ds, model, seed, split, y, pred, score, init, classes):
    rows = []
    for tier, mask in (("all", np.ones(len(y), bool)), ("initiator", init), ("receive_only", ~init)):
        if not mask.any():
            continue
        m = _metrics(y[mask], pred[mask], score[mask], classes, T.MACRO_CODES)
        rows.append({"dataset": ds, "model": model, "seed": seed, "split": split, "tier": tier, "n_nodes": int(mask.sum()),
                     "n_normal": int((y[mask] == ev.NORMAL).sum()), "n_attack": int((y[mask] != ev.NORMAL).sum()),
                     **{k: m[k] for k in ("accuracy", "f1_macro", "fpr", "recall_attack", "roc_auc")}})
    return rows


def append_csv(path, rows):
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


class Ledger:
    """Sổ ghi thời điểm chạy; từ chối chạy lại cùng (bộ, mô hình, seed, split)."""

    FIELDS = ["ts_utc", "dataset", "model", "seed", "split", "commit", "val_f1_ref", "val_f1_rebuilt", "val_f1_diff", "seconds", "note"]

    def __init__(self, path):
        self.path = Path(path)
        self.done = set()
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                self.done = {(r["dataset"], r["model"], r["seed"], r["split"]) for r in csv.DictReader(f)}

    def check(self, ds, model, seed, split):
        if (ds, model, str(seed), split) in self.done:
            raise RuntimeError(f"{ds} {model} seed {seed} split={split} đã chạy trong {self.path}: không chạy lại (mỗi mục chỉ chạy một lần)")

    def add(self, **kw):
        row = {k: kw.get(k, "") for k in self.FIELDS}
        row["ts_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        append_csv(self.path, [row])
        self.done.add((row["dataset"], row["model"], str(row["seed"]), row["split"]))


def run_dataset(cfg, ds, spec, ledger, commit, other=None):
    split, out = cfg["split"], Path(cfg["out"])
    ctx = Context(cfg, ds, spec, split)
    models = [m for m in cfg.get("models", MODELS) if not (ds == "nf_bot_iot_v3" and m == LIGHT)]
    seeds = tuple(cfg.get("seeds", SEEDS))
    y = y_of(ctx.target)
    Xt = x_of(ctx.target)
    zero = float(ctx.Xtr[:, 0].min())
    init = F.initiator_mask(Xt, zero)
    nodes = F.node_table(ctx.target_raw)
    nodes["initiator"] = init
    for i, c in enumerate(ctx.meta["features"]):
        nodes[c] = Xt[:, i]
    nodes.to_parquet(out / f"nodes_{ds}_{split}.parquet", index=False)
    print(f"{ds} [{split}]: {len(y):,} node, {init.mean():.1%} node có khởi tạo flow, lớp {ctx.classes}, mô hình {models}, seed {seeds}")
    res_csv, tier_csv, vc_path = out / f"results_{split}_{ds}.csv", out / f"tiers_{split}_{ds}.csv", out / f"val_check_{split}_{ds}.json"
    val_check = json.loads(vc_path.read_text(encoding="utf-8")) if vc_path.exists() else {}
    cross_ctx = None
    if cfg.get("cross") and other is not None:
        cross_ctx = other

    for name in models:
        for seed in seeds:
            ledger.check(ds, name, seed, split)
            t0 = time.time()
            thr, ref = ctx.entry(name, seed)
            pred_model = build(ctx, name, seed)
            proba_val = pred_model.proba(ctx.val)
            f1_val = _metrics(ctx.yva, proba_val.argmax(1), proba_val, ctx.classes, T.MACRO_CODES)["f1_macro"]
            diff = f1_val - ref
            val_check[f"{name}/s{seed}"] = {"ref": ref, "rebuilt": f1_val, "diff": diff,
                                            "val_fpr_at_threshold": float(_metrics(ctx.yva, TH.apply_threshold(proba_val, thr), proba_val, ctx.classes, T.MACRO_CODES)["fpr"])}
            vc_path.write_text(json.dumps(val_check, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{ds} {name} s{seed}] F1 val dựng lại {f1_val:.6f} so với tham chiếu {ref:.6f} (lệch {diff:+.2e})")
            if abs(diff) > VAL_TOL:
                raise RuntimeError(f"{ds} {name} seed {seed}: F1 val dựng lại lệch {diff:+.2e} > {VAL_TOL}: dừng trước khi đọc {split}")

            proba = proba_val if split == "val" else pred_model.proba(ctx.target)        # đánh giá MỘT lần
            preds = {name: proba.argmax(1), f"{name}_thr": TH.apply_threshold(proba, thr)}
            score = TH.attack_score(proba)
            n_threads = torch.get_num_threads()
            torch.set_num_threads(1)                                                      # D7: độ trễ trên CPU 1 luồng
            try:
                infer_ms = round(pred_model.latency(), 3)
            finally:
                torch.set_num_threads(n_threads)
            tiers = []
            for mname, pred in preds.items():
                m = _metrics(y, pred, proba, ctx.classes, T.MACRO_CODES)
                ev.log_result(m, dataset=ds, model=mname, task="multiclass", seed=seed, split=split, n_params=pred_model.n_params,
                              size_kb=pred_model.size_kb, train_time_s=pred_model.train_time_s, infer_ms_per_window=infer_ms, path=res_csv)
                tiers += tier_rows(ds, mname, seed, split, y, pred, score, init, ctx.classes)
                print(f"    {mname}: acc {m['accuracy']:.4f} F1-macro {m['f1_macro']:.4f} FPR {m['fpr']:.4f} recall {m['recall_attack']:.4f}")
            append_csv(tier_csv, tiers)
            pd.DataFrame({"y_true": y.astype(np.int8), "pred": preds[name].astype(np.int8), "pred_thr": preds[f"{name}_thr"].astype(np.int8),
                          "score": score.astype(np.float32)}).to_parquet(out / f"pred_{ds}_{name}_s{seed}.parquet", index=False)

            if cross_ctx is not None and name in CROSS_MODELS:
                oc = cross_ctx
                assert ctx.classes[0] == "Normal" and oc["classes"][0] == "Normal"
                cols_a, _ = common_codes(ctx.classes)
                _, lut_b = common_codes(oc["classes"])
                yb = lut_b[oc["y"]]
                keep = yb >= 0
                pb = pred_model.proba(oc["graphs_in"][ds])[keep]                         # đặc trưng của B đã chuẩn hóa theo scaler của A
                p_arg, p_thr, sc = cross_predictions(pb, cols_a, thr)
                for mname, pred in ((name, p_arg), (f"{name}_thr", p_thr)):
                    m = _metrics(yb[keep], pred, sc, COMMON, [0, 1, 2])
                    ev.log_result(m, dataset=f"{ds}→{oc['name']}", model=mname, task="multiclass", seed=seed, split=split,
                                  n_params=pred_model.n_params, size_kb=pred_model.size_kb, train_time_s=pred_model.train_time_s,
                                  infer_ms_per_window=infer_ms, path=res_csv)
                    print(f"    {ds}→{oc['name']} {mname}: F1-macro {m['f1_macro']:.4f} FPR {m['fpr']:.4f} recall {m['recall_attack']:.4f} ({int(keep.sum()):,} node)")
            ledger.add(dataset=ds, model=name, seed=seed, split=split, commit=commit, val_f1_ref=ref, val_f1_rebuilt=f1_val,
                       val_f1_diff=diff, seconds=round(time.time() - t0, 1))
            del pred_model


def load_other(cfg, ds, spec, other_ds, other_spec, ctx_scaler_path):
    """Đồ thị đánh giá của bộ `other_ds`, đã đổi sang không gian chuẩn hóa của bộ `ds` (cho thử chéo ds → other_ds)."""
    split = cfg["split"]
    graphs, meta = D.load_split(other_spec["data"])
    target = T.strip(graphs[split])
    del graphs
    sc_other = pickle.load(open(Path(other_spec["data"]) / "scaler.pkl", "rb"))
    sc_ds = pickle.load(open(ctx_scaler_path, "rb"))
    return {"name": other_ds, "classes": meta["classes"], "y": y_of(target), "graphs_in": {ds: rescale(target, sc_other, sc_ds)}}


def run(cfg):
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    split = cfg["split"]
    if split not in ("val", "test"):
        raise ValueError("split phải là val hoặc test")
    commit = ev.git_commit()
    if cfg.get("expect_commit") and commit != cfg["expect_commit"]:
        raise RuntimeError(f"Commit đang chạy {commit} khác commit bắt buộc {cfg['expect_commit']}")
    ledger = Ledger(out / f"ledger_{split}.csv")
    print(f"commit {commit}, split={split}, thiết bị CPU ({torch.get_num_threads()} luồng)")
    datasets = cfg["datasets"]
    for ds, spec in datasets.items():
        other = None
        if cfg.get("cross") and spec.get("cross_to"):
            o = spec["cross_to"]
            other = load_other(cfg, ds, spec, o, datasets[o], Path(spec["data"]) / "scaler.pkl")
        run_dataset(cfg, ds, spec, ledger, commit, other)
    print("xong")
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
