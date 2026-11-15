"""Lưới huấn luyện GNN cho ablation GAT (task 5.5), log tinh chỉnh GNN (task 5.2), bản cải tiến GAT (task 5.4). Chỉ val.

Chạy:  python -I scripts/run_gnn_grid.py <data_dir> <dataset> <out_dir> [parts=arch,improved,tuning] [window=<giây>] [seeds=42,43,44]

Cấu hình chính của GAT (task 5.3): hidden 16, heads 4, layers 2, có đặc trưng cạnh, cross-entropy có trọng số lớp. Mỗi cấu hình khác
cấu hình chính đúng một yếu tố.
- `arch`: GAT, đổi một yếu tố mỗi lần: heads {1, 2, 8}, layers {1, 3}, hidden {8, 32, 64}, không đặc trưng cạnh, cộng cấu hình chính (10 cấu
  hình). Đây cũng là log tinh chỉnh của GAT.
- `tuning`: GCN và GraphSAGE, hidden {8, 16, 32} (heads 4 nên độ rộng 32, 64, 128) × layers {2, 3} = 6 cấu hình mỗi mô hình.
- `improved`: GAT với focal loss; residual + LayerNorm; cả hai (task 5.4).
- `default`: chỉ GAT cấu hình chính (dùng cho ablation kích thước cửa sổ trên đồ thị W khác 5 giây; `window=` ghi vào cột `window_sec`).

Ghi vào <out_dir>: grid_<dataset>.csv (mỗi dòng một cấu hình và seed), curves_grid_<dataset>.csv (loss train, loss val, F1-macro val theo
epoch), thresholds_grid_<dataset>.json (ngưỡng FPR val ≤ 4,08% cho từng lần chạy), ckpt_grid/*.pt.
"""

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
from src import evaluate as ev  # noqa: E402
from src import models as M  # noqa: E402
from src import threshold as TH  # noqa: E402
from src import train as T  # noqa: E402

SEEDS = (42, 43, 44)
MAX_TRIALS = 20                                   # task 5.2 tiêu chí 3: ≤ 20 lần thử mỗi mô hình
MAIN = dict(kind="gat", hidden=16, heads=4, layers=2, use_edge=True, residual=False, layer_norm=False, loss="ce")
COLUMNS = ["dataset", "part", "factor", "value", "cfg_id", "kind", "hidden", "heads", "layers", "use_edge", "residual", "layer_norm",
           "loss", "seed", "window_sec", "n_params", "size_kb", "epochs", "best_epoch", "train_s", "p95_ms", "accuracy", "f1_macro",
           "fpr", "recall_attack", "roc_auc", "recall_botnet", "f1_per_class", "recall_per_class",
           "thr", "thr_feasible", "thr_f1_macro", "thr_fpr", "thr_recall_attack", "commit"]


def experiments(part, window=None):
    """Danh sách cấu hình của một phần. Mỗi phần tử là dict MAIN đã đổi + id, part, factor, value."""
    def exp(cfg_id, factor, value, **change):
        return {**MAIN, **change, "cfg_id": cfg_id, "part": part, "factor": factor, "value": value}

    if part == "arch":
        out = [exp("gat_default", "default", "default")]
        out += [exp(f"gat_heads{h}", "heads", h, heads=h) for h in (1, 2, 8)]
        out += [exp(f"gat_layers{n}", "layers", n, layers=n) for n in (1, 3)]
        out += [exp(f"gat_hidden{h}", "hidden", h, hidden=h) for h in (8, 32, 64)]
        out += [exp("gat_noedge", "edge", "no", use_edge=False)]
    elif part == "tuning":
        out = [exp(f"{k}_h{h}_l{n}", "tuning", f"width{h * MAIN['heads']}_layers{n}", kind=k, hidden=h, layers=n)
               for k in ("gcn", "sage") for h in (8, 16, 32) for n in (2, 3)]
    elif part == "improved":
        out = [exp("gat_focal", "improved", "focal", loss="focal"),
               exp("gat_resln", "improved", "residual+layernorm", residual=True, layer_norm=True),
               exp("gat_focal_resln", "improved", "focal+residual+layernorm", loss="focal", residual=True, layer_norm=True)]
    elif part == "default":
        out = [exp("gat_default", "window", window if window is not None else 5)]
    else:
        raise ValueError(f"part phải thuộc arch, tuning, improved, default; nhận {part!r}")
    for kind in {e["kind"] for e in out}:
        n = sum(e["kind"] == kind for e in out)
        if n > MAX_TRIALS:
            raise ValueError(f"{kind}: {n} cấu hình > {MAX_TRIALS} lần thử")
    return out


def _load_run_gnn():
    spec = importlib.util.spec_from_file_location("run_gnn", ROOT / "scripts" / "run_gnn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(data_dir, dataset, out_dir, parts=("arch", "improved", "tuning"), window=None, seeds=SEEDS):
    out = Path(out_dir)
    (out / "ckpt_grid").mkdir(parents=True, exist_ok=True)
    grid_csv, curves_csv = out / f"grid_{dataset}.csv", out / f"curves_grid_{dataset}.csv"
    thr_json = out / f"thresholds_grid_{dataset}.json"
    assert not grid_csv.exists(), f"{grid_csv} đã có từ trước"
    RG = _load_run_gnn()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs, meta = D.load_split(data_dir)
    classes, K = meta["classes"], len(meta["classes"])
    train, val = T.strip(graphs["train"]), T.strip(graphs["val"])
    window_g = T.strip([D.sample_window(graphs["val"])])[0]
    win_sec = window if window is not None else meta.get("window_sec")
    del graphs
    print(f"{dataset}: train {len(train):,} đồ thị, val {len(val):,}, lớp {classes}, cửa sổ {win_sec} s, thiết bị {device}")

    n_threads = torch.get_num_threads()
    thr, n_done = {dataset: {}}, 0
    with open(grid_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for part in parts:
            for e in experiments(part, window):
                for seed in seeds:
                    t0 = time.time()
                    T.set_seed(seed)
                    model = M.GNN(e["kind"], in_dim=len(meta["features"]), n_classes=K, hidden=e["hidden"], heads=e["heads"],
                                  layers=e["layers"], use_edge=e["use_edge"], residual=e["residual"], layer_norm=e["layer_norm"])
                    model, history, train_s = T.fit(model, train, val, classes, device, seed, loss=e["loss"], verbose=False)
                    y, proba = T.predict(model, val, device)
                    pred = proba.argmax(1)
                    m = ev.compute_metrics(y, pred, proba, class_names=classes, macro_labels=T.MACRO_CODES)
                    cal = TH.calibrate(y, proba, classes, T.MACRO_CODES)
                    recall_cls = {c: (float((pred[y == i] == i).mean()) if (y == i).any() else float("nan"))
                                  for i, c in enumerate(classes)}
                    cpu = model.to("cpu").eval()
                    torch.set_num_threads(1)
                    with torch.no_grad():
                        lat = ev.measure_latency(lambda: cpu(window_g.x, window_g.edge_index, window_g.edge_attr))
                    torch.set_num_threads(n_threads)
                    best_epoch = 1 + max(range(len(history)), key=lambda i: (history[i]["val_f1_macro"], -i))
                    row = {"dataset": dataset, "part": part, "factor": e["factor"], "value": e["value"], "cfg_id": e["cfg_id"],
                           "kind": e["kind"], "hidden": e["hidden"], "heads": e["heads"], "layers": e["layers"],
                           "use_edge": e["use_edge"], "residual": e["residual"], "layer_norm": e["layer_norm"], "loss": e["loss"],
                           "seed": seed, "window_sec": win_sec, "n_params": ev.count_params(cpu),
                           "size_kb": round(ev.model_size_kb(cpu), 1), "epochs": len(history), "best_epoch": best_epoch,
                           "train_s": round(train_s, 1), "p95_ms": round(lat["p95_ms"], 3), "accuracy": m["accuracy"],
                           "f1_macro": m["f1_macro"], "fpr": m["fpr"], "recall_attack": m["recall_attack"], "roc_auc": m["roc_auc"],
                           "recall_botnet": recall_cls.get("Botnet_Backdoor", float("nan")),
                           "f1_per_class": json.dumps(m["f1_per_class"], ensure_ascii=False),
                           "recall_per_class": json.dumps(recall_cls, ensure_ascii=False),
                           "thr": cal["threshold"], "thr_feasible": cal["feasible"], "thr_f1_macro": cal["after"]["f1_macro"],
                           "thr_fpr": cal["after"]["fpr"], "thr_recall_attack": cal["after"]["recall_attack"],
                           "commit": ev.git_commit()}
                    writer.writerow(row)
                    fh.flush()
                    RG.write_curves(curves_csv, dataset, e["cfg_id"], seed, history)
                    torch.save(cpu.state_dict(), out / "ckpt_grid" / f"{dataset}_{e['cfg_id']}_s{seed}.pt")
                    thr[dataset].setdefault(e["cfg_id"], {})[str(seed)] = cal
                    thr_json.write_text(json.dumps(thr, ensure_ascii=False, indent=1), encoding="utf-8")
                    n_done += 1
                    print(f"[{dataset} {e['cfg_id']} seed {seed}] {row['n_params']:,} tham số, {len(history)} epoch (tốt nhất {best_epoch}), "
                          f"acc {m['accuracy']:.4f} F1-macro {m['f1_macro']:.4f} FPR {m['fpr']:.4f} recall {m['recall_attack']:.4f} "
                          f"| ngưỡng {cal['threshold']:.4f} → FPR {cal['after']['fpr']:.4f} recall {cal['after']['recall_attack']:.4f} "
                          f"| {time.time() - t0:.0f} s ({n_done} lần chạy)")
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pos = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    data_dir, dataset, out_dir = pos[:3]
    return run(data_dir, dataset, out_dir, parts=tuple(opt.get("parts", "arch,improved,tuning").split(",")),
               window=int(opt["window"]) if "window" in opt else None,
               seeds=tuple(int(s) for s in opt.get("seeds", "42,43,44").split(",")))


if __name__ == "__main__":
    sys.exit(main())
