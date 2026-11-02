"""Huấn luyện GCN, GraphSAGE, GAT nhẹ (task 5.2, 5.3) và ghi kết quả val vào results.csv.

Chạy:  python -I scripts/run_gnn.py <data_dir> <dataset_name> <results.csv> <out_dir> [models=gcn,sage,gat] [json_hyperparams]
Mỗi mô hình chạy 3 seed 42, 43, 44; lưu trọng số tốt nhất vào <out_dir>/<dataset>_<model>_s<seed>.pt.
Độ trễ (`infer_ms_per_window`) là p95 của 1 lượt forward trên CPU 1 luồng cho 1 cửa sổ có số node trung vị.
Chỉ ghi split=val (test chỉ chạy 1 lần ở GĐ6).
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import models as M  # noqa: E402
from src import train as T  # noqa: E402

SEEDS = (42, 43, 44)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data_dir, dataset, results_csv, out_dir = sys.argv[1:5]
    kinds = sys.argv[5].split(",") if len(sys.argv) > 5 else list(M.KINDS)
    hp = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
    model_hp = {k: hp[k] for k in ("hidden", "heads", "layers", "dropout", "bidirectional") if k in hp}
    train_hp = {k: hp[k] for k in ("lr", "weight_decay", "epochs", "patience", "batch_size") if k in hp}
    tag = hp.get("tag", "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs, meta = D.load_split(data_dir)
    classes = meta["classes"]
    train, val = T.strip(graphs["train"]), T.strip(graphs["val"])
    window = T.strip([D.sample_window(graphs["val"])])[0]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"{dataset}: train {len(train):,} đồ thị, val {len(val):,}, lớp {classes}, thiết bị {device}")

    for kind in kinds:
        for seed in SEEDS:
            T.set_seed(seed)
            model = M.GNN(kind, in_dim=len(meta["features"]), n_classes=len(classes), **model_hp)
            print(f"== {kind}{tag} seed {seed}: {ev.count_params(model):,} tham số")
            model, history, train_s = T.fit(model, train, val, classes, device, seed, **train_hp)

            y, proba = T.predict(model, val, device)
            metrics = ev.compute_metrics(y, proba.argmax(1), proba, class_names=classes, macro_labels=T.MAIN4)

            cpu = model.to("cpu").eval()
            torch.set_num_threads(1)
            with torch.no_grad():
                lat = ev.measure_latency(lambda: cpu(window.x, window.edge_index, window.edge_attr))
            row = ev.log_result(
                metrics, dataset=dataset, model=f"{kind}{tag}", task="multiclass", seed=seed, split="val",
                n_params=ev.count_params(cpu), size_kb=round(ev.model_size_kb(cpu), 1),
                train_time_s=round(train_s, 1), infer_ms_per_window=round(lat["p95_ms"], 3), path=results_csv)
            torch.save(cpu.state_dict(), Path(out_dir) / f"{dataset}_{kind}{tag}_s{seed}.pt")
            print(f"[{kind}{tag} seed {seed}] acc {metrics['accuracy']:.4f} F1-macro4 {metrics['f1_macro']:.4f} "
                  f"FPR {metrics['fpr']:.4f} recall_tc {metrics['recall_attack']:.4f} AUC {metrics['roc_auc']:.4f} "
                  f"p95 {lat['p95_ms']:.2f} ms, {len(history)} epoch ({row['commit']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
