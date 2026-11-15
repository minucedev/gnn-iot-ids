"""Bảng đếm tham số theo lớp của GAT và trọng số attention của 2 cửa sổ mẫu (task 5.3, tiêu chí 1 và 4). Chỉ val.

Chạy:  python -I scripts/gat_explain.py <data_dir> <dataset> <out_dir> ckpt=<thư mục ckpt> [seed=42]

Nạp GAT cấu hình chính (`<ckpt>/<dataset>_gat_s<seed>.pt`), ghi vào <out_dir>:
- gat_param_table_<dataset>.csv: mỗi tham số một dòng (lớp, tên, hình dạng, số phần tử);
- gat_param_summary_<dataset>.json: tổng theo lớp, tổng toàn mô hình, kích thước file state_dict;
- attention_samples_<dataset>.json: với hai lớp mẫu (DoS_DDoS và Truy_cap_trai_phep, nhóm gồm cả dò quét), một cửa sổ val nhỏ (6–16 node)
  mà mô hình đoán đúng nhiều node thuộc lớp đó nhất; lưu IP, nhãn, dự đoán, và với từng lớp GAT các cạnh (gồm cạnh ngược và vòng lặp tự
  thân đúng như `forward` dùng) cùng hệ số attention của từng head, để GĐ H vẽ hình giải thích.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import models as M  # noqa: E402

SAMPLE_CLASSES = ("DoS_DDoS", "Truy_cap_trai_phep")
MIN_NODES, MAX_NODES, TARGET_NODES = 6, 16, 10


def param_table(model):
    """Danh sách dict (layer, param, shape, numel) theo thứ tự khai báo, layer = convs.<i> hoặc head."""
    rows = []
    for name, p in model.named_parameters():
        parts = name.split(".")
        layer = ".".join(parts[:2]) if parts[0] in ("convs", "norms", "skips") else parts[0]
        rows.append({"layer": layer, "param": name[len(layer) + 1:], "shape": "x".join(str(s) for s in p.shape), "numel": int(p.numel())})
    return rows


def param_summary(model, rows):
    by_layer = {}
    for r in rows:
        by_layer[r["layer"]] = by_layer.get(r["layer"], 0) + r["numel"]
    total = sum(by_layer.values())
    assert total == ev.count_params(model)
    return {"total_params": total, "by_layer": by_layer, "size_kb": round(ev.model_size_kb(model), 1)}


@torch.no_grad()
def pick_window(model, graphs, class_idx):
    """Chỉ số cửa sổ val nhỏ có nhiều node lớp `class_idx` được đoán đúng nhất (hòa: số node gần TARGET_NODES hơn, rồi sớm hơn)."""
    best, best_key = None, None
    for gi, g in enumerate(graphs):
        if not (MIN_NODES <= g.num_nodes <= MAX_NODES) or g.num_edges == 0:
            continue
        target = g.y == class_idx
        if not bool(target.any()):
            continue
        pred = model(g.x, g.edge_index, g.edge_attr).argmax(1)
        key = (int((pred[target] == class_idx).sum()), -abs(g.num_nodes - TARGET_NODES), -gi)
        if best_key is None or key > best_key:
            best, best_key = gi, key
    return best


@torch.no_grad()
def sample_entry(model, g, gi, class_name, classes):
    attn = model.attention(g.x, g.edge_index, g.edge_attr)
    pred = model(g.x, g.edge_index, g.edge_attr).argmax(1)
    return {
        "sample_class": class_name, "val_graph_index": int(gi), "window": int(g.win), "t_start": float(g.t_start),
        "n_nodes": int(g.num_nodes), "n_edges_original": int(g.num_edges), "ips": list(g.ips),
        "y": [classes[i] for i in g.y.tolist()], "pred": [classes[i] for i in pred.tolist()],
        "edge_index_original": g.edge_index.tolist(), "edge_attr_original": np.round(g.edge_attr.numpy(), 5).tolist(),
        "layers": [{"edge_index": ei.tolist(), "alpha": np.round(a.numpy(), 5).tolist(), "n_heads": int(a.shape[1])}
                   for ei, a in attn],
        "note": ("Cạnh của mỗi lớp = cạnh gốc, cạnh ngược (bidirectional), rồi vòng lặp tự thân; alpha (E', heads) đã softmax theo node nhận."),
    }


def run(data_dir, dataset, out_dir, ckpt_dir, seed=42):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graphs, meta = D.load_split(data_dir)
    classes = meta["classes"]
    val = graphs["val"]                                    # giữ nguyên ips, win để vẽ hình
    path = Path(ckpt_dir) / f"{dataset}_gat_s{seed}.pt"
    model = M.GNN("gat", in_dim=len(meta["features"]), n_classes=len(classes))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()

    rows = param_table(model)
    summary = {"dataset": dataset, "seed": seed, "checkpoint": path.name, "config": {"hidden": 16, "heads": 4, "layers": 2, "edge_dim": 3},
               **param_summary(model, rows)}
    import csv
    with open(out / f"gat_param_table_{dataset}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "param", "shape", "numel"])
        w.writeheader()
        w.writerows(rows)
    (out / f"gat_param_summary_{dataset}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    samples = []
    for name in SAMPLE_CLASSES:
        if name not in classes:
            print(f"Bộ {dataset} không có lớp {name}, bỏ qua")
            continue
        gi = pick_window(model, val, classes.index(name))
        if gi is None:
            print(f"Không tìm thấy cửa sổ val {MIN_NODES}-{MAX_NODES} node có lớp {name}")
            continue
        e = sample_entry(model, val[gi], gi, name, classes)
        samples.append(e)
        print(f"[{name}] cửa sổ val #{gi} (win {e['window']}), {e['n_nodes']} node, {e['n_edges_original']} cạnh; nhãn {e['y']}; đoán {e['pred']}")
    (out / f"attention_samples_{dataset}.json").write_text(
        json.dumps({"dataset": dataset, "seed": seed, "classes": classes, "samples": samples}, ensure_ascii=False), encoding="utf-8")
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pos = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    data_dir, dataset, out_dir = pos[:3]
    return run(data_dir, dataset, out_dir, opt["ckpt"], seed=int(opt.get("seed", 42)))


if __name__ == "__main__":
    sys.exit(main())
