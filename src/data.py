"""Nạp đồ thị đã dựng (graphs_*.pt, split.json) theo tập train/val/test.

Thư mục dữ liệu do scripts/build_dataset_cli.py ghi ra. Mọi mô hình dùng chung split.json
(KE_HOACH.md quy ước 6.3) nên cả baseline lẫn GNN đều đọc qua module này.
"""

import json
from pathlib import Path

import numpy as np
import torch

PARTS = ("train", "val", "test")


def load_split(data_dir):
    """Trả về (graphs_theo_tập, split_json). graphs_theo_tập[part] là danh sách Data theo thứ tự thời gian."""
    d = Path(data_dir)
    meta = json.loads((d / "split.json").read_text(encoding="utf-8"))
    graphs = torch.load(d / meta["graph_file"], weights_only=False)
    return {p: [graphs[i] for i in meta[p]["graph_idx"]] for p in PARTS}, meta


def node_arrays(graphs):
    """Ghép mọi node của danh sách đồ thị thành (X, y); X đã chuẩn hóa (scaler fit trên train)."""
    X = torch.cat([g.x for g in graphs]).numpy()
    y = torch.cat([g.y for g in graphs]).numpy()
    return X, y


def sample_window(graphs, n_nodes=None):
    """Chọn 1 đồ thị đại diện để đo độ trễ: có số node gần n_nodes nhất (mặc định số node trung vị)."""
    sizes = np.array([g.num_nodes for g in graphs])
    target = np.median(sizes) if n_nodes is None else n_nodes
    return graphs[int(np.argmin(np.abs(sizes - target)))]


def class_weights(y, n_classes):
    """Trọng số mẫu cân bằng: n / (K * số mẫu của lớp), lớp vắng mặt có trọng số 0."""
    cnt = np.bincount(y, minlength=n_classes).astype(float)
    w = np.where(cnt > 0, len(y) / (n_classes * np.maximum(cnt, 1)), 0.0)
    return w
