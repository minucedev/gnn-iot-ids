"""Vòng huấn luyện và dự đoán chung cho các mô hình GNN (task 5.1).

- Mất mát: cross-entropy có trọng số lớp cân bằng (lớp vắng mặt trong train có trọng số 0).
- Chọn epoch tốt nhất theo F1-macro 4 lớp chính trên val (dừng sớm), không bao giờ nhìn vào test.
- Mỗi mini-batch là một lô đồ thị (PyG DataLoader); thông tin `ips`, `win` bị bỏ khỏi Data để ghép lô nhanh.
"""

import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src import data as D
from src import evaluate as ev

MAIN4 = [0, 1, 2, 3]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip(graphs):
    """Giữ x, y, edge_index, edge_attr; bỏ các trường danh sách (ips) và số nguyên (win)."""
    return [Data(x=g.x, y=g.y, edge_index=g.edge_index, edge_attr=g.edge_attr) for g in graphs]


@torch.no_grad()
def predict(model, graphs, device, batch_size=256):
    """Trả về (y_true, proba) trên mọi node của danh sách đồ thị."""
    model.eval()
    ys, ps = [], []
    for b in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        b = b.to(device)
        ps.append(torch.softmax(model(b.x, b.edge_index, b.edge_attr), dim=1).cpu())
        ys.append(b.y.cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def val_f1(model, graphs, device, class_names):
    y, p = predict(model, graphs, device)
    return ev.compute_metrics(y, p.argmax(1), p, class_names=class_names, macro_labels=MAIN4)["f1_macro"]


def fit(model, train_graphs, val_graphs, class_names, device, seed, lr=5e-3, weight_decay=1e-4,
        epochs=40, patience=6, batch_size=128, verbose=True):
    """Huấn luyện, trả về (model với trọng số tốt nhất, lịch sử, thời gian huấn luyện giây)."""
    set_seed(seed)
    model = model.to(device)
    y_train = torch.cat([g.y for g in train_graphs]).numpy()
    w = torch.tensor(D.class_weights(y_train, len(class_names)), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, generator=gen)

    best, best_state, bad, history = -1.0, None, 0, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for b in loader:
            b = b.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(b.x, b.edge_index, b.edge_attr), b.y, weight=w)
            loss.backward()
            opt.step()
            total += loss.item() * b.num_nodes
        f1 = val_f1(model, val_graphs, device, class_names)
        history.append({"epoch": epoch, "train_loss": total / len(y_train), "val_f1_macro4": f1})
        if verbose:
            print(f"  epoch {epoch:2d} loss {total / len(y_train):.4f} val F1-macro(4) {f1:.4f}")
        if f1 > best:
            best, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, history, time.time() - t0
