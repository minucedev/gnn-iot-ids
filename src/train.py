"""Vòng huấn luyện và dự đoán chung cho các mô hình GNN (task 5.1).

- Mất mát: cross-entropy có trọng số lớp cân bằng (lớp vắng mặt trong train có trọng số 0). Tùy chọn `loss="focal"`
  (task 5.4): focal loss (1 - p_t)^gamma · CE với cùng trọng số lớp, chuẩn hóa bằng tổng trọng số như `F.cross_entropy`.
- Chọn epoch tốt nhất theo F1-macro trên val (dừng sớm), không bao giờ nhìn vào test. F1-macro lấy trung bình
  các lớp có mã trong `MACRO_CODES` (0-3, bỏ Khac mã 4 theo D5) và có mặt trong val: NF-ToN-IoT-v3 là 4 lớp,
  NF-BoT-IoT-v3 chỉ 3 lớp (không có Botnet_Backdoor, mã 3 tự bị bỏ).
- Mỗi epoch ghi loss train, loss val và F1-macro val vào lịch sử (task 5.2, 5.3: đường học).
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

# Các MÃ lớp được xét cho F1-macro (không phải số lớp): 0 Normal, 1 DoS_DDoS, 2 Botnet_Backdoor, 3 Truy_cap_trai_phep.
# compute_metrics bỏ mã không có mẫu thật trong val, nên bộ ít lớp hơn (NF-BoT-IoT-v3: mã 0, 1, 2) vẫn đúng.
MACRO_CODES = [0, 1, 2, 3]


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


def val_scores(model, graphs, device, class_names, class_weights):
    """(F1-macro, loss) trên một tập đồ thị bằng một lượt dự đoán. Loss là cross-entropy trung bình có trọng số
    `class_weights` (cùng công thức `F.cross_entropy(weight=...)`), để so được với loss train."""
    y, p = predict(model, graphs, device)
    f1 = ev.compute_metrics(y, p.argmax(1), p, class_names=class_names, macro_labels=MACRO_CODES)["f1_macro"]
    w = np.asarray(class_weights, dtype=np.float64)[y]
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))
    return f1, float((w * nll).sum() / w.sum())


def val_f1(model, graphs, device, class_names):
    y, p = predict(model, graphs, device)
    return ev.compute_metrics(y, p.argmax(1), p, class_names=class_names, macro_labels=MACRO_CODES)["f1_macro"]


def focal_loss(logits, y, weight, gamma=2.0):
    """Focal loss nhiều lớp có trọng số lớp: sum(w_y · (1 - p_y)^gamma · (-log p_y)) / sum(w_y). gamma = 0 trùng cross-entropy."""
    logp = F.log_softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
    fl = (1.0 - logp.exp()).pow(gamma) * (-logp)
    w = weight[y]
    return (w * fl).sum() / w.sum()


def fit(model, train_graphs, val_graphs, class_names, device, seed, lr=5e-3, weight_decay=1e-4,
        epochs=40, patience=6, batch_size=128, verbose=True, loss="ce", focal_gamma=2.0):
    """Huấn luyện, trả về (model với trọng số tốt nhất, lịch sử, thời gian huấn luyện giây).
    `loss`: "ce" (mặc định, cross-entropy có trọng số lớp) hoặc "focal"."""
    if loss not in ("ce", "focal"):
        raise ValueError(f"loss phải là 'ce' hoặc 'focal', nhận {loss!r}")
    set_seed(seed)
    model = model.to(device)
    y_train = torch.cat([g.y for g in train_graphs]).numpy()
    cw = D.class_weights(y_train, len(class_names))
    w = torch.tensor(cw, dtype=torch.float32, device=device)
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
            logits = model(b.x, b.edge_index, b.edge_attr)
            batch_loss = (F.cross_entropy(logits, b.y, weight=w) if loss == "ce"
                          else focal_loss(logits, b.y, w, focal_gamma))
            batch_loss.backward()
            opt.step()
            total += batch_loss.item() * b.num_nodes
        f1, val_loss = val_scores(model, val_graphs, device, class_names, cw)
        history.append({"epoch": epoch, "train_loss": total / len(y_train), "val_loss": val_loss, "val_f1_macro": f1})
        if verbose:
            print(f"  epoch {epoch:2d} loss {total / len(y_train):.4f} val loss {val_loss:.4f} val F1-macro {f1:.4f}")
        if f1 > best:
            best, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, history, time.time() - t0
