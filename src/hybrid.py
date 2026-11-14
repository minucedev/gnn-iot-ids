"""Hybrid GAT + LightGBM (task 5.7, quyết định D9).

Quy trình: (1) GAT nhẹ (cấu hình task 5.3) học phân loại node; (2) lấy vector ẩn sau lớp GAT cuối (64 chiều) làm
embedding; (3) ghép với 18 đặc trưng node đã chuẩn hóa (scaler của dữ liệu đã dựng) thành 82 cột; (4) LightGBM phân loại.

Chống rò rỉ: GAT và LightGBM chỉ học trên train. GAT đã thấy nhãn của node train nên embedding train tính trực tiếp
sẽ "quá sạch"; vì vậy embedding của train là out-of-fold (`oof_embeddings`): train chia K nhóm khối liên tiếp theo thời
gian, mỗi nhóm lấy embedding từ GAT huấn luyện trên K-1 nhóm còn lại. Val dùng GAT huấn luyện trên toàn bộ train.
Val chỉ dùng để chọn epoch của GAT (dừng sớm, như mọi GNN khác) và chọn cấu hình LightGBM, không dùng để học trọng số.
"""

import time

import numpy as np
import torch
from lightgbm import LGBMClassifier
from torch_geometric.loader import DataLoader

from src import data as D
from src import evaluate as ev
from src import models as M
from src import train as T

LGBM_LR = 0.1
# Cùng không gian tìm siêu tham số với baseline LightGBM (scripts/run_baselines.py, make_models) để so sánh công bằng;
# tests/test_hybrid.py kiểm hai bên khớp nhau.
LGBM_GRID = [{"n": 100, "l": 15}, {"n": 200, "l": 31}, {"n": 300, "l": 63}]
MAX_TRIALS = 20


# ---------------------------------------------------------------- GAT và embedding
def train_gat(train_graphs, val_graphs, classes, device, seed, model_hp=None, train_hp=None, verbose=False):
    """Huấn luyện GAT (cấu hình mặc định của task 5.3). Trả về (model, lịch sử, thời gian huấn luyện giây).
    Đồ thị phải đã qua `train.strip`."""
    T.set_seed(seed)
    model = M.GNN("gat", in_dim=train_graphs[0].x.shape[1], n_classes=len(classes), **(model_hp or {}))
    return T.fit(model, train_graphs, val_graphs, classes, device, seed, verbose=verbose, **(train_hp or {}))


@torch.no_grad()
def extract_embeddings(model, graphs, device=None, batch_size=256):
    """Vector ẩn của lớp GAT cuối cho từng node, hình (số node, 64), theo đúng thứ tự `data.node_arrays` (đồ thị theo
    thứ tự danh sách, node theo thứ tự trong đồ thị)."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for b in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        b = b.to(device)
        out.append(model.embed(b.x, b.edge_index, b.edge_attr).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def fold_groups(graphs, k=5):
    """Chia danh sách đồ thị (đã theo thời gian) thành k nhóm liên tiếp có số node xấp xỉ nhau. Trả về k mảng chỉ số."""
    n = np.array([g.num_nodes for g in graphs], dtype=np.float64)
    c = np.cumsum(n)
    gid = np.minimum(k - 1, ((c - n / 2) * k / c[-1]).astype(int))          # theo điểm giữa của đồ thị, không giảm
    groups = [np.flatnonzero(gid == f) for f in range(k)]
    if any(len(g) == 0 for g in groups):
        raise ValueError(f"Không chia được {len(graphs)} đồ thị thành {k} nhóm không rỗng")
    return groups


def oof_embeddings(train_graphs, val_graphs, classes, device, seed, k=5, model_hp=None, train_hp=None,
                   verbose=False, train_fn=train_gat):
    """Embedding out-of-fold cho train: nhóm f lấy embedding từ GAT huấn luyện trên các nhóm khác f.

    Trả về (embedding hình (số node train, 64) theo thứ tự train, danh sách thông tin từng nhóm). `train_fn` có
    cùng chữ ký với `train_gat` (tiêm để test kiểm tra GAT không thấy nhóm bị giữ lại)."""
    groups = fold_groups(train_graphs, k)
    assert np.array_equal(np.concatenate(groups), np.arange(len(train_graphs)))
    parts, info = [], []
    for f, held in enumerate(groups):
        rest = np.concatenate([g for j, g in enumerate(groups) if j != f])
        model, hist, secs = train_fn([train_graphs[i] for i in rest], val_graphs, classes, device, seed,
                                     model_hp, train_hp, verbose)
        parts.append(extract_embeddings(model, [train_graphs[i] for i in held], device))
        y_fit = torch.cat([train_graphs[i].y for i in rest]).numpy()
        info.append({"fold": f, "n_graphs_held": int(len(held)),
                     "n_nodes_held": int(sum(train_graphs[i].num_nodes for i in held)),
                     "epochs": len(hist), "train_s": round(secs, 1),
                     "classes_missing_in_fit": [classes[c] for c in range(len(classes)) if not (y_fit == c).any()]})
    return np.concatenate(parts), info


def build_features(emb, x):
    """Ghép 18 đặc trưng node (đã chuẩn hóa) với embedding: [x | emb], 18 + 64 = 82 cột."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) != len(emb):
        raise ValueError(f"x có {len(x)} node nhưng embedding có {len(emb)}")
    return np.hstack([x, np.asarray(emb, dtype=np.float32)])


# ---------------------------------------------------------------- LightGBM
def make_lgbm(cfg, seed, n_jobs=-1):
    return LGBMClassifier(n_estimators=cfg["n"], num_leaves=cfg["l"], learning_rate=LGBM_LR, n_jobs=n_jobs,
                          random_state=seed, verbose=-1, deterministic=True, force_row_wise=True)


def fit_lgbm(X, y, cfg, seed, n_classes, n_jobs=-1):
    """Một lần huấn luyện LightGBM với trọng số lớp cân bằng (như baseline)."""
    w = D.class_weights(y, n_classes)
    return make_lgbm(cfg, seed, n_jobs).fit(X, y, sample_weight=w[y])


def predict_proba(model, X, n_classes):
    """Xác suất (n, K) đủ K cột kể cả khi một lớp vắng mặt trong train."""
    p = model.predict_proba(X)
    if p.shape[1] == n_classes:
        return p
    full = np.zeros((len(X), n_classes))
    full[:, model.classes_] = p
    return full


def search_lgbm(Xtr, ytr, Xva, yva, classes, seed=42, grid=None):
    """Chọn cấu hình LightGBM trên val theo F1-macro các lớp có mặt. Trả về (cấu hình tốt nhất, log các lần thử)."""
    grid = LGBM_GRID if grid is None else grid
    if len(grid) > MAX_TRIALS:
        raise ValueError(f"Tối đa {MAX_TRIALS} lần thử, nhận {len(grid)}")
    log = []
    for cfg in grid:
        t0 = time.time()
        m = fit_lgbm(Xtr, ytr, cfg, seed, len(classes))
        pred = predict_proba(m, Xva, len(classes)).argmax(1)
        f1 = ev.compute_metrics(yva, pred, class_names=classes, macro_labels=T.MACRO_CODES)["f1_macro"]
        log.append({"cfg": dict(cfg), "val_f1_macro": f1, "seconds": round(time.time() - t0, 1)})
    best = max(range(len(grid)), key=lambda i: log[i]["val_f1_macro"])
    return grid[best], log


def lgbm_n_leaves(model):
    """Tổng số lá của mọi cây LightGBM (mỗi vòng boosting có K cây với K lớp)."""
    return int(sum(t["num_leaves"] for t in model.booster_.dump_model()["tree_info"]))


# ---------------------------------------------------------------- suy luận một cửa sổ
@torch.no_grad()
def predict_window(gat, lgbm, window, n_classes):
    """Cả hai bước của hybrid cho một đồ thị cửa sổ (dùng để đo độ trễ): embedding GAT → ghép → LightGBM."""
    gat.eval()
    emb = gat.embed(window.x, window.edge_index, window.edge_attr).numpy()
    return predict_proba(lgbm, build_features(emb, window.x.numpy()), n_classes)
