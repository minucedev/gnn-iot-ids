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


def _contiguous_blocks(graphs, n_blocks):
    """Chia danh sách đồ thị (đã theo thời gian) thành n_blocks khối liên tiếp có số node xấp xỉ nhau."""
    n = np.array([g.num_nodes for g in graphs], dtype=np.float64)
    c = np.cumsum(n)
    gid = np.minimum(n_blocks - 1, ((c - n / 2) * n_blocks / c[-1]).astype(int))    # theo điểm giữa của đồ thị, không giảm
    blocks = [np.flatnonzero(gid == f) for f in range(n_blocks)]
    if any(len(b) == 0 for b in blocks):
        raise ValueError(f"Không chia được {len(graphs)} đồ thị thành {n_blocks} khối không rỗng")
    return blocks


def fold_groups(graphs, k=5, interleave=0):
    """Chia đồ thị train thành k nhóm out-of-fold. Trả về k mảng chỉ số đồ thị (tăng dần).

    interleave = 0: k nhóm liên tiếp theo thời gian (đặc tả ban đầu của task 5.7). Trên dữ liệu này cách đó hỏng: mỗi
    loại tấn công chỉ xuất hiện ở một giai đoạn, nên nhóm cuối là nơi duy nhất có DoS_DDoS (BoT) hoặc Botnet_Backdoor
    và Khac (ToN); GAT huấn luyện không có nhóm cuối chưa từng thấy các lớp đó (log `classes_missing_in_fit`).
    interleave = m > 0: cắt train thành k·m khối liên tiếp theo thời gian rồi gán khối i cho nhóm i mod k; mỗi nhóm
    trải khắp mọi giai đoạn, nên GAT của mọi nhóm đều thấy đủ các lớp, mà vẫn giữ tính liên tiếp trong từng khối."""
    if interleave <= 0:
        return _contiguous_blocks(graphs, k)
    blocks = _contiguous_blocks(graphs, k * interleave)
    return [np.sort(np.concatenate(blocks[f::k])) for f in range(k)]


def _scatter(parts, groups, graphs):
    """Ghép kết quả theo nhóm (mỗi phần theo thứ tự đồ thị trong nhóm) về thứ tự node của train."""
    sizes = np.array([g.num_nodes for g in graphs])
    off = np.r_[0, np.cumsum(sizes)]
    out = np.empty((int(off[-1]), parts[0].shape[1]), dtype=parts[0].dtype)
    for part, g in zip(parts, groups):
        idx = np.concatenate([np.arange(off[i], off[i + 1]) for i in g])
        out[idx] = part
    return out


def oof_ensemble(train_graphs, val_graphs, classes, device, seed, k=5, model_hp=None, train_hp=None,
                 verbose=False, train_fn=train_gat, interleave=0):
    """Embedding out-of-fold cho train: nhóm f lấy embedding từ GAT huấn luyện trên các nhóm khác f.

    Trả về (embedding train hình (số node train, 64) theo thứ tự train, danh sách k GAT của các nhóm (đã eval, trên
    `device`), danh sách k embedding val (mỗi cái từ một GAT nhóm), thông tin từng nhóm). `train_fn` có cùng chữ ký với
    `train_gat` (tiêm để test kiểm tra GAT không thấy nhóm bị giữ lại)."""
    groups = fold_groups(train_graphs, k, interleave)
    assert np.array_equal(np.sort(np.concatenate(groups)), np.arange(len(train_graphs)))
    parts, models, val_embs, info = [], [], [], []
    for f, held in enumerate(groups):
        rest = np.concatenate([g for j, g in enumerate(groups) if j != f])
        model, hist, secs = train_fn([train_graphs[i] for i in rest], val_graphs, classes, device, seed,
                                     model_hp, train_hp, verbose)
        parts.append(extract_embeddings(model, [train_graphs[i] for i in held], device))
        val_embs.append(extract_embeddings(model, val_graphs, device))
        models.append(model.eval())
        y_fit = torch.cat([train_graphs[i].y for i in rest]).numpy()
        info.append({"fold": f, "n_graphs_held": int(len(held)),
                     "n_nodes_held": int(sum(train_graphs[i].num_nodes for i in held)),
                     "epochs": len(hist), "train_s": round(secs, 1),
                     "classes_missing_in_fit": [classes[c] for c in range(len(classes)) if not (y_fit == c).any()]})
    return _scatter(parts, groups, train_graphs), models, val_embs, info


def oof_embeddings(train_graphs, val_graphs, classes, device, seed, k=5, model_hp=None, train_hp=None,
                   verbose=False, train_fn=train_gat, interleave=0):
    """Như `oof_ensemble` nhưng chỉ trả về (embedding train, thông tin từng nhóm)."""
    emb, _, _, info = oof_ensemble(train_graphs, val_graphs, classes, device, seed, k, model_hp, train_hp, verbose,
                                   train_fn, interleave)
    return emb, info


def class_proba(model, graphs, device=None, batch_size=256):
    """Xác suất softmax (số node, K) của GAT cho từng node, theo đúng thứ tự `data.node_arrays`."""
    device = device or next(model.parameters()).device
    _, p = T.predict(model, graphs, device, batch_size)
    return p.astype(np.float32)


def oof_class_proba(train_graphs, fold_models, k=5, device=None, interleave=0):
    """Xác suất out-of-fold cho train: nhóm f lấy xác suất từ `fold_models[f]` (GAT huấn luyện trên các nhóm khác f, cùng
    cách chia `fold_groups` như `oof_embeddings`). Xác suất có ngữ nghĩa cố định giữa các GAT (cột = lớp), khác embedding."""
    groups = fold_groups(train_graphs, k, interleave)
    return _scatter([class_proba(fold_models[f], [train_graphs[i] for i in g], device)
                     for f, g in enumerate(groups)], groups, train_graphs)


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


def predict_proba_views(model, views, n_classes):
    """Trung bình xác suất của cùng các node qua nhiều "góc nhìn" đặc trưng (mỗi góc là một mảng (n, cột), cùng n).
    Ghép các góc thành một lần gọi LightGBM rồi lấy trung bình, bằng trung bình của từng lần gọi riêng."""
    n = len(views[0])
    p = predict_proba(model, np.vstack(views), n_classes)
    return p.reshape(len(views), n, n_classes).mean(0)


def search_lgbm(Xtr, ytr, Xva, yva, classes, seed=42, grid=None, val_views=None):
    """Chọn cấu hình LightGBM trên val theo F1-macro các lớp có mặt. Trả về (cấu hình tốt nhất, log các lần thử).
    `val_views`: danh sách nhiều mảng đặc trưng val (ensemble các GAT nhóm); có thì dự đoán val là trung bình xác suất."""
    grid = LGBM_GRID if grid is None else grid
    if len(grid) > MAX_TRIALS:
        raise ValueError(f"Tối đa {MAX_TRIALS} lần thử, nhận {len(grid)}")
    log = []
    for cfg in grid:
        t0 = time.time()
        m = fit_lgbm(Xtr, ytr, cfg, seed, len(classes))
        p = predict_proba_views(m, val_views, len(classes)) if val_views else predict_proba(m, Xva, len(classes))
        pred = p.argmax(1)
        f1 = ev.compute_metrics(yva, pred, class_names=classes, macro_labels=T.MACRO_CODES)["f1_macro"]
        log.append({"cfg": dict(cfg), "val_f1_macro": f1, "seconds": round(time.time() - t0, 1)})
    best = max(range(len(grid)), key=lambda i: log[i]["val_f1_macro"])
    return grid[best], log


def lgbm_n_leaves(model):
    """Tổng số lá của mọi cây LightGBM (mỗi vòng boosting có K cây với K lớp)."""
    return int(sum(t["num_leaves"] for t in model.booster_.dump_model()["tree_info"]))


# ---------------------------------------------------------------- suy luận một cửa sổ
@torch.no_grad()
def predict_window_stack(gats, lgbm, window, n_classes):
    """Hybrid stacking: trung bình xác suất lớp của các GAT nhóm → ghép với x → một lần LightGBM."""
    p = np.mean([torch.softmax(g.eval()(window.x, window.edge_index, window.edge_attr), 1).numpy() for g in gats], axis=0)
    return predict_proba(lgbm, build_features(p, window.x.numpy()), n_classes)


@torch.no_grad()
def predict_window_ensemble(gats, lgbm, window, n_classes):
    """Hybrid ensemble: embedding từ từng GAT nhóm → ghép với x → một lần LightGBM cho mọi góc → trung bình xác suất."""
    x = window.x.numpy()
    views = [build_features(g.eval().embed(window.x, window.edge_index, window.edge_attr).numpy(), x) for g in gats]
    return predict_proba_views(lgbm, views, n_classes)


@torch.no_grad()
def predict_window(gat, lgbm, window, n_classes):
    """Cả hai bước của hybrid cho một đồ thị cửa sổ (dùng để đo độ trễ): embedding GAT → ghép → LightGBM."""
    gat.eval()
    emb = gat.embed(window.x, window.edge_index, window.edge_attr).numpy()
    return predict_proba(lgbm, build_features(emb, window.x.numpy()), n_classes)


# ---------------------------------------------------------------- ablation (task 5.8) và bản nhẹ
def oof_embeddings_from_models(train_graphs, fold_models, k=5, interleave=0, device=None):
    """Embedding out-of-fold của train từ k GAT nhóm ĐÃ huấn luyện: nhóm f lấy embedding từ `fold_models[f]`, cùng cách chia
    `fold_groups` như `oof_ensemble` nên khớp từng node với embedding lúc huấn luyện hybrid (không huấn luyện lại gì)."""
    groups = fold_groups(train_graphs, k, interleave)
    return _scatter([extract_embeddings(fold_models[f], [train_graphs[i] for i in g], device)
                     for f, g in enumerate(groups)], groups, train_graphs)


def load_gat(path, n_classes, in_dim=18, model_hp=None):
    """Nạp một GAT (cấu hình task 5.3 hoặc `model_hp`) từ file state_dict, ở chế độ eval trên CPU."""
    model = M.GNN("gat", in_dim=in_dim, n_classes=n_classes, **(model_hp or {}))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def neighbor_mean(graphs):
    """Trung bình 18 đặc trưng của các node kề, tính tay (không học), hình (số node, số đặc trưng), theo thứ tự node của
    `data.node_arrays`. Node kề = đầu kia của mọi cạnh, không phân biệt hướng, mỗi hàng xóm tính một lần (cạnh trùng hoặc
    hai chiều chỉ đếm một), bỏ vòng lặp tự thân. Node không có hàng xóm có trung bình 0 (tức giá trị trung bình của train
    sau chuẩn hóa)."""
    sizes = torch.tensor([g.num_nodes for g in graphs], dtype=torch.long)
    off = torch.cumsum(sizes, 0) - sizes
    n = int(sizes.sum())
    x = torch.cat([g.x for g in graphs]).float()
    ei = torch.cat([g.edge_index.long() + o for g, o in zip(graphs, off.tolist())], dim=1)
    both = torch.cat([ei, ei.flip(0)], dim=1)
    both = both[:, both[0] != both[1]]
    key = torch.unique(both[0] * n + both[1])                    # cặp (nguồn, đích) duy nhất, hai chiều
    src, dst = key // n, key % n
    total = torch.zeros(n, x.shape[1]).index_add_(0, dst, x[src])
    deg = torch.zeros(n).index_add_(0, dst, torch.ones(len(dst)))
    return (total / deg.clamp(min=1)[:, None]).numpy().astype(np.float32)
