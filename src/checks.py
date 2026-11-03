"""Kiểm tra đồ thị đã dựng (task 1.5, 1.6): rò rỉ giữa các tập, NaN/inf, hình dạng, scaler, ngưỡng lớp và tỉ lệ.

Mọi hàm trả về danh sách (tên kiểm tra, đạt hay không, chi tiết) để ghi vào summary.json.
"""

import numpy as np
import torch

from src import build_graph as bg

MAIN = ["Normal", "DoS_DDoS", "Botnet_Backdoor", "Truy_cap_trai_phep"]


def check_graphs(graphs, split, scaler, classes, gap):
    """graphs: danh sách Data theo thứ tự thời gian; split: {part: mảng mã cửa sổ}; gap: số cửa sổ bỏ ở ranh giới."""
    checks = []
    add = lambda name, ok, detail="": checks.append((name, bool(ok), str(detail)))

    # 1. không cửa sổ nào thuộc hai tập
    sets = {p: set(int(w) for w in ws) for p, ws in split.items()}
    overlap = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
    add("Không cửa sổ trùng giữa các tập", not overlap, f"{len(overlap)} cửa sổ trùng")

    # 2. khoảng cách giữa hai cửa sổ liền nhau thuộc hai tập khác nhau >= gap + 1 (id cửa sổ)
    pairs = sorted((w, p) for p, ws in sets.items() for w in ws)
    bad = [(a, b) for (a, pa), (b, pb) in zip(pairs, pairs[1:]) if pa != pb and b - a < gap + 1]
    add(f"Gap >= {gap} cửa sổ ở mọi ranh giới giữa các tập", not bad, f"{len(bad)} ranh giới vi phạm")

    # 3. NaN/inf và hình dạng
    n_feat, n_edge_feat = len(bg.FEATURES), len(bg.EDGE_FEATURES)
    nonfinite = bad_shape = bad_index = 0
    for g in graphs:
        nonfinite += int(not torch.isfinite(g.x).all()) + int(not torch.isfinite(g.edge_attr).all())
        bad_shape += int(g.x.shape[1] != n_feat or (g.edge_attr.numel() and g.edge_attr.shape[1] != n_edge_feat))
        bad_index += int(g.edge_index.numel() > 0 and int(g.edge_index.max()) >= g.num_nodes)
    add("Không có NaN hoặc inf trong x và edge_attr", nonfinite == 0, f"{nonfinite} đồ thị lỗi")
    add(f"x có {n_feat} đặc trưng, edge_attr có {n_edge_feat}", bad_shape == 0, f"{bad_shape} đồ thị sai hình dạng")
    add("Chỉ số cạnh nằm trong số node", bad_index == 0, f"{bad_index} đồ thị lỗi")

    # 4. IP/cổng không nằm trong vector đặc trưng
    raw = [f for f in bg.FEATURES if f in ("ip", "src", "dst", "sport", "dport", "pport", "peer")]
    add("IP và cổng không nằm trong đặc trưng", not raw, f"đặc trưng: {len(bg.FEATURES)} cột thống kê")

    # 5. scaler chỉ fit trên train
    part_graphs = {p: [g for g in graphs if int(g.win) in sets[p]] for p in sets}
    x_train = torch.cat([g.x for g in part_graphs["train"]]).numpy()
    n_train = len(x_train)
    add("Scaler chỉ fit trên train (số mẫu = số node train)", scaler.n_samples_seen_ == n_train,
        f"scaler {scaler.n_samples_seen_} / train {n_train}")
    add("Đặc trưng train sau chuẩn hóa có trung bình ~0", np.abs(x_train.mean(0)).max() < 1e-3,
        f"|mean| lớn nhất {np.abs(x_train.mean(0)).max():.2e}")

    # 6. ngưỡng lớp (task 1.5.1) và tỉ lệ node (task 1.5.4)
    counts = {}
    for p, gs in part_graphs.items():
        y = torch.cat([g.y for g in gs]).numpy()
        counts[p] = np.bincount(y, minlength=len(classes))
    missing = [c for i, c in enumerate(classes) if c in MAIN
               and not (counts["train"][i] >= 100 and counts["val"][i] >= 30 and counts["test"][i] >= 30)]
    add("Mỗi lớp chính có >= 100 node train, >= 30 node val và test", not missing,
        f"thiếu: {missing}" if missing else "đủ")
    total = sum(int(c.sum()) for c in counts.values())
    ratio = {p: 100 * int(c.sum()) / total for p, c in counts.items()}
    ok_ratio = all(abs(ratio[p] - r) <= 3 for p, r in zip(("train", "val", "test"), (70, 15, 15)))
    add("Tỉ lệ node train/val/test trong 70/15/15 ± 3 điểm %", ok_ratio,
        "/".join(f"{ratio[p]:.1f}" for p in ("train", "val", "test")))
    return checks
