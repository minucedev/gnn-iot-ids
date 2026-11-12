"""Kiểm tra các hàm tính toán của scripts/eda.py trên đồ thị nhỏ tính tay được (task 2.1–2.3)."""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

_spec = importlib.util.spec_from_file_location("eda_script", Path(__file__).resolve().parents[1] / "scripts" / "eda.py")
eda = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eda)


def test_homophily_two_classes_by_hand():
    # lớp 0: 2 node, lớp 1: 2 node; cạnh 0→1 (cùng lớp 0), 2→3 (cùng lớp 1), 1→2 (khác lớp)
    M = np.array([[1, 1], [0, 1]])
    h = eda.homophily_from_mix(M, [2, 2])
    assert np.isclose(h["edge_homophily"], 2 / 3)
    # M + M^T = [[2,1],[1,2]]: h_0 = h_1 = 2/3, n_k/n = 1/2 → h_ci = (1/6 + 1/6) / (2 - 1) = 1/3
    assert np.allclose(h["class_wise_homophily"], [2 / 3, 2 / 3])
    assert np.isclose(h["class_insensitive_homophily"], 1 / 3)
    assert np.isclose(h["random_edge_homophily"], 0.5)
    assert np.isclose(h["adjusted_homophily"], (2 / 3 - 0.5) / 0.5)


def test_homophily_extremes():
    pure = eda.homophily_from_mix(np.diag([5, 7]), [3, 4])
    assert pure["edge_homophily"] == 1.0 and np.isclose(pure["class_insensitive_homophily"], 1.0)
    bip = eda.homophily_from_mix(np.array([[0, 6], [0, 0]]), [3, 3])      # mọi cạnh nối hai lớp khác nhau
    assert bip["edge_homophily"] == 0.0 and bip["class_insensitive_homophily"] == 0.0


def test_homophily_ignores_absent_class():
    h = eda.homophily_from_mix(np.array([[2, 0, 0], [1, 1, 0], [0, 0, 0]]), [3, 2, 0])
    assert np.isnan(h["class_wise_homophily"][2])
    assert 0 <= h["class_insensitive_homophily"] <= 1


def _graph(edges, y, win):
    e = torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=torch.zeros(len(y), 18), y=torch.tensor(y), edge_index=e, edge_attr=torch.zeros(e.shape[1], 3),
                win=win, t_start=float(win * 5), ips=[str(i) for i in range(len(y))])


def test_collect_arrays_and_pick_windows():
    classes = ["Normal", "DoS_DDoS", "Truy_cap_trai_phep"]
    normal = _graph([[0, 1], [1, 2], [2, 0]], [0, 0, 0], 0)
    ddos = _graph([[1, 0], [2, 0], [3, 0], [4, 0]], [1, 1, 1, 1, 1], 1)       # node 0 nhận từ 4 node: fan-in 4
    scan = _graph([[0, 1], [0, 2], [0, 3]], [2, 2, 2, 0], 2)                  # node 0 gửi tới 3 node: fan-out 3
    parts = {"train": [normal, ddos, scan], "val": [], "test": []}
    A = eda.collect_graph_arrays(parts, 3)
    assert A["n_nodes"].tolist() == [3, 5, 4] and A["n_edges"].tolist() == [3, 4, 3]
    assert A["max_in"][1, 1] == 4 and A["max_out"][2, 2] == 3
    assert A["mix"][0].sum() == 10 and A["mix"][0][1, 1] == 4 and A["mix"][0][2, 2] == 2 and A["mix"][0][2, 0] == 1
    assert A["in_deg"].sum() == A["out_deg"].sum() == 10
    picks = {k: (gi, mode) for k, gi, _, mode, _ in eda.pick_windows(A, classes)}
    assert picks == {"normal": (0, "total"), "ddos": (1, "in"), "scan": (2, "out")}


def test_format_vietnamese_numbers():
    assert eda.vn(1234567) == "1.234.567" and eda.vn(0.875, 2) == "0,88" and eda.vn(float("nan")) == "n/a"
