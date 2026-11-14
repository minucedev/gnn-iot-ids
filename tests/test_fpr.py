import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from src import fpr as F

K = 3


def _flows():
    # cửa sổ 0: A→B (Normal), A→C (tấn công 1), C→B (tấn công 1), D→D (bị bỏ)
    # cửa sổ 1: B→A (Normal), E→A (tấn công 2), E→A (tấn công 2)
    return pd.DataFrame({
        "win": [0, 0, 0, 0, 1, 1, 1],
        "src": ["A", "A", "C", "D", "B", "E", "E"],
        "dst": ["B", "C", "B", "D", "A", "A", "A"],
        "y": [0, 1, 1, 1, 0, 2, 2],
    })


def _nodes():
    return pd.DataFrame({"win": [0, 0, 0, 1, 1, 1], "ip": ["A", "B", "C", "A", "B", "E"]})


def test_flow_counts_drop_self_loops_and_split_direction():
    c = F.flow_label_counts(_flows(), K)
    assert (0, "D") not in c.index                                               # flow src == dst bị bỏ
    assert c.loc[(0, "A"), ["out_0", "out_1", "in_0"]].tolist() == [1, 1, 0]
    assert c.loc[(0, "B"), ["in_0", "in_1"]].tolist() == [1, 1]
    assert c.loc[(1, "A"), "in_2"] == 2
    assert c.loc[(1, "E"), "out_2"] == 2


def test_attach_counts_keeps_order_and_rejects_missing():
    c = F.flow_label_counts(_flows(), K)
    out, inn = F.attach_counts(_nodes(), c, K)
    assert out.shape == inn.shape == (6, K)
    assert out[0].tolist() == [1, 1, 0] and inn[1].tolist() == [1, 1, 0]
    bad = pd.DataFrame({"win": [0], "ip": ["Z"]})
    with pytest.raises(ValueError):
        F.attach_counts(bad, c, K)


def test_rules_on_hand_computed_nodes():
    c = F.flow_label_counts(_flows(), K)
    out, inn = F.attach_counts(_nodes(), c, K)
    # nút: (0,A) (0,B) (0,C) (1,A) (1,B) (1,E)
    # (0,A) và (0,B) hòa Normal/lớp 1 → lớp lớn hơn; (1,A) chỉ nhận 2 flow lớp 2
    assert F.relabel(out, inn, "hien_tai").tolist() == [1, 1, 1, 2, 0, 2]
    assert F.relabel(out, inn, "chi_khoi_tao").tolist() == [1, -1, 1, -1, 0, 2]
    assert F.relabel(out, inn, "co_flow_tan_cong").tolist() == [1, 1, 1, 2, 0, 2]


def test_tie_goes_to_larger_class_like_pipeline():
    c = F.flow_label_counts(_flows(), K)
    out, inn = F.attach_counts(_nodes(), c, K)
    # (0,A) có flow ra 1 Normal + 1 lớp 1: hòa → lớp lớn hơn = 1 (khớp thứ tự sắp xếp của build_graph)
    assert F.relabel(out, inn, "hien_tai")[0] == 1


def test_purity_table():
    out = np.array([[1, 1, 0], [0, 0, 0]])
    inn = np.array([[0, 0, 0], [3, 1, 0]])
    p = F.purity_table(out, inn)
    assert p["from_in_only"].tolist() == [False, True]
    assert p["out_normal_frac"].iloc[0] == 0.5 and np.isnan(p["out_normal_frac"].iloc[1])
    assert p["all_normal_frac"].tolist() == [0.5, 0.75]
    assert p["purity"].tolist() == [0.5, 0.75]


def test_centroids_and_nearest():
    X = np.array([[0.0, 0.0], [0.2, 0.0], [10.0, 10.0]])
    y = np.array([0, 0, 2])
    C = F.class_centroids(X, y, 3)
    assert np.isnan(C[1]).all()
    d = F.centroid_distances(np.array([[0.1, 0.0], [9.0, 9.0]]), C)
    assert np.isinf(d[:, 1]).all()
    assert d.argmin(1).tolist() == [0, 2]


def test_binary_rates_and_keep():
    y = np.array([0, 0, 0, 0, 1, 2])
    pred = np.array([0, 1, 2, 0, 1, 0])
    r = F.binary_rates(y, pred)
    assert r["n_fp"] == 2 and r["fpr"] == 0.5 and r["recall_attack"] == 0.5
    keep = np.array([True, True, False, True, True, True])
    assert F.binary_rates(y, pred, keep)["n_fp"] == 1


def test_ip_report_counts_and_shares():
    val = pd.DataFrame({"ip": ["A", "A", "B", "C"], "y": [0, 0, 0, 1]})
    fp = np.array([True, True, False, False])
    ref = pd.DataFrame({"ip": ["A", "A", "B", "C", "A"], "y": [0, 1, 0, 1, 0]})
    r = F.ip_report(val, fp, ref, top=2)
    assert r["total_fp"] == 2 and r["n_ip_normal_val"] == 2 and r["n_ip_with_fp"] == 1
    assert r["share_fp_top1"] == 1.0
    assert r["share_fp_ip_also_attack_node"] == 1.0         # IP A cũng có node tấn công ở cửa sổ khác
    assert r["top"][0]["ip"] == "A" and r["top"][0]["fp"] == 2 and r["top"][0]["ref_attack_nodes"] == 1
    assert r["all"] is not None and len(r["all"]) == 2


def test_node_table_matches_concatenation_order():
    g1 = Data(x=torch.zeros(2, 4), y=torch.tensor([0, 1]), win=7, ips=["a", "b"])
    g2 = Data(x=torch.zeros(1, 4), y=torch.tensor([2]), win=9, ips=["c"])
    t = F.node_table([g1, g2])
    assert t["ip"].tolist() == ["a", "b", "c"] and t["win"].tolist() == [7, 7, 9]
    assert t["gi"].tolist() == [0, 0, 1] and t["y"].tolist() == [0, 1, 2]
