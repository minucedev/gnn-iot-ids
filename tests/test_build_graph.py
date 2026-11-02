import numpy as np
import pandas as pd
import pytest

from src import build_graph as bg

# Một cửa sổ (W = S = 5 s, mọi ts trong [0, 5)), 7 flow; f6 là C→C tự vòng nên bị bỏ.
#        ts   src  dst sport dport proto  b_fwd b_bwd p_fwd p_bwd dur  noreply rst    est    y
FLOWS = [
    (0.0, "A", "B", 1000, 80, "tcp", 100, 200, 2, 3, 1.0, False, False, True, 0),   # f1
    (1.0, "A", "C", 1001, 80, "tcp", 50, 0, 1, 0, 0.0, True, False, False, 1),      # f2
    (2.0, "A", "C", 1002, 22, "tcp", 60, 0, 1, 0, 0.0, True, True, False, 1),       # f3
    (3.0, "B", "A", 2000, 443, "udp", 300, 100, 3, 1, 2.0, False, False, True, 0),  # f4
    (1.5, "D", "B", 3000, 53, "udp", 40, 80, 1, 1, 0.5, False, False, True, 0),     # f5
    (4.0, "C", "C", 4000, 9, "tcp", 10, 10, 1, 1, 0.1, False, False, True, 0),      # f6 (bỏ)
    (2.5, "D", "B", 3001, 53, "icmp", 20, 0, 1, 0, 0.0, True, False, False, 2),     # f7
]
COLS = ["ts", "src", "dst", "sport", "dport", "proto", "bytes_fwd", "bytes_bwd", "pkts_fwd", "pkts_bwd",
        "dur", "f_noreply", "f_rst", "f_est", "y"]

# Tính tay theo định nghĩa (bytes_sent của IP = chiều đi của flow nó khởi tạo + chiều về của flow nó nhận, ...)
# Node sắp theo IP: A=0, B=1, C=2, D=3.
HAND = {
    #       A: ra f1,f2,f3; vào f4        B: ra f4; vào f1,f5,f7     C: vào f2,f3        D: ra f5,f7
    "n_out":         [3, 1, 0, 2],
    "n_in":          [1, 3, 2, 0],
    "bytes_sent":    [100 + 50 + 60 + 100, 300 + 200 + 80 + 0, 0, 40 + 20],
    "bytes_recv":    [200 + 0 + 0 + 300, 100 + 100 + 40 + 20, 50 + 60, 80 + 0],
    "pkts_sent":     [2 + 1 + 1 + 1, 3 + 3 + 1 + 0, 0, 1 + 1],
    "pkts_recv":     [3 + 0 + 0 + 3, 1 + 2 + 1 + 1, 1 + 1, 1 + 0],
    "uniq_dst":      [2, 1, 0, 1],                 # A→{B,C}; B→{A}; D→{B}
    "uniq_dport":    [2, 1, 0, 1],                 # A: {80,22}; B: {443}; D: {53}
    "uniq_src":      [1, 2, 1, 0],                 # A←{B}; B←{A,D}; C←{A}
    "dur_mean":      [(1 + 0 + 0 + 2) / 4, (2 + 1 + 0.5 + 0) / 4, 0.0, (0.5 + 0) / 2],
    "pkt_size_mean": [810 / 11, 840 / 12, 110 / 2, 140 / 3],
    "noreply_ratio": [2 / 4, 1 / 4, 2 / 2, 1 / 2],
    "rst_ratio":     [1 / 4, 0.0, 1 / 2, 0.0],
    "est_ratio":     [2 / 4, 3 / 4, 0.0, 1 / 2],
    "tcp_ratio":     [3 / 4, 1 / 4, 1.0, 0.0],
    "udp_ratio":     [1 / 4, 2 / 4, 0.0, 1 / 2],
    "icmp_ratio":    [0.0, 1 / 4, 0.0, 1 / 2],
    "iat_mean_out":  [1.0, 5.0, 5.0, 1.0],         # A: ts 0,1,2; D: 1.5,2.5; B, C < 2 flow ra → W = 5
}
# Nhãn: A đa số flow ra là lớp 1 (f2,f3); B flow ra f4 → 0; C không có flow ra, flow vào f2,f3 → 1;
#       D flow ra f5 (0) và f7 (2) hòa → lớp lớn hơn = 2.
HAND_Y = [1, 0, 1, 2]
# Cạnh (s, d): e_flows, e_bytes (tổng 2 chiều), e_pkts
HAND_EDGES = {(0, 1): (1, 300, 5), (0, 2): (2, 110, 2), (1, 0): (1, 400, 4), (3, 1): (2, 140, 3)}


def _window():
    fl = pd.DataFrame(FLOWS, columns=COLS).sort_values("ts", kind="stable").reset_index(drop=True)
    return bg.assign_windows(fl, 5, 5)


def test_hand_computed_window():
    nodes, edges = bg.node_edge_tables(_window(), 5)
    assert list(nodes["ip"]) == ["A", "B", "C", "D"]
    assert len(bg.FEATURES) == 18 and set(HAND) == set(bg.FEATURES)
    for f in bg.FEATURES:
        expect = np.array(HAND[f], dtype=float)
        if f in bg.LOG_FEATURES:
            expect = np.log1p(expect)
        np.testing.assert_allclose(nodes[f].to_numpy(float), expect, rtol=1e-12, err_msg=f)
    assert list(nodes["y"]) == HAND_Y
    got = {(s, d): tuple(np.expm1(r).round().astype(int))
           for s, d, *r in edges[["s", "d", *bg.EDGE_FEATURES]].itertuples(index=False)}
    assert got == HAND_EDGES


def test_to_pyg_shapes():
    nodes, edges = bg.node_edge_tables(_window(), 5)
    (g,) = bg.to_pyg(nodes, edges, scaler=None, stride_sec=5)
    assert tuple(g.x.shape) == (4, 18)
    assert tuple(g.edge_index.shape) == (2, 4)
    assert tuple(g.edge_attr.shape) == (4, 3)
    assert g.ips == ["A", "B", "C", "D"] and g.y.tolist() == HAND_Y


def test_sliding_window_duplicates_flows():
    fl = pd.DataFrame(FLOWS, columns=COLS)
    fw = bg.assign_windows(fl, 10, 5)            # W = 10, S = 5: mỗi flow thuộc 2 cửa sổ
    assert len(fw) == 2 * len(fl)
    assert sorted(fw["win"].unique()) == [-1, 0]


def test_to_group_case_insensitive():
    assert list(bg.to_group(["Benign", "Backdoor", "backdoor", "XSS", "mitm", "lạ"])) == \
        ["Normal", "Botnet_Backdoor", "Botnet_Backdoor", "Truy_cap_trai_phep", "Khac", "Khac"]


def _nf_csv(path, with_ts):
    rows = []
    for i, (ts, s, d, sp, dp, proto, bf, bb, pf, pb, dur, *_rest, y) in enumerate(FLOWS):
        r = {"IPV4_SRC_ADDR": s, "L4_SRC_PORT": sp, "IPV4_DST_ADDR": d, "L4_DST_PORT": dp,
             "PROTOCOL": {"tcp": 6, "udp": 17, "icmp": 1}[proto], "IN_BYTES": bf, "IN_PKTS": pf,
             "OUT_BYTES": bb, "OUT_PKTS": pb, "TCP_FLAGS": 0, "CLIENT_TCP_FLAGS": 0, "SERVER_TCP_FLAGS": 0,
             "FLOW_DURATION_MILLISECONDS": int(dur * 1000), "Attack": ["Benign", "ddos", "Backdoor"][y]}
        if with_ts:
            r = {"FLOW_START_MILLISECONDS": 1_556_028_590_000 + int(ts * 1000), **r}
        rows.append(r)
    pd.DataFrame(rows).sort_values("FLOW_START_MILLISECONDS" if with_ts else "IN_BYTES").to_csv(path, index=False)


def test_new_dataset_needs_only_adapter(tmp_path):
    """Bộ v3 chỉ cần adapter load_nf_v3 (ts thật, giây); phần dựng đồ thị dùng chung."""
    p = tmp_path / "nf_v3.csv"
    _nf_csv(p, with_ts=True)
    flows, classes, _ = bg.load_clean("nf_ton_iot_v3", p)
    assert classes == ["Normal", "DoS_DDoS", "Botnet_Backdoor"]
    assert flows["ts"].iloc[0] == pytest.approx(1_556_028_590.0)
    graphs, split, _, _ = bg.build_dataset(flows, 5, 5, ratios=(1.0, 0.0, 0.0))
    assert len(graphs) == 1 and graphs[0].num_nodes == 4 and graphs[0].num_edges == 4


def test_fingerprint_deterministic(tmp_path):
    p = tmp_path / "nf_v2.csv"
    _nf_csv(p, with_ts=False)
    fps = []
    for _ in range(2):
        flows, _, _ = bg.load_clean("nf_ton_iot_v2", p)
        graphs, _, _, _ = bg.build_dataset(flows, 2, 2, ratios=(0.5, 0.25, 0.25))
        fps.append(bg.fingerprint(graphs))
    assert fps[0] == fps[1] and len(fps[0]) == 64


def test_gap_between_splits():
    wins = np.arange(100)
    a = bg.time_split(wins, bg.SPLIT_RATIO, 0)
    b = bg.time_split(wins, bg.SPLIT_RATIO, 1)
    assert a["val"][0] == a["train"][-1] + 1                 # gap = 0: hai tập liền kề
    assert b["val"][0] == b["train"][-1] + 2                 # gap = 1: cách 1 cửa sổ
    assert b["test"][0] == b["val"][-1] + 2
    bb = bg.time_split(wins, bg.SPLIT_RATIO, 1, mode="blocked", blocks=4)
    assert len(set(bb["train"]) | set(bb["val"]) | set(bb["test"])) == len(bb["train"]) + len(bb["val"]) + len(bb["test"])
