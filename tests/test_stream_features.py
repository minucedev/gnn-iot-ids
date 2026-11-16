"""Kiểm thử src/stream_features.py (task 8.1): đặc trưng online trùng offline, đóng cửa sổ đúng, RAM không tích lũy."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src import build_graph as bg
from src import stream_features as SF

T0 = 1_000_000.0


def make_flows(seed=0, n=700, span=100.0):
    """Flow giả: 100 giây, có khoảng trống không có flow (30-40 s), một cửa sổ chỉ gồm flow src == dst (50-55 s) và flow trùng ts."""
    rng = np.random.default_rng(seed)
    ips = [f"10.0.0.{i}" for i in range(9)]
    ts = np.sort(rng.uniform(T0, T0 + span, n))
    ts = ts[((ts < T0 + 30) | (ts >= T0 + 40)) & ((ts < T0 + 50) | (ts >= T0 + 55))]
    ts = np.sort(np.r_[ts, [T0 + 12.5, T0 + 12.5, T0 + 12.5]])                       # flow trùng ts
    m = len(ts)
    df = pd.DataFrame({
        "ts": ts, "src": rng.choice(ips, m), "dst": rng.choice(ips, m),
        "sport": rng.integers(1024, 65535, m).astype("int64"), "dport": rng.choice([22, 80, 443, 53], m).astype("int64"),
        "proto": rng.choice(["tcp", "udp", "icmp"], m),
        "bytes_fwd": rng.integers(0, 5000, m).astype("int64"), "bytes_bwd": rng.integers(0, 5000, m).astype("int64"),
        "pkts_fwd": rng.integers(1, 20, m).astype("int64"), "pkts_bwd": rng.integers(0, 20, m).astype("int64"),
        "dur": rng.uniform(0, 3, m), "f_noreply": rng.random(m) < 0.2, "f_rst": rng.random(m) < 0.1, "f_est": rng.random(m) < 0.6,
    })
    loop = pd.DataFrame({**{c: df[c].iloc[:3].to_numpy() for c in df.columns}, "ts": [T0 + 51.0, T0 + 52.0, T0 + 53.0]})
    loop["dst"] = loop["src"]                                                          # cửa sổ 50-55 s chỉ có flow src == dst
    return pd.concat([df, loop]).sort_values("ts", kind="stable").reset_index(drop=True)


def offline(df, scaler=None):
    nodes, edges = bg.build_tables(df, 5, 5)
    if scaler is None:
        scaler = StandardScaler().fit(nodes[bg.FEATURES].to_numpy(np.float64))
    return {g.win: g for g in bg.to_pyg(nodes, edges, scaler, 5)}, scaler


def assert_same(a, b):
    assert a.ips == b.ips
    assert torch.equal(a.edge_index, b.edge_index)
    assert float((a.x - b.x).abs().max()) < 1e-6 and a.x.shape == b.x.shape
    assert float((a.edge_attr - b.edge_attr).abs().max()) < 1e-6 and a.edge_attr.shape == b.edge_attr.shape
    assert a.win == b.win and a.t_start == b.t_start


@pytest.mark.parametrize("batch_sec", [1.0, 0.37, 7.0, 1000.0])
def test_online_graphs_equal_offline(batch_sec):
    df = make_flows()
    off, scaler = offline(df)
    on = dict(SF.stream_graphs(df, scaler, 5, batch_sec))
    assert set(on) == set(off) and len(on) > 10
    for w in off:
        assert_same(on[w], off[w])
    wins = np.unique(SF.window_ids(df["ts"], 5))
    assert len(wins) > len(off)                                                         # cửa sổ chỉ có flow src == dst không sinh đồ thị
    assert int(T0 // 5 + 10) not in on                                                  # cửa sổ 50-55 s
    assert not any(int((T0 + 32) // 5) == w for w in on)                                # khoảng trống không có cửa sổ


def test_scaler_is_not_refit_and_applied():
    df = make_flows()
    off, scaler = offline(df)
    mean0, scale0 = scaler.mean_.copy(), scaler.scale_.copy()
    on = dict(SF.stream_graphs(df, scaler, 5))
    assert np.array_equal(scaler.mean_, mean0) and np.array_equal(scaler.scale_, scale0)
    other = StandardScaler().fit(np.random.default_rng(1).normal(5, 3, (50, 18)))        # scaler khác cho đặc trưng khác
    on2 = dict(SF.stream_graphs(df, other, 5))
    w = next(iter(on))
    assert not torch.allclose(on[w].x, on2[w].x)


def test_assembler_closes_on_next_window_and_flush():
    df = make_flows()
    asm = SF.WindowAssembler(5)
    closed = []
    for b in SF.replay_batches(df, 1.0):
        out = asm.push(b)
        closed += out
        assert asm.n_buffered_chunks <= 6                                               # chỉ giữ cửa sổ đang mở (≤ 5 lô 1 giây + mảnh)
        assert all(cw.win < asm.open_window for cw in out)
    last = asm.flush()
    assert len(last) == 1 and asm.open_window is None and asm.flush() == []
    all_closed = closed + last
    assert [c.win for c in all_closed] == sorted(set(c.win for c in all_closed))
    assert sum(len(c.flows()) for c in all_closed) == len(df)                           # không mất flow nào
    for c in all_closed:
        assert (SF.window_ids(c.flows()["ts"], 5) == c.win).all()


def test_assembler_rejects_out_of_order():
    df = make_flows()
    asm = SF.WindowAssembler(5)
    asm.push(df.iloc[300:320])
    with pytest.raises(ValueError):
        asm.push(df.iloc[0:10])                                                          # cửa sổ cũ hơn cửa sổ đang mở
    with pytest.raises(ValueError):
        list(SF.replay_batches(df.iloc[::-1]))


def test_empty_batch_and_empty_window_graph():
    asm = SF.WindowAssembler(5)
    assert asm.push(make_flows().iloc[0:0]) == [] and asm.flush() == []
    df = make_flows()
    loops = df[df["ts"].between(T0 + 50, T0 + 55)]
    assert len(loops) == 3 and (loops["src"] == loops["dst"]).all()
    assert SF.build_window_graph(int((T0 + 50) // 5), loops, StandardScaler().fit(np.zeros((2, 18)) + [[0], [1]]), 5) is None


def test_replay_speedup_waits_in_data_time():
    import time
    df = make_flows().iloc[:60]
    span = float(df["ts"].iloc[-1] - df["ts"].iloc[0])
    t0 = time.perf_counter()
    n = sum(len(b) for b in SF.replay_batches(df, 1.0, speedup=span / 0.3))               # phát lại trong khoảng 0,3 giây thật
    el = time.perf_counter() - t0
    assert n == len(df) and 0.15 <= el < 2.0
