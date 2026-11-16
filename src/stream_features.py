"""Gom cửa sổ 5 giây từ luồng flow và dựng đồ thị bằng đúng hàm offline (task 8.1).

Luồng chạy:  replay_batches (phát lại flow đã sắp theo thời gian, tăng tốc) → WindowAssembler.push (gom theo cửa sổ, đóng cửa sổ khi
flow của cửa sổ sau xuất hiện) → ClosedWindow.flows() → build_window_graph (cùng `build_graph.node_edge_tables` và `to_pyg` với bản
offline, scaler đã fit trên train, KHÔNG fit lại).

Khác biệt duy nhất so với offline là lô cửa sổ: offline dựng 1.000 cửa sổ một lượt, online dựng từng cửa sổ; mọi đặc trưng đều
tính theo nhóm (cửa sổ, IP) nên kết quả phải trùng (tests/test_stream_features.py và notebook 20 kiểm max |sai khác| < 1e-6).

Bộ nhớ: bộ gom chỉ giữ flow của cửa sổ đang mở; đóng cửa sổ là bỏ tham chiếu, nên RAM không tăng theo thời gian chạy.
Dữ liệu thời gian thực không có nhãn: đồ thị dựng không có cột `y` (nhãn = -1), nhãn thật chỉ dùng để đánh giá.
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import build_graph as bg

# cột bảng flow chuẩn mà node_edge_tables cần (không gồm nhãn)
FLOW_COLUMNS = ["ts", "src", "dst", "sport", "dport", "proto", "bytes_fwd", "bytes_bwd", "pkts_fwd", "pkts_bwd", "dur",
                "f_noreply", "f_rst", "f_est"]


def window_ids(ts, window_sec):
    """Mã cửa sổ của từng flow: floor(ts / window_sec), cùng công thức `build_graph.assign_windows` với W = S."""
    return np.floor(np.asarray(ts, dtype=np.float64) / window_sec).astype(np.int64)


def replay_batches(flows, batch_sec=1.0, speedup=None):
    """Phát lại bảng flow đã sắp theo `ts` thành các lô, mỗi lô là flow của một khoảng `batch_sec` giây dữ liệu.

    speedup=None: phát nhanh nhất có thể (không chờ). speedup=N: chờ để thời gian thật trôi N lần chậm hơn thời gian dữ liệu
    (N = 1 là tốc độ thật, N = 10 là nhanh gấp 10 lần); chỉ dùng cho demo, đo độ trễ luôn chạy không chờ.
    """
    ts = flows["ts"].to_numpy(np.float64)
    if len(ts) == 0:
        return
    if (np.diff(ts) < 0).any():
        raise ValueError("flow phải được sắp theo ts tăng dần")
    sec = np.floor(ts / batch_sec).astype(np.int64)
    cuts = np.flatnonzero(np.diff(sec)) + 1
    starts, ends = np.r_[0, cuts], np.r_[cuts, len(ts)]
    t_wall0, t_data0 = time.perf_counter(), ts[0]
    for a, b in zip(starts, ends):
        if speedup:
            wait = (ts[a] - t_data0) / speedup - (time.perf_counter() - t_wall0)
            if wait > 0:
                time.sleep(wait)
        yield flows.iloc[a:b]


@dataclass
class ClosedWindow:
    """Một cửa sổ vừa đóng: mã cửa sổ và các mảnh flow của nó (chưa nối)."""

    win: int
    chunks: list

    def flows(self):
        return self.chunks[0] if len(self.chunks) == 1 else pd.concat(self.chunks, ignore_index=True)


class WindowAssembler:
    """Nhận flow theo lô (lô sau không lùi thời gian), đóng cửa sổ `window_sec` giây khi thấy flow của cửa sổ sau.

    Cửa sổ cuối cùng chỉ đóng khi gọi `flush()` (hết luồng). Cửa sổ không có flow nào không sinh sự kiện (giống offline: không có đồ thị).
    """

    def __init__(self, window_sec=5):
        self.window_sec = window_sec
        self._win = None
        self._chunks = []

    @property
    def open_window(self):
        return self._win

    @property
    def n_buffered_chunks(self):
        return len(self._chunks)

    def push(self, flows):
        """Thêm một lô flow; trả về danh sách cửa sổ đã đóng (theo thứ tự thời gian)."""
        closed = []
        if len(flows) == 0:
            return closed
        wins = window_ids(flows["ts"].to_numpy(), self.window_sec)
        if (np.diff(wins) < 0).any() or (self._win is not None and wins[0] < self._win):
            raise ValueError("flow không theo thứ tự thời gian: gặp cửa sổ cũ hơn cửa sổ đang mở")
        cuts = np.flatnonzero(np.diff(wins)) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(wins)]):
            w = int(wins[a])
            if self._win is not None and w != self._win:
                closed.append(self._close())
            self._win = w
            self._chunks.append(flows.iloc[a:b])
        return closed

    def flush(self):
        return [self._close()] if self._win is not None else []

    def _close(self):
        cw = ClosedWindow(self._win, self._chunks)
        self._win, self._chunks = None, []
        return cw


def build_window_graph(win, flows, scaler, window_sec=5):
    """Đồ thị (torch_geometric Data) của một cửa sổ, bằng đúng `node_edge_tables` và `to_pyg` của bản offline.

    Trả về None nếu cửa sổ không có node (mọi flow có src == dst), giống offline không sinh đồ thị cho cửa sổ đó.
    """
    keep = flows["src"].to_numpy() != flows["dst"].to_numpy()
    if not keep.any():
        return None
    fw = flows.loc[keep, FLOW_COLUMNS].assign(win=np.int64(win))
    nodes, edges = bg.node_edge_tables(fw, window_sec)
    return bg.to_pyg(nodes, edges, scaler, window_sec)[0]


def stream_graphs(flows, scaler, window_sec=5, batch_sec=1.0):
    """Tiện ích cho kiểm thử và demo nhỏ: phát lại `flows` và sinh (mã cửa sổ, đồ thị) của mọi cửa sổ có node."""
    asm = WindowAssembler(window_sec)

    def emit(closed):
        for cw in closed:
            g = build_window_graph(cw.win, cw.flows(), scaler, window_sec)
            if g is not None:
                yield cw.win, g

    for batch in replay_batches(flows, batch_sec):
        yield from emit(asm.push(batch))
    yield from emit(asm.flush())
