"""Kiểm thử đường chạy của scripts/detector.py (task 8.1-8.3): ghi windows.csv từng dòng, khớp offline, trim, probe. Dùng mô hình giả."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src import build_graph as bg
from tests.test_stream_features import make_flows

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load("detector")
CLASSES = ["Normal", "DoS_DDoS", "Botnet_Backdoor", "Truy_cap_trai_phep", "Khac"]


def fake_proba(x):
    """Xác suất giả xác định từ đặc trưng: node có x[:, 0] > 0 bị coi là tấn công lớp 1 (P(Normal) = 0,1), còn lại P(Normal) = 0,9."""
    x = np.asarray(x)
    p0 = np.where(x[:, 0] > 0, 0.1, 0.9)
    p = np.tile(((1 - p0) / 4)[:, None], (1, 5))
    p[:, 0] = p0
    return p


class FakeModel:
    threshold = 0.5

    def proba_window(self, g):
        return fake_proba(g.x.numpy())


class FakeRef:
    """Tham chiếu offline: đồ thị dựng bằng hàm offline, nhãn thật = lớp 1 nếu x[:, 0] > 0, xác suất = hàm giả."""

    def __init__(self, graphs):
        self.g = {}
        for gr in graphs:
            gr.y = torch.where(gr.x[:, 0] > 0, 1, 0)
            self.g[int(gr.win)] = (gr, fake_proba(gr.x.numpy()))

    def get(self, win):
        return self.g.get(int(win))


def setup():
    df = make_flows()
    rng = np.random.default_rng(1)
    df["label"] = rng.choice(["Normal", "DoS_DDoS"], len(df))
    nodes, edges = bg.build_tables(df.drop(columns="label"), 5, 5)
    scaler = StandardScaler().fit(nodes[bg.FEATURES].to_numpy(np.float64))
    off = bg.to_pyg(nodes, edges, scaler, 5)
    return df, scaler, off


def test_window_columns_layout():
    assert D.window_columns(5, False, False)[-2:] == ["ref_found", "warmup"]
    cols = D.window_columns(5, True, True)
    assert len(cols) == len(set(cols)) and "first_ts_4" in cols and "n_det_4" in cols and "max_dproba" in cols
    assert "first_ts_1" not in D.window_columns(5, False, True) and "n_fp" not in D.window_columns(5, True, False)


@pytest.mark.parametrize("keep_rows,trim_every", [(False, 0), (True, 3), (False, 2)])
def test_run_stream_writes_csv_and_matches_reference(tmp_path, keep_rows, trim_every):
    df, scaler, off = setup()
    seen = []
    win, n_flows = D.run_stream(FakeModel(), df, scaler, CLASSES, tmp_path, FakeRef(off), 5, 1.0, None, None, warmup=3,
                                trim_every=trim_every, probe=seen.append, keep_rows=keep_rows)
    assert len(win) == len(off) == len(seen) and seen == list(range(1, len(off) + 1))
    on_disk = pd.read_csv(tmp_path / "windows.csv")
    assert list(on_disk.columns) == D.window_columns(5, True, True) and len(on_disk) == len(win)
    assert (win["ref_found"] == 1).all() and (win["struct_ok"] == 1).all()
    assert win["max_dx"].max() < 1e-6 and win["max_de"].max() < 1e-6 and win["n_mis_thr"].sum() == 0 and win["n_mis_arg"].sum() == 0
    assert win["warmup"].tolist() == [1, 1, 1] + [0] * (len(win) - 3)
    assert n_flows == int(win["n_flows"].sum()) and (win["n_nodes"] > 0).all()
    assert (win["total_ms"] >= win["build_ms"]).all()
    alerts = pd.read_csv(tmp_path / "alerts.csv")
    assert len(alerts) == int(win["n_alert"].sum()) and set(alerts["pred_class"]) <= {"DoS_DDoS"}
    assert int(win["n_tp"].sum()) == len(alerts) and int(win["n_fp"].sum()) == 0                        # nhãn giả trùng quy tắc của mô hình giả
    assert (win["n_true_1"] == win["n_det_1"]).all()


def test_run_stream_without_reference_and_max_windows(tmp_path):
    df, scaler, _ = setup()
    win, _ = D.run_stream(FakeModel(), df.drop(columns="label"), scaler, CLASSES, tmp_path, None, 5, 1.0, None, 7, warmup=2)
    assert len(win) == 7 and (win["ref_found"] == 0).all() and "n_true_1" not in win.columns and "first_ts_1" not in win.columns
    assert win["warmup"].tolist() == [1, 1, 0, 0, 0, 0, 0]


def test_heap_stats_and_trim_do_not_fail():
    s = D.heap_stats()
    assert {"rss_mb", "py_blocks", "gc_objects"} <= set(s) and s["py_blocks"] > 0 and s["gc_objects"] > 0
    assert D.trim_memory() in (True, False)
