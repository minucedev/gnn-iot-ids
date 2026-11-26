"""Kiểm thử đường chạy của scripts/detector.py (task 8.1-8.3): ghi windows.csv từng dòng, khớp offline, trim, probe. Dùng mô hình giả."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src import build_graph as bg

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load("detector")
_SF_SPEC = importlib.util.spec_from_file_location("test_stream_features_helpers", ROOT / "tests" / "test_stream_features.py")
_SF_TEST = importlib.util.module_from_spec(_SF_SPEC)
_SF_SPEC.loader.exec_module(_SF_TEST)
make_flows = _SF_TEST.make_flows
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


# ---------------------------------------------------------------- tùy chọn cho demo (--start/--end, --print-windows, ...)
TIMING_COLS = ["build_ms", "infer_ms", "total_ms", "rss_mb"]


def test_parse_utc():
    t = D.parse_utc("2019-04-24T20:07:00")
    assert t == 1556136420.0                                              # 24/04/2019 20:07:00 UTC
    assert D.parse_utc("2019-04-24T20:07:00Z") == t == D.parse_utc("2019-04-24 20:07:00+00:00")
    assert D.parse_utc("2019-04-25T03:07:00+07:00") == t                  # có múi giờ thì đổi về UTC
    assert D.parse_utc("2019-04-24T20:09:15.5") == t + 135.5


def test_slice_time_is_half_open_and_none_is_identity():
    df = setup()[0]
    assert D.slice_time(df) is df and D.slice_time(df, None, None) is df
    t0 = float(df["ts"].min())
    part = D.slice_time(df, t0 + 20, t0 + 60)
    assert len(part) and part["ts"].min() >= t0 + 20 and part["ts"].max() < t0 + 60
    assert len(part) == int(((df["ts"] >= t0 + 20) & (df["ts"] < t0 + 60)).sum())
    assert len(D.slice_time(df, t0 + 20)) == int((df["ts"] >= t0 + 20).sum())
    assert len(D.slice_time(df, None, t0 + 60)) == int((df["ts"] < t0 + 60).sum())
    assert len(D.slice_time(df, t0 + 1e6, t0 + 2e6)) == 0
    assert D.slice_time(df, t0, float(df["ts"].max()) + 1).equals(df)


def test_window_line_format():
    t = D.parse_utc("2019-04-24T20:07:05")
    assert D.format_window_line(t, 12, 11, 103.2, 0) == "[20:07:05] cửa sổ   12 node   11 cạnh   103 ms   cảnh báo: 0"
    line = D.format_window_line(D.parse_utc("2019-04-24T20:09:20"), 41, 38, 109.4, 3)
    assert line == "[20:09:20] cửa sổ   41 node   38 cạnh   109 ms   ⚠ 3 cảnh báo"
    assert D.RED in D.format_window_line(t, 12, 11, 103.2, 3, color=True) and D.RED not in D.format_window_line(t, 12, 11, 103.2, 0, color=True)
    a = D.format_alert_line("192.168.1.39", "DoS_DDoS", 0.999, color=False)
    assert a.split() == ["⚠", "192.168.1.39", "DoS_DDoS", "p", "=", "1,00"] and a.startswith(" " * 11)
    assert D.RED in D.format_alert_line("10.0.0.1", "Khac", 0.5, color=True) and "p = 0,50" in D.format_alert_line("10.0.0.1", "Khac", 0.5)


def test_window_report_limits_alert_lines_and_sorts_by_p():
    row = {"t_close": D.parse_utc("2019-04-24T20:09:20"), "n_nodes": 41, "n_edges": 38, "total_ms": 109.0, "n_alert": 4}
    alerts = [(f"10.0.0.{i}", "DoS_DDoS", p, p) for i, p in enumerate([0.8, 0.99, 0.9, 0.95])]
    lines = D.window_report({"row": row, "alerts": alerts}, max_alert_lines=2)
    assert len(lines) == 4 and "4 cảnh báo" in lines[0]
    assert "10.0.0.1" in lines[1] and "10.0.0.3" in lines[2] and "2 cảnh báo khác" in lines[3]
    assert len(D.window_report({"row": row, "alerts": alerts}, max_alert_lines=9)) == 5
    quiet = D.window_report({"row": {**row, "n_alert": 0}, "alerts": []})
    assert len(quiet) == 1 and quiet[0].endswith("cảnh báo: 0")


def _read_det(path):
    """windows.csv và alerts.csv của một lần chạy, bỏ các cột đo thời gian/RAM (mỗi lần chạy một khác)."""
    win = pd.read_csv(path / "windows.csv")
    win = win.drop(columns=[c for c in TIMING_COLS if c in win.columns])
    return win, pd.read_csv(path / "alerts.csv").drop(columns=["proc_ms"])


def _iso(t):
    return D.datetime.fromtimestamp(t, tz=D.timezone.utc).isoformat()


def test_run_stream_callbacks_do_not_change_results(tmp_path):
    df, scaler, off = setup()
    seen = []
    base, _ = D.run_stream(FakeModel(), df, scaler, CLASSES, tmp_path / "a", FakeRef(off), 5, 1.0, None, None, warmup=3)
    cb, _ = D.run_stream(FakeModel(), df, scaler, CLASSES, tmp_path / "b", FakeRef(setup()[2]), 5, 1.0, None, None, warmup=3, on_window=seen.append)
    wa, aa = _read_det(tmp_path / "a")
    wb, ab = _read_det(tmp_path / "b")
    assert wa.equals(wb) and aa.equals(ab)
    assert len(seen) == len(base) == len(cb) and all(set(s) == {"row", "alerts", "graph", "pred", "score", "iso"} for s in seen)
    assert [s["row"]["n_alert"] for s in seen] == base["n_alert"].tolist() and [len(s["alerts"]) for s in seen] == base["n_alert"].tolist()
    assert sum(len(s["alerts"]) for s in seen) == len(aa)


def test_stop_after_alert_stops_at_first_alert_window(tmp_path):
    df, scaler, _ = setup()
    full, _ = D.run_stream(FakeModel(), df, scaler, CLASSES, tmp_path / "full", None, 5, 1.0, None, None, warmup=0)
    k = int(np.flatnonzero(full["n_alert"].to_numpy() > 0)[0])
    stopped, _ = D.run_stream(FakeModel(), df, scaler, CLASSES, tmp_path / "stop", None, 5, 1.0, None, None, warmup=0, stop_after_alert=True)
    assert len(stopped) == k + 1 and stopped["n_alert"].iloc[-1] > 0 and (stopped["n_alert"].iloc[:-1] == 0).all()
    assert stopped["win"].tolist() == full["win"].iloc[:k + 1].tolist()
    alert_idx = np.flatnonzero(full["n_alert"].to_numpy() > 0)
    assert len(alert_idx) >= 2                                                                    # có từ hai cửa sổ cảnh báo trở lên để thử alert_from
    k2 = int(alert_idx[1])
    late, _ = D.run_stream(FakeModel(), df, scaler, CLASSES, tmp_path / "late", None, 5, 1.0, None, None, warmup=0, stop_after_alert=True,
                           alert_from=float(full["t_start"].iloc[k2]))                          # bỏ qua cửa sổ cảnh báo đầu: dừng ở cửa sổ cảnh báo kế tiếp
    assert len(late) == k2 + 1 and late["win"].tolist() == full["win"].iloc[:k2 + 1].tolist()
    none_after, _ = D.run_stream(FakeModel(), df, scaler, CLASSES, tmp_path / "none", None, 5, 1.0, None, None, warmup=0, stop_after_alert=True,
                                 alert_from=float(full["t_close"].max()) + 1)
    assert len(none_after) == len(full)                                                           # không có cảnh báo nào từ mốc đó: chạy hết


def test_graph_dump_describes_window(tmp_path):
    df, scaler, _ = setup()
    got = []
    D.run_stream(FakeModel(), df, scaler, CLASSES, tmp_path, None, 5, 1.0, None, 4, warmup=0, on_window=got.append)
    info = got[0]
    d = D.graph_dump(info, CLASSES)
    g = info["graph"]
    assert d["n_nodes"] == len(d["nodes"]) == g.num_nodes and d["n_edges"] == len(d["edges"]) == g.num_edges
    assert [n["ip"] for n in d["nodes"]] == list(g.ips)
    assert {n["pred"] for n in d["nodes"]} <= {"Normal", "DoS_DDoS"} and all(0 <= n["p_attack"] <= 1 for n in d["nodes"])
    assert sum(n["pred"] != "Normal" for n in d["nodes"]) == info["row"]["n_alert"]
    assert all(0 <= a < g.num_nodes and 0 <= b < g.num_nodes for a, b in d["edges"])
    import json
    json.dumps(d)                                                                  # ghi được ra JSON


def test_parser_defaults_leave_old_behavior():
    a = D.build_parser().parse_args(["--config", "c.json", "--out", "o"])
    assert (a.start, a.end, a.flows_csv, a.dump_first_alert, a.alert_from) == (None, None, None, None, None)
    assert not (a.partial_flows or a.print_windows or a.color or a.stop_after_alert) and a.max_alert_lines == 5
    assert (a.model, a.seed, a.restrict, a.batch_sec, a.speedup, a.max_windows, a.trim_every, a.no_reference) == \
           (D.FT.LIGHT, 42, "test", 1.0, None, None, D.TRIM_EVERY, False)


class FakeSession:
    """Phiên đã nạp giả cho `main(argv, session)`: mô hình giả, không đối chiếu offline."""

    def __init__(self, scaler):
        self.name, self.seed, self.ctx, self.model, self.ref = D.FT.LIGHT, 42, object(), FakeModel(), None
        self.model.n_params, self.model.size_kb = 1, 1.0
        self.classes, self.scaler, self.window_sec, self.frozen_p95 = CLASSES, scaler, 5, 1.0


def run_main(tmp_path, monkeypatch, name, df, scaler, *extra):
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"dataset": "x", "flows_csv": "unused.csv"}', encoding="utf-8")
    seen = {}

    def fake_load(cfg_, ctx, restrict="test", allow_partial=False):
        seen.update(csv=cfg_["flows_csv"], allow_partial=allow_partial, restrict=restrict)
        return df.copy()

    monkeypatch.setattr(D, "load_stream_flows", fake_load)
    out = tmp_path / name
    out.mkdir()
    rc = D.main(["--config", str(cfg), "--out", str(out), "--no-reference", *extra], session=FakeSession(scaler))
    assert rc == 0
    return out, seen


def test_main_default_equals_full_range_with_options(tmp_path, monkeypatch, capsys):
    df, scaler, _ = setup()
    t0, t1 = float(df["ts"].min()), float(df["ts"].max())
    a, seen_a = run_main(tmp_path, monkeypatch, "a", df, scaler)
    quiet = capsys.readouterr().out
    assert "] cửa sổ" not in quiet and "⚠" not in quiet                              # mặc định không in từng cửa sổ
    b, seen_b = run_main(tmp_path, monkeypatch, "b", df, scaler, "--start", _iso(t0 - 1), "--end", _iso(t1 + 1), "--print-windows", "--partial-flows",
                         "--flows-csv", "cut.csv", "--dump-first-alert", str(tmp_path / "first.json"))
    printed = capsys.readouterr().out
    assert _read_det(a)[0].equals(_read_det(b)[0]) and _read_det(a)[1].equals(_read_det(b)[1])
    assert seen_a == {"csv": "unused.csv", "allow_partial": False, "restrict": "test"} and seen_b == {"csv": "cut.csv", "allow_partial": True, "restrict": "test"}
    import json
    sa, sb = (json.loads((p / "detector_summary.json").read_text(encoding="utf-8")) for p in (a, b))
    assert "start" not in sa and "end" not in sa and sb["start"] and sb["end"]
    assert {k: v for k, v in sb.items() if k not in ("start", "end", "total_seconds", "stream_seconds", "ram_start_mb", "ram_end_mb")} == \
           {k: v for k, v in sa.items() if k not in ("total_seconds", "stream_seconds", "ram_start_mb", "ram_end_mb")}
    n_win = int(len(_read_det(a)[0]))
    assert printed.count("] cửa sổ") == n_win and "⚠" in printed                    # mỗi cửa sổ đúng một dòng
    first = json.loads((tmp_path / "first.json").read_text(encoding="utf-8"))
    al = pd.read_csv(a / "alerts.csv")
    assert first["win"] == int(al["window"].iloc[0]) and first["t_close"] == al["t_close"].iloc[0]
    assert sum(n["pred"] != "Normal" for n in first["nodes"]) == int((al["window"] == al["window"].iloc[0]).sum())


def test_main_runs_only_the_selected_segment(tmp_path, monkeypatch, capsys):
    df, scaler, _ = setup()
    t0 = float(df["ts"].min())
    s, e = (t0 // 5) * 5 + 20, (t0 // 5) * 5 + 60                                    # bội của 5 s
    full, _ = run_main(tmp_path, monkeypatch, "full", df, scaler)
    seg, _ = run_main(tmp_path, monkeypatch, "seg", df, scaler, "--start", _iso(s), "--end", _iso(e))
    wf, ws = pd.read_csv(full / "windows.csv"), pd.read_csv(seg / "windows.csv")
    assert len(ws) and ws["t_start"].min() >= s and ws["t_close"].max() <= e and len(ws) < len(wf)
    assert ws["win"].tolist() == wf[(wf["t_start"] >= s) & (wf["t_close"] <= e)]["win"].tolist()
    cols = [c for c in wf.columns if c not in TIMING_COLS + ["warmup"]]
    assert ws[cols].reset_index(drop=True).equals(wf[wf["win"].isin(ws["win"])][cols].reset_index(drop=True))     # đoạn cho kết quả y hệt cửa sổ tương ứng
    with pytest.raises(SystemExit):
        run_main(tmp_path, monkeypatch, "bad", df, scaler, "--start", _iso(e), "--end", _iso(s))


def test_main_stop_after_alert_from_and_dump(tmp_path, monkeypatch, capsys):
    df, scaler, _ = setup()
    full, _ = run_main(tmp_path, monkeypatch, "full", df, scaler)
    wf = pd.read_csv(full / "windows.csv")
    capsys.readouterr()
    idx = np.flatnonzero(wf["n_alert"].to_numpy() > 0)
    k2 = int(idx[1])
    out, _ = run_main(tmp_path, monkeypatch, "stop", df, scaler, "--print-windows", "--stop-after-alert", "--alert-from", _iso(float(wf["t_start"].iloc[k2])),
                      "--dump-first-alert", str(tmp_path / "g.json"))
    printed = capsys.readouterr().out
    ws = pd.read_csv(out / "windows.csv")
    assert len(ws) == k2 + 1 and printed.count("] cửa sổ") == k2 + 1                              # mọi cửa sổ vẫn in, kể cả cửa sổ cảnh báo trước alert_from
    import json
    g = json.loads((tmp_path / "g.json").read_text(encoding="utf-8"))
    assert g["win"] == int(wf["win"].iloc[k2]) and g["n_nodes"] == int(wf["n_nodes"].iloc[k2])      # đồ thị của cửa sổ cảnh báo đầu tiên từ alert_from
