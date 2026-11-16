"""Kiểm thử scripts/realtime_eval.py (task 8.3, 8.4) trên bảng cửa sổ giả có đáp án tính tay."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RE = _load("realtime_eval")
CLASSES = ["Normal", "DoS_DDoS", "Botnet_Backdoor", "Truy_cap_trai_phep", "Khac"]


def make_win(n=40):
    """40 cửa sổ 5 s (win = 0..39, t_start = 5·win). 5 cửa sổ đầu là warm-up (chậm bất thường).
    Lớp 1: đợt A ở cửa sổ 10-14 (phát hiện lần đầu ở cửa sổ 12), đợt B ở 25-26 (phát hiện ngay ở 25). Lớp 2: cửa sổ 30-31, không bao giờ phát hiện.
    Cảnh báo giả: 2 ở cửa sổ 3 và 1 ở cửa sổ 20 (cả hai cửa sổ chỉ có Normal)."""
    r = []
    for w in range(n):
        row = dict(win=w, t_start=5.0 * w, t_close=5.0 * (w + 1), n_flows=100 + w, n_nodes=10, n_edges=20, build_ms=6.0, infer_ms=4.0 + (w % 5),
                   total_ms=10.0 + (w % 5), rss_mb=1000.0, n_alert=0, ref_found=1, n_normal=10, n_true_attack=0, n_fp=0, n_tp=0, struct_ok=1,
                   max_dx=1e-8, max_de=0.0, n_mis_thr=0, n_mis_arg=0, max_dproba=1e-7, warmup=int(w < 5))
        for k in (1, 2, 3, 4):
            row.update({f"n_true_{k}": 0, f"n_det_{k}": 0, f"first_ts_{k}": np.nan})
        if w < 5:
            row.update(total_ms=5000.0, build_ms=3000.0, infer_ms=2000.0)
        if w in (10, 11, 12, 13, 14, 25, 26):
            row.update(n_true_1=3, n_true_attack=3, n_normal=7, first_ts_1=5.0 * w + 2.0 if w in (10, 25) else 5.0 * w + 1.0)
            if w in (12, 13, 14, 25, 26):
                row.update(n_det_1=2, n_alert=2, n_tp=2)
        if w in (30, 31):
            row.update(n_true_2=4, n_true_attack=4, n_normal=6, first_ts_2=5.0 * w, n_alert=0)
        if w == 3:
            row.update(n_alert=2, n_fp=2)
        if w == 20:
            row.update(n_alert=1, n_fp=1)
        r.append(row)
    return pd.DataFrame(r)


SUMMARY = {"model": "hybrid_light", "seed": 42, "dataset": "nf_ton_iot_v3", "split": "test", "window_sec": 5, "classes": CLASSES, "threshold": 0.5,
           "n_params": 30934, "size_kb": 985.7, "n_flows": 5000, "frozen_style_p95_ms": 7.0, "env": {"cpu_model": "x"}}


def test_latency_report_excludes_warmup():
    win = make_win()
    lat = RE.latency_report(win, SUMMARY)
    assert lat["n_windows_total"] == 40 and lat["n_warmup"] == 5 and lat["n_windows_measured"] == 35
    m = win[win["warmup"] == 0]["total_ms"].to_numpy()
    assert abs(lat["total_ms"]["p95"] - np.percentile(m, 95)) < 1e-9 and lat["total_ms"]["max"] == 14.0     # warm-up 5000 ms không lọt vào
    assert abs(lat["build_ms"]["p50"] - 6.0) < 1e-9 and lat["p95_ok"] and lat["n_windows_over_limit"] == 0
    assert lat["throughput"]["flows_per_s"] > 0 and lat["by_nodes"][0]["n_windows"] == 35
    assert RE.latency_report(win.assign(total_ms=2000.0), SUMMARY)["p95_ok"] is False


def test_agreement_report():
    win = make_win()
    a = RE.agreement_report(win)
    assert a["n_windows_compared"] == 40 and a["n_nodes"] == 400 and a["features_ok"] and a["pred_ok"] and a["pred_agreement_threshold"] == 1.0
    win2 = win.copy()
    win2.loc[7, "n_mis_thr"] = 1                                                      # 1 node lệch trên 400: 99,75% < 99,9%
    a2 = RE.agreement_report(win2)
    assert abs(a2["pred_agreement_threshold"] - (1 - 1 / 400)) < 1e-12 and a2["pred_ok"] is False and a2["windows_with_mismatch_threshold"] == 1
    win3 = win.copy()
    win3.loc[7, "max_dx"] = 1e-3
    assert RE.agreement_report(win3)["features_ok"] is False
    win4 = win.copy()
    win4.loc[8, "struct_ok"] = 0
    assert RE.agreement_report(win4)["features_ok"] is False


def test_false_alarms_per_hour_hand_computed():
    """Cửa sổ chỉ Normal = 40 - 7 (lớp 1) - 2 (lớp 2) = 31; 31 · 5 s = 155 s = 155/3600 giờ; 3 cảnh báo giả → 3 / (155/3600) ≈ 69,68 mỗi giờ."""
    fa = RE.false_alarm_report(make_win(), 5)
    assert fa["n_windows_normal_only"] == 31 and fa["n_false_alerts"] == 3 and fa["windows_with_false_alert"] == 2
    assert abs(fa["hours_normal_only"] - 155 / 3600) < 1e-12 and abs(fa["false_alerts_per_hour"] - 3 / (155 / 3600)) < 1e-9


def test_detection_times_hand_computed():
    """Đợt A (cửa sổ 10-14): bắt đầu 52,0 s, báo đầu tiên ở cửa sổ 12 đóng lúc 65 s, xử lý 12 ms → 65 - 52 + 0,012 = 13,012 s.
    Đợt B (25-26): bắt đầu 127,0, báo ở cửa sổ 25 đóng lúc 130, xử lý 10 ms → 3,010 s. Lớp 2 (30-31) không phát hiện."""
    win = make_win()
    e = RE.episodes(win, 1, gap_s=20)
    assert len(e) == 2 and e[0]["detected"] and e[0]["first_alert_window"] == 12 and e[0]["n_windows"] == 5
    assert abs(e[0]["detection_s"] - 13.012) < 1e-9 and abs(e[1]["detection_s"] - 3.010) < 1e-9
    assert len(RE.episodes(win, 1, gap_s=60)) == 1                                       # cách nhau 50 s ≤ 60 s: gộp một đợt
    d = RE.detection_report(win, CLASSES, gap_s=20)
    assert d["n_episodes"] == 3 and d["n_detected"] == 2 and d["n_undetected"] == 1
    assert abs(d["detection_median_s"] - (13.012 + 3.010) / 2) < 1e-9 and abs(d["detection_max_s"] - 13.012) < 1e-9
    assert d["per_class"]["DoS_DDoS"]["n_detected"] == 2 and d["per_class"]["Botnet_Backdoor"]["n_episodes"] == 1
    assert d["per_class"]["Botnet_Backdoor"]["max_s"] is None


def test_node_report_and_ram():
    win = make_win()
    n = RE.node_report(win)
    assert n["n_attack"] == 7 * 3 + 2 * 4 and n["n_normal"] == 400 - 29 and abs(n["fpr"] - 3 / 371) < 1e-12 and abs(n["recall_attack"] - 10 / 29) < 1e-12
    r = RE.ram_report(win)
    assert r["stable"] and abs(r["growth_late_vs_early"]) < 1e-12 and r["ram_max_mb"] == 1000.0
    grow = win.assign(rss_mb=np.linspace(1000, 1500, len(win)))
    g = RE.ram_report(grow)
    assert g["stable"] is False and g["growth_late_vs_early"] > 0.3 and g["slope_mb_per_1000_windows_second_half"] > 0


def test_targets_table_marks_pass_and_fail():
    win = make_win()
    summary = dict(SUMMARY)
    lat = RE.latency_report(win, summary)
    ram, agree = RE.ram_report(win), RE.agreement_report(win)
    t = RE.targets_table(lat, ram, agree, RE.false_alarm_report(win), RE.detection_report(win, CLASSES), RE.node_report(win), summary)
    by = {r["chi_tieu"]: r for r in t}
    assert by["Độ trễ p95 end-to-end ≤ 1 s mỗi cửa sổ"]["dat"] is True
    assert by["Tổng tham số ≤ 100.000"]["dat"] is True and by["Kích thước file mô hình ≤ 1 MB"]["dat"] is True
    big = dict(summary, n_params=144000, size_kb=9700.0)
    t2 = {r["chi_tieu"]: r for r in RE.targets_table(lat, ram, agree, None, None, None, big)}
    assert t2["Tổng tham số ≤ 100.000"]["dat"] is False and t2["Kích thước file mô hình ≤ 1 MB"]["dat"] is False
    assert by["Cảnh báo giả mỗi giờ (đoạn chỉ có Normal)"]["dat"] is None                 # chưa có chỉ tiêu: chỉ báo cáo
    assert by["Recall tấn công mức node (online)"]["dat"] is False


def test_alerts_sample_and_evaluate_end_to_end(tmp_path):
    csv = tmp_path / "alerts.csv"
    pd.DataFrame({"t_close": range(2500), "window": range(2500), "ip": "10.0.0.1", "pred_class": "DoS_DDoS", "p_attack": 0.9, "p_class": 0.8,
                  "proc_ms": 12.0}).to_csv(csv, index=False)
    out = tmp_path / "s.csv"
    assert RE.alerts_sample(csv, out, 1000) == 1000 and len(pd.read_csv(out)) == 1000
    assert pd.read_csv(out)["window"].iloc[0] == 0 and pd.read_csv(out)["window"].iloc[-1] == 2499      # trải đều từ đầu đến cuối
    small = tmp_path / "small.csv"
    pd.DataFrame({"a": range(10)}).to_csv(small, index=False)
    assert RE.alerts_sample(small, tmp_path / "o2.csv", 1000) == 10

    det = tmp_path / "det"
    det.mkdir()
    make_win().to_csv(det / "windows.csv", index=False)
    (det / "detector_summary.json").write_text(json.dumps(SUMMARY), encoding="utf-8")
    csv.replace(det / "alerts.csv")
    lat, ev = RE.evaluate(det, tmp_path / "tables", tmp_path / "figs")
    for f in ("tables/latency.json", "tables/realtime_eval.json", "tables/alerts_sample.csv", "figs/gd8_latency_hist.png", "figs/gd8_latency_vs_nodes.png"):
        assert (tmp_path / f).stat().st_size > 0, f
    loaded = json.loads((tmp_path / "tables" / "realtime_eval.json").read_text(encoding="utf-8"))
    assert loaded["false_alarms"]["n_false_alerts"] == 3 and loaded["detection"]["n_episodes"] == 2      # gap 60 s: lớp 1 gộp thành một đợt, lớp 2 một đợt
    assert loaded["online_vs_offline"]["pred_ok"] and len(loaded["targets"]) >= 8
