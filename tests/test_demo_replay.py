"""Kiểm thử src/demo_replay.py (task 8.2): đóng gói cửa sổ, rút gọn đồ thị, bảng tốc độ và bảng giới hạn từ kết quả đã có trong reports/."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src import demo_replay as DR

ROOT = Path(__file__).resolve().parents[1]


def make_win():
    return pd.DataFrame({"win": [100, 101, 102], "t_close": [1556136425, 1556136430, 1556136435], "n_nodes": [12, 9, 41], "n_edges": [11, 8, 38],
                         "total_ms": [103.24, 101.0, 109.46], "n_alert": [0, 0, 3], "n_true_attack": [0, 0, 20]})


def make_alerts():
    return pd.DataFrame({"t_close": ["x"] * 3, "window": [102, 102, 102], "ip": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
                         "pred_class": ["DoS_DDoS", "DoS_DDoS", "Khac"], "p_attack": [0.8, 0.99912, 0.9], "p_class": [0.5, 0.5, 0.5], "proc_ms": [109.0] * 3})


def test_vn_number_format():
    assert DR.vn(144560) == "144.560" and DR.vn(5.0921, 1) == "5,1" and DR.vn(1937.4) == "1.937" and DR.vn(0.5, 2) == "0,50"


def test_pack_windows():
    rows = DR.pack_windows(make_win(), make_alerts(), max_alerts=2)
    assert [r["t"] for r in rows] == ["20:07:05", "20:07:10", "20:07:15"]
    assert rows[0] == {"t": "20:07:05", "t_close": 1556136425, "nodes": 12, "edges": 11, "ms": 103.2, "n_alert": 0, "alerts": [], "truth": 0}
    last = rows[2]
    assert last["n_alert"] == 3 and last["truth"] == 20 and last["ms"] == 109.5
    assert [a["ip"] for a in last["alerts"]] == ["10.0.0.2", "10.0.0.3"] and last["alerts"][0] == {"ip": "10.0.0.2", "cls": "DoS_DDoS", "p": 0.999}
    json.dumps(rows)


def test_pack_windows_without_alerts_and_truth():
    empty = make_alerts().iloc[0:0]
    rows = DR.pack_windows(make_win().drop(columns="n_true_attack").assign(n_alert=0), empty)
    assert len(rows) == 3 and all(r["alerts"] == [] and r["truth"] is None for r in rows)


def graph(n, attack_idx, edges):
    return {"win": 7, "t_close": "2019-04-24T20:09:20+00:00", "n_nodes": n, "n_edges": len(edges), "total_ms": 109.4,
            "nodes": [{"ip": f"10.0.0.{i}", "pred": "DoS_DDoS" if i in attack_idx else "Normal", "p_attack": 0.99 if i in attack_idx else 0.01} for i in range(n)],
            "edges": edges}


def test_shrink_graph_keeps_small_graph_unchanged():
    g = graph(5, {1}, [[0, 1], [2, 1], [3, 1]])
    s = DR.shrink_graph(g, max_nodes=10)
    assert not s["truncated"] and s["t"] == "20:09:20" and s["n_nodes_total"] == 5 and s["n_edges_total"] == 3 and s["n_alert_total"] == 1
    assert [n["ip"] for n in s["nodes"]] == [f"10.0.0.{i}" for i in range(5)] and s["edges"] == g["edges"]
    assert s["nodes"][1] == {"ip": "10.0.0.1", "pred": "DoS_DDoS", "p": 0.99}


def test_shrink_graph_truncates_to_max_nodes_and_keeps_alerts():
    n = 50
    edges = [[i, 0] for i in range(1, n)] + [[0, 1]]                       # node 0 là đích chung (fan-in), node 1..49 chỉ có 1-2 cạnh
    s = DR.shrink_graph(graph(n, {0, 1, 2}, edges), max_nodes=10)
    assert s["truncated"] and len(s["nodes"]) == 10 and s["n_nodes_total"] == 50 and s["n_edges_total"] == len(edges) and s["n_alert_total"] == 3
    ips = [x["ip"] for x in s["nodes"]]
    assert {"10.0.0.0", "10.0.0.1", "10.0.0.2"} <= set(ips)                  # node bị cảnh báo luôn được giữ
    assert all(0 <= a < 10 and 0 <= b < 10 for a, b in s["edges"]) and all(s["nodes"][a]["ip"] != s["nodes"][b]["ip"] for a, b in s["edges"])
    assert len(s["edges"]) < len(edges)
    pairs = {(ips[a], ips[b]) for a, b in s["edges"]}
    assert ("10.0.0.1", "10.0.0.0") in pairs and ("10.0.0.0", "10.0.0.1") in pairs   # cạnh giữa hai node được giữ vẫn còn, đúng chiều


def test_shrink_graph_many_alerts_capped_to_half():
    n = 40
    s = DR.shrink_graph(graph(n, set(range(30)), [[i, i + 1] for i in range(n - 1)]), max_nodes=10)
    assert len(s["nodes"]) == 10 and sum(x["pred"] != "Normal" for x in s["nodes"]) >= 5 and s["n_alert_total"] == 30


def test_plot_first_alert_writes_png(tmp_path):
    pytest.importorskip("networkx")
    pytest.importorskip("matplotlib")
    n = 30
    edges = [[i, 0] for i in range(1, n)] + [[0, 1], [2, 3]]
    g = graph(n, {0, 1}, edges)
    info = DR.plot_first_alert(g, tmp_path / "g.png", title_prefix="D1 ", max_nodes=20)
    assert (tmp_path / "g.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert info == {"in_max": info["in_max"], "out_max": info["out_max"], "n_nodes": 30, "n_edges": len(edges), "n_alert": 2, "t": "20:09:20"}
    assert info["in_max"] >= 15 and info["out_max"] >= 1                      # node 0 là đích chung của các node còn lại (fan-in) trong phần được vẽ


def test_pick_episode():
    eps = [{"start_ts": 100.0, "detection_s": 5.0}, {"start_ts": 5000.2, "detection_s": 4.9}]
    assert DR.pick_episode(eps, 5000.0, tol=1.0)["detection_s"] == 4.9
    with pytest.raises(ValueError):
        DR.pick_episode(eps, 3000.0)
    with pytest.raises(ValueError):
        DR.pick_episode([], 1.0)


def test_speed_rows_from_committed_results():
    nums = DR.speed_numbers(ROOT)
    m, l = nums[DR.MAIN_MODEL], nums[DR.LIGHT_MODEL]
    assert (m["n_params"], l["n_params"]) == (144560, 30934) and m["threads"] == l["threads"] == 1 and m["limit_ms"] == 1000
    assert m["agreement"] == l["agreement"] == 1.0 and 1500 < m["flows_per_s"] < 2500 and m["p95_ms"] < 150 and l["p95_ms"] < 150
    rows = {r[0]: r[1:] for r in DR.speed_rows(nums)}
    assert rows["Số tham số / dung lượng"] == ("144.560 / 9,7 MB", "30.934 / 986 KB")
    assert rows["Dự đoán online khớp offline"] == ("100%", "100%")
    lat, build = rows["Độ trễ mỗi cửa sổ (trung vị / p95)"][0], rows["Phần dựng đồ thị / phần dự đoán (p95)"][0]
    assert lat.endswith(" ms") and " / " in lat and build.endswith(" ms") and " / " in build


def test_limits_rows_from_committed_results():
    rows = DR.limits_rows(ROOT)
    by = {r["class"]: r for r in rows}
    assert [(by[c]["n"], by[c]["detected"]) for c in DR.ATTACK_CLASSES] == [(4, 4), (5, 4), (5, 2), (13, 3)]
    assert (by["all"]["n"], by["all"]["detected"]) == (27, 13) and "14 đợt không phát hiện" in by["all"]["note"] and "6 đợt chỉ kéo dài 1 cửa sổ" in by["all"]["note"]
    assert by["DoS_DDoS"]["note"].startswith("Phát hiện hết") and by["DoS_DDoS"]["note"].endswith(" giây")
    assert "11 flow tấn công" in by["Botnet_Backdoor"]["note"]
    assert "1–2 flow" in by["Truy_cap_trai_phep"]["note"] and "3 đợt chỉ kéo dài 1 cửa sổ" in by["Truy_cap_trai_phep"]["note"]
    assert by["Khac"]["label"] == "Khac (mitm, ransomware)"
    json.dumps(rows, ensure_ascii=False)
