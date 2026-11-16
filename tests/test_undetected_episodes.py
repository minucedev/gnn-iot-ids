"""Kiểm thử scripts/undetected_episodes.py (GĐ J): lý do không phát hiện, mô tả đợt trên bảng node giả có đáp án tính tay."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


UE = _load("undetected_episodes")
CLASSES = ["Normal", "DoS_DDoS", "Botnet_Backdoor", "Truy_cap_trai_phep", "Khac"]
THR = 0.72


def test_classify_reason_boundaries():
    assert UE.classify_reason(0.10, 3, THR) == ("bi_gan_normal", ["bi_gan_normal"])
    assert UE.classify_reason(0.499, 1, THR) == ("bi_gan_normal", ["bi_gan_normal", "qua_ngan"])        # ngắn nhưng nguyên nhân chính vẫn là bị gán Normal
    assert UE.classify_reason(0.50, 3, THR) == ("duoi_nguong", ["duoi_nguong"])                          # đúng 0,5 thì không còn "gán Normal"
    assert UE.classify_reason(THR, 1, THR) == ("duoi_nguong", ["duoi_nguong", "qua_ngan"])                # điểm bằng ngưỡng vẫn không báo (điểm > ngưỡng mới báo)


def make_nodes():
    """Cửa sổ 10-11: 4 node lớp 1 (2 chỉ nhận), điểm 0,6 / 0,3 / 0,2 / 0,1, không báo. Cửa sổ 30: 2 node lớp 2, điểm 0,9 và 0,2, một node báo.
    Cửa sổ 50: 1 node lớp 3 điểm 0,05, không báo. Thêm node Normal để chắc rằng chúng bị bỏ qua."""
    rows = []

    def add(win, ip, y, score, recv, argmax):
        rows.append(dict(win=win, ip=ip, y=y, score=score, pred=(argmax if score > THR else 0), argmax=argmax, receive_only=recv))

    add(10, "a", 1, 0.6, False, 0)
    add(10, "b", 1, 0.3, True, 0)
    add(11, "a", 1, 0.2, False, 0)
    add(11, "c", 1, 0.1, True, 0)
    add(10, "n1", 0, 0.99, False, 1)                                    # Normal bị báo nhầm: không tính vào đợt lớp 1
    add(30, "a", 2, 0.9, False, 2)
    add(30, "b", 2, 0.2, True, 0)
    add(50, "z", 3, 0.05, True, 0)
    return pd.DataFrame(rows)


def make_flow_counts():
    idx = pd.MultiIndex.from_tuples([(10, "DoS_DDoS"), (10, "Normal"), (11, "DoS_DDoS"), (12, "DoS_DDoS"), (30, "Botnet_Backdoor"), (50, "Truy_cap_trai_phep")],
                                    names=["win", "label"])
    return pd.Series([100, 40, 60, 999, 7, 3], index=idx)


def test_describe_undetected_episode_hand_computed():
    ep = {"class_code": 1, "start_ts": 50.5, "end_ts": 60.0, "detected": False, "first_window": 10, "last_window": 11}
    d = UE.describe_episode(ep, CLASSES, make_nodes(), make_flow_counts(), THR, 5)
    assert d["class"] == "DoS_DDoS" and d["n_windows"] == 2 and d["n_nodes"] == 4 and d["n_ips"] == 3     # ip a lặp ở 2 cửa sổ
    assert d["share_receive_only"] == 0.5 and d["duration_s"] == 9.5
    assert d["n_flows_class"] == 160 and d["n_flows_all"] == 200                                          # cửa sổ 12 ngoài đợt nên không tính
    assert d["score_max"] == 0.6 and abs(d["score_median"] - 0.25) < 1e-12 and abs(d["score_mean"] - 0.3) < 1e-12
    assert d["n_nodes_score_ge_half"] == 1 and d["n_windows_score_ge_half"] == 1 and d["n_nodes_pred_attack"] == 0
    assert d["share_argmax_normal"] == 1.0 and d["argmax_distribution"] == {"Normal": 4}
    assert d["reason_primary"] == "duoi_nguong" and d["reason_tags"] == ["duoi_nguong"]
    assert d["start_utc"].startswith("1970-01-01T00:00:50")


def test_describe_detected_and_single_window_short_episode():
    det = UE.describe_episode({"class_code": 2, "start_ts": 150.0, "end_ts": 155.0, "detected": True, "first_window": 30, "last_window": 30,
                               "detection_s": 5.1, "first_alert_window": 30}, CLASSES, make_nodes(), make_flow_counts(), THR, 5)
    assert det["detected"] and det["n_nodes_pred_attack"] == 1 and det["detection_s"] == 5.1 and "reason_primary" not in det
    short = UE.describe_episode({"class_code": 3, "start_ts": 250.0, "end_ts": 255.0, "detected": False, "first_window": 50, "last_window": 50},
                                CLASSES, make_nodes(), make_flow_counts(), THR, 5)
    assert short["n_windows"] == 1 and short["reason_primary"] == "bi_gan_normal" and short["reason_tags"] == ["bi_gan_normal", "qua_ngan"]
    assert short["n_flows_class"] == 3 and short["share_receive_only"] == 1.0


def test_summarize_counts():
    nodes, fc = make_nodes(), make_flow_counts()
    eps = [UE.describe_episode({"class_code": 1, "start_ts": 50.5, "end_ts": 60.0, "detected": False, "first_window": 10, "last_window": 11}, CLASSES, nodes, fc, THR, 5),
           UE.describe_episode({"class_code": 3, "start_ts": 250.0, "end_ts": 255.0, "detected": False, "first_window": 50, "last_window": 50}, CLASSES, nodes, fc, THR, 5),
           UE.describe_episode({"class_code": 2, "start_ts": 150.0, "end_ts": 155.0, "detected": True, "first_window": 30, "last_window": 30,
                                "detection_s": 5.1, "first_alert_window": 30}, CLASSES, nodes, fc, THR, 5)]
    s = UE.summarize(eps)
    assert s["n_undetected"] == 2 and s["reasons_primary"] == {"duoi_nguong": 1, "bi_gan_normal": 1}
    assert s["by_class"]["DoS_DDoS"]["n_undetected"] == 1 and s["by_class"]["Truy_cap_trai_phep"]["reasons"] == {"bi_gan_normal": 1}
    assert s["n_single_window"] == 1 and s["n_receive_only_majority"] == 1                                # chỉ đợt lớp 3 có > 50% node chỉ nhận
    assert np.isclose(s["by_class"]["Truy_cap_trai_phep"]["score_max_median"], 0.05)
