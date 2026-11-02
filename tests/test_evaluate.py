import csv
import math

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src import evaluate as ev

# Ví dụ tính tay: 12 node, 5 lớp (0 Normal, 1 DoS_DDoS, 2 Botnet_Backdoor, 3 Truy_cap_trai_phep, 4 Khac)
#   idx     0  1  2  3  4  5  6  7  8  9 10 11
Y_TRUE = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4]
Y_PRED = [0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 3, 4]
# Đúng ở idx 0,1,2,4,5,6,8,10,11 -> accuracy = 9/12
# Nhị phân (Normal = âm): 5 Normal, idx 3 bị báo tấn công -> FP = 1, TN = 4 -> FPR = 1/5
#                          7 tấn công, idx 7 bị báo Normal -> FN = 1, TP = 6 -> recall tấn công = 6/7
# F1 từng lớp: lớp 0: P = 4/5, R = 4/5 -> 4/5
#              lớp 1: dự đoán 1 ở idx 3,5,6,9 (đúng 5,6) -> P = 2/4, R = 2/3 -> 4/7
#              lớp 2: P = 1/1, R = 1/2 -> 2/3;  lớp 3: 1;  lớp 4: 1
F1_EACH = [4 / 5, 4 / 7, 2 / 3, 1.0, 1.0]


def test_hand_computed_metrics():
    m = ev.compute_metrics(Y_TRUE, Y_PRED)
    assert m["accuracy"] == pytest.approx(9 / 12)
    assert m["fpr"] == pytest.approx(1 / 5)
    assert m["recall_attack"] == pytest.approx(6 / 7)
    assert m["f1_macro"] == pytest.approx(sum(F1_EACH) / 5)
    assert list(m["f1_per_class"].values()) == pytest.approx(F1_EACH)
    assert list(m["f1_per_class"]) == ev.CLASSES


def test_f1_macro_on_4_main_classes():
    m = ev.compute_metrics(Y_TRUE, Y_PRED, macro_labels=[0, 1, 2, 3])
    assert m["f1_macro"] == pytest.approx(sum(F1_EACH[:4]) / 4)


def test_dummy_classifier_most_frequent():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 18))
    y = np.array([0] * 70 + [1] * 20 + [3] * 10)
    y_pred = DummyClassifier(strategy="most_frequent").fit(X, y).predict(X)
    m = ev.compute_metrics(y, y_pred)
    # Luôn đoán Normal: đúng 70/100, không báo nhầm Normal nào, không bắt được tấn công nào
    assert m["accuracy"] == pytest.approx(0.70)
    assert m["fpr"] == 0.0
    assert m["recall_attack"] == 0.0
    assert m["f1_per_class"]["Normal"] == pytest.approx(2 * 0.7 / 1.7)


def test_fpr_is_binary_normal_negative():
    # Nhầm giữa hai lớp tấn công không phải là dương tính giả
    m = ev.compute_metrics([0, 0, 1, 2], [0, 0, 2, 1])
    assert m["accuracy"] == 0.5
    assert m["fpr"] == 0.0
    assert m["recall_attack"] == 1.0
    # Không có Normal thì FPR không xác định
    assert math.isnan(ev.compute_metrics([1, 2], [1, 2])["fpr"])


def test_roc_auc_binary_from_proba():
    y = [0, 0, 1, 2]
    proba = np.array([[0.9, 0.1, 0, 0, 0], [0.6, 0.4, 0, 0, 0], [0.3, 0.7, 0, 0, 0], [0.1, 0, 0.9, 0, 0]])
    assert ev.compute_metrics(y, [0, 0, 1, 2], y_score=proba)["roc_auc"] == 1.0


def _log(path, **kw):
    m = ev.compute_metrics(Y_TRUE, Y_PRED)
    args = dict(dataset="toy", model="dummy", task="multiclass", seed=42, split="val", n_params=0,
                size_kb=0.0, train_time_s=0.0, infer_ms_per_window=0.0, commit="test", path=path)
    args.update(kw)
    return ev.log_result(m, **args)


def test_results_csv_has_exactly_20_columns(tmp_path):
    p = tmp_path / "results.csv"
    _log(p)
    _log(p, seed=43)
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ev.RESULT_COLUMNS
    assert len(ev.RESULT_COLUMNS) == 20
    assert len(rows) == 3 and all(len(r) == 20 for r in rows)


def test_log_result_rejects_bad_header_and_values(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text("dataset,model\n", encoding="utf-8")
    with pytest.raises(ValueError, match="20 cột"):
        _log(p)
    with pytest.raises(ValueError):
        _log(tmp_path / "ok.csv", split="validation")


def test_measure_latency_warmup_and_runs():
    calls = []
    r = ev.measure_latency(lambda: calls.append(1))
    assert len(calls) == 20 + 200
    assert {"mean_ms", "p50_ms", "p95_ms", "p99_ms"} <= set(r)
    assert r["p50_ms"] <= r["p95_ms"] <= r["p99_ms"]
    with pytest.raises(ValueError):
        ev.measure_latency(lambda: None, n_runs=100)


def test_count_params_and_size():
    torch = pytest.importorskip("torch")
    lin = torch.nn.Linear(18, 5)
    assert ev.count_params(lin) == 18 * 5 + 5
    assert ev.model_size_kb(lin) > 0
    from sklearn.linear_model import LogisticRegression
    X = np.random.default_rng(0).normal(size=(30, 4))
    y = np.array([0, 1, 2] * 10)
    assert ev.count_params(LogisticRegression(max_iter=200).fit(X, y)) == 3 * 4 + 3


def test_plots_written(tmp_path):
    proba = np.eye(5)[Y_PRED] * 0.8 + 0.04
    p1 = ev.plot_confusion(Y_TRUE, Y_PRED, tmp_path / "gd0_confusion.png")
    p2 = ev.plot_roc_pr(Y_TRUE, proba, tmp_path / "gd0_roc_pr.png")
    assert p1.stat().st_size > 0 and p2.stat().st_size > 0


def test_count_params_boosting_and_mlp():
    from lightgbm import LGBMClassifier
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = np.array([0, 1, 2] * 40)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=30, random_state=0).fit(X, y)
    assert ev.count_params(mlp) == 4 * 8 + 8 * 3 + 8 + 3
    assert ev.count_params(XGBClassifier(n_estimators=5, max_depth=2).fit(X, y)) > 0
    assert ev.count_params(LGBMClassifier(n_estimators=5, num_leaves=4, verbose=-1).fit(X, y)) > 0


def test_macro_ignores_class_absent_from_y_true():
    # lớp 2 không có mẫu thật: macro chỉ tính trên lớp 0, 1, 3
    y_true = [0, 0, 1, 1, 3, 3]
    y_pred = [0, 0, 1, 0, 3, 3]
    m = ev.compute_metrics(y_true, y_pred, macro_labels=[0, 1, 2, 3])
    f = m["f1_per_class"]
    assert m["f1_macro"] == pytest.approx((f["Normal"] + f["DoS_DDoS"] + f["Truy_cap_trai_phep"]) / 3)
    assert f["Botnet_Backdoor"] == 0.0
