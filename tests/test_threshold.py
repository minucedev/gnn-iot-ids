import numpy as np
import pytest

from src import threshold as TH

CLASSES = ["Normal", "DoS_DDoS", "Truy_cap_trai_phep"]


def _data(n_normal=1000, n_attack=400, seed=0):
    """Xác suất 3 lớp giả: node Normal có điểm tấn công thấp, node tấn công cao, hai nhóm chồng nhau một phần."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n_normal, int), rng.integers(1, 3, n_attack)]
    s = np.r_[rng.beta(1, 6, n_normal), rng.beta(5, 2, n_attack)]               # điểm tấn công
    cls = np.where(y == 1, 0.8, 0.2)
    proba = np.c_[1 - s, s * cls, s * (1 - cls)]
    return y, proba / proba.sum(1, keepdims=True)


def test_threshold_gives_fpr_at_most_limit_and_is_minimal():
    y, p = _data()
    t = TH.choose_threshold(y, p, 0.0408)
    score = TH.attack_score(p)
    n_neg = int((y == 0).sum())
    fp = int(((score > t) & (y == 0)).sum())
    assert fp <= int(0.0408 * n_neg) and fp / n_neg <= 0.0408
    lower = np.nextafter(t, -np.inf)                                            # ngưỡng nhỏ hơn một chút thì vượt giới hạn
    assert int(((score > lower) & (y == 0)).sum()) > int(0.0408 * n_neg)


def test_apply_threshold_rule_and_class_choice():
    p = np.array([[0.9, 0.05, 0.05], [0.4, 0.5, 0.1], [0.4, 0.1, 0.5], [0.6, 0.1, 0.3]])
    # điểm tấn công = 0.1, 0.6, 0.6, 0.4
    np.testing.assert_array_equal(TH.apply_threshold(p, 0.5), [0, 1, 2, 0])
    np.testing.assert_array_equal(TH.apply_threshold(p, 0.35), [0, 1, 2, 2])    # điểm 0.4 > 0.35: lớp tấn công cao nhất
    np.testing.assert_array_equal(TH.apply_threshold(p, 0.0), [1, 1, 2, 2])


def test_calibrate_reports_before_after_and_flags():
    y, p = _data()
    cal = TH.calibrate(y, p, CLASSES, macro_labels=[0, 1, 2], limit=0.0408)
    assert cal["after"]["fpr"] <= 0.0408 and cal["feasible"] is True
    assert cal["before"]["fpr"] == pytest.approx(((p.argmax(1) != 0) & (y == 0)).sum() / (y == 0).sum())
    assert cal["n_val_normal"] == 1000 and cal["n_val_attack"] == 400
    assert cal["meets_recall_target"] == (cal["after"]["recall_attack"] >= 0.95)
    # nới giới hạn thì ngưỡng giảm, recall tăng
    loose = TH.calibrate(y, p, CLASSES, macro_labels=[0, 1, 2], limit=0.2)
    assert loose["threshold"] < cal["threshold"] and loose["after"]["recall_attack"] >= cal["after"]["recall_attack"]


def test_calibrate_infeasible_when_all_normal_scores_are_tied_at_the_top():
    """Mọi node Normal và tấn công đều có điểm 1: không ngưỡng nào giữ FPR ≤ giới hạn mà còn bắt được tấn công."""
    y = np.r_[np.zeros(100, int), np.ones(20, int)]
    p = np.tile([0.0, 1.0, 0.0], (120, 1))
    cal = TH.calibrate(y, p, CLASSES, macro_labels=[0, 1, 2])
    assert cal["feasible"] is False and cal["after"]["recall_attack"] == 0.0


def test_no_normal_nodes_raises():
    with pytest.raises(ValueError):
        TH.choose_threshold(np.ones(5, int), np.tile([0.1, 0.9, 0.0], (5, 1)))
