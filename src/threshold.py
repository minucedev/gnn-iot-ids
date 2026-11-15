"""Hiệu chỉnh ngưỡng quyết định trên val (task 5.8, GĐ G).

Điểm tấn công của một node là `1 - P(Normal)` (cùng định nghĩa với `evaluate.roc_auc`). Luật dự đoán sau hiệu chỉnh:
node bị coi là tấn công khi điểm > ngưỡng `t`, và khi đó lớp dự đoán là lớp tấn công có xác suất cao nhất (cột 1..K-1);
ngược lại là Normal. Ngưỡng chọn trên val là ngưỡng NHỎ NHẤT (tức recall tấn công lớn nhất) mà FPR trên val ≤ `limit`
(mặc định 4,08%, ngưỡng của đề tài). Ngưỡng được giữ nguyên khi chạy test (GĐ H); hàm ở đây không bao giờ đọc test.
"""

import math

import numpy as np

from src import evaluate as ev

LIMIT = 0.0408          # FPR nhị phân tối đa (KE_HOACH mục 3)
RECALL_TARGET = 0.95    # recall tấn công tối thiểu (KE_HOACH mục 3), chỉ để ghi cờ


def attack_score(proba):
    """Điểm tấn công 1 - P(Normal) (float64), từ xác suất (n, K) có Normal là cột 0."""
    return 1.0 - np.asarray(proba, dtype=np.float64)[:, ev.NORMAL]


def apply_threshold(proba, t):
    """Dự đoán lớp (mã số nguyên) theo ngưỡng: điểm > t thì lớp tấn công có xác suất cao nhất, ngược lại Normal (0)."""
    proba = np.asarray(proba, dtype=np.float64)
    attack = attack_score(proba) > t
    return np.where(attack, 1 + proba[:, 1:].argmax(1), ev.NORMAL)


def choose_threshold(y_true, proba, limit=LIMIT):
    """Ngưỡng nhỏ nhất mà số FP (node Normal có điểm > t) ≤ floor(limit · số node Normal). Trả về float.

    Với N node Normal được sắp theo điểm giảm dần, cho phép tối đa a = floor(limit · N) FP, nên t = điểm của node Normal
    thứ a + 1 (chỉ số a): chỉ các node có điểm > t bị báo, tối đa a node. t nhỏ hơn thì ít nhất a + 1 node Normal bị báo.
    """
    score = attack_score(proba)
    neg = np.sort(score[np.asarray(y_true) == ev.NORMAL])[::-1]
    if len(neg) == 0:
        raise ValueError("Val không có node Normal, không hiệu chỉnh được ngưỡng")
    allowed = int(math.floor(limit * len(neg) + 1e-9))
    return float(neg[allowed]) if allowed < len(neg) else float("-inf")


def _summary(y_true, pred, proba, classes, macro_labels):
    m = ev.compute_metrics(y_true, pred, proba, class_names=classes, macro_labels=macro_labels)
    return {k: float(m[k]) for k in ("accuracy", "f1_macro", "fpr", "recall_attack")}


def calibrate(y_true, proba, classes, macro_labels=(0, 1, 2, 3), limit=LIMIT):
    """Chọn ngưỡng trên val và ghi chỉ số trước (argmax) và sau (theo ngưỡng).

    `feasible` = sau hiệu chỉnh FPR ≤ limit VÀ vẫn còn phát hiện được tấn công (recall > 0). `feasible = False` nghĩa là
    không có ngưỡng nào vừa giữ FPR ≤ limit vừa cho recall dương (ví dụ nhiều node Normal có điểm tối đa bằng nhau);
    khi đó ghi rõ, không ép số. `meets_recall_target` = recall sau ≥ 95%.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    macro_labels = list(macro_labels)
    before = _summary(y_true, proba.argmax(1), proba, classes, macro_labels)
    t = choose_threshold(y_true, proba, limit)
    after = _summary(y_true, apply_threshold(proba, t), proba, classes, macro_labels)
    return {
        "threshold": t, "limit": limit,
        "feasible": bool(after["fpr"] <= limit + 1e-12 and after["recall_attack"] > 0),
        "meets_recall_target": bool(after["recall_attack"] >= RECALL_TARGET),
        "n_val_normal": int((y_true == ev.NORMAL).sum()), "n_val_attack": int((y_true != ev.NORMAL).sum()),
        "before": before, "after": after,
    }
