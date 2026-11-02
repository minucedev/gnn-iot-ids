"""Module đánh giá chung (task 0.5). Mọi kết quả ghi vào reports/tables/results.csv qua log_result().

Quy ước (docs/gnn_ids/KE_HOACH.md mục 3 và 6):
- Nhãn là số nguyên, Normal luôn là lớp 0 (giống pipeline Tuần 1 của Phong).
- FPR và recall tấn công luôn tính trên bài toán nhị phân: Normal là lớp âm, mọi lớp tấn công gộp là lớp dương.
- roc_auc là AUC nhị phân (tấn công so với Normal), điểm tấn công = 1 - P(Normal).
- results.csv có đúng 20 cột RESULT_COLUMNS.
"""

import csv
import json
import math
import pickle
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "reports" / "tables" / "results.csv"

CLASSES = ["Normal", "DoS_DDoS", "Botnet_Backdoor", "Truy_cap_trai_phep", "Khac"]
NORMAL = 0

RESULT_COLUMNS = [
    "dataset", "model", "task", "seed", "split",
    "accuracy", "balanced_acc", "precision_macro", "recall_macro", "f1_macro", "f1_per_class",
    "fpr", "recall_attack", "roc_auc",
    "n_params", "size_kb", "train_time_s", "infer_ms_per_window",
    "author", "commit",
]
TASKS = {"multiclass", "binary"}
SPLITS = {"train", "val", "test"}


def _attack_score(y_score):
    """Điểm tấn công từ xác suất: mảng 1 chiều giữ nguyên, mảng (n, K) lấy 1 - P(Normal)."""
    s = np.asarray(y_score, dtype=float)
    return s if s.ndim == 1 else 1.0 - s[:, NORMAL]


def compute_metrics(y_true, y_pred, y_score=None, class_names=CLASSES, macro_labels=None):
    """Tính các chỉ số của một lần đánh giá.

    y_true, y_pred: nhãn số nguyên (Normal = 0). y_score: xác suất (n, K) hoặc điểm tấn công (n,), tùy chọn.
    class_names: tên lớp theo thứ tự mã 0..K-1. macro_labels: các mã lớp dùng cho chỉ số macro
    (mặc định mọi lớp; ví dụ [0, 1, 2, 3] để bỏ lớp Khac theo D5).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))
    macro = labels if macro_labels is None else list(macro_labels)

    attack_true = y_true != NORMAL
    attack_pred = y_pred != NORMAL
    tp = int(np.sum(attack_true & attack_pred))
    fn = int(np.sum(attack_true & ~attack_pred))
    fp = int(np.sum(~attack_true & attack_pred))
    tn = int(np.sum(~attack_true & ~attack_pred))

    f1_each = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    roc_auc = math.nan
    if y_score is not None and attack_true.any() and (~attack_true).any():
        roc_auc = float(roc_auc_score(attack_true, _attack_score(y_score)))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=macro, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=macro, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=macro, average="macro", zero_division=0)),
        "f1_per_class": {class_names[i]: float(f1_each[i]) for i in labels},
        "fpr": fp / (fp + tn) if fp + tn else math.nan,
        "recall_attack": tp / (tp + fn) if tp + fn else math.nan,
        "roc_auc": roc_auc,
    }


def git_commit():
    """Hash commit hiện tại (thêm "-dirty" nếu còn thay đổi chưa commit)."""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        sha = run("rev-parse", "--short", "HEAD").strip()
        return sha + ("-dirty" if run("status", "--porcelain", "--untracked-files=no").strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_result(metrics, *, dataset, model, task, seed, split, n_params, size_kb,
               train_time_s, infer_ms_per_window, author="Phuc", commit=None, path=RESULTS_CSV):
    """Ghi 1 dòng vào results.csv (tạo file và header nếu chưa có). Trả về dòng đã ghi."""
    if task not in TASKS:
        raise ValueError(f"task phải thuộc {sorted(TASKS)}, nhận {task!r}")
    if split not in SPLITS:
        raise ValueError(f"split phải thuộc {sorted(SPLITS)}, nhận {split!r}")
    row = {
        "dataset": dataset, "model": model, "task": task, "seed": seed, "split": split,
        **{k: metrics[k] for k in ("accuracy", "balanced_acc", "precision_macro", "recall_macro", "f1_macro")},
        "f1_per_class": json.dumps(metrics["f1_per_class"], ensure_ascii=False),
        **{k: metrics[k] for k in ("fpr", "recall_attack", "roc_auc")},
        "n_params": n_params, "size_kb": size_kb, "train_time_s": train_time_s,
        "infer_ms_per_window": infer_ms_per_window,
        "author": author, "commit": commit or git_commit(),
    }
    assert list(row) == RESULT_COLUMNS

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        if header != RESULT_COLUMNS:
            raise ValueError(f"{path} có header khác quy ước 20 cột: {header}")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def count_params(model):
    """Số tham số: torch.nn.Module (tổng numel), sklearn tuyến tính (coef_ + intercept_),
    sklearn cây/rừng (tổng số node)."""
    if hasattr(model, "parameters") and callable(model.parameters):
        return int(sum(p.numel() for p in model.parameters()))
    if hasattr(model, "coef_"):
        return int(np.size(model.coef_) + np.size(getattr(model, "intercept_", 0)))
    if hasattr(model, "tree_"):
        return int(model.tree_.node_count)
    if hasattr(model, "estimators_"):
        ests = np.ravel(model.estimators_)
        return int(sum(e.tree_.node_count for e in ests))
    raise TypeError(f"Chưa hỗ trợ đếm tham số cho {type(model).__name__}")


def model_size_kb(model):
    """Kích thước file mô hình (KB): torch lưu state_dict bằng torch.save, mô hình khác dùng pickle."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "model.bin"
        if hasattr(model, "state_dict"):
            import torch
            torch.save(model.state_dict(), p)
        else:
            with open(p, "wb") as f:
                pickle.dump(model, f)
        return p.stat().st_size / 1024


def measure_latency(fn, n_warmup=20, n_runs=200, cuda_sync=False):
    """Đo độ trễ của fn(): warm-up n_warmup lần, đo n_runs lần (≥ 200), trả về ms.

    cuda_sync=True gọi torch.cuda.synchronize() sau mỗi lần để đo đúng thời gian trên GPU.
    """
    if n_runs < 200:
        raise ValueError("Quy ước đo hiệu năng: n_runs ≥ 200")
    sync = (lambda: __import__("torch").cuda.synchronize()) if cuda_sync else (lambda: None)
    for _ in range(n_warmup):
        fn()
    sync()
    times = np.empty(n_runs)
    for i in range(n_runs):
        t0 = time.perf_counter()
        fn()
        sync()
        times[i] = (time.perf_counter() - t0) * 1000
    return {
        "mean_ms": float(times.mean()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "n_warmup": n_warmup,
        "n_runs": n_runs,
    }


def plot_confusion(y_true, y_pred, path, class_names=CLASSES, normalize="true", title="Ma trận nhầm lẫn"):
    """Lưu confusion matrix (mặc định chuẩn hóa theo hàng = recall từng lớp) ra PNG dpi 150."""
    import matplotlib.pyplot as plt

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 3, 1.2 * len(labels) + 2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, label="tỉ lệ" if normalize else "số node")
    for i in labels:
        for j in labels:
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks(labels, class_names, rotation=30, ha="right")
    ax.set_yticks(labels, class_names)
    ax.set_xlabel("Lớp dự đoán")
    ax.set_ylabel("Lớp thật")
    ax.set_title(title)
    return _save(fig, path)


def plot_roc_pr(y_true, y_score, path, title="ROC và PR (tấn công so với Normal)"):
    """Lưu đường ROC và Precision-Recall của bài toán nhị phân ra PNG dpi 150."""
    import matplotlib.pyplot as plt

    attack = np.asarray(y_true) != NORMAL
    score = _attack_score(y_score)
    fpr, tpr, _ = roc_curve(attack, score)
    prec, rec, _ = precision_recall_curve(attack, score)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.5))
    a1.plot(fpr, tpr, label=f"AUC = {roc_auc_score(attack, score):.4f}")
    a1.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    a1.set(xlabel="FPR (tỉ lệ)", ylabel="Recall tấn công (tỉ lệ)", title="ROC")
    a2.plot(rec, prec, label=f"AP = {average_precision_score(attack, score):.4f}")
    a2.set(xlabel="Recall tấn công (tỉ lệ)", ylabel="Precision (tỉ lệ)", title="Precision-Recall")
    for a in (a1, a2):
        a.legend(loc="lower right")
        a.grid(alpha=0.3)
    fig.suptitle(title)
    return _save(fig, path)


def _save(fig, path):
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
