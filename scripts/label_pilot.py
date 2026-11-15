"""Thí nghiệm nhỏ: chỉ gán nhãn và đánh giá node CÓ khởi tạo flow (n_out > 0), bỏ node chỉ nhận (đề xuất từ phân tích FPR, 5.3).

Chạy:  python -I scripts/label_pilot.py <data_dir> <dataset> <baseline_search.json> <out.json>

Không đổi dữ liệu và không dựng lại đồ thị: node chỉ nhận vẫn nằm trong đồ thị (làm ngữ cảnh), chỉ bị loại khỏi huấn luyện
và đánh giá bằng mặt nạ `initiator_mask` (suy từ đặc trưng n_out). So trên cùng tập node val có khởi tạo:
- "cũ": LightGBM huấn luyện trên mọi node train (nhãn hiện tại), đánh giá trên node val có khởi tạo;
- "mới": LightGBM huấn luyện chỉ trên node train có khởi tạo, đánh giá trên node val có khởi tạo.
LightGBM cấu hình tốt nhất của baseline_search, seed 42. Chỉ train và val, không đọc test. Chỉ là thí nghiệm: kết quả này
chưa phải kết quả của mô hình nào trong báo cáo.
"""

import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import fpr as F  # noqa: E402
from src import train as T  # noqa: E402


def best_cfg(search_json):
    log = json.loads(Path(search_json).read_text(encoding="utf-8"))["lightgbm"]
    b = next(e["best"] for e in log if "best" in e)
    return int(b["n"]), int(b["l"])


def fit(X, y, K, n, leaves):
    w = D.class_weights(y, K)
    return LGBMClassifier(n_estimators=n, num_leaves=leaves, learning_rate=0.1, n_jobs=-1, random_state=42,
                          verbose=-1).fit(X, y, sample_weight=w[y])


def metrics(y, proba, classes):
    m = ev.compute_metrics(y, proba.argmax(1), proba, class_names=classes, macro_labels=T.MACRO_CODES)
    return {k: float(m[k]) for k in ("accuracy", "f1_macro", "fpr", "recall_attack", "roc_auc")}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data_dir, ds, search_json, out = sys.argv[1:5]
    graphs, meta = D.load_split(data_dir)
    classes, K = meta["classes"], len(meta["classes"])
    Xtr, ytr = D.node_arrays(graphs["train"])
    Xva, yva = D.node_arrays(graphs["val"])
    zero = float(Xtr[:, 0].min())
    mtr, mva = F.initiator_mask(Xtr, zero), F.initiator_mask(Xva, zero)
    n, leaves = best_cfg(search_json)
    res = {"dataset": ds, "classes": classes, "lightgbm_cfg": {"n_estimators": n, "num_leaves": leaves},
           "n_train": int(len(ytr)), "n_val": int(len(yva)),
           "train_initiator_share": float(mtr.mean()), "val_initiator_share": float(mva.mean()),
           "n_val_non_initiator": int((~mva).sum()),
           "class_counts_val_all": {c: int((yva == k).sum()) for k, c in enumerate(classes)},
           "class_counts_val_initiators": {c: int((yva[mva] == k).sum()) for k, c in enumerate(classes)}}
    print(f"{ds}: train {len(ytr):,} ({mtr.mean():.1%} khởi tạo), val {len(yva):,} ({mva.mean():.1%} khởi tạo, "
          f"{int((~mva).sum()):,} node chỉ nhận)")

    old = fit(Xtr, ytr, K, n, leaves)
    p_old = old.predict_proba(Xva)
    res["old_trained_all_eval_all_val"] = metrics(yva, p_old, classes)
    res["old_trained_all_eval_initiator_val"] = metrics(yva[mva], p_old[mva], classes)
    new = fit(Xtr[mtr], ytr[mtr], K, n, leaves)
    p_new = new.predict_proba(Xva[mva])
    res["new_trained_initiator_eval_initiator_val"] = metrics(yva[mva], p_new, classes)
    res["new_classes_in_train"] = [classes[int(c)] for c in new.classes_]
    for k in ("old_trained_all_eval_all_val", "old_trained_all_eval_initiator_val", "new_trained_initiator_eval_initiator_val"):
        m = res[k]
        print(f"  {k:44s} F1-macro {m['f1_macro']:.4f} FPR {m['fpr']:.4f} recall {m['recall_attack']:.4f} AUC {m['roc_auc']:.4f}")
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
