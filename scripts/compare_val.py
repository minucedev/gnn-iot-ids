"""So sánh trên val: mean ± std (3 seed) của F1-macro, FPR, recall tấn công cho mọi mô hình, và hybrid có vượt mô hình
tốt nhất (không tính hybrid) hơn 2 lần std hay không (task 5.7, yêu cầu "minh chứng hybrid cao hơn baseline").

Chạy:  python -I scripts/compare_val.py <dataset> <compare_val.csv> <results1.csv> [<results2.csv> ...]
Đọc các results.csv (chỉ dòng split=val, task=multiclass, đúng bộ) của baseline, GNN và hybrid; ghi một dòng mỗi mô hình.

Quy ước: std lấy ddof = 1 trên các seed. Mô hình tốt nhất tính riêng cho từng chỉ số (F1 cao nhất, FPR thấp nhất,
recall cao nhất), chỉ trong nhóm không phải hybrid. "Vượt hơn 2 std" nghĩa là hybrid tốt hơn mô hình đó một khoảng
> 2 × max(std hybrid, std mô hình đó); max (thay vì trung bình) để không tuyên bố vượt khi một bên dao động rộng.
Mô hình cây không ngẫu nhiên theo seed có std ≈ 0, khi đó chỉ std của hybrid hạn chế tuyên bố. Chỉ 3 seed nên std thô.
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np

HYBRID = ("hybrid_gat_lgbm", "hybrid_gat_lgbm_nooof")
METRICS = {"f1_macro": +1, "fpr": -1, "recall_attack": +1}      # +1: càng cao càng tốt, -1: càng thấp càng tốt
COLUMNS = ["dataset", "model", "is_hybrid", "n_seeds",
           *[f"{m}_{s}" for m in METRICS for s in ("mean", "std")],
           *[f"{m}_{s}" for m in METRICS for s in ("best_model", "best_mean", "diff_vs_best", "diff_over_std", "beats_2std")]]


def read_rows(paths, dataset):
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            rows += [r for r in csv.DictReader(f)
                     if r["dataset"] == dataset and r["split"] == "val" and r["task"] == "multiclass"]
    return rows


def summarize(rows):
    """model -> {metric: (mean, std ddof=1, số seed)}; mỗi (model, seed) phải đúng một dòng."""
    by = {}
    for r in rows:
        by.setdefault(r["model"], {}).setdefault(int(r["seed"]), []).append(r)
    out = {}
    for model, seeds in by.items():
        if any(len(v) != 1 for v in seeds.values()):
            raise ValueError(f"{model}: có seed trùng dòng {[s for s, v in seeds.items() if len(v) != 1]}")
        out[model] = {}
        for m in METRICS:
            v = np.array([float(seeds[s][0][m]) for s in sorted(seeds)])
            out[model][m] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v))
    return out


def compare(rows, dataset):
    summ = summarize(rows)
    others = {k: v for k, v in summ.items() if k not in HYBRID}
    if not others:
        raise ValueError("Không có mô hình nào ngoài hybrid để so sánh")
    best = {m: (max if sign > 0 else min)(others, key=lambda k: others[k][m][0]) for m, sign in METRICS.items()}
    out = []
    for model in sorted(summ, key=lambda k: (k in HYBRID, k)):
        s = summ[model]
        row = {"dataset": dataset, "model": model, "is_hybrid": model in HYBRID,
               "n_seeds": s["f1_macro"][2]}
        for m, sign in METRICS.items():
            row[f"{m}_mean"], row[f"{m}_std"] = s[m][0], s[m][1]
            if model in HYBRID:
                b = best[m]
                diff = sign * (s[m][0] - others[b][m][0])               # > 0: hybrid tốt hơn
                spread = max(s[m][1], others[b][m][1])
                row[f"{m}_best_model"], row[f"{m}_best_mean"] = b, others[b][m][0]
                row[f"{m}_diff_vs_best"] = diff
                row[f"{m}_diff_over_std"] = diff / spread if spread > 0 else (math.inf if diff > 0 else None)
                row[f"{m}_beats_2std"] = bool(diff > 2 * spread)
        out.append(row)
    return out


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8")
    dataset, out_csv, *paths = argv
    rows = compare(read_rows(paths, dataset), dataset)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{dataset}: {len(rows)} mô hình → {out_csv}")
    for r in rows:
        line = (f"  {r['model']:22s} F1 {r['f1_macro_mean']:.4f}±{r['f1_macro_std']:.4f}  FPR {r['fpr_mean']:.4f}±{r['fpr_std']:.4f}"
                f"  recall {r['recall_attack_mean']:.4f}±{r['recall_attack_std']:.4f}")
        if r["is_hybrid"]:
            line += "  | vượt >2std: " + ", ".join(f"{m}={r[f'{m}_beats_2std']}" for m in METRICS)
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
