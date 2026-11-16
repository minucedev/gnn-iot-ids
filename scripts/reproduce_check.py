"""Kiểm tra tái lập (GĐ J, task 9.3): so các bảng do `scripts/aggregate_h.py` sinh ra ở một lần chạy mới với bản trong repo.

Chạy:  python -I scripts/reproduce_check.py <thư mục bảng mới> <thư mục bảng trong repo> <ghi_json> [--tol 0.005]
So:    results.csv (khóa dataset, model, seed, split), summary_test.csv, summary_val.csv (khóa dataset, base, mode),
       summary_test_tiers.csv (khóa dataset, base, mode, tier). Mọi cột số; chỉ tiêu chính là lệch F1-macro ≤ `tol` (0,005 = 0,5 điểm %).
Đạt khi: cùng tập dòng ở cả bốn bảng và max |lệch F1-macro| ≤ tol ở results.csv và mọi cột *f1_macro_mean của các bảng tổng hợp.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = {"results.csv": ["dataset", "model", "seed", "split"], "summary_test.csv": ["dataset", "base", "mode"],
          "summary_val.csv": ["dataset", "base", "mode"], "summary_test_tiers.csv": ["dataset", "base", "mode", "tier"]}


def compare_table(new, ref, keys, tol):
    """So hai bảng theo `keys`. Trả về dict: số dòng, dòng chỉ có ở một bên, lệch tuyệt đối lớn nhất của từng cột số, lệch F1-macro lớn nhất."""
    new = new.set_index(keys).sort_index()
    ref = ref.set_index(keys).sort_index()
    only_new, only_ref = new.index.difference(ref.index), ref.index.difference(new.index)
    both = new.index.intersection(ref.index)
    num = [c for c in ref.columns if c in new.columns and pd.api.types.is_numeric_dtype(ref[c]) and pd.api.types.is_numeric_dtype(new[c])]
    diff = {}
    for c in num:
        a, b = new.loc[both, c].astype(float), ref.loc[both, c].astype(float)
        if (a.isna() != b.isna()).any():
            diff[c] = float("inf")                                              # một bên có số, một bên trống
            continue
        d = (a - b).abs().dropna()                                              # cả hai trống thì coi là không lệch
        diff[c] = float(d.max()) if len(d) else 0.0
    f1_cols = [c for c in diff if "f1_macro" in c and not c.endswith("_std") and not c.endswith("_fmt")]
    f1_max = max((diff[c] for c in f1_cols), default=0.0)
    worst = None
    if f1_cols:
        c = max(f1_cols, key=lambda c: diff[c])
        d = (new.loc[both, c].astype(float) - ref.loc[both, c].astype(float)).abs()
        worst = {"column": c, "row": [str(x) for x in (d.idxmax() if isinstance(d.idxmax(), tuple) else (d.idxmax(),))], "abs_diff": diff[c]}
    return {"rows_new": int(len(new)), "rows_ref": int(len(ref)), "rows_only_in_new": [list(map(str, i)) for i in only_new][:20],
            "rows_only_in_ref": [list(map(str, i)) for i in only_ref][:20], "max_abs_diff_by_column": diff, "max_abs_diff_f1_macro": f1_max,
            "worst_f1_macro": worst, "same_rows": len(only_new) == 0 and len(only_ref) == 0, "f1_within_tol": bool(f1_max <= tol)}


def reproduce_check(new_dir, ref_dir, tol=0.005):
    out = {"tolerance_f1_macro": tol, "tables": {}}
    for fn, keys in TABLES.items():
        out["tables"][fn] = compare_table(pd.read_csv(Path(new_dir) / fn), pd.read_csv(Path(ref_dir) / fn), keys, tol)
    t = out["tables"]
    out["max_abs_diff_f1_macro"] = max(v["max_abs_diff_f1_macro"] for v in t.values())
    out["max_abs_diff_any_numeric_column"] = max(max(v["max_abs_diff_by_column"].values(), default=0.0) for v in t.values())
    out["passed"] = bool(all(v["same_rows"] and v["f1_within_tol"] for v in t.values()))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tol = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 0.005
    if "--tol" in sys.argv:
        args.remove(sys.argv[sys.argv.index("--tol") + 1])
    new_dir, ref_dir, out = args[:3]
    res = reproduce_check(new_dir, ref_dir, tol)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for fn, v in res["tables"].items():
        print(f"{fn}: {v['rows_new']} dòng (bản repo {v['rows_ref']}), cùng tập dòng: {v['same_rows']}, lệch F1-macro lớn nhất {v['max_abs_diff_f1_macro']:.2e}")
    print(f"Lệch F1-macro lớn nhất {res['max_abs_diff_f1_macro']:.2e} (ngưỡng {tol}); lệch lớn nhất mọi cột số {res['max_abs_diff_any_numeric_column']:.2e}; "
          f"{'ĐẠT' if res['passed'] else 'KHÔNG ĐẠT'}")
    return 0 if res["passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
