"""Kiểm thử scripts/reproduce_check.py (GĐ J): so bảng mới với bản repo, có đáp án tính tay."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RC = _load("reproduce_check")


def write_all(d, f1_shift=0.0, drop_row=False, nan_cell=False):
    d.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame({"dataset": ["ds"] * 3, "model": ["a", "a", "b"], "seed": [42, 43, 42], "split": ["test"] * 3,
                        "f1_macro": [0.9, 0.8, 0.7], "fpr": [0.1, 0.2, 0.3], "commit": ["x", "x", "x"]})
    res.loc[0, "f1_macro"] += f1_shift
    if nan_cell:
        res.loc[1, "fpr"] = float("nan")
    if drop_row:
        res = res.iloc[:2]
    res.to_csv(d / "results.csv", index=False)
    for fn in ("summary_test.csv", "summary_val.csv"):
        pd.DataFrame({"dataset": ["ds", "ds"], "base": ["a", "b"], "mode": ["thr", "thr"], "f1_macro_mean": [0.85 + f1_shift / 2, 0.7],
                      "f1_macro_std": [0.01, 0.01], "f1_macro_fmt": ["x", "y"]}).to_csv(d / fn, index=False)
    pd.DataFrame({"dataset": ["ds"], "base": ["a"], "mode": ["thr"], "tier": ["all"], "f1_macro_mean": [0.5]}).to_csv(d / "summary_test_tiers.csv", index=False)


def test_identical_tables_pass(tmp_path):
    write_all(tmp_path / "new")
    write_all(tmp_path / "ref")
    r = RC.reproduce_check(tmp_path / "new", tmp_path / "ref")
    assert r["passed"] and r["max_abs_diff_f1_macro"] == 0.0 and all(t["same_rows"] for t in r["tables"].values())


def test_f1_shift_boundary(tmp_path):
    write_all(tmp_path / "ref")
    write_all(tmp_path / "new", f1_shift=0.004)                         # 0,4 điểm % ≤ 0,5 điểm %: đạt
    r = RC.reproduce_check(tmp_path / "new", tmp_path / "ref")
    assert r["passed"] and abs(r["tables"]["results.csv"]["max_abs_diff_f1_macro"] - 0.004) < 1e-12
    assert r["tables"]["results.csv"]["worst_f1_macro"]["row"] == ["ds", "a", "42", "test"]
    write_all(tmp_path / "new2", f1_shift=0.006)                        # 0,6 điểm % > 0,5: không đạt
    assert not RC.reproduce_check(tmp_path / "new2", tmp_path / "ref")["passed"]


def test_missing_row_and_nan_mismatch_fail(tmp_path):
    write_all(tmp_path / "ref")
    write_all(tmp_path / "new", drop_row=True)
    r = RC.reproduce_check(tmp_path / "new", tmp_path / "ref")
    assert not r["passed"] and r["tables"]["results.csv"]["rows_only_in_ref"] == [["ds", "b", "42", "test"]]
    write_all(tmp_path / "new2", nan_cell=True)
    r2 = RC.reproduce_check(tmp_path / "new2", tmp_path / "ref")
    assert r2["tables"]["results.csv"]["max_abs_diff_by_column"]["fpr"] == float("inf")                 # một bên trống, một bên có số
