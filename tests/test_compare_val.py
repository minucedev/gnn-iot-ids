import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_val", ROOT / "scripts" / "compare_val.py")
CV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CV)


def _row(model, seed, f1, fpr, rec, ds="d", split="val", task="multiclass"):
    return {"dataset": ds, "model": model, "seed": str(seed), "split": split, "task": task,
            "f1_macro": str(f1), "fpr": str(fpr), "recall_attack": str(rec)}


def _rows(model, f1s, fprs, recs):
    return [_row(model, 42 + i, f, p, r) for i, (f, p, r) in enumerate(zip(f1s, fprs, recs))]


def test_summarize_mean_and_sample_std():
    s = CV.summarize(_rows("m", [0.9, 0.92, 0.94], [0.1] * 3, [0.95] * 3))
    assert s["m"]["f1_macro"][0] == pytest.approx(0.92) and s["m"]["f1_macro"][1] == pytest.approx(0.02)
    assert s["m"]["fpr"][1] == 0.0 and s["m"]["f1_macro"][2] == 3


def test_duplicate_seed_rows_are_rejected():
    with pytest.raises(ValueError):
        CV.summarize(_rows("m", [0.9, 0.9], [0.1] * 2, [0.9] * 2) + [_row("m", 42, 0.8, 0.1, 0.9)])


def test_hybrid_beats_only_when_gap_exceeds_two_std_and_best_is_per_metric():
    rows = (_rows("lightgbm", [0.90, 0.90, 0.90], [0.08] * 3, [0.95] * 3)
            + _rows("mlp", [0.89, 0.91, 0.90], [0.05] * 3, [0.90] * 3)
            + _rows("hybrid_gat_lgbm", [0.93, 0.94, 0.95], [0.06, 0.06, 0.06], [0.96] * 3)
            + _rows("hybrid_gat_lgbm_nooof", [0.90, 0.91, 0.92], [0.04] * 3, [0.95] * 3))
    out = {r["model"]: r for r in CV.compare(rows, "d")}
    h = out["hybrid_gat_lgbm"]
    assert h["f1_macro_best_model"] == "lightgbm" or h["f1_macro_best_model"] == "mlp"
    assert h["fpr_best_model"] == "mlp"                                    # FPR thấp nhất trong nhóm không phải hybrid
    assert h["f1_macro_beats_2std"] is True                                # 0,94 - 0,90 = 0,04 > 2 × 0,01
    assert h["fpr_beats_2std"] is False                                    # 0,06 so với mlp 0,05: kém hơn
    assert h["fpr_diff_vs_best"] == pytest.approx(-0.01)
    n = out["hybrid_gat_lgbm_nooof"]
    assert n["f1_macro_beats_2std"] is False                               # 0,91 - 0,90 = 0,01 < 2 × 0,01
    assert n["fpr_beats_2std"] is True                                     # std cả hai bằng 0, hơn 0,01 FPR (xem ghi chú std = 0)
    assert out["lightgbm"]["f1_macro_mean"] == pytest.approx(0.90)
    assert "f1_macro_beats_2std" not in out["lightgbm"]


def test_read_rows_filters_split_task_and_dataset(tmp_path):
    p = tmp_path / "r.csv"
    rows = [_row("a", 42, 1, 1, 1), _row("a", 43, 1, 1, 1, split="test"), _row("a", 44, 1, 1, 1, task="binary"),
            _row("a", 45, 1, 1, 1, ds="other")]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    got = CV.read_rows([p], "d")
    assert [r["seed"] for r in got] == ["42"]


def test_main_writes_all_models(tmp_path):
    src = tmp_path / "r.csv"
    rows = _rows("lightgbm", [0.9] * 3, [0.1] * 3, [0.9] * 3) + _rows("hybrid_gat_lgbm", [0.95] * 3, [0.05] * 3, [0.9] * 3)
    with open(src, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "c.csv"
    assert CV.main(["d", str(out), str(src)]) == 0
    got = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [r["model"] for r in got] == ["lightgbm", "hybrid_gat_lgbm"]
    assert list(got[0]) == CV.COLUMNS
