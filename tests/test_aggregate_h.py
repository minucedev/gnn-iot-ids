"""Kiểm thử các hàm thuần của scripts/aggregate_h.py (GĐ H): McNemar, tên chế độ, mean ± std, kết luận hybrid so với baseline."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aggregate_h", ROOT / "scripts" / "aggregate_h.py")
AH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AH)


def test_split_mode_by_suffix():
    assert AH.split_mode("lightgbm") == ("lightgbm", "argmax")
    assert AH.split_mode("lightgbm_thr") == ("lightgbm", "thr")
    assert AH.split_mode("hybrid_gat_lgbm_oofavg_il_thr") == ("hybrid_gat_lgbm_oofavg_il", "thr")
    assert AH.split_mode("lgbm_neighbor_mean") == ("lgbm_neighbor_mean", "argmax")


def test_mcnemar_exact_and_chi2_known_values():
    a = np.array([True] * 8 + [False] * 2)             # A đúng, B sai: 8; A sai, B đúng: 2
    b = np.array([False] * 8 + [True] * 2)
    t = AH.mcnemar_test(a, b)
    assert (t["b"], t["c"], t["method"]) == (8, 2, "exact")
    assert t["p_value"] == pytest.approx(0.109375)      # 2 · P(X ≤ 2 | n = 10, p = 0,5) = 2 · 56/1024
    big_a = np.array([True] * 300 + [False] * 100 + [True] * 50)
    big_b = np.array([False] * 300 + [True] * 100 + [True] * 50)
    t2 = AH.mcnemar_test(big_a, big_b)
    assert (t2["b"], t2["c"], t2["method"]) == (300, 100, "chi2_cc")
    assert t2["statistic"] == pytest.approx((200 - 1) ** 2 / 400)
    assert t2["p_value"] < 1e-10
    same = AH.mcnemar_test(a, a)
    assert same["p_value"] == 1.0 and same["method"] == "no_discordant"


def test_mean_std_and_beats_use_ddof1_and_two_std_rule():
    mean, std = AH.ms([0.90, 0.92, 0.94])
    assert mean == pytest.approx(0.92) and std == pytest.approx(0.02)
    assert AH.ms([0.5, 0.5, 0.5]) == (0.5, 0.0)
    assert AH.beats(0.05, 0.01, 0.02) and not AH.beats(0.03, 0.01, 0.02)
    assert AH.beats(0.001, 0.0, 0.0) and not AH.beats(0.0, 0.0, 0.0)


def _rows(ds, model, f1s, **kw):
    return [{"dataset": ds, "model": model, "seed": 42 + i, "split": "test", "task": "multiclass", "accuracy": 0.9 + i * 0.001, "balanced_acc": 0.9,
             "precision_macro": 0.9, "recall_macro": 0.9, "f1_macro": f, "fpr": 0.05, "recall_attack": 0.95, "roc_auc": 0.97, "n_params": 1000, "size_kb": 10.0,
             "train_time_s": 5.0, "infer_ms_per_window": 1.0, **kw} for i, f in enumerate(f1s)]


def test_summarize_groups_by_dataset_base_and_mode():
    df = pd.DataFrame(_rows("d", "lightgbm", [0.90, 0.92, 0.94]) + _rows("d", "lightgbm_thr", [0.8, 0.8, 0.8]))
    s = AH.summarize(AH.add_keys(df))
    assert len(s) == 2
    assert AH.get(s, "d", "lightgbm", "argmax", "f1_macro_mean") == pytest.approx(0.92)
    assert AH.get(s, "d", "lightgbm", "argmax", "f1_macro_std") == pytest.approx(0.02)
    assert AH.get(s, "d", "lightgbm", "thr", "f1_macro_std") == 0.0
    assert math.isnan(AH.get(s, "d", "mlp", "argmax", "f1_macro_mean"))
    assert s.loc[s["mode"] == "argmax", "f1_macro_fmt"].iloc[0] == "0.9200 ± 0.0200"


def test_best_by_val_and_conclusions_verdicts():
    ds = AH.MAIN_DATASET
    val = pd.DataFrame(sum((_rows(ds, m, [f] * 3) for m, f in (("logreg", 0.7), ("lightgbm", 0.8), ("random_forest", 0.78), ("xgboost", 0.79), ("mlp", 0.75),
                                                              (AH.NBR, 0.9), ("gcn", 0.6), ("sage", 0.85), ("gat", 0.8))), []))
    sv = AH.summarize(AH.add_keys(val))
    best = AH.best_by_val(sv, ds)
    assert best == {"ML": "lightgbm", "MLP": "mlp", "GNN": "sage", "c": AH.NBR}
    test = pd.DataFrame(sum((_rows(ds, m, f) for m, f in (("lightgbm", [0.80, 0.80, 0.81]), ("mlp", [0.70, 0.71, 0.70]), ("sage", [0.88, 0.89, 0.88]),
                                                         (AH.NBR, [0.93, 0.93, 0.93]), (AH.HYBRID, [0.93, 0.94, 0.935]))), []))
    st = AH.summarize(AH.add_keys(test))
    mcn = pd.DataFrame([{"dataset": ds, "hybrid": AH.HYBRID, "baseline": b, "group": AH.GROUP[b], "selected_best_by_val": True, "seed": 42 + i, "mode": "argmax",
                         "b": 100, "c": 10 if b != AH.NBR else 95, "p_value": 1e-9 if b != AH.NBR else 0.6, "significant": b != AH.NBR,
                         "better": "hybrid"} for b in ("lightgbm", "mlp", "sage", AH.NBR) for i in range(3)])
    conc = AH.conclusions_table(st, mcn).set_index("baseline")
    assert conc.loc["lightgbm", "verdict"] == "hybrid VƯỢT" and conc.loc["mlp", "verdict"] == "hybrid VƯỢT" and conc.loc["sage", "verdict"] == "hybrid VƯỢT"
    assert conc.loc[AH.NBR, "verdict"] == "KHÔNG phân biệt được"            # (c) ngang: McNemar không có ý nghĩa
    assert conc.loc["lightgbm", "diff"] == pytest.approx(0.935 - 0.80333333, abs=1e-6) and bool(conc.loc["lightgbm", "diff_gt_2std"])
    tg = AH.targets_table(st, conc.reset_index(), 0.0, "test")
    row = tg[(tg["target"].str.startswith("Hybrid > baseline")) & (tg["model"] == AH.HYBRID) & (tg["mode"] == "argmax")].iloc[0]
    assert not row["passed"] and row["value"] == 3.0 and row["threshold"] == 4.0          # vượt 3/4 nhóm (không vượt (c)) → chưa đạt
    fpr_row = tg[(tg["target"].str.startswith("FPR")) & (tg["model"] == AH.HYBRID) & (tg["mode"] == "argmax")].iloc[0]
    assert fpr_row["value"] == pytest.approx(0.05) and not fpr_row["passed"]


def test_mcnemar_table_reads_prediction_files(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)

    def write(model, seed, acc):
        pred = np.where(rng.random(500) < acc, y, (y + 1) % 3)
        pd.DataFrame({"y_true": y.astype(np.int8), "pred": pred.astype(np.int8), "pred_thr": pred.astype(np.int8), "score": rng.random(500).astype(np.float32)}).to_parquet(
            tmp_path / f"pred_{{ds}}_{model}_s{seed}.parquet".replace("{ds}", AH.MAIN_DATASET), index=False)
        pd.DataFrame({"y_true": y.astype(np.int8), "pred": pred.astype(np.int8), "pred_thr": pred.astype(np.int8), "score": rng.random(500).astype(np.float32)}).to_parquet(
            tmp_path / f"pred_nf_bot_iot_v3_{model}_s{seed}.parquet", index=False)

    for seed in (42, 43, 44):
        for m in AH.ALL_MODELS:
            write(m, seed, 0.95 if m in (AH.HYBRID, AH.LIGHT) else 0.75)
    val = pd.DataFrame(sum((_rows(ds, m, [0.8, 0.8, 0.8]) for ds in AH.DATASETS for m in AH.SINGLE), []))
    sv = AH.summarize(AH.add_keys(val))
    mcn = AH.mcnemar_table(tmp_path, sv)
    # bộ chính: 2 hybrid × 9 mô hình đơn × 3 seed × 2 chế độ; BoT chỉ hybrid chính
    assert len(mcn[mcn["dataset"] == AH.MAIN_DATASET]) == 2 * 9 * 3 * 2 and len(mcn[mcn["dataset"] == "nf_bot_iot_v3"]) == 1 * 9 * 3 * 2
    assert mcn["selected_best_by_val"].sum() == (2 + 1) * 4 * 3 * 2
    assert (mcn["acc_hybrid"] > mcn["acc_baseline"]).all() and mcn["significant"].all() and (mcn["better"] == "hybrid").all()
