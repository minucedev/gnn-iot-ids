"""Kiểm thử src/ahp.py (task 7.2): ví dụ 3×3 tính tay, chuẩn hóa, độ nhạy và chạy trên số liệu thật của repo.

Không dùng fixture pytest để chạy được cả bằng `python3 -c "import tests.test_ahp ..."` ở nơi không có pytest."""

import csv
import json
import math
import tempfile
from pathlib import Path

from src import ahp

ROOT = Path(__file__).resolve().parents[1]


def close(a, b, tol=1e-9):
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


# ---------------------------------------------------------------- ví dụ 3×3 tính tay
def test_consistent_3x3_hand_computed():
    """Ma trận nhất quán hoàn toàn A = [[1,2,4],[1/2,1,2],[1/4,1/2,1]]: các cột tỉ lệ nhau.
    Tổng mỗi cột là 7/4, 7/2, 7 nên mỗi cột chuẩn hóa đều là (4/7, 2/7, 1/7) = vector riêng; λmax = 3, CI = CR = 0."""
    A = [[1, 2, 4], [1 / 2, 1, 2], [1 / 4, 1 / 2, 1]]
    w, lam = ahp.priority_vector(A)
    assert all(close(a, b) for a, b in zip(w, [4 / 7, 2 / 7, 1 / 7]))
    assert close(lam, 3.0)
    c = ahp.consistency(A)
    assert close(c["ci"], 0.0) and close(c["cr"], 0.0) and c["ri"] == 0.58
    assert all(close(a, b) for a, b in zip(ahp.geometric_mean_weights(A), [4 / 7, 2 / 7, 1 / 7]))   # trung bình nhân trùng khi nhất quán


def test_saaty_3x3_not_consistent_hand_computed():
    """A = [[1,3,5],[1/3,1,3],[1/5,1/3,1]] (không nhất quán). Tính tay: chuẩn hóa theo cột rồi lấy trung bình hàng cho xấp xỉ
    (0,633; 0,260; 0,106); lặp lũy thừa hội tụ về w = (0,6370; 0,2583; 0,1047). Aw = (1,9355; 0,7848; 0,3182) nên Aw/w ≈ 3,0385
    cho cả ba hàng: λmax = 3,0385, CI = (3,0385 - 3)/2 = 0,01926, CR = CI / 0,58 = 0,0332 < 0,10."""
    A = [[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]]
    colsum = [sum(A[i][j] for i in range(3)) for j in range(3)]
    approx = [sum(A[i][j] / colsum[j] for j in range(3)) / 3 for i in range(3)]       # cách tính tay
    assert all(abs(a - b) < 0.005 for a, b in zip(approx, [0.6370, 0.2583, 0.1047]))
    w, lam = ahp.priority_vector(A)
    assert all(abs(a - b) < 5e-5 for a, b in zip(w, [0.6370, 0.2583, 0.1047]))
    assert abs(sum(w) - 1) < 1e-12
    c = ahp.consistency(A)
    assert abs(lam - 3.0385) < 5e-5 and abs(c["ci"] - 0.01926) < 5e-5 and abs(c["cr"] - 0.0332) < 5e-4
    assert c["cr"] < ahp.CR_LIMIT
    assert max(abs(a - b) for a, b in zip(w, ahp.geometric_mean_weights(A))) < 1e-6    # nhất quán gần đúng: hai cách gần trùng


def test_ri_table_and_inconsistent_matrix_fails_cr():
    assert [ahp.RI[n] for n in (3, 4, 5, 6, 7, 8, 9, 10)] == [0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
    bad = [[1, 9, 1 / 9], [1 / 9, 1, 9], [9, 1 / 9, 1]]                                # vòng tròn: a > b > c > a
    assert ahp.consistency(bad)["cr"] > ahp.CR_LIMIT


def test_check_matrix_rejections():
    assert raises(lambda: ahp.check_matrix([[1, 2], [2, 1]]))                          # không nghịch đảo
    assert raises(lambda: ahp.check_matrix([[1, 10], [1 / 10, 1]]))                    # ngoài thang 1-9
    assert raises(lambda: ahp.check_matrix([[2, 1], [1, 1]]))                          # đường chéo khác 1
    assert raises(lambda: ahp.check_matrix([[1, 2, 3], [1 / 2, 1]]))                   # không vuông
    ahp.check_matrix([[1, 10], [1 / 10, 1]], saaty=False)


def test_matrix_from_pairs_and_aggregate():
    ids = ["a", "b", "c"]
    A = ahp.matrix_from_pairs(ids, [{"i": "a", "j": "b", "value": "3"}, {"i": "a", "j": "c", "value": 5},
                                    {"i": "b", "j": "c", "value": "3"}])
    assert A[0][1] == 3 and close(A[1][0], 1 / 3) and close(A[2][0], 1 / 5)
    assert raises(lambda: ahp.matrix_from_pairs(ids, [{"i": "a", "j": "b", "value": "3"}]))                         # thiếu cặp
    assert raises(lambda: ahp.matrix_from_pairs(ids, [{"i": "a", "j": "b", "value": "3"}, {"i": "b", "j": "a", "value": "2"},
                                                      {"i": "a", "j": "c", "value": 1}, {"i": "b", "j": "c", "value": 1}]))  # lặp cặp
    B = ahp.matrix_from_pairs(ids, [{"i": "a", "j": "b", "value": "1/3"}, {"i": "a", "j": "c", "value": "1/5"},
                                    {"i": "b", "j": "c", "value": "1/3"}])
    G = ahp.aggregate_matrices([A, B])                                                  # trung bình nhân của x và 1/x là 1
    assert all(close(G[i][j], 1.0) for i in range(3) for j in range(3))
    assert all(close(x, y) for x, y in zip(sum(ahp.aggregate_matrices([A, A]), []), sum(A, [])))
    ahp.check_matrix(G)


# ---------------------------------------------------------------- chuẩn hóa
def test_normalize_methods():
    v = [2.0, 4.0, 10.0]
    assert all(close(a, b) for a, b in zip(ahp.normalize(v, True, "minmax"), [0.0, 0.25, 1.0]))
    assert all(close(a, b) for a, b in zip(ahp.normalize(v, False, "minmax"), [1.0, 0.75, 0.0]))          # đảo chiều
    assert all(close(a, b) for a, b in zip(ahp.normalize(v, True, "ratio"), [0.2, 0.4, 1.0]))
    assert all(close(a, b) for a, b in zip(ahp.normalize(v, False, "ratio"), [1.0, 0.5, 0.2]))
    lg = ahp.normalize([1.0, 10.0, 100.0], False, "log_minmax")                                           # log10 = 0, 1, 2
    assert all(close(a, b) for a, b in zip(lg, [1.0, 0.5, 0.0]))
    assert ahp.normalize([3.0, 3.0], False, "minmax") == [1.0, 1.0]                                       # cột hằng
    assert raises(lambda: ahp.normalize([0.0, 1.0], True, "log_minmax"))
    assert raises(lambda: ahp.normalize([1.0, 2.0], True, "lạ"))


def test_totals_and_ranking():
    S = {"A": [1.0, 0.0], "B": [0.0, 1.0], "C": [0.5, 0.5]}
    t = ahp.totals(S, [0.7, 0.3])
    assert close(t["A"], 0.7) and close(t["B"], 0.3) and close(t["C"], 0.5)
    assert ahp.ranking(t) == ["A", "C", "B"]


# ---------------------------------------------------------------- độ nhạy
def test_tipping_points_hand_computed():
    """A = (1, 0), B = (0, 1), w = (0,7; 0,3): điểm A = t, điểm B = 1 - t khi trọng số tiêu chí 0 là t, cắt nhau tại t = 0,5.
    Giảm trọng số tiêu chí 0 từ 0,7 xuống 0,5 thì B vượt A (đổi -28,57%); tiêu chí 1 tăng từ 0,3 lên 0,5 (đổi +66,67%)."""
    S = {"A": [1.0, 0.0], "B": [0.0, 1.0]}
    tp = ahp.tipping_points(S, [0.7, 0.3], ["c0", "c1"])
    d0 = tp["c0"]["down"]["top_change"]
    assert close(d0["weight"], 0.5) and close(d0["relative_change"], 0.5 / 0.7 - 1) and d0["new_top"] == "B"
    assert tp["c0"]["up"]["top_change"] is None and tp["c0"]["up"]["rank_swap"] is None
    u1 = tp["c1"]["up"]["top_change"]
    assert close(u1["weight"], 0.5) and close(u1["relative_change"], 0.5 / 0.3 - 1) and u1["new_top"] == "B"


def test_perturbation_flags_rank_change():
    S = {"A": [1.0, 0.0], "B": [0.0, 1.0]}
    rows = ahp.perturbation(S, [0.7, 0.3], ["c0", "c1"])
    assert len(rows) == 4 and not any(r["top_changed"] for r in rows)                  # 0,7/0,3 chịu được ±20%
    assert all(close(sum(r["weights"]), 1.0) for r in rows)
    # sát nhau: w = (0,52; 0,48). Giảm 20% trọng số c0 → (0,464; 0,536) và tăng 20% trọng số c1 → (0,4745; 0,5255): B vượt A.
    # Hai phép còn lại (tăng c0, giảm c1) giữ A đứng đầu.
    rows = ahp.perturbation(S, [0.52, 0.48], ["c0", "c1"])
    flips = sorted((r["criterion"], round(r["change"], 2)) for r in rows if r["top_changed"])
    assert flips == [("c0", -0.2), ("c1", 0.2)] and all(r["top"] == "B" for r in rows if r["top_changed"])


def test_conditional_choices_prioritises_criterion():
    S = {"A": [1.0, 0.0], "B": [0.0, 1.0]}
    c = ahp.conditional_choices(S, [0.7, 0.3], ["c0", "c1"], share=0.5)
    assert c["c0"]["top"] in ("A", "B") and close(sum(c["c0"]["weights"]), 1.0)
    c = ahp.conditional_choices(S, [0.7, 0.3], ["c0", "c1"], share=0.9)
    assert c["c0"]["top"] == "A" and c["c1"]["top"] == "B"


# ---------------------------------------------------------------- ma trận đã duyệt và số liệu thật
def test_approved_matrix_is_consistent_and_hand_weights():
    ids, A, spec = ahp.load_matrix(ROOT / "docs" / "ahp_matrix.json")
    assert ids == ["f1_macro", "fpr", "recall_attack", "latency_p95", "size"] and len(spec["pairs"]) == 10
    assert all(p["reason"].strip() for p in spec["pairs"])                              # mỗi cặp có lý do bằng lời
    c = ahp.consistency(A)
    assert c["cr"] < ahp.CR_LIMIT and abs(c["lambda_max"] - 5.0441) < 5e-4 and abs(c["cr"] - 0.0098) < 5e-4
    w, _ = ahp.priority_vector(A)
    assert abs(sum(w) - 1) < 1e-12 and w[1] == max(w) and w[3] == min(w)                # FPR lớn nhất, độ trễ nhỏ nhất


def test_run_ahp_on_real_summary():
    res = ahp.run_ahp(ROOT / "docs" / "ahp_matrix.json", ROOT / "reports" / "tables" / "summary_test.csv")
    names = [n for n, _ in ahp.ALTERNATIVES]
    assert sorted(res["ranking"]) == sorted(names) and res["recommended"] == res["ranking"][0]
    assert close(sum(res["weights"].values()), 1.0) and res["consistency"]["ok"]
    assert res["weights_max_abs_diff_eig_vs_gm"] < 0.01
    for a in res["alternatives"].values():
        assert all(0.0 <= s <= 1.0 for s in a["scores"].values())
    best = {cid: max(a["scores"][cid] for a in res["alternatives"].values()) for cid in res["criteria"]}
    assert all(close(v, 1.0) for v in best.values())                                    # mỗi tiêu chí có phương án tốt nhất = 1
    # đảo chiều: FPR nhỏ nhất phải có điểm FPR cao nhất, kích thước nhỏ nhất có điểm kích thước cao nhất
    raw = {n: a["raw"] for n, a in res["alternatives"].items()}
    assert min(raw, key=lambda n: raw[n]["fpr"]) == max(res["alternatives"], key=lambda n: res["alternatives"][n]["scores"]["fpr"])
    assert min(raw, key=lambda n: raw[n]["size"]) == max(res["alternatives"], key=lambda n: res["alternatives"][n]["scores"]["size"])
    assert set(res["sensitivity"]["tipping"]) == set(res["criteria"]) and len(res["sensitivity"]["perturbation"]) == 10
    assert len(res["statements"]) >= 6 and set(res["normalization_robustness"]) == {"per_criterion", "minmax", "log_minmax", "ratio"}
    json.dumps(res)                                                                      # ghi được ra JSON


def test_run_ahp_synthetic_csv_and_missing_alternative():
    cols = ["dataset", "base", "mode", "f1_macro_mean", "fpr_mean", "recall_attack_mean", "infer_ms_per_window_mean",
            "size_kb_mean", "n_params_mean"]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for k, (name, _) in enumerate(ahp.ALTERNATIVES):
                w.writerow({"dataset": "ds", "base": name, "mode": "thr", "f1_macro_mean": 0.9 + 0.01 * k, "fpr_mean": 0.05 - 0.002 * k,
                            "recall_attack_mean": 0.9 + 0.005 * k, "infer_ms_per_window_mean": 1 + k, "size_kb_mean": 10 * (k + 1),
                            "n_params_mean": 1000 * (k + 1)})
            w.writerow({"dataset": "ds", "base": "gat", "mode": "argmax", "f1_macro_mean": 0, "fpr_mean": 1, "recall_attack_mean": 0,
                        "infer_ms_per_window_mean": 1, "size_kb_mean": 1, "n_params_mean": 1})   # chế độ khác phải bị bỏ qua
        res = ahp.run_ahp(ROOT / "docs" / "ahp_matrix.json", p, dataset="ds", mode="thr")
        assert res["alternatives"]["gat"]["raw"]["f1_macro"] == 0.9 + 0.01 * 7
        assert res["feasible_only"]["alternatives"] == [n for n, _ in ahp.ALTERNATIVES]    # mọi phương án đạt ngưỡng nhẹ
        assert raises(lambda: ahp.run_ahp(ROOT / "docs" / "ahp_matrix.json", p, dataset="khong_co"))
