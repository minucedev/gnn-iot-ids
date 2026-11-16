"""AHP chọn mô hình triển khai (task 7.1, 7.2, quyết định D6). Python thuần (không cần numpy) để kiểm tra được ở mọi nơi.

Quy trình (Saaty 1980):
  1. Ma trận so sánh cặp 5×5 giữa các tiêu chí (docs/ahp_matrix.json, thang 1-9, a_ji = 1/a_ij).
  2. Trọng số = vector riêng chính (power iteration), kiểm bằng trung bình nhân các hàng; λmax, CI = (λmax - n)/(n - 1),
     CR = CI / RI(n) với RI chuẩn của Saaty (n = 3: 0,58; n = 4: 0,90; n = 5: 1,12 ...). Yêu cầu CR < 0,10.
  3. Điểm phương án theo từng tiêu chí lấy từ số liệu thật (summary_test.csv), chuẩn hóa về [0, 1], tiêu chí càng nhỏ càng
     tốt (FPR, độ trễ, kích thước) được đảo chiều. Điểm tổng = tổng trọng số × điểm chuẩn hóa.
  4. Độ nhạy: (a) nhân từng trọng số với 0,8 và 1,2 rồi chuẩn hóa lại; (b) điểm đảo chiều: giá trị trọng số của tiêu chí i
     (các trọng số còn lại co giãn theo tỉ lệ) mà tại đó hai phương án đổi thứ hạng, tính chính xác vì điểm tổng là hàm
     tuyến tính của trọng số đó; (c) kết luận có điều kiện "nếu ưu tiên tiêu chí X (trọng số 50%) thì chọn ...".

Chạy:  python -I src/ahp.py <summary_test.csv> <ahp_matrix.json> <ahp_result.json> [dataset] [mode]
"""

import csv
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}   # chỉ số ngẫu nhiên Saaty
CR_LIMIT = 0.10
PERTURB = 0.20
MAX_PARAMS, MAX_SIZE_KB = 100_000, 1024      # ngưỡng nghiệm thu "nhẹ" (KE_HOACH mục 3)

# id tiêu chí -> cột trong summary_test.csv, chiều tốt (benefit: càng lớn càng tốt)
CRITERIA = {
    "f1_macro": {"column": "f1_macro_mean", "benefit": True},
    "fpr": {"column": "fpr_mean", "benefit": False},
    "recall_attack": {"column": "recall_attack_mean", "benefit": True},
    "latency_p95": {"column": "infer_ms_per_window_mean", "benefit": False},
    "size": {"column": "size_kb_mean", "benefit": False},
}
DEFAULT_NORM = {"f1_macro": "minmax", "fpr": "minmax", "recall_attack": "minmax", "latency_p95": "log_minmax", "size": "log_minmax"}

# phương án: (tên trong summary_test.csv, nhãn)
ALTERNATIVES = [
    ("hybrid_gat_lgbm_oofavg_il", "Hybrid chính"),
    ("hybrid_light", "Hybrid nhẹ"),
    ("lgbm_neighbor_mean", "LightGBM + TB hàng xóm (c)"),
    ("lightgbm", "LightGBM"),
    ("random_forest", "Random Forest"),
    ("mlp", "MLP"),
    ("sage", "GraphSAGE"),
    ("gat", "GAT"),
]


# ---------------------------------------------------------------- ma trận so sánh cặp
def check_matrix(A, tol=1e-6, saaty=True):
    """Ma trận vuông dương, đường chéo 1, nghịch đảo (a_ij · a_ji = 1); saaty=True: mọi phần tử trong [1/9, 9]."""
    n = len(A)
    if n < 1 or any(len(r) != n for r in A):
        raise ValueError("Ma trận phải vuông")
    for i in range(n):
        if abs(A[i][i] - 1) > tol:
            raise ValueError(f"Đường chéo phải bằng 1 (a[{i}][{i}] = {A[i][i]})")
        for j in range(n):
            if A[i][j] <= 0:
                raise ValueError("Phần tử phải dương")
            if abs(A[i][j] * A[j][i] - 1) > tol:
                raise ValueError(f"Không nghịch đảo tại ({i},{j}): {A[i][j]} · {A[j][i]} ≠ 1")
            if saaty and not (1 / 9 - tol <= A[i][j] <= 9 + tol):
                raise ValueError(f"Ngoài thang Saaty 1-9 tại ({i},{j}): {A[i][j]}")


def matrix_from_pairs(ids, pairs):
    """Ma trận n×n từ danh sách cặp {i, j, value}; value là số hoặc chuỗi như '1/3'. Mỗi cặp i < j phải có đúng một lần."""
    n = len(ids)
    pos = {c: k for k, c in enumerate(ids)}
    A = [[None] * n for _ in range(n)]
    for k in range(n):
        A[k][k] = 1.0
    for p in pairs:
        i, j = pos[p["i"]], pos[p["j"]]
        if i == j or A[i][j] is not None:
            raise ValueError(f"Cặp lặp hoặc tự so sánh: {p['i']}, {p['j']}")
        v = float(Fraction(str(p["value"])))
        A[i][j], A[j][i] = v, 1.0 / v
    if any(x is None for r in A for x in r):
        raise ValueError("Thiếu cặp so sánh")
    return A


def load_matrix(path):
    """Đọc docs/ahp_matrix.json. Trả về (ids, A, JSON gốc); kiểm ma trận ghi sẵn trùng ma trận dựng từ các cặp."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [c["id"] for c in spec["criteria"]]
    A = matrix_from_pairs(ids, spec["pairs"])
    if "matrix" in spec:
        for ra, rb in zip(A, spec["matrix"]):
            if any(abs(a - b) > 1e-9 for a, b in zip(ra, rb)):
                raise ValueError("'matrix' không khớp với 'pairs'")
    check_matrix(A)
    return ids, A, spec


def priority_vector(A, tol=1e-14, max_iter=100000):
    """(trọng số, λmax) theo vector riêng chính của ma trận dương: lặp lũy thừa w ← Aw / Σ(Aw)."""
    n = len(A)
    w = [1.0 / n] * n
    for _ in range(max_iter):
        v = [sum(A[i][j] * w[j] for j in range(n)) for i in range(n)]
        s = sum(v)
        nw = [x / s for x in v]
        done = max(abs(a - b) for a, b in zip(nw, w)) < tol
        w = nw
        if done:
            break
    Aw = [sum(A[i][j] * w[j] for j in range(n)) for i in range(n)]
    return w, sum(Aw[i] / w[i] for i in range(n)) / n


def geometric_mean_weights(A):
    """Trọng số theo trung bình nhân các hàng (xấp xỉ khác của vector riêng; trùng khi ma trận nhất quán hoàn toàn)."""
    n = len(A)
    g = [math.exp(sum(math.log(x) for x in r) / n) for r in A]
    s = sum(g)
    return [x / s for x in g]


def consistency(A):
    """{lambda_max, ci, ri, cr}. n ≤ 2 luôn nhất quán (CI = CR = 0)."""
    n = len(A)
    if n > max(RI):
        raise ValueError(f"Không có RI cho n = {n}")
    _, lam = priority_vector(A)
    ci = (lam - n) / (n - 1) if n > 2 else 0.0
    ri = RI[n]
    return {"lambda_max": lam, "ci": ci, "ri": ri, "cr": ci / ri if ri > 0 else 0.0}


def aggregate_matrices(mats):
    """Gộp nhiều ma trận cá nhân bằng trung bình nhân từng phần tử (vẫn nghịch đảo). Dùng khi có thêm người chấm."""
    n = len(mats[0])
    return [[math.exp(sum(math.log(m[i][j]) for m in mats) / len(mats)) for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------- điểm phương án
def normalize(values, benefit=True, method="minmax"):
    """Chuẩn hóa một cột số liệu về [0, 1], 1 là tốt nhất.

    minmax:     (x - min) / (max - min); tiêu chí càng nhỏ càng tốt (benefit=False) thì đảo thành 1 - ·.
    log_minmax: như minmax nhưng trên log10(x) (x > 0), cho đại lượng trải nhiều bậc độ lớn.
    ratio:      x / max (benefit) hoặc min / x (cost); x > 0.
    Cột hằng (max = min): mọi phương án 1,0 (tiêu chí không phân biệt được phương án).
    """
    v = [float(x) for x in values]
    if method in ("log_minmax", "ratio") and min(v) <= 0:
        raise ValueError(f"{method} cần giá trị dương, nhận {min(v)}")
    if method == "ratio":
        return [x / max(v) for x in v] if benefit else [min(v) / x for x in v]
    if method not in ("minmax", "log_minmax"):
        raise ValueError(f"Kiểu chuẩn hóa lạ: {method}")
    if method == "log_minmax":
        v = [math.log10(x) for x in v]
    lo, hi = min(v), max(v)
    if hi - lo < 1e-300:
        return [1.0] * len(v)
    s = [(x - lo) / (hi - lo) for x in v]
    return s if benefit else [1.0 - x for x in s]


def load_alternatives(summary_csv, dataset="nf_ton_iot_v3", mode="thr", alternatives=ALTERNATIVES):
    """Số liệu thật của từng phương án: {tên: {id tiêu chí: giá trị}} từ summary_test.csv (dòng của bộ, chế độ ngưỡng)."""
    rows = {}
    with open(summary_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == dataset and r["mode"] == mode:
                rows[r["base"]] = r
    out = {}
    for name, _ in alternatives:
        if name not in rows:
            raise ValueError(f"Thiếu phương án {name} (bộ {dataset}, chế độ {mode}) trong {summary_csv}")
        out[name] = {cid: float(rows[name][spec["column"]]) for cid, spec in CRITERIA.items()}
        out[name]["n_params"] = float(rows[name]["n_params_mean"])        # không phải tiêu chí AHP, chỉ để lọc phương án đạt ngưỡng
    return out


def score_table(data, ids, norm):
    """Điểm chuẩn hóa {tên: [điểm theo từng tiêu chí trong ids]}. `norm`: kiểu chuẩn hóa mỗi tiêu chí (dict) hoặc một chuỗi chung."""
    names = list(data)
    cols = []
    for cid in ids:
        method = norm if isinstance(norm, str) else norm[cid]
        cols.append(normalize([data[n][cid] for n in names], CRITERIA[cid]["benefit"], method))
    return {n: [cols[c][k] for c in range(len(ids))] for k, n in enumerate(names)}


def totals(S, w):
    return {n: sum(wi * si for wi, si in zip(w, row)) for n, row in S.items()}


def ranking(tot):
    """Tên phương án theo điểm giảm dần (hòa điểm: theo thứ tự xuất hiện)."""
    return sorted(tot, key=lambda n: -tot[n])


# ---------------------------------------------------------------- độ nhạy
def _renorm(w, i, t):
    """Đặt trọng số tiêu chí i bằng t, các trọng số còn lại co giãn theo tỉ lệ để tổng = 1."""
    rest = 1.0 - w[i]
    return [t if k == i else wk * (1.0 - t) / rest for k, wk in enumerate(w)]


def perturbation(S, w, ids, pct=PERTURB):
    """Nhân từng trọng số với (1 - pct) và (1 + pct), chuẩn hóa lại; ghi thứ hạng, phương án đứng đầu, có đảo thứ hạng không."""
    base = ranking(totals(S, w))
    out = []
    for i, cid in enumerate(ids):
        for sign in (-1, +1):
            w2 = list(w)
            w2[i] *= 1 + sign * pct
            s = sum(w2)
            w2 = [x / s for x in w2]
            tot = totals(S, w2)
            rk = ranking(tot)
            out.append({"criterion": cid, "change": sign * pct, "weights": w2, "totals": tot, "ranking": rk, "top": rk[0],
                        "top_changed": rk[0] != base[0], "ranking_changed": rk != base})
    return out


def tipping_points(S, w, ids):
    """Với từng tiêu chí i: trọng số gần nhất (tăng và giảm) làm hai phương án đổi thứ hạng, và làm phương án đứng đầu đổi.

    Điểm của phương án a theo trọng số t của tiêu chí i là t·s_ai + (1 - t)·R_a (R_a = điểm các tiêu chí còn lại sau khi co giãn),
    tuyến tính theo t nên giao điểm của hai phương án giải được chính xác. Trả về {cid: {"weight", "up", "down"}}; mỗi hướng
    gồm "rank_swap" và "top_change" (None nếu không có giao điểm trong (0, 1))."""
    names = list(S)
    top = ranking(totals(S, w))[0]
    out = {}
    for i, cid in enumerate(ids):
        rest = 1.0 - w[i]
        R = {n: sum(w[k] * S[n][k] for k in range(len(w)) if k != i) / rest for n in names}
        cross = []                                  # (t*, a, b)
        for x in range(len(names)):
            for y in range(x + 1, len(names)):
                a, b = names[x], names[y]
                den = (S[a][i] - R[a]) - (S[b][i] - R[b])
                if abs(den) < 1e-15:
                    continue
                t = (R[b] - R[a]) / den
                if 1e-12 < t < 1 - 1e-12 and abs(t - w[i]) > 1e-12:
                    cross.append((t, a, b))
        entry = {"weight": w[i]}
        for direction, pick in (("up", lambda c: c[0] > w[i]), ("down", lambda c: c[0] < w[i])):
            cs = sorted((c for c in cross if pick(c)), key=lambda c: abs(c[0] - w[i]))

            def fmt(c, new_top=None):
                if c is None:
                    return None
                d = {"weight": c[0], "relative_change": c[0] / w[i] - 1, "pair": [c[1], c[2]]}
                if new_top:
                    d["new_top"] = new_top
                return d

            swap = cs[0] if cs else None
            tc = next((c for c in cs if top in (c[1], c[2])), None)
            other = None if tc is None else (tc[2] if tc[1] == top else tc[1])
            entry[direction] = {"rank_swap": fmt(swap), "top_change": fmt(tc, other)}
        out[cid] = entry
    return out


def conditional_choices(S, w, ids, share=0.5):
    """Phương án đứng đầu khi một tiêu chí chiếm `share` tổng trọng số (các tiêu chí còn lại co giãn theo tỉ lệ)."""
    out = {}
    for i, cid in enumerate(ids):
        w2 = _renorm(w, i, share)
        tot = totals(S, w2)
        out[cid] = {"weights": w2, "ranking": ranking(tot), "top": ranking(tot)[0], "score": max(tot.values())}
    return out


# ---------------------------------------------------------------- chạy toàn bộ
def run_ahp(matrix_json, summary_csv, dataset="nf_ton_iot_v3", mode="thr"):
    ids, A, spec = load_matrix(matrix_json)
    if set(ids) != set(CRITERIA):
        raise ValueError(f"Tiêu chí trong ma trận {ids} khác {list(CRITERIA)}")
    w, lam = priority_vector(A)
    gm = geometric_mean_weights(A)
    cons = consistency(A)
    data = load_alternatives(summary_csv, dataset, mode)
    labels = dict(ALTERNATIVES)
    norm = spec.get("normalization", {}).get("per_criterion", DEFAULT_NORM)
    S = score_table(data, ids, norm)
    tot = totals(S, w)
    rk = ranking(tot)
    robust = {}
    for key, nm in (("per_criterion", norm), ("minmax", "minmax"), ("log_minmax", "log_minmax"), ("ratio", "ratio")):
        t2 = totals(score_table(data, ids, nm), w)
        robust[key] = {"ranking": ranking(t2), "top": ranking(t2)[0], "totals": t2}
    cond = conditional_choices(S, w, ids)
    sens = {"perturbation": perturbation(S, w, ids), "tipping": tipping_points(S, w, ids)}
    top = rk[0]
    ok = [n for n in data if data[n]["n_params"] <= MAX_PARAMS and data[n]["size"] <= MAX_SIZE_KB]
    feasible = None
    if len(ok) >= 2:                      # phân tích phụ: chỉ xét phương án đạt ngưỡng nhẹ, chuẩn hóa lại trong nhóm đó
        sub = {n: data[n] for n in ok}
        S2 = score_table(sub, ids, norm)
        t2 = totals(S2, w)
        feasible = {"alternatives": ok, "ranking": ranking(t2), "top": ranking(t2)[0], "totals": t2,
                    "conditional": {cid: c["top"] for cid, c in conditional_choices(S2, w, ids).items()}}
    flips = [p for p in sens["perturbation"] if p["top_changed"]]
    statements = [f"Cân bằng theo ma trận đã duyệt: chọn {labels[top]}."]
    for cid in ids:
        c = cond[cid]
        statements.append(f"Nếu ưu tiên {_label(spec, cid)} (trọng số 50%) thì chọn {labels[c['top']]}"
                          + ("." if c["top"] != top else " (không đổi)."))
    if feasible:
        limit = f"{MAX_PARAMS:,}".replace(",", ".")
        statements.append(f"Nếu chỉ xét phương án đạt ngưỡng nhẹ (≤ {limit} tham số, ≤ {MAX_SIZE_KB} KB) thì chọn "
                          f"{labels[feasible['top']]} (phân tích phụ, ngoài ma trận đã duyệt).")
    return {
        "dataset": dataset, "mode": mode, "criteria": ids,
        "criteria_labels": {c["id"]: c["label"] for c in spec["criteria"]},
        "matrix": A, "weights": dict(zip(ids, w)), "weights_geometric_mean": dict(zip(ids, gm)),
        "weights_max_abs_diff_eig_vs_gm": max(abs(a - b) for a, b in zip(w, gm)),
        "consistency": {**cons, "cr_limit": CR_LIMIT, "ok": cons["cr"] < CR_LIMIT},
        "alternatives": {n: {"label": labels[n], "raw": data[n], "scores": dict(zip(ids, S[n])), "total": tot[n]} for n in data},
        "normalization": norm, "ranking": rk, "recommended": top, "recommended_label": labels[top],
        "normalization_robustness": robust,
        "conditional": cond, "sensitivity": sens, "feasible_only": feasible,
        "top_flips_under_20pct": [{"criterion": p["criterion"], "change": p["change"], "new_top": p["top"]} for p in flips],
        "statements": statements,
    }


def screen(data, max_params=MAX_PARAMS, max_size_kb=MAX_SIZE_KB):
    """Sàng lọc cứng: (phương án giữ lại, {phương án bị loại: [lý do]}). Giữ khi n_params ≤ max_params VÀ size ≤ max_size_kb."""
    kept, dropped = [], {}
    for n, d in data.items():
        why = []
        if d["n_params"] > max_params:
            why.append(f"{d['n_params']:,.0f} tham số > {max_params:,}".replace(",", "."))
        if d["size"] > max_size_kb:
            why.append(f"{d['size']:,.0f} KB > {max_size_kb:,} KB".replace(",", "."))
        if why:
            dropped[n] = why
        else:
            kept.append(n)
    return kept, dropped


def run_constrained(matrix_json, summary_csv, dataset="nf_ton_iot_v3", mode="thr", max_params=MAX_PARAMS, max_size_kb=MAX_SIZE_KB,
                    fpr_limit=0.0408, recall_min=0.95):
    """Kịch bản "sàng lọc rồi xếp hạng" (GĐ J): dùng đúng ma trận đã duyệt, LOẠI phương án vượt ràng buộc cứng rồi mới chuẩn hóa và xếp hạng
    trong nhóm còn lại (chuẩn hóa lại trong nhóm). Trả về cả kịch bản (a) không ràng buộc để so sánh cạnh nhau.

    Ngoài ra ghi: (1) thứ hạng của nhóm còn lại nếu GIỮ điểm chuẩn hóa của 8 phương án (không chuẩn hóa lại), để thấy kết quả phụ thuộc
    việc chuẩn hóa lại; (2) độ bền theo cách chuẩn hóa; (3) độ nhạy ±20% và kết luận có điều kiện trong nhóm; (4) từng phương án có đạt
    chỉ tiêu nghiệm thu FPR ≤ fpr_limit và recall tấn công ≥ recall_min hay không (số trung bình 3 seed, chế độ ngưỡng) và thứ hạng nếu
    ngoài ràng buộc nhẹ còn bắt buộc đạt cả hai chỉ tiêu đó (phân tích mở rộng, ngoài yêu cầu gốc)."""
    ids, A, spec = load_matrix(matrix_json)
    w, _ = priority_vector(A)
    data = load_alternatives(summary_csv, dataset, mode)
    labels = dict(ALTERNATIVES)
    norm = spec.get("normalization", {}).get("per_criterion", DEFAULT_NORM)
    S_all = score_table(data, ids, norm)
    tot_all = totals(S_all, w)
    rk_all = ranking(tot_all)
    kept, dropped = screen(data, max_params, max_size_kb)
    if len(kept) < 2:
        raise ValueError(f"Chỉ còn {len(kept)} phương án sau sàng lọc: không xếp hạng được")
    sub = {n: data[n] for n in kept}
    S = score_table(sub, ids, norm)
    tot = totals(S, w)
    rk = ranking(tot)
    robust = {}
    for key, nm in (("per_criterion", norm), ("minmax", "minmax"), ("log_minmax", "log_minmax"), ("ratio", "ratio")):
        t2 = totals(score_table(sub, ids, nm), w)
        robust[key] = {"ranking": ranking(t2), "top": ranking(t2)[0], "totals": t2}
    keep_scores = {n: S_all[n] for n in kept}                       # điểm của 8 phương án, không chuẩn hóa lại
    t_keep = totals(keep_scores, w)
    cond = conditional_choices(S, w, ids)
    sens = {"perturbation": perturbation(S, w, ids), "tipping": tipping_points(S, w, ids)}
    flips = [p for p in sens["perturbation"] if p["top_changed"]]
    meets = {n: {"fpr": data[n]["fpr"], "recall_attack": data[n]["recall_attack"], "meets_fpr": data[n]["fpr"] <= fpr_limit,
                 "meets_recall": data[n]["recall_attack"] >= recall_min} for n in data}
    both = [n for n in kept if meets[n]["meets_fpr"] and meets[n]["meets_recall"]]
    plus = None
    if len(both) >= 2:
        S3 = score_table({n: data[n] for n in both}, ids, norm)
        t3 = totals(S3, w)
        plus = {"alternatives": both, "ranking": ranking(t3), "top": ranking(t3)[0], "totals": t3}
    top_all, top = rk_all[0], rk[0]
    top_row, all_row = data[top], data[top_all]
    delta = {c: top_row[c] - all_row[c] for c in ("f1_macro", "fpr", "recall_attack", "latency_p95", "size", "n_params")}
    return {
        "dataset": dataset, "mode": mode, "criteria": ids, "criteria_labels": {c["id"]: c["label"] for c in spec["criteria"]},
        "weights": dict(zip(ids, w)), "normalization": norm,
        "constraints": {"max_params": max_params, "max_size_kb": max_size_kb,
                        "note": "ngưỡng nhẹ của KE_HOACH mục 3; 1 MB = 1.024 KB (kết quả không đổi nếu dùng 1.000 KB vì mọi phương án giữ lại đều ≤ 986 KB)"},
        "alternatives": {n: {"label": labels[n], "raw": data[n], "feasible": n in kept, "excluded_because": dropped.get(n, []),
                             "meets_targets": meets[n]} for n in data},
        "scenario_a_unconstrained": {"ranking": rk_all, "top": top_all, "top_label": labels[top_all], "totals": tot_all,
                                     "scores": {n: dict(zip(ids, S_all[n])) for n in data}},
        "scenario_b_constrained": {"alternatives": kept, "excluded": dropped, "ranking": rk, "top": top, "top_label": labels[top],
                                   "totals": tot, "scores": {n: dict(zip(ids, S[n])) for n in kept},
                                   "ranking_without_renormalization": ranking(t_keep), "totals_without_renormalization": t_keep,
                                   "normalization_robustness": robust, "conditional": cond, "sensitivity": sens,
                                   "top_flips_under_20pct": [{"criterion": p["criterion"], "change": p["change"], "new_top": p["top"]} for p in flips]},
        "tradeoff_b_vs_a": {"from": top_all, "to": top, "delta": delta},
        "extension_constrained_and_targets": {"fpr_limit": fpr_limit, "recall_min": recall_min, "result": plus,
                                              "kept_meeting_both": both},
    }


def _label(spec, cid):
    return next(c["label"] for c in spec["criteria"] if c["id"] == cid)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    summary_csv, matrix_json, out = sys.argv[1:4]
    dataset = sys.argv[4] if len(sys.argv) > 4 else "nf_ton_iot_v3"
    mode = sys.argv[5] if len(sys.argv) > 5 else "thr"
    res = run_ahp(matrix_json, summary_csv, dataset, mode)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Trọng số:", {k: round(v, 4) for k, v in res["weights"].items()})
    print("CR:", round(res["consistency"]["cr"], 4), "(đạt)" if res["consistency"]["ok"] else "(KHÔNG ĐẠT)")
    print("Xếp hạng:", res["ranking"])
    print("\n".join(res["statements"]))
    return 0 if res["consistency"]["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
