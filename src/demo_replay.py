"""Dữ liệu cho demo thời gian thực (task 8.2, kịch bản report/KICH_BAN_DEMO.md).

Đóng gói kết quả chạy `scripts/detector.py` trên từng đoạn demo thành `demo_replay.json` (cho trang `report/demo/demo_replay.html`),
và dựng bảng tốc độ/độ nhẹ (cảnh 4), bảng giới hạn (cảnh 5) từ các file kết quả đã có trong `reports/`.
Chỉ đọc và định dạng; không tính lại mô hình, không đổi ngưỡng.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MAIN_MODEL = "hybrid_gat_lgbm_oofavg_il"
LIGHT_MODEL = "hybrid_light"
ATTACK_CLASSES = ["DoS_DDoS", "Botnet_Backdoor", "Truy_cap_trai_phep", "Khac"]
CLASS_LABEL = {"DoS_DDoS": "DoS_DDoS", "Botnet_Backdoor": "Botnet_Backdoor", "Truy_cap_trai_phep": "Truy_cap_trai_phep", "Khac": "Khac (mitm, ransomware)"}
RT_DIR = "reports/kaggle/gnn-iot-ids-20-realtime"
MAX_ALERTS = 5                  # số cảnh báo giữ lại mỗi cửa sổ (mạnh nhất trước), giống số dòng in của detector
MAX_GRAPH_NODES = 120           # đồ thị cửa sổ cảnh báo lớn hơn thì rút gọn để vẽ


def hms(t):
    """Giờ UTC HH:MM:SS của giây epoch."""
    return datetime.fromtimestamp(float(t), tz=timezone.utc).strftime("%H:%M:%S")


def iso(t):
    """Giờ UTC ISO (đến giây) của giây epoch."""
    return datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat(timespec="seconds")


def vn(x, nd=0):
    """Số kiểu Việt Nam: dấu chấm ngăn nghìn, dấu phẩy thập phân."""
    return f"{x:,.{nd}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- đóng gói một lần chạy
def pack_windows(win, alerts, max_alerts=MAX_ALERTS):
    """Từ windows.csv và alerts.csv của detector: mỗi cửa sổ một dict {t, nodes, edges, ms, n_alert, alerts[{ip, cls, p}], truth}.

    `t` là giờ UTC lúc đóng cửa sổ; `p` = 1 - P(Normal); `alerts` chỉ giữ `max_alerts` cảnh báo mạnh nhất; `truth` = số node tấn công thật
    của cửa sổ (có khi chạy cùng đồ thị offline, nếu không thì None) để trang phát lại đánh dấu mốc tấn công.
    """
    by_win = {}
    if len(alerts):
        for w, g in alerts.sort_values("p_attack", ascending=False, kind="stable").groupby("window"):
            by_win[int(w)] = g
    has_truth = "n_true_attack" in win.columns
    rows = []
    for _, r in win.iterrows():
        g = by_win.get(int(r["win"]))
        top = [] if g is None else [{"ip": str(ip), "cls": str(c), "p": round(float(p), 3)}
                                    for ip, c, p in zip(g["ip"], g["pred_class"], g["p_attack"])][:max_alerts]
        rows.append({"t": hms(r["t_close"]), "t_close": int(r["t_close"]), "nodes": int(r["n_nodes"]), "edges": int(r["n_edges"]),
                     "ms": round(float(r["total_ms"]), 1), "n_alert": int(r["n_alert"]), "alerts": top,
                     "truth": int(r["n_true_attack"]) if has_truth else None})
    return rows


def shrink_graph(g, max_nodes=MAX_GRAPH_NODES):
    """Đồ thị một cửa sổ (từ `detector.graph_dump`) rút gọn để vẽ: giữ nguyên nếu ≤ max_nodes node.

    Lớn hơn thì giữ các node bị cảnh báo (nhiều cạnh nhất trước, tối đa một nửa) rồi các node nhiều cạnh nhất còn lại; chỉ giữ cạnh giữa các node
    được giữ. Ghi số node/cạnh gốc để trang hiển thị "đang vẽ N trong M node"."""
    nodes, edges = g["nodes"], g["edges"]
    n = len(nodes)
    deg = np.zeros(n, dtype=np.int64)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    attack = [i for i, x in enumerate(nodes) if x["pred"] != "Normal"]
    if n <= max_nodes:
        keep = list(range(n))
    else:
        top_attack = sorted(attack, key=lambda i: (-deg[i], i))[:max_nodes // 2]
        chosen = set(top_attack)
        rest = [i for i in sorted(range(n), key=lambda i: (-deg[i], i)) if i not in chosen][:max_nodes - len(top_attack)]
        keep = sorted(top_attack + rest)
    idx = {old: new for new, old in enumerate(keep)}
    return {"win": g["win"], "t": g["t_close"][11:19], "t_close": g["t_close"], "total_ms": g["total_ms"],
            "n_nodes_total": n, "n_edges_total": len(edges), "n_alert_total": len(attack), "truncated": n > max_nodes,
            "nodes": [{"ip": nodes[i]["ip"], "pred": nodes[i]["pred"], "p": round(float(nodes[i]["p_attack"]), 3)} for i in keep],
            "edges": [[idx[a], idx[b]] for a, b in edges if a in idx and b in idx]}


def pick_episode(episodes, onset_ts, tol=1.0):
    """Đợt tấn công (từ `realtime_eval.episodes`) bắt đầu tại mốc `onset_ts` (giây epoch, sai số ≤ tol giây)."""
    best = min(episodes, key=lambda e: abs(e["start_ts"] - onset_ts), default=None)
    if best is None or abs(best["start_ts"] - onset_ts) > tol:
        raise ValueError(f"Không có đợt nào bắt đầu tại {onset_ts} (các đợt: {[e['start_ts'] for e in episodes]})")
    return best


# ---------------------------------------------------------------- bảng cảnh 4: tốc độ và độ nhẹ
def speed_numbers(repo):
    """Số đo tốc độ/độ nhẹ của hai mô hình từ kết quả notebook 20 (reports/tables/latency.json và extra_hybrid_light)."""
    repo = Path(repo)
    srcs = {MAIN_MODEL: ("reports/tables/latency.json", f"{RT_DIR}/detector_summary.json", "reports/tables/realtime_eval.json"),
            LIGHT_MODEL: (f"{RT_DIR}/extra_hybrid_light/latency.json", f"{RT_DIR}/extra_hybrid_light/detector_summary.json",
                          f"{RT_DIR}/extra_hybrid_light/realtime_eval.json")}
    out = {}
    for model, (lat_p, sum_p, ev_p) in srcs.items():
        lat, sm, ev = _load(repo / lat_p), _load(repo / sum_p), _load(repo / ev_p)
        if not (lat["model"] == sm["model"] == ev["model"] == model):
            raise ValueError(f"File kết quả không đúng mô hình {model}: {lat['model']}, {sm['model']}, {ev['model']}")
        out[model] = {"p50_ms": lat["total_ms"]["p50"], "p95_ms": lat["total_ms"]["p95"], "build_p95_ms": lat["build_ms"]["p95"],
                      "infer_p95_ms": lat["infer_ms"]["p95"], "n_params": sm["n_params"], "size_kb": sm["size_kb"],
                      "flows_per_s": lat["throughput"]["flows_per_s"], "agreement": ev["online_vs_offline"]["pred_agreement_threshold"],
                      "n_windows": lat["n_windows_measured"], "limit_ms": lat["p95_limit_ms"], "threads": lat["threads"]}
    return out


def fmt_size(kb):
    return f"{vn(kb / 1000, 1)} MB" if kb >= 1000 else f"{vn(kb)} KB"


def speed_rows(nums):
    """Bảng cảnh 4 dạng [(chỉ tiêu, hybrid chính, hybrid nhẹ)] từ `speed_numbers`."""
    m, l = nums[MAIN_MODEL], nums[LIGHT_MODEL]

    def both(f):
        return f(m), f(l)

    rows = [("Độ trễ mỗi cửa sổ (trung vị / p95)", *both(lambda x: f"{vn(x['p50_ms'])} / {vn(x['p95_ms'])} ms")),
            ("Phần dựng đồ thị / phần dự đoán (p95)", *both(lambda x: f"{vn(x['build_p95_ms'])} / {vn(x['infer_p95_ms'])} ms")),
            ("Số tham số / dung lượng", *both(lambda x: f"{vn(x['n_params'])} / {fmt_size(x['size_kb'])}")),
            ("Thông lượng (flow mỗi giây, 1 luồng)", *both(lambda x: vn(x["flows_per_s"]))),
            ("Dự đoán online khớp offline", *both(lambda x: "100%" if round(100 * x["agreement"], 2) == 100 else f"{vn(100 * x['agreement'], 2)}%")),
            ("Ngưỡng độ trễ p95 / số luồng CPU", *both(lambda x: f"{vn(x['limit_ms'])} ms / {x['threads']}"))]
    return rows


# ---------------------------------------------------------------- bảng cảnh 5: giới hạn
def limits_rows(repo):
    """Bảng cảnh 5 từ reports/tables/undetected_episodes.json: mỗi nhóm tấn công một dòng {class, label, n, detected, note}, thêm dòng tổng."""
    u = _load(Path(repo) / "reports/tables/undetected_episodes.json")
    rows = []
    for cls in ATTACK_CLASSES:
        det = [e for e in u["episodes_detected"] if e["class"] == cls]
        und = [e for e in u["episodes_undetected"] if e["class"] == cls]
        n, nd = len(det) + len(und), len(det)
        if not und:
            med = float(np.median([e["detection_s"] for e in det])) if det else None
            note = "Phát hiện hết, thời gian phát hiện trung vị " + (f"{vn(med, 1)} giây" if med is not None else "không có")
        else:
            fl = [e["n_flows_class"] for e in und]
            single = sum(e["n_windows"] == 1 for e in und)
            note = f"{len(und)} đợt bỏ sót: {min(fl)}–{max(fl)} flow tấn công" if min(fl) != max(fl) else f"{len(und)} đợt bỏ sót: {fl[0]} flow tấn công"
            if single:
                note += f", {single} đợt chỉ kéo dài 1 cửa sổ"
            gan = sum(e["reason_primary"] == "bi_gan_normal" for e in und)
            note += f"; mô hình tin là Normal ở {gan}/{len(und)} đợt"
        rows.append({"class": cls, "label": CLASS_LABEL[cls], "n": n, "detected": nd, "note": note})
    rows.append({"class": "all", "label": "Tổng", "n": sum(r["n"] for r in rows), "detected": sum(r["detected"] for r in rows),
                 "note": f"{u['n_undetected']} đợt không phát hiện; {u['summary']['n_single_window']} đợt chỉ kéo dài 1 cửa sổ"})
    return rows
