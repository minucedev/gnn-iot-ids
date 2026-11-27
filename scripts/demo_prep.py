"""Chuẩn bị demo (task 8.2, notebook 22a_demo_prep): cắt 3 đoạn D1, D2, D3 từ NF-ToN-IoT-v3.csv, kiểm tra, chạy bộ phát hiện, xuất demo_replay.json.

Chạy (CPU, trên Kaggle):
    python -I scripts/demo_prep.py --config detector_config.json --out /kaggle/working --commit <hash>

Các bước (kịch bản report/KICH_BAN_DEMO.md, mục 3 và 7):
  1. Đọc CSV gốc theo khối, giữ mọi flow trong ba khoảng thời gian (không lấy mẫu); ghi demo_data/D1.csv, D2.csv, D3.csv.
  2. Tìm mốc tấn công thật (flow đầu tiên của lớp tấn công từ mốc kịch bản); kiểm tra phần trước mốc chỉ có Normal, nếu không thì dời mốc bắt đầu
     tới cửa sổ 5 giây ngay sau flow lạ cuối cùng và ghi lại (segments.json: start_moved).
  3. Làm sạch file đoạn bằng đúng hàm offline, kiểm mọi cửa sổ thuộc tập test.
  4. Chạy `detector.main` (có đối chiếu đồ thị offline để biết nhãn thật) trên từng đoạn với hybrid chính và hybrid nhẹ; tính thời gian phát hiện
     bằng đúng hàm `realtime_eval.episodes` và so với số trong kịch bản.
  5. Ghi demo_replay.json (từng cửa sổ, đồ thị cửa sổ có cảnh báo đầu tiên từ mốc tấn công, bảng tốc độ, bảng giới hạn), segments.json, segments_result.csv.
Không sửa mô hình, cấu hình, ngưỡng đã đóng băng.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import build_graph as bg  # noqa: E402
from src import demo_replay as DR  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PLAN = {
    "D1": dict(title="Yên tĩnh rồi DDoS", cls="DoS_DDoS", start="2019-04-24T20:07:00", end="2019-04-24T20:12:00", onset="2019-04-24T20:09:15", doc_s=5.0),
    "D2": dict(title="Dò quét", cls="Truy_cap_trai_phep", start="2019-04-26T18:29:30", end="2019-04-26T18:33:00", onset="2019-04-26T18:30:45", doc_s=5.1),
    "D3": dict(title="Backdoor", cls="Botnet_Backdoor", start="2019-04-28T15:27:00", end="2019-04-28T15:31:00", onset="2019-04-28T15:28:35", doc_s=4.4),
}
MARGIN_S = 600                      # lấy rộng thêm 10 phút phía trước để còn dời mốc bắt đầu
TS_COL, LAB_COL = "FLOW_START_MILLISECONDS", "Attack"
WINDOW_S = 5
MIN_CALM_S = 30                     # phần yên tĩnh tối thiểu trước mốc tấn công


def epoch(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def iso(t):
    return datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat(timespec="seconds")


def jdefault(o):
    return o.item() if hasattr(o, "item") else str(o)


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "không rõ"


@contextmanager
def step(info, name):
    t0 = time.time()
    try:
        yield
    finally:
        info["steps_s"][name] = round(time.time() - t0, 1)
        print(f"--- {name}: {info['steps_s'][name]} s", flush=True)


# ---------------------------------------------------------------- 1. cắt đoạn
def read_ranges(csv_path, chunk, max_chunks=None):
    """Đọc CSV theo khối (mọi cột là chuỗi, giữ nguyên giá trị) và giữ flow trong khoảng mở rộng của từng đoạn. Trả về (RAW, tên cột ts, tên cột nhãn, số dòng đã đọc)."""
    lo = {k: (epoch(v["start"]) - MARGIN_S) * 1000 for k, v in PLAN.items()}
    hi = {k: epoch(v["end"]) * 1000 for k, v in PLAN.items()}
    parts = {k: [] for k in PLAN}
    n_read, tcol, lcol = 0, None, None
    for i, df in enumerate(pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_filter=False, chunksize=chunk, encoding="utf-8-sig")):
        if max_chunks and i >= max_chunks:
            break
        if tcol is None:
            names = {c.strip(): c for c in df.columns}
            tcol, lcol = names[TS_COL], names[LAB_COL]
        n_read += len(df)
        t = pd.to_numeric(df[tcol], errors="coerce")
        for k in PLAN:
            m = ((t >= lo[k]) & (t < hi[k])).to_numpy()
            if m.any():
                parts[k].append(df[m])
        if i % 20 == 0:
            print(f"    đã đọc {n_read:,} dòng", flush=True)
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}, tcol, lcol, n_read


def plan_segment(k, raw, tcol, lcol, data_dir):
    """Mốc tấn công thật, kiểm tra phần trước mốc, dời mốc bắt đầu nếu cần, ghi file đoạn. Trả về dict mô tả đoạn."""
    p = PLAN[k]
    tm = pd.to_numeric(raw[tcol]).to_numpy(np.int64)
    t = tm / 1000.0
    lab = np.asarray(bg.to_group(raw[lcol]))
    nominal = epoch(p["onset"])
    cand = t[(lab == p["cls"]) & (t >= nominal)]          # mốc kịch bản là đầu một cửa sổ 5 s; mốc thật = flow tấn công đầu tiên từ đó (khớp cách `realtime_eval` tính đợt)
    if not len(cand):
        raise RuntimeError(f"{k}: không có flow {p['cls']} từ mốc {p['onset']}")
    onset = float(cand.min())
    if onset - nominal >= 2.0:
        print(f"    CẢNH BÁO {k}: flow {p['cls']} đầu tiên lúc {iso(onset)}, muộn hơn mốc kịch bản {p['onset']} hơn 2 s")
    w_on = math.floor(onset / WINDOW_S) * WINDOW_S         # đầu cửa sổ chứa mốc tấn công: phần trước đó phải chỉ có Normal
    start0, end = epoch(p["start"]), epoch(p["end"])
    bad = (t >= start0) & (t < w_on) & (lab != "Normal")
    near = (t >= onset - 120) & (t < onset) & (lab != "Normal")
    print(f"{k}: flow không phải Normal trong 120 s trước mốc thật {iso(onset)}: {int(near.sum())}", flush=True)
    for i in np.flatnonzero(near)[:40]:
        print(f"    {t[i] - onset:+9.3f} s so với mốc | {lab[i]} | {raw[lcol].iloc[i]}")
    start, moved, violation = start0, None, None
    if bad.any():                                      # có lưu lượng tấn công/khác trước mốc: dời mốc bắt đầu tới cửa sổ ngay sau flow đó nếu còn đủ đoạn yên tĩnh
        last_bad = float(t[bad].max())
        new_start = math.ceil((last_bad + 1e-3) / WINDOW_S) * WINDOW_S
        info = {"n_non_normal_before": int(bad.sum()), "labels": pd.Series(lab[bad]).value_counts().to_dict(), "last_non_normal": iso(last_bad),
                "offsets_s": [round(float(x), 3) for x in (t[bad] - onset)[:40]]}
        if w_on - new_start >= MIN_CALM_S:
            start, moved = new_start, {"from": p["start"], "to": iso(new_start), **info}
        else:
            violation = {"would_move_to": iso(new_start), **info}   # không còn đủ đoạn yên tĩnh: giữ mốc, ghi lại để quyết định
            print(f"    CẢNH BÁO {k}: không dời được mốc bắt đầu (chỉ còn {w_on - new_start:.0f} s yên tĩnh): {violation}", flush=True)
    sel = (tm >= start * 1000) & (tm < end * 1000)
    path = data_dir / f"{k}.csv"
    raw[sel].to_csv(path, index=False, encoding="utf-8")
    pre = pd.Series(lab[(t >= start) & (t < w_on)]).value_counts().to_dict()
    post = pd.Series(lab[(t >= onset) & (t < end)]).value_counts().to_dict()
    seg = dict(id=k, title=p["title"], cls=p["cls"], start=iso(start), start_planned=p["start"], end=p["end"], onset_nominal=p["onset"], onset_ts=onset,
               onset=datetime.fromtimestamp(onset, tz=timezone.utc).isoformat(timespec="milliseconds"),
               onset_window=iso(w_on), start_moved=moved, calm_violation=violation, n_flows=int(sel.sum()), file=str(path), doc_s=p["doc_s"],
               flows_before_onset=pre, flows_after_onset=post, duration_s=end - start, calm_s=w_on - start)
    print(f"{k} {p['title']}: {iso(start)} → {p['end']}, mốc thật {seg['onset']}, {seg['n_flows']:,} flow ({os.path.getsize(path) / 1024:.0f} KB)")
    print(f"    trước mốc: {pre} | từ mốc: {post}" + (f" | ĐÃ DỜI MỐC BẮT ĐẦU: {moved}" if moved else (" | CÓ FLOW LẠ TRƯỚC MỐC, KHÔNG DỜI ĐƯỢC" if violation else " | phần trước mốc chỉ có Normal, không dời")), flush=True)
    return seg


# ---------------------------------------------------------------- 3. kiểm tra file đoạn
def verify_segment(seg, test_wins, dataset):
    """Làm sạch file đoạn bằng đúng hàm offline; mọi cửa sổ thuộc tập test; trước mốc chỉ có Normal."""
    flows, classes, _ = bg.load_clean(dataset, seg["file"])
    wins = np.unique(np.floor(flows["ts"].to_numpy() / WINDOW_S).astype(np.int64))
    outside = [int(w) for w in wins if int(w) not in test_wins]
    pre = flows[flows["ts"] < math.floor(seg["onset_ts"] / WINDOW_S) * WINDOW_S]
    chk = {"n_flows_clean": int(len(flows)), "n_windows": int(len(wins)), "classes": list(classes), "windows_outside_test": len(outside),
           "labels_before_onset": sorted(set(pre["label"])), "n_before_onset": int(len(pre)), "dropped_by_clean": int(seg["n_flows"] - len(flows))}
    print(seg["id"], chk, flush=True)
    if outside:
        raise RuntimeError(f"{seg['id']}: {len(outside)} cửa sổ không thuộc tập test")
    chk["normal_only_before_onset"] = set(pre["label"]) == {"Normal"}
    if not chk["normal_only_before_onset"]:
        print(f"    CẢNH BÁO {seg['id']}: trước cửa sổ chứa mốc tấn công còn nhãn {set(pre['label'])} ({int((pre['label'] != 'Normal').sum())} flow); xem segments.json")
    if chk["dropped_by_clean"] > 0.01 * seg["n_flows"]:
        raise RuntimeError(f"{seg['id']}: làm sạch bỏ quá nhiều flow ({chk['dropped_by_clean']})")
    return chk


# ---------------------------------------------------------------- 4. chạy bộ phát hiện
def run_one(DET, RE, sess, seg, model, tmp, cfg_path, seed):
    out = Path(tmp) / f"{seg['id']}_{model}"
    out.mkdir(parents=True, exist_ok=True)
    DET.main(["--config", cfg_path, "--model", model, "--seed", str(seed), "--out", str(out), "--flows-csv", seg["file"], "--partial-flows",
              "--start", seg["start"], "--end", seg["end"], "--alert-from", seg["onset_window"], "--dump-first-alert", str(out / "first_alert.json")], sess)
    win = pd.read_csv(out / "windows.csv")
    alerts = pd.read_csv(out / "alerts.csv")
    k_cls = sess.classes.index(seg["cls"])
    ep = DR.pick_episode(RE.episodes(win, k_cls, 60.0, WINDOW_S), seg["onset_ts"])
    w0 = math.floor(seg["onset_ts"] / WINDOW_S) * WINDOW_S
    calm = win[(win["t_start"] < w0) & (win["n_alert"] > 0)]
    after = win[(win["t_start"] >= w0) & (win["n_alert"] > 0)]
    return dict(out=str(out), win=win, alerts=alerts, ep=ep, calm_alerts=[{"t": DR.hms(r.t_close), "n_alert": int(r.n_alert)} for r in calm.itertuples()],
                first_alert_t=None if after.empty else DR.hms(after.iloc[0]["t_close"]),
                agreement=dict(n_windows=int(len(win)), n_ref=int((win["ref_found"] == 1).sum()), n_mis_thr=int(win["n_mis_thr"].sum()),
                               n_mis_arg=int(win["n_mis_arg"].sum()), struct_bad=int((win["struct_ok"] == 0).sum()), max_dx=float(win["max_dx"].max()),
                               max_de=float(win["max_de"].max()), max_dproba=float(win["max_dproba"].max())),
                p95_ms=float(np.percentile(win["total_ms"], 95)))


def result_table(SEG, RUNS, models):
    rows = []
    for k, s in SEG.items():
        for m in models:
            R = RUNS[(k, m)]
            d = R["ep"]["detection_s"]
            rows.append({"đoạn": k, "mô hình": m, "mốc tấn công thật": s["onset"][11:23], "phát hiện": R["ep"]["detected"],
                         "cảnh báo đầu tiên từ mốc": R["first_alert_t"], "thời gian phát hiện (s)": None if d is None else round(d, 3),
                         "kịch bản (s)": s["doc_s"], "chỉ Normal trước mốc": s["check"]["normal_only_before_onset"], "báo nhầm trước mốc (cửa sổ)": len(R["calm_alerts"]), "p95 ms/cửa sổ": round(R["p95_ms"], 1),
                         "lệch offline (node)": R["agreement"]["n_mis_thr"], "cửa sổ lệch cấu trúc": R["agreement"]["struct_bad"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. demo_replay.json
def build_replay(SEG, RUNS, SESS, models, meta_extra):
    speed = DR.speed_numbers(ROOT)
    replay = {"meta": {**meta_extra, "generated_utc": iso(time.time()), "dataset": "nf_ton_iot_v3", "split": "test", "window_sec": WINDOW_S, "speedup_default": 5,
                       "main_model": models[0], "light_model": models[1], "threads": 1,
                       "alert_definition": "p = 1 - P(Normal), vượt ngưỡng đã đóng băng",
                       "models": {m: {"threshold": SESS[m].model.threshold, "n_params": SESS[m].model.n_params, "size_kb": SESS[m].model.size_kb} for m in models}},
              "segments": [], "speed": {"numbers": speed, "rows": DR.speed_rows(speed)}, "limits": DR.limits_rows(ROOT)}
    for k, s in SEG.items():
        runs = {}
        for m in models:
            R = RUNS[(k, m)]
            fa = json.loads(Path(R["out"], "first_alert.json").read_text(encoding="utf-8"))
            if fa is None:
                raise RuntimeError(f"{k} {m}: không có cảnh báo nào từ mốc tấn công")
            runs[m] = {"windows": DR.pack_windows(R["win"], R["alerts"]), "first_alert_graph": DR.shrink_graph(fa), "detected": R["ep"]["detected"],
                       "detection_s": R["ep"]["detection_s"], "first_alert_t": R["first_alert_t"], "calm_alerts": R["calm_alerts"]}
        replay["segments"].append({"id": k, "title": s["title"], "cls": s["cls"], "start": s["start"], "end": s["end"], "onset": s["onset"],
                                   "onset_t": s["onset"][11:19], "n_flows": s["n_flows"], "start_moved": s["start_moved"], "runs": runs})
    return replay


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="detector_config.json (flows_csv = NF-ToN-IoT-v3.csv gốc)")
    ap.add_argument("--out", required=True, help="thư mục ghi demo_replay.json, segments.json, demo_data/")
    ap.add_argument("--tmp", default="/tmp/prep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--commit", default="")
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--max-chunks", type=int, default=None, help="chỉ để thử nhanh quy trình, không dùng cho kết quả thật")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    out, data_dir = Path(a.out), Path(a.out) / "demo_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    models = [json.loads((ROOT / "reports/tables/ahp_result.json").read_text(encoding="utf-8"))["recommended"], "hybrid_light"]
    info = {"steps_s": {}, "models": models, "cpu": cpu_model(), "commit": a.commit}

    with step(info, "doc_csv_cat_doan"):
        print(f"Đọc {cfg['flows_csv']} ({os.path.getsize(cfg['flows_csv']) / 2**30:.2f} GB)", flush=True)
        RAW, tcol, lcol, n_read = read_ranges(cfg["flows_csv"], a.chunk, a.max_chunks)
    info["csv_rows_read"] = n_read
    print(f"Đã đọc {n_read:,} dòng CSV; flow trong khoảng mở rộng:", {k: len(v) for k, v in RAW.items()}, flush=True)

    SEG = {k: plan_segment(k, RAW[k], tcol, lcol, data_dir) for k in PLAN}
    del RAW

    meta = json.loads((Path(cfg["datasets"][cfg["dataset"]]["data"]) / "split.json").read_text(encoding="utf-8"))
    test_wins = set(int(w) for w in meta["test"]["windows"])
    with step(info, "kiem_tra_file_doan"):
        for s in SEG.values():
            s["check"] = verify_segment(s, test_wins, cfg["dataset"])

    DET, RE = _load_script("detector"), _load_script("realtime_eval")
    SESS = {}
    for m in models:
        with step(info, f"nap_{m}"):
            SESS[m] = DET.prepare(cfg, m, a.seed, reference=True)
    RUNS = {}
    for k, s in SEG.items():
        for m in models:
            with step(info, f"chay_{k}_{m}"):
                RUNS[(k, m)] = run_one(DET, RE, SESS[m], s, m, a.tmp, a.config, a.seed)

    tab = result_table(SEG, RUNS, models)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n" + tab.to_string(index=False), flush=True)
    for k, s in SEG.items():
        s["detection"] = {m: {"detected": RUNS[(k, m)]["ep"]["detected"], "detection_s": RUNS[(k, m)]["ep"]["detection_s"], "calm_alerts": RUNS[(k, m)]["calm_alerts"],
                              "first_alert_t": RUNS[(k, m)]["first_alert_t"], "agreement": RUNS[(k, m)]["agreement"], "p95_ms": RUNS[(k, m)]["p95_ms"]} for m in models}
        for m in models:
            d = RUNS[(k, m)]["ep"]["detection_s"]
            if d is not None and abs(d - s["doc_s"]) > 0.05:
                print(f"LỆCH KỊCH BẢN: {k} {m}: đo được {d:.3f} s, kịch bản ghi {s['doc_s']} s")

    replay = build_replay(SEG, RUNS, SESS, models, {"commit": a.commit, "seed": a.seed, "cpu": info["cpu"]})
    (out / "demo_replay.json").write_text(json.dumps(replay, ensure_ascii=False, separators=(",", ":"), default=jdefault), encoding="utf-8")
    (out / "segments.json").write_text(json.dumps(SEG, ensure_ascii=False, indent=1, default=jdefault), encoding="utf-8")
    tab.to_csv(out / "segments_result.csv", index=False)
    info["result_table"] = tab.to_dict("records")
    info["replay_json_kb"] = round((out / "demo_replay.json").stat().st_size / 1024, 1)
    (out / "demo_prep_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=jdefault), encoding="utf-8")
    print("demo_replay.json:", info["replay_json_kb"], "KB;", {s["id"]: {m: len(s["runs"][m]["windows"]) for m in models} for s in replay["segments"]})
    for s in replay["segments"]:
        for m in models:
            g = s["runs"][m]["first_alert_graph"]
            print(f"Đồ thị cửa sổ cảnh báo đầu tiên {s['id']} {m}: đóng lúc {g['t']}, {g['n_nodes_total']} node, {g['n_edges_total']} cạnh, {g['n_alert_total']} node bị cảnh báo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
