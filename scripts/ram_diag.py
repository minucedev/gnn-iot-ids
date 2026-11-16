"""Chẩn đoán RAM tăng khi phát lại luồng (task 8.1): tách từng thành phần và đo heap native, đối tượng Python, RSS mỗi 200 cửa sổ.

Chạy (Kaggle CPU, cùng detector_config.json của notebook 20):
    python -I scripts/ram_diag.py --config detector_config.json --model hybrid_gat_lgbm_oofavg_il --phase stream --tag A --out diag_out \
        [--max-windows 4000] [--keep-rows] [--trim-every 200] [--flows-cache /tmp/flows_test.parquet]

Các pha (mỗi pha chạy trong MỘT tiến trình riêng để RSS không lẫn nhau):
  model_only   lặp lại suy luận mô hình trên các đồ thị test đã dựng sẵn (không dựng đồ thị từ flow)
  build_only   phát lại flow, gom cửa sổ, dựng đồ thị rồi bỏ, không suy luận
  stream       toàn bộ đường của detector.run_stream (không nạp đồ thị offline tham chiếu); --keep-rows tái hiện bản cũ (giữ dict mỗi cửa sổ),
               --trim-every N gọi malloc_trim sau mỗi N cửa sổ

Ghi vào --out: probe_<tag>.csv (mỗi 200 cửa sổ: rss_mb, in_use_mb, free_mb, heap_arena_mb, py_blocks, gc_objects) và diag_<tag>.json (tóm tắt:
tăng RSS cuối so với đầu, độ dốc MB mỗi 1.000 cửa sổ ở nửa sau, tương tự cho in_use_mb và py_blocks).
Không đổi mô hình, ngưỡng hay cách dự đoán.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_detector():
    spec = importlib.util.spec_from_file_location("detector_mod", ROOT / "scripts" / "detector.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load_detector()
SF = D.SF
EVERY = 200


def load_flows(cfg, ctx, cache):
    """Flow test đã làm sạch (như detector); có cache parquet để các pha sau khỏi đọc lại CSV 27 triệu dòng."""
    if cache and Path(cache).exists():
        return pd.read_parquet(cache)
    flows = D.load_stream_flows(cfg, ctx, "test")
    if cache:
        flows.to_parquet(cache)
    return flows


def summarize(rows, warmup=50):
    df = pd.DataFrame(rows)
    out = {"n_probes": int(len(df))}
    if len(df) < 6:
        return out
    n = df["n"].to_numpy(dtype=float)
    k = max(1, int(len(df) * 0.1))
    for col in ("rss_mb", "in_use_mb", "free_mb", "heap_arena_mb", "py_blocks", "gc_objects"):
        if col not in df:
            continue
        v = df[col].to_numpy(dtype=float)
        half = len(df) // 2
        slope = float(np.polyfit(n[half:] / 1000.0, v[half:], 1)[0])
        early, late = float(np.median(v[1:1 + k])), float(np.median(v[-k:]))
        out[col] = {"first": float(v[0]), "early_median": early, "late_median": late, "growth_late_vs_early": late / early - 1 if early else None,
                    "slope_per_1000_windows_second_half": slope}
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="hybrid_gat_lgbm_oofavg_il")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase", required=True, choices=["model_only", "build_only", "stream"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-windows", type=int, default=4000)
    ap.add_argument("--keep-rows", action="store_true")
    ap.add_argument("--trim-every", type=int, default=0)
    ap.add_argument("--flows-cache", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    ds = cfg["dataset"]
    ctx = D.FT.Context(cfg, ds, cfg["datasets"][ds], cfg.get("split", "test"))
    model = D.OnlineModel(ctx, a.model, a.seed)
    model.single_thread()
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    window_sec, classes, scaler = ctx.meta["window_sec"], ctx.classes, ctx.scaler
    target = ctx.target                                                    # đồ thị test đã dựng sẵn (cho pha model_only)
    for attr in ("train", "val", "Xtr", "ytr", "Xva", "yva", "Ftr_nbr", "window", "target_raw"):
        setattr(ctx, attr, None)
    if a.phase != "model_only":
        ctx.target = None
        target = None
    D.gc.collect()
    D.trim_memory()

    flows = load_flows(cfg, ctx, a.flows_cache) if a.phase != "model_only" else None
    if flows is not None:
        D.gc.collect()
        D.trim_memory()
    rows = []

    def probe(n):
        if n == 1 or n % EVERY == 0:
            rows.append({"n": n, **D.heap_stats()})

    print(f"[diag {a.tag}] pha {a.phase}, mô hình {a.model}, RAM trước khi chạy {D.rss_mb():.0f} MB", flush=True)
    t0 = time.time()
    if a.phase == "model_only":
        n = len(target)
        for i in range(a.max_windows):
            model.proba_window(target[i % n])
            probe(i + 1)
    elif a.phase == "build_only":
        asm = SF.WindowAssembler(window_sec)
        n = 0
        for batch in SF.replay_batches(flows, 1.0):
            for cw in asm.push(batch):
                g = SF.build_window_graph(cw.win, cw.flows(), scaler, window_sec)
                if g is None:
                    continue
                n += 1
                probe(n)
            if n >= a.max_windows:
                break
    else:
        D.run_stream(model, flows, scaler, classes, out / f"stream_{a.tag}", None, window_sec, 1.0, None, a.max_windows,
                     trim_every=a.trim_every, probe=probe, keep_rows=a.keep_rows)
    secs = time.time() - t0
    pd.DataFrame(rows).to_csv(out / f"probe_{a.tag}.csv", index=False)
    summ = {"tag": a.tag, "phase": a.phase, "model": a.model, "max_windows": a.max_windows, "keep_rows": a.keep_rows, "trim_every": a.trim_every,
            "seconds": round(secs, 1), **summarize(rows)}
    (out / f"diag_{a.tag}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    r = summ.get("rss_mb", {})
    iu = summ.get("in_use_mb", {})
    print(f"[diag {a.tag}] xong {secs:.0f} s | RSS sớm {r.get('early_median', float('nan')):.0f} → muộn {r.get('late_median', float('nan')):.0f} MB "
          f"({(r.get('growth_late_vs_early') or 0):+.1%}), độ dốc {r.get('slope_per_1000_windows_second_half', float('nan')):+.1f} MB/1000 cửa sổ | "
          f"in_use độ dốc {iu.get('slope_per_1000_windows_second_half', float('nan')):+.1f} MB/1000", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
