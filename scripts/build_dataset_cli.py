"""Dựng và lưu đồ thị chính thức của một bộ dữ liệu (task 1.5, 1.6).

Chạy:  python -I scripts/build_dataset_cli.py <dataset> <csv> <window> <stride> <mode> <gap> <outdir> [<tag>]
Ví dụ: python -I scripts/build_dataset_cli.py nf_ton_iot_v3 NF-ToN-IoT-v3.csv 5 5 blocked 1 processed/nf_ton_iot_v3

Ghi vào <outdir>: graphs_<tag>.pt, split.json, scaler.pkl, summary.json (số node mỗi lớp mỗi tập, hash).
Quy ước (KE_HOACH.md mục 6): scaler chỉ fit trên train; mọi tập dùng chung split.json.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build_graph as bg  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    name, csv, window, stride, mode, gap, outdir = sys.argv[1:8]
    window, stride, gap = int(window), int(stride), int(gap)
    tag = sys.argv[8] if len(sys.argv) > 8 else f"{name}_w{window}s{stride}_{mode}"
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    flows, classes, log = bg.load_clean(name, csv)
    ts_is_order = bg.TS_IS_ORDER.get(name, False)
    print(f"Làm sạch {time.time() - t0:.0f} s | lớp: {classes}")
    print(log.to_string())

    t0 = time.time()
    graphs, split, scaler, _ = bg.build_dataset(flows, window, stride, mode=mode, gap=gap)
    print(f"Dựng {len(graphs):,} đồ thị trong {time.time() - t0:.0f} s")

    win_pos = {g.win: i for i, g in enumerate(graphs)}
    parts = {}
    for part, ws in split.items():
        parts[part] = {"graph_idx": [win_pos[int(w)] for w in ws], "windows": [int(w) for w in ws]}
        if not ts_is_order:
            parts[part]["t_start_utc"] = str(pd.to_datetime(ws[0] * stride, unit="s"))
            parts[part]["t_end_utc"] = str(pd.to_datetime(ws[-1] * stride + window, unit="s"))

    fp = bg.fingerprint(graphs)
    graph_file = out / f"graphs_{tag}.pt"
    torch.save(graphs, graph_file)
    split_json = {
        "dataset": name, "graph_file": graph_file.name, "seed": 42,
        "ts_unit": "flow" if ts_is_order else "giây",
        "ts_note": "ts = số thứ tự dòng (giả định theo thời gian)" if ts_is_order else "ts = giây epoch (FLOW_START_MILLISECONDS / 1000)",
        "split_mode": mode, "split_blocks": bg.SPLIT_BLOCKS if mode == "blocked" else None,
        "window_sec": window, "stride_sec": stride, "ratios": bg.SPLIT_RATIO, "gap_windows": gap,
        "classes": classes, "features": bg.FEATURES, "log1p_features": bg.LOG_FEATURES,
        "edge_features": bg.EDGE_FEATURES, "edge_direction": "src->dst (có hướng)",
        "node_label_rule": "đa số nhãn flow IP khởi tạo; nếu không có thì flow IP nhận; hòa → lớp lớn hơn",
        "fingerprint_sha256": fp, **parts,
    }
    (out / "split.json").write_text(json.dumps(split_json, ensure_ascii=False), encoding="utf-8")
    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    rows = {}
    for part, p in parts.items():
        gs = [graphs[i] for i in p["graph_idx"]]
        y = torch.cat([g.y for g in gs]).numpy()
        rows[part] = {"n_graphs": len(gs), "n_nodes": int(len(y)),
                      **{c: int((y == i).sum()) for i, c in enumerate(classes)}}
    summary = {"dataset": name, "tag": tag, "n_graphs": len(graphs), "fingerprint_sha256": fp,
               "clean_log": log.to_dict(orient="records"), "split_counts": rows,
               "files": {p.name: p.stat().st_size for p in out.iterdir()}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(pd.DataFrame(rows).T.to_string())
    print("Hash:", fp)
    for n, s in summary["files"].items():
        print(f"{n:50s} {s / 2**20:8.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
