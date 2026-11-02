"""Thống kê đồ thị và số node mỗi lớp ở train/val/test cho từng cách chia (số liệu cho quyết định D3).

Chạy:  python -I scripts/split_stats.py <dataset> <csv> <window> <stride> <out.json>
Ví dụ: python -I scripts/split_stats.py nf_ton_iot_v3 NF-ToN-IoT-v3.csv 5 5 out.json
"""

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build_graph as bg  # noqa: E402


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20   # KB trên Linux


def main():
    name, csv, window, stride, out_path = sys.argv[1:6]
    window, stride = int(window), int(stride)
    result = {"dataset": name, "window": window, "stride": stride}

    t0 = time.time()
    flows, classes, log = bg.load_clean(name, csv)
    result["classes"] = classes
    result["clean_log"] = log.to_dict(orient="records")
    result["flow_class_counts"] = {k: int(v) for k, v in flows["label"].value_counts().items()}
    result["ts_range_utc"] = [str(pd.to_datetime(flows["ts"].iloc[0], unit="s")),
                              str(pd.to_datetime(flows["ts"].iloc[-1], unit="s"))]
    print(f"Làm sạch {time.time() - t0:.0f} s, RAM đỉnh {rss_gb():.1f} GB | lớp: {classes}")
    print(log.to_string())

    t0 = time.time()
    nodes, edges = bg.build_tables(flows, window, stride)
    del flows
    wins = np.unique(nodes["win"].to_numpy())
    print(f"Dựng bảng node/cạnh {time.time() - t0:.0f} s, RAM đỉnh {rss_gb():.1f} GB | "
          f"{len(wins):,} cửa sổ, {len(nodes):,} node, {len(edges):,} cạnh")

    n_per = nodes.groupby("win").size()
    e_per = edges.groupby("win").size()
    q = [0.5, 0.95, 0.99, 1.0]
    result["graph_stats"] = {
        "n_windows": int(len(wins)), "n_nodes": int(len(nodes)), "n_edges": int(len(edges)),
        "nodes_per_window": {f"p{int(x * 100)}": float(n_per.quantile(x)) for x in q},
        "edges_per_window": {f"p{int(x * 100)}": float(e_per.quantile(x)) for x in q},
        "node_class_counts": {classes[int(k)]: int(v) for k, v in nodes["y"].value_counts().items()},
    }
    print(json.dumps(result["graph_stats"], indent=1, ensure_ascii=False))

    gap = window // stride - 1
    result["splits"] = {}
    for mode in ("chronological", "blocked"):
        sp = bg.time_split(wins, bg.SPLIT_RATIO, gap, mode=mode)
        rows = {}
        for part, ws in sp.items():
            sub = nodes.loc[nodes["win"].isin(ws), "y"]
            cnt = sub.value_counts()
            rows[part] = {"n_windows": int(len(ws)), "n_nodes": int(len(sub)),
                          **{c: int(cnt.get(i, 0)) for i, c in enumerate(classes)}}
        result["splits"][mode] = rows
        print(f"\n== {mode}")
        print(pd.DataFrame(rows).T.to_string())

    result["peak_ram_gb"] = round(rss_gb(), 1)
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Đã ghi", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
