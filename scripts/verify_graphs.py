"""Kiểm tra lại đồ thị đã lưu (graphs_*.pt, split.json, scaler.pkl) bằng src/checks.py, không dựng lại.

Chạy:  python -I scripts/verify_graphs.py <data_dir> [<out.json>]
Thoát mã 3 nếu có kiểm tra không đạt.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checks import check_graphs  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d = Path(sys.argv[1])
    meta = json.loads((d / "split.json").read_text(encoding="utf-8"))
    graphs = torch.load(d / meta["graph_file"], weights_only=False)
    scaler = pickle.loads((d / "scaler.pkl").read_bytes())
    split = {p: np.array(meta[p]["windows"]) for p in ("train", "val", "test")}
    checks = check_graphs(graphs, split, scaler, meta["classes"], meta["gap_windows"])
    for name, ok, detail in checks:
        print(f"[{'ĐẠT' if ok else 'KHÔNG ĐẠT'}] {name} ({detail})")
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps([{"name": n, "ok": ok, "detail": dt} for n, ok, dt in checks],
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if all(ok for _, ok, _ in checks) else 3


if __name__ == "__main__":
    sys.exit(main())
