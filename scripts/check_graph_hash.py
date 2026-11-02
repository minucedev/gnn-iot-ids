"""Dựng đồ thị từ một file NetFlow và so hash với giá trị mong đợi (task 0.4, tiêu chí 1).

Chạy:  python -I scripts/check_graph_hash.py <dataset> <csv> <window> <stride> <hash_16_ky_tu>
Ví dụ: python -I scripts/check_graph_hash.py nf_ton_iot_v2 NF-ToN-IoT-v2.csv 1000 1000 cb8842d60f9a3a5a
Thoát mã 1 nếu hash khác.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build_graph as bg  # noqa: E402


def main():
    name, csv, window, stride, expected = sys.argv[1:6]
    window, stride = int(window), int(stride)

    t0 = time.time()
    flows, classes, log = bg.load_clean(name, csv)
    print(f"Làm sạch {time.time() - t0:.0f} s | lớp: {classes}")
    print(log.to_string())

    t0 = time.time()
    graphs, split, _, _ = bg.build_dataset(flows, window, stride)
    fp = bg.fingerprint(graphs)
    print(f"Dựng {len(graphs):,} đồ thị trong {time.time() - t0:.0f} s")
    print({p: len(w) for p, w in split.items()})
    print("Hash:", fp[:16], "| mong đợi:", expected, "| trùng:", fp.startswith(expected))
    print("Hash đầy đủ:", fp)
    return 0 if fp.startswith(expected) else 1


if __name__ == "__main__":
    sys.exit(main())
