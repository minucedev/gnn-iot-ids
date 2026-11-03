"""Tìm cấu hình chia tập đạt tiêu chí task 1.5, chỉ dựa trên đếm nhãn (không dùng kết quả mô hình).

Chạy:  python -I scripts/split_search.py <dataset> <csv> <window> <stride> <out.json>
Dựng bảng node một lần, rồi thử: chronological, và blocked với blocks thuộc {10, 20, 50, 100, 200}, mỗi cái chia theo
số cửa sổ hoặc theo số node. Với mỗi cấu hình ghi số node mỗi lớp mỗi tập, tỉ lệ node, và đạt hay không:
  (a) mỗi lớp chính (Normal, DoS_DDoS, Botnet_Backdoor, Truy_cap_trai_phep) >= 100 node ở train, >= 30 ở val và test;
  (b) tỉ lệ node train/val/test nằm trong 70/15/15 ± 3 điểm %.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build_graph as bg  # noqa: E402

MAIN = ["Normal", "DoS_DDoS", "Botnet_Backdoor", "Truy_cap_trai_phep"]
GAP = 1


def class_ok(rows, classes):
    """Trả về danh sách lớp chính không đạt ngưỡng 100/30/30."""
    return [c for c in classes if c in MAIN
            and not (rows["train"][c] >= 100 and rows["val"][c] >= 30 and rows["test"][c] >= 30)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    name, csv, window, stride, out_path = sys.argv[1:6]
    window, stride = int(window), int(stride)
    flows, classes, _ = bg.load_clean(name, csv)
    nodes, _ = bg.build_tables(flows, window, stride)
    del flows

    wins, n_per = np.unique(nodes["win"].to_numpy(), return_counts=True)
    pos = np.searchsorted(wins, nodes["win"].to_numpy())
    per_win_cls = np.zeros((len(wins), len(classes)), dtype=np.int64)       # số node mỗi lớp trong mỗi cửa sổ
    np.add.at(per_win_cls, (pos, nodes["y"].to_numpy()), 1)
    win_pos = {int(w): i for i, w in enumerate(wins)}

    configs = [("chronological", 1, False), ("chronological", 1, True)]
    configs += [("blocked", b, bal) for b in (10, 20, 50, 100, 200) for bal in (False, True)]
    results = []
    for mode, blocks, bal in configs:
        sp = bg.time_split(wins, bg.SPLIT_RATIO, GAP, mode=mode, blocks=blocks, weights=n_per if bal else None)
        rows = {}
        for part, ws in sp.items():
            idx = np.array([win_pos[int(w)] for w in ws], dtype=int)
            cnt = per_win_cls[idx].sum(0) if len(idx) else np.zeros(len(classes), dtype=np.int64)
            rows[part] = {c: int(cnt[i]) for i, c in enumerate(classes)}
            rows[part]["n_nodes"] = int(cnt.sum())
            rows[part]["n_windows"] = int(len(idx))
        total = sum(r["n_nodes"] for r in rows.values())
        if total == 0 or blocks > len(wins):
            print(f"{mode:13s} blocks={blocks:3d} balance={int(bal)} | bỏ qua (số khối > số cửa sổ {len(wins)})")
            continue
        ratio = {k: 100 * r["n_nodes"] / total for k, r in rows.items()}
        miss = class_ok(rows, classes)
        ok_ratio = all(abs(ratio[k] - r) <= 3 for k, r in zip(("train", "val", "test"), (70, 15, 15)))
        results.append({"mode": mode, "blocks": blocks, "balance_nodes": bal, "ratio_pct": ratio,
                        "missing_classes": miss, "ok_classes": not miss, "ok_ratio": ok_ratio, "counts": rows})
        print(f"{mode:13s} blocks={blocks:3d} balance={int(bal)} | node {ratio['train']:.1f}/{ratio['val']:.1f}/"
              f"{ratio['test']:.1f} | lớp {'ĐẠT' if not miss else 'THIẾU ' + ','.join(miss)} | "
              f"tỉ lệ {'ĐẠT' if ok_ratio else 'lệch'}")
    Path(out_path).write_text(json.dumps({"dataset": name, "classes": classes, "window": window, "stride": stride,
                                          "gap": GAP, "configs": results}, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
