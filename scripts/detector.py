"""Bộ phát hiện xâm nhập thời gian thực (task 8.2, 8.3): phát lại flow, gom cửa sổ 5 giây, dựng đồ thị, dự đoán, ghi cảnh báo.

Chạy một lệnh (CPU, 1 luồng, không cần GPU):
    python -I scripts/detector.py --config detector_config.json --model hybrid_light --out demo_out [--max-windows 200]

`detector_config.json` có cấu trúc của config `scripts/final_test.py` (đường dẫn mô hình đã đóng băng, ngưỡng) cùng hai khóa:
  dataset     bộ dữ liệu, ví dụ "nf_ton_iot_v3"
  flows_csv   file flow NetFlow v3 gốc (cùng định dạng NF-ToN-IoT-v3.csv)
Notebook `notebooks/20_realtime.ipynb` sinh file này trên Kaggle.

Quy trình: nạp mô hình đã đóng băng và ngưỡng của nó (đúng code của final_test.py) → đọc và làm sạch flow bằng đúng hàm offline →
chỉ giữ flow của các cửa sổ thuộc tập test (`--restrict none` để phát lại mọi flow) → phát lại theo thứ tự thời gian
(`--speedup N` chờ cho thời gian thật chậm hơn thời gian dữ liệu N lần; mặc định không chờ) → `WindowAssembler` đóng cửa sổ 5 s →
`build_window_graph` (scaler đã fit trên train) → dự đoán → ngưỡng đã đóng băng.

Ghi vào --out:
  alerts.csv          mỗi node bị dự đoán là tấn công: t_close (UTC, thời điểm đóng cửa sổ theo thời gian dữ liệu), window, ip, pred_class,
                      p_attack (= 1 - P(Normal)), p_class (xác suất lớp dự đoán), proc_ms (dựng đồ thị + suy luận của cửa sổ đó)
  windows.csv         mỗi cửa sổ một dòng: số flow/node/cạnh, build_ms, infer_ms, total_ms, RAM, số cảnh báo; khi có đồ thị offline để đối chiếu
                      (tập test): nhãn thật, sai khác đặc trưng và dự đoán so với offline
  detector_summary.json  mô hình, ngưỡng, thời gian, môi trường

Độ trễ của một cửa sổ = từ lúc cửa sổ đóng (lấy flow ra khỏi bộ gom) đến lúc có dự đoán: `build_ms` (nối flow, nhóm theo IP, đặc trưng, chuẩn hóa,
dựng Data) + `infer_ms` (suy luận mô hình và áp ngưỡng). Việc ghi cảnh báo và đối chiếu offline nằm ngoài khoảng đo.
"""

import argparse
import csv
import ctypes
import gc
import importlib.util
import json
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import build_graph as bg  # noqa: E402
from src import hybrid as H  # noqa: E402
from src import models as M  # noqa: E402
from src import stream_features as SF  # noqa: E402
from src import threshold as TH  # noqa: E402
from src import train as T  # noqa: E402

N_WARMUP = 50                       # số cửa sổ đầu không tính vào thống kê độ trễ
TRIM_EVERY = 200                     # malloc_trim sau mỗi 200 cửa sổ, ngoài khoảng đo độ trễ (chẩn đoán RAM, task 8.1: notebook 20b-20d)
PERF = time.perf_counter


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FT = _load_script("final_test")


# ---------------------------------------------------------------- môi trường
def rss_mb():
    """RAM thường trú (RSS) của tiến trình, MB, đọc từ /proc (Linux)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError):
        return float("nan")


_LIBC = None


def _libc():
    global _LIBC
    if _LIBC is None:
        _LIBC = ctypes.CDLL("libc.so.6")
    return _LIBC


def trim_memory():
    """Trả vùng rỗi của bộ cấp phát glibc cho hệ điều hành (malloc_trim). Ngoài Linux/glibc thì không làm gì; trả về True nếu gọi được."""
    try:
        _libc().malloc_trim(0)
        return True
    except (OSError, AttributeError):
        return False


class _MallInfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


def heap_stats():
    """Số liệu chẩn đoán RAM: rss_mb; in_use_mb = bộ nhớ native đang cấp phát (heap + mmap); free_mb = vùng rỗi glibc còn giữ; heap_arena_mb =
    heap do hệ điều hành cấp; py_blocks = số khối của bộ cấp phát Python; gc_objects = số đối tượng gc theo dõi.
    rss lớn hơn nhiều so với heap_arena + mmap nghĩa là bộ nhớ nằm ngoài heap glibc."""
    out = {"rss_mb": rss_mb(), "py_blocks": sys.getallocatedblocks(), "gc_objects": len(gc.get_objects())}
    try:
        fn = _libc().mallinfo2
        fn.restype = _MallInfo2
        m = fn()
        out.update(heap_arena_mb=m.arena / 2**20, in_use_mb=(m.uordblks + m.hblkhd) / 2**20, free_mb=m.fordblks / 2**20)
    except (OSError, AttributeError):
        pass
    return out


def env_info():
    """CPU, RAM, hệ điều hành, phiên bản thư viện, số luồng (task 8.3 tiêu chí 3)."""
    import lightgbm, sklearn, torch_geometric

    cpu = "không rõ"
    mem_gb = None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                mem_gb = round(int(line.split()[1]) / 2**20, 2)
                break
    except OSError:
        pass
    limit = None
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            limit = None if v == "max" else round(int(v) / 2**30, 2)
            break
        except (OSError, ValueError):
            continue
    try:
        usable = len(os.sched_getaffinity(0))
    except AttributeError:
        usable = os.cpu_count()
    return {"cpu_model": cpu, "cpu_count": os.cpu_count(), "cpu_usable": usable, "ram_total_gb": mem_gb, "cgroup_memory_limit_gb": limit,
            "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__,
            "torch_geometric": torch_geometric.__version__, "torch_threads": torch.get_num_threads()}


# ---------------------------------------------------------------- mô hình đã đóng băng, dự đoán từng cửa sổ
class OnlineModel:
    """Mô hình đã đóng băng (nạp bằng đúng `final_test.build`) kèm đường dự đoán nhanh cho từng cửa sổ.

    `proba_batch(graphs)`: đường offline (cả tập, theo lô) dùng làm tham chiếu.
    `proba_window(g)`: đường online cho một đồ thị. Hybrid và GNN gọi thẳng mô hình (không qua DataLoader, giống phép đo độ trễ của GĐ H);
    các mô hình còn lại (baseline, lgbm_neighbor_mean) gọi lại `proba_batch` với một đồ thị.
    """

    def __init__(self, ctx, name, seed):
        self.name, self.seed, self.K = name, seed, ctx.K
        self.pred = FT.build(ctx, name, seed)
        self.threshold, self.val_ref_f1 = ctx.entry(name, seed)
        self.n_params, self.size_kb = self.pred.n_params, self.pred.size_kb
        self._fast = None
        if name in (FT.HYBRID, FT.LIGHT):
            d = Path(ctx.spec["folds"]) / f"{ctx.ds}_s{seed}"
            light = name == FT.LIGHT
            gats = [H.load_gat(d / f"gat_fold{f}.pt", ctx.K) for f in range(FT.LIGHT_FOLDS if light else 5)]
            lgbm_path = (Path(ctx.spec["light"]) / f"{ctx.ds}_s{seed}" / FT.LIGHT_CELL) if light else (d / "lgbm_oofavg_il.pkl")
            lgbm = pickle.load(open(lgbm_path, "rb"))
            lgbm.set_params(n_jobs=1)
            self.gats, self.lgbm = gats, lgbm                        # để scripts/ram_diag.py đo riêng từng phần
            self._fast = lambda g: H.predict_window_ensemble(gats, lgbm, g, ctx.K)
        elif name in FT.GNNS:
            model = M.GNN(name, in_dim=len(ctx.meta["features"]), n_classes=ctx.K)
            model.load_state_dict(torch.load(Path(ctx.spec["ckpt"]) / f"{ctx.ds}_{name}_s{seed}.pt", map_location="cpu", weights_only=True))
            model.eval()

            def fast(g):
                with torch.no_grad():
                    return torch.softmax(model(g.x, g.edge_index, g.edge_attr), 1).numpy()

            self._fast = fast

    def proba_batch(self, graphs):
        return self.pred.proba(graphs)

    def proba_window(self, g):
        return self._fast(g) if self._fast is not None else self.pred.proba(T.strip([g]))

    def single_thread(self):
        """Chuyển mọi mô hình sang 1 luồng và trả về p95 (ms) theo cách đo của GĐ H (gọi `Predictor.latency`, đặt n_jobs=1 cho cây)."""
        torch.set_num_threads(1)
        return float(self.pred.latency())


# ---------------------------------------------------------------- dữ liệu
def load_stream_flows(cfg, ctx, restrict="test"):
    """Flow đã làm sạch bằng đúng hàm offline (`build_graph.load_clean`), sắp theo thời gian; restrict="test": chỉ flow của cửa sổ tập test."""
    flows, classes, _ = bg.load_clean(ctx.ds, cfg["flows_csv"])
    if list(classes) != list(ctx.classes):
        raise RuntimeError(f"Lớp của dữ liệu {classes} khác lớp của mô hình {ctx.classes}")
    if restrict == "test":
        keep = np.isin(SF.window_ids(flows["ts"].to_numpy(), ctx.meta["window_sec"]), np.asarray(ctx.meta["test"]["windows"], dtype=np.int64))
        flows = flows[keep].reset_index(drop=True)
    return flows


class Reference:
    """Đồ thị và xác suất offline của tập test (tham chiếu để đối chiếu và lấy nhãn thật)."""

    def __init__(self, ctx, model):
        self.graphs = ctx.target_raw
        self.index = {int(g.win): i for i, g in enumerate(self.graphs)}
        self.offset = np.r_[0, np.cumsum([g.num_nodes for g in self.graphs])]
        self.proba = np.asarray(model.proba_batch(ctx.target))          # (số node, K) theo thứ tự đồ thị
        assert len(self.proba) == self.offset[-1]

    def get(self, win):
        i = self.index.get(int(win))
        return None if i is None else (self.graphs[i], self.proba[self.offset[i]:self.offset[i + 1]])


# ---------------------------------------------------------------- vòng chạy
def window_columns(K, has_label, has_ref):
    """Cột của windows.csv (cố định từ đầu để ghi từng dòng khi cửa sổ vừa xử lý xong, không giữ danh sách trong RAM)."""
    cols = ["win", "t_start", "t_close", "n_flows", "n_nodes", "n_edges", "build_ms", "infer_ms", "total_ms", "rss_mb", "n_alert"]
    if has_label:
        cols += [f"first_ts_{k}" for k in range(1, K)]
    cols.append("ref_found")
    if has_ref:
        cols += ["n_normal", "n_true_attack", "n_fp", "n_tp"]
        for k in range(1, K):
            cols += [f"n_true_{k}", f"n_det_{k}"]
        cols += ["struct_ok", "max_dx", "max_de", "n_mis_thr", "n_mis_arg", "max_dproba"]
    return cols + ["warmup"]


def run_stream(model, flows, scaler, classes, out_dir, ref=None, window_sec=5, batch_sec=1.0, speedup=None, max_windows=None, warmup=N_WARMUP,
               trim_every=0, probe=None, keep_rows=False):
    """Phát lại `flows`, xử lý từng cửa sổ khi đóng. Trả về (bảng cửa sổ đọc lại từ windows.csv, số flow đã phát).

    Bộ nhớ: mỗi cửa sổ xử lý xong được ghi thẳng vào windows.csv, không giữ trong RAM (bản trước giữ một dict mỗi cửa sổ).
    `trim_every` > 0: cứ sau chừng đó cửa sổ gọi `malloc_trim` (ngoài khoảng đo độ trễ).
    `probe(n_cửa_sổ_đã_xử_lý)` được gọi sau mỗi cửa sổ (chẩn đoán RAM). `keep_rows=True` giữ thêm danh sách dict như bản cũ, chỉ để tái hiện
    bản cũ trong chẩn đoán (scripts/ram_diag.py)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    K = len(classes)
    attack_names = classes[1:]
    has_label = "label" in flows.columns
    kept_rows = []
    n_done = 0
    n_flows_total = 0
    asm = SF.WindowAssembler(window_sec)
    fw = open(out_dir / "windows.csv", "w", newline="", encoding="utf-8")
    ww = csv.DictWriter(fw, fieldnames=window_columns(K, has_label, ref is not None), restval="")
    ww.writeheader()
    fa = open(out_dir / "alerts.csv", "w", newline="", encoding="utf-8")
    wa = csv.writer(fa)
    wa.writerow(["t_close", "window", "ip", "pred_class", "p_attack", "p_class", "proc_ms"])
    done = False

    def process(cw):
        nonlocal n_flows_total, n_done
        t0 = PERF()
        fl = cw.flows()
        g = SF.build_window_graph(cw.win, fl, scaler, window_sec)
        t1 = PERF()
        if g is None:
            return
        proba = model.proba_window(g)
        pred = TH.apply_threshold(proba, model.threshold)
        t2 = PERF()
        # ---- ngoài khoảng đo: ghi cảnh báo, đối chiếu, thống kê
        n_flows_total += len(fl)
        total_ms = (t2 - t0) * 1000
        t_close = (cw.win + 1) * window_sec
        iso = datetime.fromtimestamp(t_close, tz=timezone.utc).isoformat(timespec="seconds")
        score = TH.attack_score(proba)
        for j in np.flatnonzero(pred != 0):
            wa.writerow([iso, cw.win, g.ips[j], classes[pred[j]], f"{score[j]:.6f}", f"{proba[j, pred[j]]:.6f}", f"{total_ms:.3f}"])
        row = {"win": cw.win, "t_start": cw.win * window_sec, "t_close": t_close, "n_flows": len(fl), "n_nodes": g.num_nodes,
               "n_edges": g.num_edges, "build_ms": (t1 - t0) * 1000, "infer_ms": (t2 - t1) * 1000, "total_ms": total_ms,
               "rss_mb": rss_mb(), "n_alert": int((pred != 0).sum())}
        r = ref.get(cw.win) if ref is not None else None
        if has_label:
            lab = fl["label"].to_numpy()
            fts = fl["ts"].to_numpy()
            for k, nm in enumerate(attack_names, start=1):
                m = lab == nm
                row[f"first_ts_{k}"] = float(fts[m].min()) if m.any() else float("nan")
        if r is not None:
            rg, po = r
            y = rg.y.numpy()
            row.update(ref_found=1, n_normal=int((y == 0).sum()), n_true_attack=int((y != 0).sum()),
                       n_fp=int(((y == 0) & (pred != 0)).sum()), n_tp=int(((y != 0) & (pred != 0)).sum()))
            for k in range(1, K):
                row[f"n_true_{k}"] = int((y == k).sum())
                row[f"n_det_{k}"] = int(((y == k) & (pred != 0)).sum())
            same_shape = g.x.shape == rg.x.shape and g.edge_attr.shape == rg.edge_attr.shape
            row["struct_ok"] = int(same_shape and g.ips == rg.ips and torch.equal(g.edge_index, rg.edge_index))
            row["max_dx"] = float((g.x - rg.x).abs().max()) if same_shape else float("inf")
            row["max_de"] = float((g.edge_attr - rg.edge_attr).abs().max()) if same_shape and g.num_edges else (0.0 if same_shape else float("inf"))
            if len(po) == len(proba):
                row["n_mis_thr"] = int((pred != TH.apply_threshold(po, model.threshold)).sum())
                row["n_mis_arg"] = int((proba.argmax(1) != po.argmax(1)).sum())
                row["max_dproba"] = float(np.abs(proba - po).max())
            else:
                row.update(n_mis_thr=len(po), n_mis_arg=len(po), max_dproba=float("inf"))
        else:
            row["ref_found"] = 0
        row["warmup"] = int(n_done < warmup)
        ww.writerow(row)
        if keep_rows:
            kept_rows.append(row)
        n_done += 1
        if n_done % 500 == 0:
            fw.flush()
            fa.flush()
        if trim_every and n_done % trim_every == 0:
            trim_memory()
        if probe is not None:
            probe(n_done)

    for batch in SF.replay_batches(flows, batch_sec, speedup):
        for cw in asm.push(batch):
            process(cw)
            if max_windows and n_done >= max_windows:
                done = True
                break
        if done:
            break
    if not done:
        for cw in asm.flush():
            process(cw)
    fa.close()
    fw.close()
    return pd.read_csv(out_dir / "windows.csv"), n_flows_total


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=FT.LIGHT, choices=FT.MODELS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--restrict", default="test", choices=["test", "none"], help="test: chỉ phát lại flow của cửa sổ tập test")
    ap.add_argument("--batch-sec", type=float, default=1.0)
    ap.add_argument("--speedup", type=float, default=None, help="chờ để thời gian thật chậm hơn thời gian dữ liệu N lần (mặc định: không chờ)")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--no-reference", action="store_true", help="không nạp đồ thị offline để đối chiếu (demo trên flow bất kỳ)")
    ap.add_argument("--trim-every", type=int, default=TRIM_EVERY, help="gọi malloc_trim sau mỗi N cửa sổ (0 = không), ngoài khoảng đo độ trễ")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    ds = cfg["dataset"]
    split = cfg.get("split", "test")
    t_start = time.time()
    print(f"[detector] {ds} mô hình {a.model} seed {a.seed}, RAM {rss_mb():.0f} MB")
    ctx = FT.Context(cfg, ds, cfg["datasets"][ds], split)
    model = OnlineModel(ctx, a.model, a.seed)
    classes = ctx.classes
    scaler = ctx.scaler
    print(f"[detector] ngưỡng đóng băng {model.threshold:.6f}, {model.n_params:,} tham số, {model.size_kb:.1f} KB")

    ref = None
    frozen_p95 = None
    if not a.no_reference:
        t0 = time.time()
        ref = Reference(ctx, model)
        print(f"[detector] dự đoán offline {len(ref.proba):,} node trong {time.time() - t0:.0f} s")
    frozen_p95 = model.single_thread()                                  # từ đây mọi mô hình chạy 1 luồng
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    window_sec = ctx.meta["window_sec"]
    for attr in ("train", "val", "Xtr", "ytr", "Xva", "yva", "Ftr_nbr", "window"):    # giải phóng dữ liệu huấn luyện: tiến trình chỉ giữ mô hình và tham chiếu
        setattr(ctx, attr, None)
    import gc
    gc.collect()

    t0 = time.time()
    flows = load_stream_flows(cfg, ctx, a.restrict)
    print(f"[detector] đọc và làm sạch flow: {len(flows):,} flow trong {time.time() - t0:.0f} s, RAM {rss_mb():.0f} MB")
    gc.collect()
    try:                                                                # trả bộ nhớ đã giải phóng (bảng flow gốc, dữ liệu huấn luyện) cho hệ điều hành
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    ram_start = rss_mb()
    print(f"[detector] RAM trước khi phát lại: {ram_start:.0f} MB")
    t0 = time.time()
    win, n_flows = run_stream(model, flows, scaler, classes, a.out, ref, window_sec, a.batch_sec, a.speedup, a.max_windows, trim_every=a.trim_every)
    run_s = time.time() - t0
    summary = {"dataset": ds, "split": split, "model": a.model, "seed": a.seed, "threshold": model.threshold, "n_params": model.n_params,
               "size_kb": model.size_kb, "classes": classes, "window_sec": window_sec, "n_windows": int(len(win)), "n_flows": int(n_flows),
               "warmup_windows": N_WARMUP, "trim_every": a.trim_every, "restrict": a.restrict, "speedup": a.speedup, "reference": ref is not None,
               "n_ref_windows": None if ref is None else len(ref.index),
               "n_ref_windows_not_streamed": None if ref is None else len(set(ref.index) - set(int(w) for w in win["win"])),
               "frozen_style_p95_ms": frozen_p95, "stream_seconds": round(run_s, 1), "total_seconds": round(time.time() - t_start, 1),
               "ram_start_mb": ram_start, "ram_end_mb": rss_mb(), "env": env_info()}
    (Path(a.out) / "detector_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[detector] xong: {len(win):,} cửa sổ, {n_flows:,} flow trong {run_s:.0f} s; p95 tổng {np.percentile(win['total_ms'], 95):.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
