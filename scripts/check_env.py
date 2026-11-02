"""Kiểm tra môi trường của dự án gnn-iot-ids.

Chạy:  .venv\\Scripts\\python.exe -I scripts/check_env.py [--cpu-only]

Các bước: in phiên bản thư viện và GPU; chạy GATv2Conv (heads=4, edge_dim=3)
trên đồ thị ngẫu nhiên 50 node / 200 cạnh trên CPU và GPU; ghi rồi đọc lại
1 file parquet trong thư mục tạm. Thoát mã 1 nếu có bước lỗi.
"""

import argparse
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path

N_NODES, N_EDGES = 50, 200
NODE_DIM, EDGE_DIM = 18, 3  # 18 đặc trưng node, 3 đặc trưng cạnh (KE_HOACH.md mục 1)
HIDDEN, HEADS = 16, 4
SEED = 42


def step_versions():
    import numpy, pandas, pyarrow, sklearn, torch, torch_geometric

    print(f"Python           : {sys.version.split()[0]} ({platform.platform()})")
    print(f"torch            : {torch.__version__} (CUDA build {torch.version.cuda})")
    print(f"torch_geometric  : {torch_geometric.__version__}")
    print(f"numpy            : {numpy.__version__}")
    print(f"pandas           : {pandas.__version__}")
    print(f"pyarrow          : {pyarrow.__version__}")
    print(f"sklearn          : {sklearn.__version__}")
    print(f"cuda.is_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU              : {props.name}, {props.total_memory / 2**30:.1f} GiB, "
              f"compute capability {props.major}.{props.minor}")
    else:
        print("GPU              : (không có)")


def _random_graph():
    import torch

    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(N_NODES, NODE_DIM, generator=g)
    edge_index = torch.randint(0, N_NODES, (2, N_EDGES), generator=g)
    edge_attr = torch.randn(N_EDGES, EDGE_DIM, generator=g)
    return x, edge_index, edge_attr


def _make_conv():
    import torch
    from torch_geometric.nn import GATv2Conv

    torch.manual_seed(SEED)
    return GATv2Conv(NODE_DIM, HIDDEN, heads=HEADS, edge_dim=EDGE_DIM).eval()


def _run_conv(device):
    import torch

    conv = _make_conv().to(device)
    x, edge_index, edge_attr = (t.to(device) for t in _random_graph())
    with torch.no_grad():
        out = conv(x, edge_index, edge_attr)
    if device.type == "cuda":
        torch.cuda.synchronize()
    n_params = sum(p.numel() for p in conv.parameters())
    assert out.shape == (N_NODES, HEADS * HIDDEN), f"shape sai: {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "đầu ra có NaN/Inf"
    print(f"GATv2Conv[{device.type}] : in {tuple(x.shape)}, edge_index {tuple(edge_index.shape)}, "
          f"edge_attr {tuple(edge_attr.shape)} -> out {tuple(out.shape)}, tham số = {n_params}")
    return out.cpu()


def step_gat_cpu():
    import torch

    step_gat_cpu.out = _run_conv(torch.device("cpu"))


def step_gat_gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() = False")
    out = _run_conv(torch.device("cuda"))
    ref = getattr(step_gat_cpu, "out", None)
    if ref is not None:
        diff = (out - ref).abs().max().item()
        print(f"Chênh lệch CPU/GPU: max |Δ| = {diff:.2e}")
        assert diff < 1e-4, "kết quả CPU và GPU lệch nhau"


def step_parquet():
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({
        "src_ip": ["192.168.1.1", "10.0.0.2", "10.0.0.3"],
        "bytes": np.array([100, 2000, 30000], dtype="int64"),
        "duration": np.array([0.1, 1.5, 3.25], dtype="float32"),
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "check.parquet"
        df.to_parquet(path, engine="pyarrow", index=False)
        back = pd.read_parquet(path, engine="pyarrow")
        size = path.stat().st_size
    pd.testing.assert_frame_equal(df, back)
    print(f"Parquet          : ghi/đọc {len(back)} dòng x {back.shape[1]} cột ({size} byte), khớp dữ liệu gốc")


def main():
    # Windows chuyển stdout qua pipe dùng cp1252; -I bỏ qua PYTHONIOENCODING
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu-only", action="store_true",
                        help="bỏ qua bước GPU (máy không có GPU)")
    args = parser.parse_args()

    steps = [("Phiên bản & GPU", step_versions),
             ("GATv2Conv CPU", step_gat_cpu),
             ("GATv2Conv GPU", step_gat_gpu),
             ("Parquet", step_parquet)]
    failed = []
    for name, fn in steps:
        print(f"\n== {name}")
        if args.cpu_only and fn is step_gat_gpu:
            print("BỎ QUA (--cpu-only)")
            continue
        t0 = time.perf_counter()
        try:
            fn()
            print(f"OK ({time.perf_counter() - t0:.2f} s)")
        except Exception:
            traceback.print_exc(file=sys.stdout)
            print("LỖI")
            failed.append(name)

    print("\n== Tổng kết")
    if failed:
        print(f"LỖI ở {len(failed)} bước: {', '.join(failed)}")
        return 1
    print("Tất cả các bước OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
