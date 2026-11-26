"""Máy chủ nhỏ cho notebook 22_demo (task 8.2): nạp mô hình đã đóng băng MỘT lần, rồi chạy từng đoạn demo theo lệnh.

Vì sao là tiến trình riêng: notebook cài gói đúng phiên bản (numpy, pandas, torch ...) ngay trong phiên, nên tiến trình kernel (đã nạp numpy cũ khi khởi động)
không import được các thư viện đó nữa; mọi tính toán chạy trong tiến trình con này, giống các notebook khác của dự án.

Chạy:  python -u scripts/demo_server.py --config detector_config.json --model <mô hình> --seed 42
Giao thức (mỗi dòng một JSON; kết quả trả về trên stdout, dòng bắt đầu bằng "<<"):
  khởi động xong      <<READY>> {thông tin máy và mô hình}
  {"cmd": "run", "model": m, "argv": [...]}     chạy `detector.main(argv, phiên của m)`; các dòng in ra của detector được chuyển nguyên văn
  {"cmd": "plot", "json": first_alert.json, "png": đường dẫn, "title": tiền tố}   vẽ đồ thị cửa sổ cảnh báo đầu tiên ra PNG
  {"cmd": "tables"}                             bảng tốc độ (cảnh 4) và bảng giới hạn (cảnh 5)
  {"cmd": "quit"}
  mỗi lệnh xong   <<DONE>> {kết quả}      lỗi: in traceback rồi <<ERROR>> {thông báo}
"""

import argparse
import importlib.util
import io
import json
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src import demo_replay as DR  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reply(tag, payload=None):
    print(f"<<{tag}>> " + ("" if payload is None else json.dumps(payload, ensure_ascii=False, default=str)), flush=True)


def quiet_prepare(DET, cfg, model, seed):
    with redirect_stdout(io.StringIO()):                              # ẩn nhật ký nạp để màn hình gọn
        return DET.prepare(cfg, model, seed, reference=False)


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="thư mục ghi kết quả khởi động nóng")
    ap.add_argument("--warm-windows", type=int, default=30)
    a = ap.parse_args()

    DET = _load_script("detector")
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    t0 = time.time()
    sessions = {a.model: quiet_prepare(DET, cfg, a.model, a.seed)}
    t_load = time.time() - t0
    S = sessions[a.model]

    t0 = time.time()                                                  # khởi động nóng: chạy thử vài chục cửa sổ đầu của file đoạn, không in
    with redirect_stdout(io.StringIO()):
        DET.main(["--config", a.config, "--model", a.model, "--seed", str(a.seed), "--out", str(Path(a.out) / "warmup"), "--partial-flows", "--no-reference",
                  "--max-windows", str(a.warm_windows)], S)
    t_warm = time.time() - t0
    try:
        import os
        usable = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        usable = None
    reply("READY", {"model": a.model, "threshold": S.model.threshold, "n_params": S.model.n_params, "size_kb": S.model.size_kb, "classes": S.classes,
                    "torch_threads": torch.get_num_threads(), "cpu_usable": usable, "load_s": round(t_load, 1), "warm_s": round(t_warm, 1), "warm_windows": a.warm_windows,
                    "env": DET.env_info()})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
            kind = cmd["cmd"]
            if kind == "quit":
                reply("DONE", {"bye": True})
                return 0
            if kind == "run":
                m = cmd.get("model", a.model)
                if m not in sessions:
                    t0 = time.time()
                    sessions[m] = quiet_prepare(DET, cfg, m, a.seed)
                    print(f"[server] đã nạp thêm {m} trong {time.time() - t0:.0f} s", flush=True)
                t0 = time.time()
                DET.main(cmd["argv"], sessions[m])
                reply("DONE", {"wall_s": round(time.time() - t0, 2)})
            elif kind == "plot":
                g = json.loads(Path(cmd["json"]).read_text(encoding="utf-8"))
                reply("DONE", DR.plot_first_alert(g, cmd["png"], cmd.get("title", "")))
            elif kind == "tables":
                nums = DR.speed_numbers(ROOT)
                reply("DONE", {"speed_rows": DR.speed_rows(nums), "speed_numbers": nums, "limits": DR.limits_rows(ROOT)})
            else:
                raise ValueError(f"lệnh lạ: {kind}")
        except SystemExit as e:
            reply("ERROR", f"detector thoát: {e}")
        except Exception as e:                                        # lỗi một lệnh không làm mất phiên đã nạp
            traceback.print_exc(file=sys.stdout)
            reply("ERROR", f"{type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
