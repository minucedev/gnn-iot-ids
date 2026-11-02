# Ghi chú môi trường (task 0.3)

Ngày kiểm tra: 02/11/2026. Người chạy: Phúc (qua Claude Code).

## Máy cá nhân

| Mục | Giá trị |
|---|---|
| Hệ điều hành | Windows 11 Pro 10.0.26200 |
| CPU | Intel Core i7-12800H |
| RAM | 32 GB |
| GPU | NVIDIA RTX A1000 Laptop GPU, 4 GB VRAM, compute capability 8.6 |
| Driver NVIDIA | 596.47 (hỗ trợ tới CUDA 13.2 theo `nvidia-smi`) |
| Ổ dữ liệu | D: còn trống khoảng 171 GB |
| Công cụ | uv 0.13.2, git 2.56.0, GitHub CLI (`gh`, tài khoản `minucedev`), Docker 29.8 |
| Python | 3.11.17, do `uv python install 3.11` cài (Windows không có Python thật; WSL có Python 3.14 nhưng không dùng cho dự án này) |

## Chọn bản torch

- Index chính thức của PyTorch cho Python 3.11/Windows: `cu121` chỉ có tới torch 2.5.1, `cu124` chỉ có tới 2.6.0; `cu126` và `cu130` có tới 2.14.1.
- Notebook Tuần 1 của Phong chạy torch 2.14.1 với PyG 2.8.1. Phúc chốt **torch 2.14.1+cu126** để cùng phiên bản với notebook (02/11/2026). Driver 596.47 chạy được cu126.
- PyG cài bản pure-python (`torch-geometric==2.8.1`), không dùng `pyg-lib`, `torch-scatter`, `torch-sparse`.

## Thời gian cài

| Lần | Điều kiện | Thời gian |
|---|---|---|
| 1 | Tải mới từ mạng, khoảng 1,7 MB/s (mạng dùng chung với một tiến trình cài khác) | torch 2,4 GB: khoảng 35 phút (20:38 → 21:13); 19 gói còn lại: khoảng 4 phút |
| 2 | Xóa `.venv`, cài lại theo đúng README, dùng cache của uv | **34,1 s** (python 0,1 s; venv 0,0 s; torch 18,5 s; requirements 14,6 s; kernel 0,8 s) |

Cache của uv ở ổ C: còn repo ở ổ D:, nên uv in cảnh báo `Failed to hardlink files; falling back to full copy` (không ảnh hưởng kết quả). Đặt `$env:UV_LINK_MODE = "copy"` nếu muốn tắt cảnh báo.

## Output `check_env.py` (sau khi cài lại lần 2)

Lệnh: `.venv\Scripts\python.exe -I scripts\check_env.py`, **mã thoát 0**, tổng 86,3 s. Lần chạy đầu sau khi cài chậm vì Windows Defender quét các DLL CUDA và Python biên dịch `.pyc`; các lần sau khoảng 7 s.

```
== Phiên bản & GPU
Python           : 3.11.17 (Windows-10-10.0.26200-SP0)
torch            : 2.14.1+cu126 (CUDA build 12.6)
torch_geometric  : 2.8.1
numpy            : 2.4.6
pandas           : 3.1.0
pyarrow          : 26.0.0
sklearn          : 1.9.1
cuda.is_available: True
GPU              : NVIDIA RTX A1000 Laptop GPU, 4.0 GiB, compute capability 8.6
OK (81.49 s)

== GATv2Conv CPU
GATv2Conv[cpu] : in (50, 18), edge_index (2, 200), edge_attr (200, 3) -> out (50, 64), tham số = 2752
OK (0.15 s)

== GATv2Conv GPU
GATv2Conv[cuda] : in (50, 18), edge_index (2, 200), edge_attr (200, 3) -> out (50, 64), tham số = 2752
Chênh lệch CPU/GPU: max |Δ| = 3.58e-07
OK (2.13 s)

== Parquet
Parquet          : ghi/đọc 3 dòng x 3 cột (2306 byte), khớp dữ liệu gốc
OK (1.05 s)

== Tổng kết
Tất cả các bước OK
```

Số tham số 2752 khớp với tính tay cho `GATv2Conv(18, 16, heads=4, edge_dim=3)`: `lin_l` 18·64 + 64 = 1216, `lin_r` 1216, `lin_edge` 3·64 = 192 (không bias), `att` 64, `bias` 64.

Khi import torch_geometric có cảnh báo `FutureWarning: torch.jit.script is deprecated` từ torch 2.14. Cảnh báo này không ảnh hưởng kết quả.

## Kaggle

[notebooks/00_kaggle_env_check.ipynb](../notebooks/00_kaggle_env_check.ipynb) được đẩy và chạy bằng Kaggle API (`kaggle kernels push`, notebook riêng tư `minucelowcode/gnn-iot-ids-00-env-check`, GPU T4, Internet bật), ngày 02/11/2026.

- **Bản 1:** lỗi `SyntaxError` ở ô cài đặt (chuỗi `"\n"` bị hỏng khi sinh notebook). Đã sửa ở commit `c58965a`.
- **Bản 2:** clone `main` tại `c58965a`; **`check_env.py` thoát mã 0**. Tổng thời gian khoảng 123 s, trong đó pip cài mất 82 s.

| Mục | Kaggle | Máy cá nhân |
|---|---|---|
| Python | 3.13.15 | 3.11.17 |
| torch | **2.11.0+cu128** (có sẵn của Kaggle, giữ nguyên) | 2.14.1+cu126 |
| GPU | Tesla T4, 14,6 GiB, compute capability 7.5 | RTX A1000, 4 GiB, 8.6 |
| torch_geometric, numpy, pandas, pyarrow, sklearn | 2.8.1, 2.4.6, 3.1.0, 26.0.0, 1.9.1 (cài theo `requirements.txt`) | giống |
| GATv2Conv | out (50, 64), 2752 tham số, chênh lệch CPU/GPU 2,38e-07 | giống, 3,58e-07 |

Khi cài trên Kaggle, pip báo xung đột với một số gói Kaggle cài sẵn mà dự án không dùng: `ydata-profiling`, `cudf-cu12`, `numba` (cần numpy < 2.3), `google-colab`, `gradio`, `grpcio-*` (cần protobuf < 6). Nếu sau này cần `numba` thì phải xử lý riêng. pandas cũng cảnh báo `bottleneck` 1.4.2 cũ, không ảnh hưởng.

**Lưu ý tái lập:** torch trên Kaggle (2.11) khác bản local (2.14). Khi so số liệu giữa hai nơi (KE_HOACH mục 3, chỉ tiêu "Tái lập"), cần ghi rõ phiên bản torch của từng lần chạy.
