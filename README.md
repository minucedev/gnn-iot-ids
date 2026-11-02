# gnn-iot-ids

Đề tài: **Ứng dụng mạng Lightweight GNN gợi ý phát hiện xâm nhập IoT thời gian thực.**

Luồng NetFlow IoT (bộ chính NF-ToN-IoT) được chia theo cửa sổ thời gian; mỗi cửa sổ là một đồ thị có IP là node (18 đặc trưng) và flow là cạnh (3 đặc trưng).
Mô hình GAT nhẹ multi-head (≤ 100K tham số) phân loại từng node vào 5 lớp, với độ trễ p95 ≤ 1 s mỗi cửa sổ trên CPU 1 luồng.
Kết quả được so với các baseline ML, sau đó dùng AHP để gợi ý mô hình nên triển khai trên gateway.

Nhóm: Nguyễn Minh Phúc (nhóm trưởng), Lê Trung Phong, Văn Bá Phạm Tấn.

> **Repo này là nơi lưu bản chính thức** của kế hoạch, danh sách task, phiếu dataset và notebook. Các bản ngoài repo (thư mục `plan/`, `docs/` cũ trên máy) chỉ là bản gốc để tham khảo, không cập nhật nữa.

## Tài liệu

- [docs/gnn_ids/KE_HOACH.md](docs/gnn_ids/KE_HOACH.md): đề bài, chỉ tiêu nghiệm thu, quy ước bắt buộc, log quyết định
- [docs/gnn_ids/TASKS.md](docs/gnn_ids/TASKS.md): danh sách task, trạng thái, bằng chứng
- [docs/plan/](docs/plan/): kế hoạch gốc 3 tuần của nhóm
- [docs/dataset_cards/](docs/dataset_cards/): phiếu đánh giá 4 bộ dữ liệu (Tuần 1)
- [docs/notes_env.md](docs/notes_env.md): cấu hình máy và output `check_env.py`

## Cài đặt trên Windows (uv)

Yêu cầu: [uv](https://docs.astral.sh/uv/), git, driver NVIDIA hỗ trợ CUDA ≥ 12.6 (kiểm tra bằng `nvidia-smi`). Không cần cài Python, vì uv tự tải Python 3.11.

```powershell
winget install --id astral-sh.uv -e          # nếu chưa có uv
git clone https://github.com/minucedev/gnn-iot-ids.git
cd gnn-iot-ids
git config core.hooksPath .githooks          # bật hook chặn file > 50 MB

uv python install 3.11
uv venv --python 3.11 --python-preference only-managed .venv

# torch bản CUDA 12.6 lấy từ index riêng của PyTorch (PyPI chỉ có bản CPU cho Windows)
uv pip install --python .venv\Scripts\python.exe torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
# các gói còn lại (torch==2.14.1 trong file đã được thỏa bởi bản +cu126 ở trên)
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

# kernel Jupyter tên gnn-iot-ids
.venv\Scripts\python.exe -m ipykernel install --user --name gnn-iot-ids --display-name "Python (gnn-iot-ids)"
```

Kích hoạt môi trường trong PowerShell: `.venv\Scripts\Activate.ps1`.

Nếu uv báo `Failed to hardlink files` (cache ở ổ C:, repo ở ổ khác) thì không sao. Đặt `$env:UV_LINK_MODE = "copy"` để tắt cảnh báo.

Trên Linux/Kaggle/Colab: thay lệnh cài torch bằng `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126`, hoặc giữ torch có sẵn của nền tảng. Xem [notebooks/00_kaggle_env_check.ipynb](notebooks/00_kaggle_env_check.ipynb).

## Kiểm tra môi trường

```powershell
.venv\Scripts\python.exe -I scripts\check_env.py
```

Script in phiên bản Python/torch/PyG/pandas/pyarrow/sklearn và GPU, chạy `GATv2Conv` (heads=4, edge_dim=3) trên đồ thị ngẫu nhiên 50 node / 200 cạnh ở cả CPU và GPU, rồi ghi và đọc lại một file parquet. Script thoát mã 0 nếu mọi bước OK. Máy không có GPU thì thêm `--cpu-only`.

## Cấu trúc thư mục

```
gnn-iot-ids/
  data/
    raw/              # dữ liệu gốc tải về (KHÔNG commit)
    processed/        # parquet sạch, graphs_*.pt, split.json, scaler.pkl (KHÔNG commit)
  docs/
    gnn_ids/          # KE_HOACH.md, TASKS.md
    plan/             # kế hoạch gốc 3 tuần
    dataset_cards/    # phiếu dataset
    notes_env.md      # ghi chú môi trường
  notebooks/
    00_kaggle_env_check.ipynb
    phong_tuan1/      # 2 notebook Tuần 1 của Phong (giữ nguyên output)
  src/                # build_graph.py, evaluate.py, models/, ahp.py, ...
  tests/              # pytest
  scripts/            # check_env.py và các script dòng lệnh
  models/             # trọng số mô hình (*.pt, *.pkl, *.onnx không commit)
  reports/
    figures/          # gd<số>_<tên>.png, dpi ≥ 150
    tables/           # results.csv (20 cột, ghi qua evaluate.py)
  docker/             # Dockerfile mô phỏng gateway
  .githooks/          # pre-commit chặn file > 50 MB
  requirements.txt
```

## Dữ liệu

Đặt file gốc vào `data/raw/<tên_bộ>/` (ví dụ `data/raw/nf_ton_iot_v3/`). Toàn bộ nội dung `data/raw/` và `data/processed/` cùng mọi file `*.parquet` đều bị `.gitignore` bỏ qua, **không commit dữ liệu**. Link tải và license của từng bộ ghi trong phiếu ở [docs/dataset_cards/](docs/dataset_cards/).

## Quy ước làm việc

- Commit và push thẳng lên `main`, không tạo nhánh hay PR.
- Commit ghi `<type>(<mã-task>): <mô tả>`, ví dụ `feat(0.4): tách build_graph.py`. Các type: `feat`, `fix`, `refactor`, `chore`, `docs`.
- Tự kiểm đủ tiêu chí của task trong [TASKS.md](docs/gnn_ids/TASKS.md) và ghi bằng chứng vào dòng **Kết quả** trước khi chuyển task sang "Xong".
- Seed, cách chia tập, cột `results.csv`, cách đặt tên hình: theo [KE_HOACH.md mục 6](docs/gnn_ids/KE_HOACH.md#6-quy-ước-bắt-buộc).
