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

## Demo thời gian thực (task 8.2)

`scripts/detector.py` phát lại flow NetFlow v3 theo thứ tự thời gian, gom cửa sổ 5 giây, dựng đồ thị bằng đúng hàm offline
([src/stream_features.py](src/stream_features.py)), dự đoán bằng mô hình đã đóng băng và ngưỡng của nó, rồi ghi cảnh báo. Chỉ cần CPU (1 luồng).
Dữ liệu và mô hình đã đóng băng nằm trên Kaggle, nên demo chạy trong notebook [20_realtime](notebooks/20_realtime.ipynb), một lệnh:

```
python -I scripts/detector.py --config detector_config.json --model hybrid_gat_lgbm_oofavg_il --out demo_out [--max-windows 200] [--speedup 10]
```

- `detector_config.json`: cấu trúc như config của `scripts/final_test.py`, thêm `dataset` (ví dụ `nf_ton_iot_v3`) và `flows_csv` (file `NF-ToN-IoT-v3.csv`); notebook 20 tự sinh file này.
- `--model`: `hybrid_gat_lgbm_oofavg_il` (AHP gợi ý), `hybrid_light`, hoặc bất kỳ mô hình đã đóng băng nào của `final_test.py`.
- `--speedup N`: chờ cho thời gian thật chậm hơn thời gian dữ liệu N lần (mặc định không chờ). `--max-windows N`: dừng sau N cửa sổ. `--no-reference`: không đối chiếu với đồ thị offline.
- Kết quả trong `--out`: `alerts.csv` (t_close, window, ip, pred_class, p_attack, p_class, proc_ms), `windows.csv` (độ trễ từng cửa sổ, RAM, đối chiếu offline), `detector_summary.json` (mô hình, ngưỡng, CPU, phiên bản thư viện).
- Đánh giá: `python -I scripts/realtime_eval.py <thư mục --out> <thư mục bảng> <thư mục hình>` ghi `latency.json`, `realtime_eval.json`, `alerts_sample.csv`, `gd8_*.png`.

## Báo cáo và slide

- Báo cáo cuối kỳ: [report/BAO_CAO.docx](report/BAO_CAO.docx) (bản nộp), nguồn [report/BAO_CAO.md](report/BAO_CAO.md).
- Slide: [report/SLIDE.pptx](report/SLIDE.pptx), nguồn [report/SLIDE_DAN_Y.md](report/SLIDE_DAN_Y.md).
- Tạo lại hai file này (từ thư mục ngoài repo, vì Windows có thể chặn `python.exe` của `.venv`):
  `uv run --no-project --python 3.11 --with pypandoc_binary --with python-docx python <repo>/report/build_docx.py` và
  `uv run --no-project --python 3.11 --with python-pptx --with pillow python <repo>/report/build_pptx.py`.

## Chạy lại toàn bộ trên Kaggle

Mọi tính toán chạy trên Kaggle (tài khoản có quyền đọc dataset `athena21/netflow-v3-datasets`, bật Internet để `git clone`). Mỗi notebook trong
[notebooks/](notebooks/) clone repo, cài `requirements.txt` (trừ torch, dùng bản có sẵn của Kaggle), chạy `scripts/*.py` và ghi `run_info.json`
(commit, thời gian, thiết bị, RAM đỉnh). Notebook sau lấy đầu vào từ output của notebook trước (thêm làm "notebook data source"). Thứ tự (slug Kaggle `gnn-iot-ids-<số>-<tên>`):

| Bước | Notebook | Việc | Máy |
|---|---|---|---|
| 1 | `04_build_graphs`, `11_build_graphs_bot` | Làm sạch NetFlow v3, dựng đồ thị cửa sổ 5 s, chia tập theo khối (ToN, BoT) | CPU |
| 2 | `12_eda` | EDA mức flow và đồ thị, `reports/figures/gd2_*` | CPU |
| 3 | `05_baselines`, `13_baselines`, `07_train_gnn`, `14_train_gnn_bot` | Baseline ML, MLP và GNN đơn (GCN, GraphSAGE, GAT), 3 seed | CPU |
| 4 | `15_hybrid`, `15b`, `15e`, `15f_hybrid_il_ton` (GPU T4) | Hybrid GAT + LightGBM (out-of-fold xen kẽ) | GPU |
| 5 | `16_ablation`, `16a`–`16h` | Ablation, hybrid nhẹ, bản cải tiến GAT, hiệu chỉnh ngưỡng trên val | GPU, CPU |
| 6 | `17a_dryrun_val`, `17_final_test` | Dựng lại mô hình đã đóng băng (tag `freeze-v1`), chạy test **một lần** | CPU |
| 7 | `18_aggregate` | `scripts/aggregate_h.py`: `results.csv`, `summary_*.csv`, McNemar, `targets.csv`, hình `gd6_*` | CPU |
| 8 | `19_ahp`, `19b_ahp_constrained` | AHP không ràng buộc và có ràng buộc cứng (`src/ahp.py`) | CPU |
| 9 | `20_realtime` | Phát lại toàn bộ flow test: độ trễ, khớp online/offline, cảnh báo giả, thời gian phát hiện, RAM | CPU |
| 10 | `20a_undetected` | Vì sao một số đợt tấn công không có cảnh báo | CPU |
| 11 | `21_reproduce` | Chạy lại bước 7 từ output đã lưu, so với bảng trong repo (lệch F1-macro ≤ 0,5 điểm %) | CPU |

Công cụ chẩn đoán RAM: `20b`, `20c`, `20d` (`scripts/ram_diag.py`). Kết quả đã kéo về lưu ở `reports/kaggle/<slug>/`, bảng ở `reports/tables/`,
hình ở `reports/figures/`. Chạy kiểm thử trên Kaggle (`pytest -q`, mỗi notebook làm bước này); WSL không có pip nên không chạy được pytest ở máy cá nhân.

**Kiểm tra tái lập (GĐ J):** xem dòng "Tái lập" trong [reports/tables/targets.csv](reports/tables/targets.csv) và
[reports/kaggle/gnn-iot-ids-21-reproduce/reproduce_check.json](reports/kaggle/gnn-iot-ids-21-reproduce/reproduce_check.json).

## Quy ước làm việc

- Commit và push thẳng lên `main`, không tạo nhánh hay PR.
- Commit ghi `<type>(<mã-task>): <mô tả>`, ví dụ `feat(0.4): tách build_graph.py`. Các type: `feat`, `fix`, `refactor`, `chore`, `docs`.
- Tự kiểm đủ tiêu chí của task trong [TASKS.md](docs/gnn_ids/TASKS.md) và ghi bằng chứng vào dòng **Kết quả** trước khi chuyển task sang "Xong".
- Seed, cách chia tập, cột `results.csv`, cách đặt tên hình: theo [KE_HOACH.md mục 6](docs/gnn_ids/KE_HOACH.md#6-quy-ước-bắt-buộc).
