# Các giai đoạn thực hiện (12/11 – 19/11/2026)

Mỗi file trong thư mục này là **một giai đoạn**, đủ thông tin để mở một phiên làm việc mới và làm đúng giai đoạn đó. Tổng quan, ngân sách GPU và bảng theo dõi ở [../LO_TRINH_KAGGLE.md](../LO_TRINH_KAGGLE.md). Tiêu chí chi tiết từng task ở [../TASKS.md](../TASKS.md).

| GĐ | Ngày | Việc | Máy | Cần xong trước |
|---|---|---|---|---|
| [A](GD_A_thu_hoi_ket_qua.md) | 12/11 | Thu hồi kết quả đã chạy trên Kaggle | — | — |
| [B](GD_B_do_thi_bot_iot.md) | 12–13/11 | Dựng đồ thị NF-BoT-IoT-v3 | CPU | — |
| [C](GD_C_eda.md) | 13/11 | EDA và trực quan hóa đồ thị | CPU | B |
| [D](GD_D_baseline.md) | 13/11 | Baseline ML và MLP | CPU | A, B |
| [E](GD_E_gnn_bot_iot.md) | 14/11 | GNN đơn trên NF-BoT-IoT-v3 | GPU | B |
| [F](GD_F_hybrid.md) | 14–15/11 | Hybrid GAT + LightGBM | GPU | A, E |
| [G](GD_G_ablation.md) | 15/11 | Ablation, bản cải tiến GAT, hiệu chỉnh ngưỡng | GPU | D, F |
| [H](GD_H_dong_bang_test.md) | 16/11 | Đóng băng, chạy test một lần, tổng hợp | GPU, CPU | G |
| [I](GD_I_ahp_realtime.md) | 17/11 | AHP và mô phỏng thời gian thực | CPU | H |
| [J](GD_J_bao_cao.md) | 18–19/11 | Báo cáo, kiểm tra tái lập, nộp | CPU | H, I |

C và D chạy song song được sau B. E chạy song song với C, D.

## Bối cảnh chung (đọc trước mọi giai đoạn)

**Đề tài:** phát hiện xâm nhập IoT bằng GNN nhẹ. Mỗi cửa sổ 5 giây của lưu lượng NetFlow là một đồ thị: node là IP (18 đặc trưng), cạnh là flow (3 đặc trưng). Nhiệm vụ là phân loại node vào 5 lớp: Normal, DoS_DDoS, Botnet_Backdoor, Truy_cap_trai_phep, Khac. Mô hình chính là **hybrid GAT + LightGBM**. Hai bộ dữ liệu: **NF-ToN-IoT-v3** (chính) và **NF-BoT-IoT-v3**. Chi tiết ở [../KE_HOACH.md](../KE_HOACH.md).

**Quy tắc bắt buộc:**
1. **Mọi xử lý, huấn luyện, đánh giá, tổng hợp đều chạy trên Kaggle** (tài khoản `minucelowcode`). Máy cá nhân chỉ sửa code, đẩy GitHub, điều khiển Kaggle qua API, tải file kết quả nhỏ (`.csv`, `.json`, `.png` dưới 5 MB) về `reports/kaggle/<tên-notebook>/`. Không tải CSV gốc, `.parquet`, `.pt`, `ckpt/`.
2. Máy Windows chặn `python.exe` của `.venv` và `kaggle.exe`. Kiểm tra nhỏ không cần thư viện thì dùng `wsl -d Ubuntu --cd /mnt/d/E/uni/ATTT/gnn-iot-ids -- python3 -I <script>`; WSL không có pip nên không chạy được `pytest`, test chạy trên Kaggle.
3. Commit thẳng lên `main`, message dạng `<type>(<mã-task>): <mô tả>`.
4. Chọn cấu hình chỉ trên **val**. Test chạy **một lần** ở GĐ H, sau tag `freeze-v1`. Mọi mô hình chạy 3 seed: 42, 43, 44.
5. Một giai đoạn chỉ coi là xong khi đã tự kiểm từng tiêu chí và ghi bằng chứng vào dòng **Kết quả** của task trong TASKS.md, đồng thời cập nhật bảng theo dõi trong LO_TRINH_KAGGLE.md.

## Cách làm một notebook Kaggle

Notebook theo mẫu các notebook đã có trong `notebooks/` (ví dụ `07_train_gnn.ipynb`):
1. Ô 1: `git clone --depth 1 https://github.com/minucedev/gnn-iot-ids.git /tmp/gnn-iot-ids`, in commit hash, cài các gói trong `requirements.txt` trừ torch (dùng torch có sẵn của Kaggle).
2. Ô 2: tìm đầu vào trong `/kaggle/input/**`, gọi `scripts/*.py` bằng `subprocess`, ghi output vào `/kaggle/working/`.
3. Ô cuối: ghi `run_info.json` gồm commit hash, thời gian chạy (giây), loại máy (CPU hoặc GPU và tên GPU), RAM đỉnh.

Viết notebook dạng `.py` rồi chuyển sang `.ipynb` bằng `scripts/py_to_ipynb.py` nếu tiện. Đặt tên `notebooks/<số>_<việc>.ipynb`, slug Kaggle `gnn-iot-ids-<số>-<việc>`.

## Gọi Kaggle API từ máy Windows

Token nằm ở `~/.kaggle/access_token`. Dùng `curl` với header `Authorization: Bearer <token>`, gốc URL `https://www.kaggle.com/api/v1`:

| Việc | Lệnh |
|---|---|
| Xem trạng thái | `GET /kernels/status?userName=minucelowcode&kernelSlug=<slug>` |
| Liệt kê file output | `GET /kernels/output?userName=minucelowcode&kernelSlug=<slug>`; mỗi file có trường `url` để tải bằng `curl -L -o` |
| Tải mã nguồn notebook | `GET /kernels/pull?userName=<user>&kernelSlug=<slug>`; mã nằm ở `blob.source` |
| Đẩy notebook mới hoặc phiên bản mới | `POST /kernels/push` với JSON gồm `slug`, `newTitle`, `text` (nội dung `.ipynb`), `language: "python"`, `kernelType: "notebook"`, `isPrivate`, `enableGpu`, `enableInternet: true`, `datasetDataSources`, `kernelDataSources`. **Chưa thử trên máy này**; nếu lỗi thì tải `.ipynb` lên bằng giao diện web Kaggle (New Notebook → File → Import) và gắn dữ liệu bằng tay |

Dữ liệu trên Kaggle: CSV NetFlow v3 ở dataset `athena21/netflow-v3-datasets`; đồ thị đã dựng của NF-ToN-IoT-v3 là output của notebook `gnn-iot-ids-04-build-graphs` (gắn bằng `kernelDataSources`).
