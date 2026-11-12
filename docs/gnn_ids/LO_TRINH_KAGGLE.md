# Lộ trình thực hiện trên Kaggle (12/11 – 19/11/2026)

Lập ngày 12/11/2026 theo yêu cầu bổ sung của thầy ([KE_HOACH.md mục 0](KE_HOACH.md#0-yêu-cầu-bổ-sung-của-thầy-12112026)) và quyết định D8–D12. **File này là lịch hiện hành**, thay cho mục 5b và 5c của KE_HOACH.md. Chi tiết từng giai đoạn nằm trong thư mục [giai_doan/](giai_doan/README.md); tiêu chí từng task ở [TASKS.md](TASKS.md).

## 1. Nguyên tắc

1. **Mọi xử lý dữ liệu, huấn luyện, đánh giá, tổng hợp số liệu đều chạy trên Kaggle** (tài khoản `minucelowcode`, GPU T4 khi cần). Máy cá nhân **không** train, **không** xử lý dữ liệu, **không** tải file dữ liệu lớn (`.csv` gốc, `.parquet`, `.pt`, checkpoint).
2. Máy cá nhân chỉ dùng để: sửa code và tài liệu, đẩy lên GitHub, điều khiển Kaggle qua API (đẩy notebook, xem trạng thái), **tải về các file kết quả nhỏ** (`.csv`, `.json`, `.png`) để lưu vào repo.
3. Số liệu trong báo cáo chỉ lấy từ file kết quả do Kaggle sinh ra; bảng và hình tổng hợp cũng do một notebook Kaggle sinh ra (GĐ H).
4. Quy ước cũ vẫn giữ: chọn cấu hình chỉ trên val; test chạy **một lần** sau khi đóng băng; 3 seed (42, 43, 44); commit thẳng lên `main`.

Lý do thêm: từ 12/11 máy Windows chặn `python.exe` của `.venv` và `kaggle.exe` (Application Control), nên local cũng không chạy được Python của dự án.

## 2. Quy trình chuẩn cho mỗi notebook

```
sửa code trong repo ──► push GitHub (main)
        │
        ▼
notebook Kaggle: git clone repo ──► cài requirements ──► chạy scripts/… ──► ghi /kaggle/working/
        │                                           (đầu vào: dataset Kaggle hoặc output
        │                                            của notebook trước qua kernel_sources)
        ▼
local: gọi Kaggle API xem trạng thái ──► tải file nhỏ về reports/kaggle/<tên-notebook>/ ──► commit
```

Chi tiết cách viết notebook và cách gọi Kaggle API bằng `curl`: [giai_doan/README.md](giai_doan/README.md).

## 3. Ngân sách Kaggle

- Quota GPU 30 giờ mỗi tuần. **Tài khoản `minucelowcode` đang dùng chung cho một dự án khác** (các notebook `gd6`–`gd12` chạy từ 05/11 đến 12/11), nên phải kiểm tra số giờ GPU còn lại trên trang Kaggle trước GĐ E.
- Notebook không cần GPU (dựng đồ thị, EDA, baseline cây, tổng hợp, AHP) chạy bằng CPU, không tốn quota GPU.
- Mỗi phiên chạy có giới hạn thời gian (notebook 05 từng bị dừng giữa chừng), nên tách việc lớn thành nhiều notebook nhỏ.
- **Ước lượng GPU** cho cả lộ trình: khoảng 6–10 giờ (GĐ E, F, G, H). Đây là ước lượng; con số thật ghi vào `run_info.json`. Nếu tổng thực tế vượt 25 giờ thì báo Phúc trước khi chạy tiếp.

## 4. Hiện trạng trên Kaggle (kiểm tra ngày 12/11)

| Notebook | Trạng thái | Output có sẵn (chưa kéo về repo) |
|---|---|---|
| `04-build-graphs` | Xong | Đồ thị đã chia tập cho NF-ToN-IoT-v3, CIC, v2 (`processed/<bộ>/`) |
| `05-baselines` | Bị dừng giữa chừng | `results_baselines_nf_ton_iot_v3.csv`, `baseline_search_nf_ton_iot_v3.json`, một phần CIC, `verify_*.json` |
| `07-train-gnn` | Xong | `results_gnn_{nf_ton_iot_v3, nf_cicids2018_v3, nf_ton_iot_v2}.csv`, 27 checkpoint |
| `09-baselines-cic` | Xong | Baseline CIC còn thiếu (XGBoost, LightGBM, MLP) |
| `10-baselines-v2` | Xong | Baseline NF-ToN-IoT-v2 |

## 5. Các giai đoạn

| GĐ | Ngày | Việc | Notebook Kaggle | Máy | Cần xong trước | Task |
|---|---|---|---|---|---|---|
| [A](giai_doan/GD_A_thu_hoi_ket_qua.md) | 12/11 | Thu hồi kết quả đã chạy | — | — | — | 4.1, 5.2, 5.3 |
| [B](giai_doan/GD_B_do_thi_bot_iot.md) | 12–13/11 | Dựng đồ thị NF-BoT-IoT-v3 | `11-build-graphs-bot` | CPU | — | 1.8 |
| [C](giai_doan/GD_C_eda.md) | 13/11 | EDA và trực quan hóa đồ thị | `12-eda` | CPU | B | 2.1–2.3 |
| [D](giai_doan/GD_D_baseline.md) | 13/11 | Baseline ML và MLP | `13-baselines` | CPU | A, B | 4.1 |
| [E](giai_doan/GD_E_gnn_bot_iot.md) | 14/11 | GNN đơn trên NF-BoT-IoT-v3 | `14-train-gnn-bot` | GPU | B | 5.2, 5.3 |
| [F](giai_doan/GD_F_hybrid.md) | 14–15/11 | Hybrid GAT + LightGBM | `15-hybrid` | GPU | A, E | 5.7 |
| [G](giai_doan/GD_G_ablation.md) | 15/11 | Ablation, bản cải tiến GAT, hiệu chỉnh ngưỡng | `16-ablation` | GPU | D, F | 5.4, 5.5, 5.8 |
| [H](giai_doan/GD_H_dong_bang_test.md) | 16/11 | **Đóng băng**, chạy test một lần, tổng hợp | `17-final-test`, `18-aggregate` | GPU, CPU | G | 6.1–6.3, 5.6 |
| [I](giai_doan/GD_I_ahp_realtime.md) | 17/11 | AHP và mô phỏng thời gian thực | `19-ahp`, `20-realtime` | CPU | H | 7.1, 7.2, 8.1–8.4 |
| [J](giai_doan/GD_J_bao_cao.md) | 18–19/11 | Báo cáo, kiểm tra tái lập, nộp | `21-reproduce` | CPU | H, I | 3.1–3.4, 9.1–9.3 |

C và D chạy song song được sau B; E chạy song song với C, D. **Mốc cứng: 16/11** đóng băng mô hình bằng tag `freeze-v1` trước khi xem tập test.

## 6. Bảng theo dõi

Cập nhật sau mỗi lần chạy. Giờ GPU lấy từ `run_info.json`.

| GĐ | Notebook | Phiên bản | Commit repo | Trạng thái | Giờ GPU | File đã kéo về |
|---|---|---|---|---|---|---|
| A | 05, 07, 09, 10 | 05: v2 (bị dừng), 07: v2, 09: v1, 10: v1 | 05, 07: `9d7761a`; 09, 10: `4d98421` | Xong (12/11) | 0 | 12 file `.csv`/`.json` trong `reports/kaggle/<slug>/`, không tải `ckpt/`; chi tiết ở [reports/kaggle/README.md](../../reports/kaggle/README.md) |
| B | 11-build-graphs-bot | | | Chưa làm | 0 | |
| C | 12-eda | | | Chưa làm | 0 | |
| D | 13-baselines | | | Chưa làm | 0 | |
| E | 14-train-gnn-bot | | | Chưa làm | | |
| F | 15-hybrid | | | Chưa làm | | |
| G | 16-ablation | | | Chưa làm | | |
| H | 17-final-test, 18-aggregate | | | Chưa làm | | |
| I | 19-ahp, 20-realtime | | | Chưa làm | 0 | |
| J | 21-reproduce | | | Chưa làm | 0 | |
