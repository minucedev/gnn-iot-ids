# Lộ trình thực hiện trên Kaggle (12/11 – 19/11/2026)

Lập ngày 12/11/2026 theo yêu cầu bổ sung của thầy ([KE_HOACH.md mục 0](KE_HOACH.md#0-yêu-cầu-bổ-sung-của-thầy-12112026)) và quyết định D8–D11. **File này là lịch hiện hành**, thay cho mục 5b và 5c của KE_HOACH.md. Chi tiết tiêu chí từng task vẫn ở [TASKS.md](TASKS.md).

## 1. Nguyên tắc

1. **Mọi xử lý dữ liệu, huấn luyện, đánh giá, tổng hợp số liệu đều chạy trên Kaggle** (tài khoản `minucelowcode`, GPU T4 khi cần). Máy cá nhân **không** train, **không** xử lý dữ liệu, **không** tải file dữ liệu lớn (`.csv` gốc, `.parquet`, `.pt`, checkpoint).
2. Máy cá nhân chỉ dùng để: sửa code và tài liệu, đẩy lên GitHub, điều khiển Kaggle qua API (đẩy notebook, xem trạng thái), **tải về các file kết quả nhỏ** (`.csv`, `.json`, `.png`) để lưu vào repo.
3. Số liệu trong báo cáo chỉ lấy từ file kết quả do Kaggle sinh ra; bảng và hình tổng hợp cũng do một notebook Kaggle sinh ra (giai đoạn H).
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

- **Tên notebook:** `gnn-iot-ids-<số>-<việc>`, số tiếp nối các notebook đã có (01–10).
- **Đầu notebook:** in commit hash của repo đã clone. Commit hash này phải khớp cột `commit` trong file kết quả.
- **Cuối notebook:** ghi `run_info.json` gồm thời gian chạy, loại máy (CPU hay GPU), RAM đỉnh, commit hash.
- **Tải kết quả về:** chỉ tải `.csv`, `.json`, `.png`, `.md` dưới 5 MB vào `reports/kaggle/<tên-notebook>/`. Không tải thư mục `ckpt/`, `processed/`.
- **Gọi API Kaggle từ máy Windows:** dùng `curl` với token ở `~/.kaggle/access_token`, header `Authorization: Bearer <token>`:
  - xem trạng thái: `GET https://www.kaggle.com/api/v1/kernels/status?userName=minucelowcode&kernelSlug=<slug>`
  - liệt kê file output: `GET .../kernels/output?userName=minucelowcode&kernelSlug=<slug>`

## 3. Ngân sách Kaggle

- Quota GPU 30 giờ mỗi tuần. **Tài khoản `minucelowcode` đang dùng chung cho một dự án khác** (các notebook `gd6`–`gd12` chạy từ 05/11 đến 12/11), nên phải kiểm tra số giờ GPU còn lại trên trang Kaggle trước giai đoạn E.
- Notebook không cần GPU (dựng đồ thị, EDA, baseline cây, tổng hợp, AHP) chạy bằng CPU, không tốn quota GPU.
- Mỗi phiên chạy có giới hạn thời gian (notebook 05 từng bị dừng giữa chừng), nên tách việc lớn thành nhiều notebook nhỏ.
- **Ước lượng GPU** cho cả lộ trình: khoảng 6–10 giờ (giai đoạn E, F, G, I). Đây là ước lượng; con số thật ghi vào `run_info.json`. Nếu tổng thực tế vượt 25 giờ thì báo Phúc trước khi chạy tiếp.

## 4. Hiện trạng trên Kaggle (kiểm tra ngày 12/11)

| Notebook | Trạng thái | Output có sẵn (chưa kéo về repo) |
|---|---|---|
| `04-build-graphs` | Xong | Đồ thị đã chia tập cho NF-ToN-IoT-v3, CIC, v2 (`processed/<bộ>/`) |
| `05-baselines` | Bị dừng giữa chừng | `results_baselines_nf_ton_iot_v3.csv`, `baseline_search_nf_ton_iot_v3.json`, một phần CIC, `verify_*.json` |
| `07-train-gnn` | Xong | `results_gnn_{nf_ton_iot_v3, nf_cicids2018_v3, nf_ton_iot_v2}.csv`, 27 checkpoint |
| `09-baselines-cic` | Xong | Baseline CIC còn thiếu (XGBoost, LightGBM, MLP) |
| `10-baselines-v2` | Xong | Baseline NF-ToN-IoT-v2 |

Tức là đã có kết quả val trên **cách chia tập mới** cho GNN (3 bộ) và một phần baseline. Cần kiểm tra file của notebook 05 có đủ 5 mô hình × 3 seed cho NF-ToN-IoT-v3 không.

## 5. Các giai đoạn

| GĐ | Ngày | Việc | Notebook Kaggle | Máy | Task |
|---|---|---|---|---|---|
| A | 12/11 | Thu hồi kết quả đã chạy | — | — | 4.1, 5.2, 5.3 |
| B | 12–13/11 | Dựng đồ thị NF-BoT-IoT-v3 | `11-build-graphs-bot` | CPU | 1.8 |
| C | 13/11 | EDA và trực quan hóa đồ thị | `12-eda` | CPU | 2.1–2.3 |
| D | 13/11 | Baseline ML và MLP | `13-baselines` | CPU | 4.1 |
| E | 14/11 | GNN đơn trên NF-BoT-IoT-v3 | `14-train-gnn-bot` | GPU | 5.2, 5.3 |
| F | 14–15/11 | Hybrid GAT + LightGBM | `15-hybrid` | GPU | 5.7 |
| G | 15/11 | Ablation và hiệu chỉnh ngưỡng | `16-ablation` | GPU | 5.4, 5.5, 5.8 |
| H | 16/11 | **Đóng băng**, chạy test một lần, tổng hợp | `17-final-test`, `18-aggregate` | GPU, CPU | 6.1–6.3, 5.6 |
| I | 17/11 | AHP và mô phỏng thời gian thực | `19-ahp`, `20-realtime` | CPU | 7.1, 7.2, 8.1–8.4 |
| J | 18–19/11 | Báo cáo, slide, kiểm tra tái lập, nộp | `21-reproduce` | CPU | 3.1–3.4, 9.1–9.3 |

### GĐ A. Thu hồi kết quả đã chạy (12/11)

- **Mục tiêu:** đưa kết quả val đã có trên Kaggle vào repo, biết chính xác còn thiếu gì.
- **Làm:** tải các file `.csv`, `.json` của notebook 05, 07, 09, 10 (không tải `ckpt/`) về `reports/kaggle/<tên-notebook>/`; ghi `reports/kaggle/README.md` liệt kê file và commit sinh ra file đó.
- **Không** tổng hợp số ở bước này (việc tổng hợp làm trên Kaggle ở GĐ H).
- **Đạt khi:** biết được NF-ToN-IoT-v3 đã có đủ 5 baseline × 3 seed và 3 GNN × 3 seed trên val hay chưa; thiếu ở đâu thì ghi vào GĐ D, E.

### GĐ B. Dựng đồ thị NF-BoT-IoT-v3 (12–13/11, task 1.8)

- **Đầu vào:** NF-BoT-IoT-v3 trên Kaggle (`athena21/netflow-v3-datasets`, đối chiếu với `seyhed/nf-bot-iot-v3` mà Tấn dùng); `src/build_graph.py`.
- **Làm:**
  - thêm ánh xạ nhãn: DoS, DDoS → DoS_DDoS; Reconnaissance, Theft → Truy_cap_trai_phep; Benign → Normal;
  - tìm số khối chia tập bằng `scripts/split_search.py`;
  - dựng đồ thị cửa sổ 5 giây, chia blocked, gap 1;
  - chạy `scripts/verify_graphs.py`.
- **Đầu ra trên Kaggle:** `processed/nf_bot_iot_v3/` (đồ thị, `split.json`, `scaler.pkl`).
- **Kéo về:** `summary.json`, `graph_stats.csv`, `split_search_nf_bot_iot_v3.json`.
- **Đạt khi:** đủ 5 tiêu chí của task 1.8 (số dòng khớp UQ, mọi lớp có mặt đủ node ở 3 tập, không lấy mẫu flow ngẫu nhiên, kiểm tra rò rỉ và tái lập đạt).

### GĐ C. EDA và trực quan hóa đồ thị (13/11, task 2.1–2.3)

- **Đầu vào:** CSV gốc của 2 bộ; output của `04-build-graphs` (NF-ToN-IoT-v3) và `11-build-graphs-bot`; phần EDA trong notebook của Tấn (`phamta/dss-cky-bot`).
- **Làm:**
  - **EDA mức flow:** phân bố lớp, phân bố theo thời gian, boxplot log theo lớp, tương quan, mutual information;
  - **EDA mức đồ thị:** số node, cạnh mỗi cửa sổ; bậc vào/ra theo lớp; homophily;
  - **Vẽ đồ thị:** ≥ 3 cửa sổ mẫu (bình thường, DDoS, dò quét hoặc backdoor).
- **Kéo về:** các hình `gd2_*.png` vào `reports/figures/`; `eda_summary.json` (bảng số kèm từng hình).
- **Đạt khi:** ≥ 12 hình cho 2 bộ, mỗi hình có tiêu đề, nhãn trục, đơn vị; nhận xét 1–2 câu cho mỗi hình nằm trong `eda_summary.json` hoặc notebook.

### GĐ D. Baseline ML và MLP (13/11, task 4.1)

- **Đầu vào:** đồ thị của 2 bộ; `scripts/run_baselines.py`.
- **Làm:**
  - LR, RF, XGBoost, LightGBM, MLP trên 18 đặc trưng node;
  - chọn cấu hình trên val (seed 42), rồi chạy 3 seed;
  - chỉ ghi `split=val`;
  - chỉ chạy những phần GĐ A cho thấy còn thiếu.
- **Kéo về:** `results_baselines_<bộ>.csv`, `baseline_search_<bộ>.json`.
- **Đạt khi:** mỗi bộ có đủ 5 mô hình × 3 seed; có số tham số và độ trễ mỗi cửa sổ.

### GĐ E. GNN đơn trên NF-BoT-IoT-v3 (14/11, task 5.2, 5.3)

- **Đầu vào:** output `11-build-graphs-bot`; `scripts/run_gnn.py`. NF-ToN-IoT-v3 đã có kết quả từ `07-train-gnn`, chỉ chạy lại nếu GĐ A thấy thiếu.
- **Làm:** GCN, GraphSAGE, GAT nhẹ, mỗi mô hình 3 seed; lưu checkpoint GAT để dùng cho GĐ F.
- **Kéo về:** `results_gnn_nf_bot_iot_v3.csv`, `run_info.json`.
- **Đạt khi:** đủ 3 GNN × 3 seed; GAT ≤ 100.000 tham số.

### GĐ F. Hybrid GAT + LightGBM (14–15/11, task 5.7)

- **Đầu vào:** đồ thị của 2 bộ; `src/hybrid.py` (viết mới, có `tests/test_hybrid.py`).
- **Làm:**
  1. **Embedding out-of-fold cho train:** chia train thành 5 nhóm khối thời gian. Với mỗi nhóm, huấn luyện GAT trên 4 nhóm còn lại rồi lấy embedding 64 chiều cho nhóm đó.
  2. **Embedding cho val:** dùng GAT huấn luyện trên toàn bộ train.
  3. Ghép embedding với 18 đặc trưng gốc thành 82 cột; LightGBM học trên train, chọn cấu hình trên val.
  4. Chạy 3 seed × 2 bộ; chạy thêm bản không dùng out-of-fold để so sánh.
- **Kéo về:** `results_hybrid_<bộ>.csv`, `hybrid_search_<bộ>.json`, `run_info.json`.
- **Đạt khi:** đủ 5 tiêu chí của task 5.7.

### GĐ G. Ablation và hiệu chỉnh ngưỡng (15/11, task 5.4, 5.5, 5.8)

- **Ablation hybrid (5 phương án):**
  - (a) LightGBM chỉ 18 đặc trưng;
  - (b) GAT đơn;
  - (c) LightGBM + đặc trưng hàng xóm tính tay (không học);
  - (d) LightGBM chỉ embedding;
  - (e) hybrid đầy đủ.
- **Ablation GAT (5 chiều):** số head, số lớp, độ rộng, có hoặc không đặc trưng cạnh, kích thước cửa sổ. Riêng chiều kích thước cửa sổ cần dựng thêm đồ thị W = 1, 2, 10 giây cho NF-ToN-IoT-v3 (chạy CPU trong `16-ablation` hoặc tách notebook riêng nếu quá giới hạn thời gian).
- **Bản cải tiến của GAT** (task 5.4): thử focal loss, residual + LayerNorm.
- **Hiệu chỉnh ngưỡng quyết định:** chọn trên val để FPR ≤ 4,08%; ghi ngưỡng vào `thresholds.json`.
- **Kéo về:** `ablation_hybrid_<bộ>.csv`, `ablation_gat.csv`, `thresholds.json`.
- **Đạt khi:** chỉ dùng val; mỗi điểm có 3 seed; quyết định có công nhận bản cải tiến hay không theo đúng tiêu chí của task 5.4.

### GĐ H. Đóng băng, chạy test, tổng hợp (16/11, task 6.1–6.3, 5.6)

- **Đóng băng:** tạo tag git `freeze-v1` trước khi chạy test. Sau tag này không chỉnh mô hình theo kết quả test.
- **`17-final-test` (GPU):**
  - mọi mô hình của 2 bộ chạy test **một lần**, 3 seed, dùng ngưỡng ở `thresholds.json`;
  - lưu dự đoán từng node để làm kiểm định McNemar;
  - thử chéo: train bộ này, test bộ kia, chỉ tính các lớp chung (task 5.6).
- **`18-aggregate` (CPU):** đọc output của các notebook trước qua `kernel_sources`, sinh:
  - `results.csv` tổng (đúng 20 cột của `src/evaluate.py`);
  - bảng mean ± std;
  - McNemar (hybrid so với baseline tốt nhất);
  - hình: cột F1-macro và FPR có vạch 4,08%, confusion matrix, ROC/PR, F1 theo số tham số, F1 theo độ trễ, đường huấn luyện.
- **Kéo về:** `reports/tables/results.csv`, các bảng `.csv`, hình `gd6_*.png`.
- **Đạt khi:** mọi số trong báo cáo truy được về file đã kéo về; bảng ghi rõ chỉ tiêu nào đạt, chỉ tiêu nào không.

### GĐ I. AHP và mô phỏng thời gian thực (17/11, task 7.1, 7.2, 8.1–8.4)

- **`19-ahp` (CPU):**
  - đầu vào là ma trận so sánh cặp do Phúc chấm (`docs/ahp_matrix.json`, viết tay ở local) và bảng kết quả của GĐ H;
  - đầu ra: trọng số, CR, xếp hạng, phân tích độ nhạy ±20%.
- **`20-realtime` (CPU):** `src/stream_features.py` phát lại flow của tập test theo thời gian, `detector.py` xuất cảnh báo, đo độ trễ p50/p95/p99 khi giới hạn 1 luồng.
- **Thay đổi so với D7:** vì không chạy trên máy cá nhân, độ trễ đo trên **CPU của Kaggle giới hạn 1 luồng**, ghi rõ loại CPU trong `run_info.json`. Ghi điều chỉnh này vào KE_HOACH.md mục 7.
- **Kéo về:** `ahp_result.json`, `latency.json`, `alerts_sample.csv`, hình `gd7_*.png`, `gd8_*.png`.

### GĐ J. Báo cáo và nộp (18–19/11, task 3.1–3.4, 9.1–9.3)

- Viết báo cáo, slide ở local, chỉ dùng số và hình đã kéo về.
- **`21-reproduce` (CPU):** clone repo sạch tại tag cuối, chạy lại `18-aggregate`, so khớp `results.csv` (lệch ≤ 0,5 điểm %).
- **Đạt khi:** báo cáo có đủ các mục thầy yêu cầu: lý do chọn mô hình hybrid, 2 bộ dữ liệu, EDA, môi trường Kaggle, minh chứng hybrid so với baseline ML và DL.

## 6. Bảng theo dõi

Cập nhật sau mỗi lần chạy. Giờ GPU lấy từ `run_info.json`.

| GĐ | Notebook | Phiên bản | Commit repo | Trạng thái | Giờ GPU | File đã kéo về |
|---|---|---|---|---|---|---|
| A | 05, 07, 09, 10 | — | — | Chưa làm | 0 | |
| B | 11-build-graphs-bot | | | Chưa làm | 0 | |
| C | 12-eda | | | Chưa làm | 0 | |
| D | 13-baselines | | | Chưa làm | 0 | |
| E | 14-train-gnn-bot | | | Chưa làm | | |
| F | 15-hybrid | | | Chưa làm | | |
| G | 16-ablation | | | Chưa làm | | |
| H | 17-final-test, 18-aggregate | | | Chưa làm | | |
| I | 19-ahp, 20-realtime | | | Chưa làm | 0 | |
| J | 21-reproduce | | | Chưa làm | 0 | |
