# Kết quả test của GĐ H (notebook gnn-iot-ids-17-final-test)

Repo `minucedev/gnn-iot-ids` tại tag `freeze-v1`, commit `abcb5f84a9dcf0adeb61800aa03c71fb591c2f9a` (cột `commit` của mọi dòng là `abcb5f8`). Chạy trên CPU Kaggle (0 giờ GPU),
16/11/2026 từ 04:48 đến 05:27 UTC (thời điểm từng mục ở `ledger_test.csv`).

## Quy ước tên mô hình trong `results_test_<bộ>.csv`

- Mỗi (bộ, mô hình, seed) chạy test đúng **một lần**; từ cùng một lần dự đoán ghi **hai dòng**:
  - tên mô hình giữ nguyên (ví dụ `lightgbm`): chế độ **argmax** (lớp có xác suất cao nhất);
  - tên mô hình thêm hậu tố `_thr` (ví dụ `lightgbm_thr`): chế độ **ngưỡng đã hiệu chỉnh** trên val (đóng băng ở `reports/kaggle/gnn-iot-ids-16c-aggregate/thresholds.json`, riêng
    `lgbm_neighbor_mean` ở `reports/kaggle/gnn-iot-ids-16h-c-threshold/`): điểm tấn công = 1 − P(Normal); điểm > ngưỡng thì lớp tấn công có xác suất cao nhất, ngược lại Normal.
- Dòng thử chéo (task 5.6) có cột `dataset` dạng `A→B` (mô hình huấn luyện trên A, đánh giá trên test của B); chỉ giữ node B thuộc lớp chung (Normal, DoS_DDoS, Truy_cap_trai_phep) và chỉ xét các
  lớp chung khi dự đoán; đặc trưng node của B được chuẩn hóa lại theo scaler của A. Mô hình: `hybrid_gat_lgbm_oofavg_il`, `gat`, `lightgbm`, `lgbm_neighbor_mean`.
- `hybrid_gat_lgbm_oofavg_il` = hybrid chính (5 GAT nhóm + LightGBM); `hybrid_light` = hybrid nhẹ (chỉ NF-ToN-IoT-v3: 2 GAT nhóm + LightGBM n=100, l=15). Ở NF-BoT-IoT-v3 hybrid nhẹ trùng hybrid
  chính nên không có dòng `hybrid_light` riêng. `lgbm_neighbor_mean` = (c): LightGBM trên 18 đặc trưng + trung bình 18 đặc trưng của node kề.
- `n_params`: baseline cây và `lgbm_neighbor_mean` đếm theo `evaluate.count_params` (tổng số node của mọi cây); hybrid = tham số các GAT + số lá LightGBM (như GĐ F). `train_time_s` của GNN và hybrid
  là thời gian huấn luyện lúc đó (đọc từ CSV val cũ; trống với `hybrid_light`), của baseline là thời gian huấn luyện lại. `infer_ms_per_window` là p95 của 200 lần, CPU 1 luồng, một cửa sổ trung vị.
- `tiers_test_<bộ>.csv`: chỉ số theo tầng: `all`, `initiator` (node có khởi tạo flow, n_out > 0), `receive_only` (node chỉ nhận flow).
- `val_check_test_<bộ>.json`: F1-macro val dựng lại so với F1 val lúc hiệu chỉnh ngưỡng (dừng nếu lệch > 5e-4; thực tế lệch 0). `ledger_test.csv`: thời điểm (UTC) chạy từng mục.
- `pred_<bộ>_<mô hình>_s<seed>.parquet` (cột `y_true`, `pred` = argmax, `pred_thr`, `score` = 1 − P(Normal)) và `nodes_<bộ>_test.parquet` chỉ để trên Kaggle, không tải về.
