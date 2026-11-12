# GĐ D. Baseline ML và MLP

| Ngày | Task | Máy | Cần xong trước | Notebook |
|---|---|---|---|---|
| 13/11 | 4.1 | Kaggle CPU | A, B | `13_baselines` (slug `gnn-iot-ids-13-baselines`) |

Đọc trước: [README.md](README.md), task 4.1 trong [../TASKS.md](../TASKS.md), danh sách "còn thiếu" do GĐ A ghi ở dòng **Kết quả** của task 4.1.

## Mục tiêu

Có đủ mốc so sánh **ML** (Logistic Regression, Random Forest, XGBoost, LightGBM) và **DL** (MLP) trên cả 2 bộ, cùng cách chia tập với hybrid. Đây là nửa "baseline" của yêu cầu "minh chứng hybrid cao hơn baseline ML, DL".

## Đầu vào

- Đồ thị: output `gnn-iot-ids-04-build-graphs` (NF-ToN-IoT-v3), `gnn-iot-ids-11-build-graphs-bot`.
- `scripts/run_baselines.py` (có tham số `models=` và `fast=1`).

## Việc cần làm

1. Chỉ chạy phần còn thiếu. NF-ToN-IoT-v3 chạy những mô hình mà GĐ A ghi là thiếu; NF-BoT-IoT-v3 chạy cả 5 mô hình.
2. Lệnh mẫu: `python -I scripts/run_baselines.py <data_dir> <bộ> results_baselines_<bộ>.csv baseline_search_<bộ>.json models=<danh sách>`.
3. Nếu sợ quá giới hạn thời gian của một phiên Kaggle thì tách 2 notebook (mỗi bộ một notebook). Notebook 05 từng bị dừng vì chạy cả 3 bộ trong một lần.
4. Ghi `run_info.json`.

## Đầu ra

- **Kéo về** `reports/kaggle/gnn-iot-ids-13-baselines/`: `results_baselines_<bộ>.csv`, `baseline_search_<bộ>.json`, `run_info.json`.

## Tiêu chí đạt (task 4.1)

1. Dùng đúng `split.json` của từng bộ, không tự chia lại.
2. Chọn cấu hình trên val với ≤ 30 lần thử mỗi mô hình, có log; rồi chạy 3 seed.
3. Mỗi bộ có đủ 5 mô hình × 3 seed (gộp với phần đã có từ GĐ A).
4. Có số tham số hoặc kích thước mô hình, và độ trễ mỗi cửa sổ.
5. Chỉ có dòng `split=val`.

## Sau khi xong

- Cập nhật dòng **Kết quả** của task 4.1 và bảng theo dõi dòng D.
- Commit: `chore(4.1): kéo kết quả baseline 2 bộ từ Kaggle`.
