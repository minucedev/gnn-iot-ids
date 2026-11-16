# Tổng hợp GĐ H (notebook gnn-iot-ids-18-aggregate)

Đầu vào: output của `gnn-iot-ids-17-final-test` (test, commit `abcb5f8` = tag `freeze-v1`) và `gnn-iot-ids-17a-dryrun-val` (dòng val của cùng mô hình và cùng mã, commit `9cae412`).
Mã: `scripts/aggregate_h.py`. Bảng đã kéo về `reports/tables/`, hình `gd6_*.png` về `reports/figures/`; thư mục này giữ `run_info.json`, `error_analysis.json`, `aggregate_h_summary.json`.

- Tên mô hình có hậu tố `_thr` là chế độ ngưỡng đã hiệu chỉnh trên val, không có hậu tố là argmax (xem `../gnn-iot-ids-17-final-test/README.md`).
- Mean ± std của 3 seed, std với ddof = 1. "Hybrid vượt baseline": chênh F1-macro trung bình > 2 · max(std hai bên) VÀ McNemar p < 0,05 ở cả 3 seed.
- Mô hình tốt nhất của từng nhóm (ML, MLP, GNN đơn) trong McNemar chọn theo F1-macro **val**, không theo test.
- `results.csv`: 20 cột; dòng `split=val` mang commit `9cae412` (lần chạy thử 17a), dòng `split=test` mang `abcb5f8`.
