# GĐ I. AHP và mô phỏng thời gian thực

| Ngày | Task | Máy | Cần xong trước | Notebook |
|---|---|---|---|---|
| 17/11 | 7.1, 7.2, 8.1–8.4 | Kaggle CPU | H | `19_ahp`, `20_realtime` |

Đọc trước: [README.md](README.md), task 7.1, 7.2, 8.1–8.4 trong [../TASKS.md](../TASKS.md), quyết định D6, D7 (điều chỉnh), D10 trong KE_HOACH.md mục 7.

## Mục tiêu

Hoàn thành phần "gợi ý" (AHP chọn mô hình triển khai) và phần "thời gian thực" (đo độ trễ khi phát lại lưu lượng) của đề tài gốc.

## Phần 1. AHP (`19_ahp`)

**Local (chỉ viết tay, không tính toán):**
- Phúc chấm ma trận so sánh cặp 5 tiêu chí theo thang Saaty 1–9: F1-macro, FPR, recall tấn công, độ trễ, kích thước mô hình. Ghi vào `docs/ahp_matrix.json`, kèm một câu lý do cho mỗi cặp. Nên nhờ Phong và Tấn chấm thêm.
- Viết `src/ahp.py`: tính trọng số (vector riêng hoặc trung bình nhân), CI, CR; chuẩn hóa điểm phương án, đảo chiều các tiêu chí càng nhỏ càng tốt; phân tích độ nhạy ±20%. Kèm `tests/test_ahp.py` với ví dụ 3×3 tính tay.

**Kaggle:** đọc `summary_test.csv` (GĐ H) và `docs/ahp_matrix.json`, chạy `src/ahp.py`, xuất kết quả.

**Kéo về:** `reports/tables/ahp_result.json` (trọng số, CR, xếp hạng, mô hình gợi ý), hình `reports/figures/gd7_*.png` (trọng số, độ nhạy).

**Đạt khi:** CR < 0,10 cho mọi ma trận; test 3×3 qua; kết luận có điều kiện, ví dụ "nếu ưu tiên độ trễ thì chọn …".

## Phần 2. Mô phỏng thời gian thực (`20_realtime`)

**Code (local, đẩy GitHub):**
- `src/stream_features.py`: đọc flow theo thứ tự thời gian, gom cửa sổ 5 giây, dựng đồ thị bằng đúng hàm offline, dùng scaler đã fit trên train.
- `detector.py`: nạp mô hình được AHP gợi ý, dự đoán từng cửa sổ, xuất `alerts.csv` (thời gian, cửa sổ, IP, lớp, xác suất, thời gian xử lý).

**Kaggle CPU:**
- Phát lại flow của tập test NF-ToN-IoT-v3, `torch.set_num_threads(1)`.
- Đo độ trễ từ lúc đóng cửa sổ đến lúc có dự đoán, tách phần dựng đồ thị và phần suy luận, p50, p95, p99, trên ≥ 1.000 cửa sổ có warm-up.
- So khớp đặc trưng online với offline (sai khác < 1e-6) và dự đoán online với offline (khớp ≥ 99,9%).
- Tính số cảnh báo giả mỗi giờ (trên đoạn chỉ có Normal) và thời gian phát hiện của từng đợt tấn công (median, max).
- Ghi loại CPU vào `run_info.json`.

**Kéo về:** `reports/tables/latency.json`, `reports/tables/realtime_eval.json`, `reports/kaggle/gnn-iot-ids-20-realtime/alerts_sample.csv` (≤ 1.000 dòng), hình `reports/figures/gd8_*.png`.

**Đạt khi:** p95 ≤ 1 giây; online khớp offline đúng ngưỡng trên; có bảng đạt hoặc không đạt cho từng chỉ tiêu triển khai.

## Sau khi xong

- Cập nhật dòng **Kết quả** của task 7.1, 7.2, 8.1–8.4 và bảng theo dõi dòng I.
- Commit: `feat(7.1, 7.2): AHP chọn mô hình triển khai` và `feat(8.1-8.4): mô phỏng thời gian thực và đo độ trễ trên Kaggle CPU`.
