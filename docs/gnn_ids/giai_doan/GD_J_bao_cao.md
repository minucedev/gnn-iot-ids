# GĐ J. Báo cáo, kiểm tra tái lập, nộp

| Ngày | Task | Máy | Cần xong trước | Notebook |
|---|---|---|---|---|
| 18–19/11 | 3.1–3.4, 9.1–9.3 | Local (chỉ viết), Kaggle CPU (tái lập) | H, I | `21_reproduce` |

Đọc trước: [README.md](README.md), task 3.1–3.4, 9.1–9.3 trong [../TASKS.md](../TASKS.md), mục 0 của KE_HOACH.md (yêu cầu của thầy).

## Mục tiêu

Báo cáo và slide trả lời đủ các yêu cầu của thầy, mọi con số lấy từ file kết quả đã kéo về, và repo chạy lại được từ đầu.

## Việc cần làm

### 1. Báo cáo (local, chỉ viết)

Dàn ý, mỗi chương ghi rõ lấy số liệu từ file nào:

| Chương | Nội dung | Nguồn |
|---|---|---|
| 1. Tổng quan | Bài toán, mục tiêu, chỉ tiêu | KE_HOACH.md mục 1, 3 |
| 2. Cơ sở lý thuyết | IDS cho IoT, NetFlow, GNN, GAT, LightGBM, **lý do chọn hybrid**, AHP | Task 3.1–3.3; ≥ 15 tài liệu, ≥ 6 từ 2022 trở lại |
| 3. Dữ liệu và EDA | Khảo sát 8 bộ, lý do chọn NF-ToN-IoT-v3 và NF-BoT-IoT-v3, EDA, dựng đồ thị, chia tập | `docs/dataset_cards/`, `reports/figures/gd2_*`, `reports/tables/graphs_v2/` |
| 4. Mô hình và thực nghiệm | Baseline ML, DL, GNN đơn, hybrid; môi trường Kaggle T4 và số giờ đã dùng | `reports/kaggle/*/run_info.json` |
| 5. Kết quả | Bảng so sánh, **minh chứng hybrid so với baseline**, ablation, McNemar, phân tích lỗi, AHP, độ trễ | `reports/tables/*`, `reports/figures/gd6_*`, `gd7_*`, `gd8_*` |
| 6. Kết luận | Đạt, chưa đạt, hạn chế, hướng phát triển | `targets.csv` |

Báo cáo phải có mục hạn chế, ghi trung thực chỉ tiêu nào không đạt.

### 2. Slide và demo

- Slide 15–20 trang theo dàn ý trên.
- Video demo 2–3 phút: chạy `20_realtime` trên Kaggle hoặc chiếu log và `alerts_sample.csv`.

### 3. Notebook `21_reproduce` (Kaggle CPU)

Clone repo sạch tại tag cuối, chạy lại `18_aggregate` từ các output đã lưu, so `results.csv` với bản trong repo (lệch ≤ 0,5 điểm %).

## Tiêu chí đạt

1. Báo cáo có đủ 5 mục thầy yêu cầu ngày 12/11: 2 bộ dữ liệu, lý do chọn hybrid, Kaggle T4, EDA, minh chứng hybrid so với baseline ML và DL.
2. Mọi con số trong báo cáo truy được về file trong `reports/`.
3. `21_reproduce` cho kết quả khớp.
4. README của repo hướng dẫn được người ngoài chạy lại trên Kaggle.

## Sau khi xong

- Cập nhật dòng **Kết quả** của các task và bảng theo dõi dòng J.
- Tag `final` trên commit nộp bài.
- Commit: `docs(9.2, 9.3): báo cáo, slide và kiểm tra tái lập`.
