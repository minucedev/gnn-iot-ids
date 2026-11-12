# GĐ C. EDA và trực quan hóa đồ thị

| Ngày | Task | Máy | Cần xong trước | Notebook |
|---|---|---|---|---|
| 13/11 | 2.1, 2.2, 2.3 | Kaggle CPU | B | `12_eda` (slug `gnn-iot-ids-12-eda`) |

Đọc trước: [README.md](README.md), task 2.1–2.3 trong [../TASKS.md](../TASKS.md).

## Mục tiêu

Đáp ứng yêu cầu "Xử lý EDA" của thầy cho **cả 2 bộ**, và có bằng chứng hình ảnh rằng cấu trúc đồ thị phân biệt được các kiểu tấn công. Đây là luận cứ cho việc dùng GNN trong hybrid.

## Đầu vào

- CSV gốc: `athena21/netflow-v3-datasets` (NF-ToN-IoT-v3, NF-BoT-IoT-v3).
- Đồ thị: output của `gnn-iot-ids-04-build-graphs` (NF-ToN-IoT-v3) và `gnn-iot-ids-11-build-graphs-bot`.
- Phần EDA trong notebook của Tấn (`phamta/dss-cky-bot`, mục 4) để dùng lại cho NF-BoT-IoT-v3.

## Việc cần làm

Viết `scripts/eda.py` (hàm vẽ dùng chung cho 2 bộ) rồi gọi trong notebook.

**Mức flow, mỗi bộ:**
1. Phân bố nhãn gốc và theo 5 lớp (thang log).
2. Số flow theo thời gian, tô màu theo loại tấn công.
3. Boxplot `log(1+x)` của các đặc trưng chính theo lớp.
4. Heatmap tương quan các đặc trưng số.
5. Mutual information của từng đặc trưng với nhãn; chỉ ra ≥ 3 đặc trưng mạnh nhất.
6. Bảng thống kê: min, median, max, % thiếu.

**Mức đồ thị, mỗi bộ:**
7. Phân bố số node và số cạnh mỗi cửa sổ.
8. Bậc vào và bậc ra theo lớp; so sánh DDoS với Normal (dấu hiệu fan-in), dò quét với Normal (fan-out).
9. Edge homophily và class-insensitive homophily.

**Vẽ đồ thị:**
10. ≥ 3 cửa sổ mẫu (bình thường, DDoS, dò quét hoặc backdoor), mỗi hình ≤ 300 node, màu theo lớp, kích thước theo bậc, cùng layout seed.
11. 1 hình sơ đồ quy trình: flow → cửa sổ → đồ thị.

## Đầu ra

- **Kéo về** `reports/figures/`: `gd2_<bộ>_<tên>.png` (dpi ≥ 150).
- **Kéo về** `reports/kaggle/gnn-iot-ids-12-eda/`: `eda_summary.json` (số liệu kèm từng hình và nhận xét 1–2 câu), `run_info.json`.

## Tiêu chí đạt

1. ≥ 12 hình tổng cho 2 bộ; mỗi hình có tiêu đề, nhãn trục, đơn vị.
2. Mỗi hình có nhận xét 1–2 câu dựa trên số liệu trong `eda_summary.json`.
3. Có homophily của cả 2 bộ, kèm một câu nói ý nghĩa với GNN.
4. Notebook chạy "Run All" không lỗi.

## Sau khi xong

- Cập nhật dòng **Kết quả** của task 2.1, 2.2, 2.3 và bảng theo dõi dòng C.
- Commit: `feat(2.1-2.3): EDA và trực quan hóa đồ thị cho NF-ToN-IoT-v3 và NF-BoT-IoT-v3`.
