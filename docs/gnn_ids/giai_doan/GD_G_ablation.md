# GĐ G. Ablation, bản cải tiến GAT, hiệu chỉnh ngưỡng

| Ngày | Task | Máy | Cần xong trước | Notebook |
|---|---|---|---|---|
| 15/11 | 5.4, 5.5, 5.8 | Kaggle GPU T4 (dựng đồ thị W khác chạy CPU) | D, F | `16_ablation` (slug `gnn-iot-ids-16-ablation`); có thể tách thêm `16b_windows` |

Đọc trước: [README.md](README.md), task 5.4, 5.5, 5.8 trong [../TASKS.md](../TASKS.md).

## Mục tiêu

Tạo **minh chứng** cho yêu cầu "hybrid cao hơn baseline" của thầy: chứng minh từng phần của hybrid có đóng góp, và chọn ngưỡng quyết định để FPR ≤ 4,08%. **Mọi việc chỉ dùng val.**

## Việc cần làm

### 1. Ablation hybrid (task 5.8), cả 2 bộ, 3 seed

| Phương án | Đầu vào của bộ phân loại | Trả lời câu hỏi |
|---|---|---|
| (a) | LightGBM, chỉ 18 đặc trưng | Mốc ML tốt nhất (lấy từ GĐ D) |
| (b) | GAT đơn | Mốc GNN (lấy từ GĐ E và notebook 07) |
| (c) | LightGBM, 18 đặc trưng + trung bình 18 đặc trưng của các node kề (tính tay, không học) | Thông tin hàng xóm tính tay đã đủ chưa |
| (d) | LightGBM, chỉ embedding GAT | Embedding một mình mạnh tới đâu |
| (e) | Hybrid đầy đủ | Lấy từ GĐ F |

Phương án (c) và (d) cần viết thêm vào `src/hybrid.py`.

### 2. Ablation GAT (task 5.5), NF-ToN-IoT-v3, 3 seed

Đổi một yếu tố mỗi lần, còn lại giữ cấu hình chính:
- số head: 1, 2, 4, 8;
- số lớp: 1, 2, 3;
- độ rộng mỗi head: 8, 16, 32, 64;
- đặc trưng cạnh: có, không;
- kích thước cửa sổ: 1, 2, 5, 10 giây.

Chiều cửa sổ cần dựng thêm đồ thị W = 1, 2, 10 giây bằng `scripts/build_dataset_cli.py`. Việc này chạy CPU và tốn RAM, nên tách notebook `16b_windows` nếu một phiên không đủ thời gian.

### 3. Bản cải tiến GAT (task 5.4)

Thử focal loss, và residual + LayerNorm. Chỉ công nhận nếu đạt tiêu chí của task 5.4 (tăng F1-macro hoặc recall lớp hiếm lớn hơn 2 lần std, hoặc giảm ≥ 30% tham số mà F1 giảm ≤ 1 điểm %). Nếu được công nhận thì dùng bản này trong hybrid.

### 4. Hiệu chỉnh ngưỡng quyết định

Với hybrid và mọi baseline, chọn trên val ngưỡng xác suất "tấn công" sao cho FPR ≤ 4,08% mà recall tấn công cao nhất. Ghi vào `thresholds.json` theo (bộ, mô hình, seed). Ngưỡng này giữ nguyên khi chạy test ở GĐ H.

## Đầu ra

- **Kéo về** `reports/kaggle/gnn-iot-ids-16-ablation/`: `ablation_hybrid_<bộ>.csv`, `ablation_gat.csv`, `improved_gat.csv`, `thresholds.json`, `run_info.json`.

## Tiêu chí đạt

1. Chỉ dùng val; mỗi điểm có 3 seed.
2. Ablation hybrid đủ 5 phương án trên cả 2 bộ.
3. Ablation GAT đủ 5 chiều.
4. Kết luận về bản cải tiến ghi rõ đạt hay không theo tiêu chí của task 5.4.
5. `thresholds.json` có ngưỡng cho mọi mô hình sẽ chạy test.

## Sau khi xong

- Cập nhật dòng **Kết quả** của task 5.4, 5.5, 5.8 và bảng theo dõi dòng G.
- Chốt danh sách mô hình và cấu hình sẽ chạy test, ghi vào `docs/gnn_ids/freeze_list.md`. Đây là đầu vào của GĐ H.
- Commit: `chore(5.4, 5.5, 5.8): kết quả ablation và ngưỡng từ Kaggle`.
