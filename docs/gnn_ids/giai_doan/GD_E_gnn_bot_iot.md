# GĐ E. GNN đơn trên NF-BoT-IoT-v3

| Ngày | Task | Máy | Cần xong trước | Notebook |
|---|---|---|---|---|
| 14/11 | 5.2, 5.3 | Kaggle GPU T4 | B | `14_train_gnn_bot` (slug `gnn-iot-ids-14-train-gnn-bot`) |

Đọc trước: [README.md](README.md), task 5.2, 5.3 trong [../TASKS.md](../TASKS.md). **Kiểm tra số giờ GPU còn lại** trên Kaggle trước khi chạy (tài khoản dùng chung với một dự án khác).

## Mục tiêu

Có kết quả GCN, GraphSAGE, GAT nhẹ trên NF-BoT-IoT-v3, để (1) làm baseline "GNN đơn" cho hybrid và (2) có checkpoint GAT cho GĐ F. NF-ToN-IoT-v3 đã có kết quả từ notebook 07; chỉ chạy lại nếu GĐ A ghi là thiếu.

## Đầu vào

- Output `gnn-iot-ids-11-build-graphs-bot`.
- `scripts/run_gnn.py`; mô hình trong `src/models.py` (GAT: 2 lớp GATv2, 4 head × 16, dùng đặc trưng cạnh, khoảng 11.700 tham số).

## Việc cần làm

1. `python -I scripts/run_gnn.py <data_dir> nf_bot_iot_v3 results_gnn_nf_bot_iot_v3.csv /kaggle/working/ckpt models=gcn,sage,gat`.
2. Kiểm tra F1-macro chỉ tính trên các lớp có mặt (bộ này không có Botnet_Backdoor). Nếu `MAIN4` trong `src/train.py` gây lỗi hoặc làm điểm sai thì sửa ở local, đẩy lên GitHub rồi chạy lại.
3. Ghi `run_info.json`, kèm số giờ GPU đã dùng.

## Đầu ra

- **Trên Kaggle**, làm đầu vào cho GĐ F, G, H: `ckpt/nf_bot_iot_v3_<model>_s<seed>.pt`.
- **Kéo về** `reports/kaggle/gnn-iot-ids-14-train-gnn-bot/`: `results_gnn_nf_bot_iot_v3.csv`, `run_info.json`.

## Tiêu chí đạt (task 5.2, 5.3)

1. Đủ 3 GNN × 3 seed, chỉ `split=val`.
2. GAT ≤ 100.000 tham số, file ≤ 1 MB.
3. Std accuracy giữa 3 seed ≤ 1,5 điểm %; nếu lớn hơn thì ghi rõ.
4. Có đường loss và F1 trên val theo epoch (lưu trong CSV hoặc JSON).

## Sau khi xong

- Cập nhật dòng **Kết quả** của task 5.2, 5.3 và bảng theo dõi dòng E (ghi số giờ GPU).
- Commit: `chore(5.2, 5.3): kết quả GNN trên NF-BoT-IoT-v3 từ Kaggle`.
