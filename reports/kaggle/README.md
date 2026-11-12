# Kết quả kéo về từ Kaggle

Tài khoản Kaggle: `minucelowcode`. Kéo ngày 12/11/2026 (GĐ A, xem [giai_doan/GD_A_thu_hoi_ket_qua.md](../../docs/gnn_ids/giai_doan/GD_A_thu_hoi_ket_qua.md)) bằng `curl` gọi Kaggle API. Chỉ tải `.csv` và `.json` nhỏ; không tải `ckpt/`. Cả 12 file đều khớp kích thước với `Content-Length` Kaggle trả về.

Toàn bộ kết quả dưới đây là `split=val`, `task=multiclass`, cách chia tập mới (v2). Không lẫn với `reports/tables/prelim_split_v1/` (cách chia cũ, không dùng).

## Bảng notebook

| Notebook | Phiên bản hiện tại | Trạng thái | Commit repo đã clone | Ngày chạy | File đã tải |
|---|---|---|---|---|---|
| [gnn-iot-ids-05-baselines](gnn-iot-ids-05-baselines/) | 2 | Bị dừng giữa chừng (`cancelAcknowledged`), vẫn có output | `9d7761a` (03/11/2026 08:31 +07) | 03/11/2026 (*) | `baseline_search_nf_ton_iot_v3.json` (1903 B), `results_baselines_nf_ton_iot_v3.csv` (6412 B), `results_baselines_nf_cicids2018_v3.csv` (3327 B), `verify_nf_ton_iot_v3.json` (1234 B), `verify_nf_cicids2018_v3.json` (1236 B) |
| [gnn-iot-ids-07-train-gnn](gnn-iot-ids-07-train-gnn/) | 2 (GPU T4) | Xong (`complete`) | `9d7761a` | 03/11/2026 (*) | `results_gnn_nf_ton_iot_v3.csv` (3870 B), `results_gnn_nf_cicids2018_v3.csv` (3634 B), `results_gnn_nf_ton_iot_v2.csv` (3882 B). Bỏ qua 27 file `ckpt/*.pt` |
| [gnn-iot-ids-09-baselines-cic](gnn-iot-ids-09-baselines-cic/) | 1 | Xong | `4d98421` (03/11/2026 12:40 +07) | 03/11/2026 (*) | `results_baselines_nf_cicids2018_v3.csv` (3683 B), `baseline_search_nf_cicids2018_v3.json` (814 B) |
| [gnn-iot-ids-10-baselines-v2](gnn-iot-ids-10-baselines-v2/) | 1 | Xong | `4d98421` | 03/11/2026 (*) | `results_baselines_nf_ton_iot_v2.csv` (6354 B), `baseline_search_nf_ton_iot_v2.json` (1900 B) |

(*) Ngày chạy lấy theo lời Phúc (03/11) và ngày của commit đã clone (cùng ngày). CSV không có cột ngày. Trường `lastRunTime` của Kaggle API báo `2026-11-09T14:11–14:12Z` cho cả 4 notebook, cách nhau vài giây, nên không phải thời điểm chạy thật và không dùng. "Phiên bản hiện tại" là `currentVersionNumber` lúc kéo; API output luôn trả output của phiên bản mới nhất.

Output của 4 notebook không có `run_info.json`, nên chưa có giờ GPU và RAM đỉnh.

Cột `commit` của các CSV chỉ có hai giá trị, `9d7761a` và `4d98421` (05 và 07 dùng `9d7761a`; 09 và 10 dùng `4d98421`). Đã kiểm bằng `git cat-file -t`: cả hai đều là `commit` có thật trong repo.

## Số dòng theo (dataset, model, task, seed, split)

Chỉ đếm. Mỗi ô là số seed có dòng kết quả; mọi dòng đều `task=multiclass`, `split=val`, không có dòng trùng.

### NF-ToN-IoT-v3 (bộ chính)

| Model | Seed có kết quả | Nguồn |
|---|---|---|
| logreg | 42, 43, 44 | 05 |
| random_forest | 42, 43, 44 | 05 |
| xgboost | 42, 43, 44 | 05 |
| lightgbm | 42, 43, 44 | 05 |
| mlp | 42, 43, 44 | 05 |
| gcn | 42, 43, 44 | 07 |
| sage | 42, 43, 44 | 07 |
| gat | 42, 43, 44 | 07 |

15 dòng baseline + 9 dòng GNN = 24 dòng.

### Phụ lục (không còn bắt buộc theo quyết định D8)

| Dataset | Nguồn | Model × seed |
|---|---|---|
| NF-CICIDS2018-v3 | 05 | logreg 42, 43, 44; random_forest 42, 43, 44; xgboost 42, 43 (dừng giữa chừng) |
| NF-CICIDS2018-v3 | 09 | xgboost 42, 43, 44; lightgbm 42, 43, 44; mlp 42, 43, 44 |
| NF-CICIDS2018-v3 | 07 | gcn, sage, gat × 42, 43, 44 |
| NF-ToN-IoT-v2 | 10 | logreg, random_forest, xgboost, lightgbm, mlp × 42, 43, 44 |
| NF-ToN-IoT-v2 | 07 | gcn, sage, gat × 42, 43, 44 |

Lưu ý với NF-CICIDS2018-v3: xgboost seed 42, 43 có mặt ở cả 05 (commit `9d7761a`) và 09 (commit `4d98421`). Hai CSV cùng tên `results_baselines_nf_cicids2018_v3.csv` nằm ở hai thư mục khác nhau, không gộp.

### Log tìm siêu tham số và kiểm tra đồ thị

Số lần thử mỗi mô hình trong `baseline_search_*.json` (đều ≤ 30):

| File | logreg | random_forest | xgboost | lightgbm | mlp |
|---|---|---|---|---|---|
| `baseline_search_nf_ton_iot_v3.json` (05) | 3 | 3 | 3 | 3 | 2 |
| `baseline_search_nf_ton_iot_v2.json` (10) | 3 | 3 | 3 | 3 | 2 |
| `baseline_search_nf_cicids2018_v3.json` (09) | — | — | 2 | 2 | 1 |

`verify_nf_ton_iot_v3.json` và `verify_nf_cicids2018_v3.json`: mỗi file 10 kiểm tra, 10 đạt.

## NF-ToN-IoT-v3: còn thiếu gì

- Baseline (task 4.1): đủ 5 mô hình × 3 seed trên val, **chỉ `task=multiclass`**. Thiếu: dòng `task=binary` (tiêu chí 6). Kích thước và độ trễ đã có trong CSV (`n_params`, `size_kb`, `infer_ms_per_window`).
- GNN (task 5.2, 5.3): đủ gcn, sage, gat × 3 seed trên val, **chỉ `task=multiclass`**. Thiếu: dòng `task=binary`; log tìm siêu tham số GNN (5.2 tiêu chí 3); đường loss và F1 theo epoch (5.2 tiêu chí 4); bảng đếm tham số theo từng lớp và trọng số attention của GAT (5.3 tiêu chí 1, 4). 07 chỉ cho CSV kết quả và `ckpt/`; chưa biết `ckpt/` có đường học hay không vì không tải.
- Cả hai nhóm: toàn bộ NF-BoT-IoT-v3 chưa chạy.
