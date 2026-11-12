# GĐ A. Thu hồi kết quả đã chạy trên Kaggle

| Ngày | Task | Máy | Cần xong trước | Notebook |
|---|---|---|---|---|
| 12/11 | 4.1, 5.2, 5.3 | Không chạy notebook | — | — |

Đọc trước: [README.md](README.md) (bối cảnh chung, cách gọi Kaggle API).

## Mục tiêu

Ngày 03/11 đã có 4 notebook chạy trên Kaggle với cách chia tập mới, nhưng kết quả chưa được tải về repo. Giai đoạn này tải các file đó về và xác định chính xác còn thiếu gì, để GĐ D và E không chạy lại phần đã có.

## Đầu vào

| Notebook Kaggle | Trạng thái (12/11) | File cần tải |
|---|---|---|
| `gnn-iot-ids-05-baselines` | Bị dừng giữa chừng | `results_baselines_nf_ton_iot_v3.csv`, `baseline_search_nf_ton_iot_v3.json`, `results_baselines_nf_cicids2018_v3.csv`, `verify_nf_ton_iot_v3.json`, `verify_nf_cicids2018_v3.json` |
| `gnn-iot-ids-07-train-gnn` | Xong | `results_gnn_nf_ton_iot_v3.csv`, `results_gnn_nf_cicids2018_v3.csv`, `results_gnn_nf_ton_iot_v2.csv` (**không** tải `ckpt/`) |
| `gnn-iot-ids-09-baselines-cic` | Xong | `results_baselines_nf_cicids2018_v3.csv`, `baseline_search_nf_cicids2018_v3.json` |
| `gnn-iot-ids-10-baselines-v2` | Xong | `results_baselines_nf_ton_iot_v2.csv`, `baseline_search_nf_ton_iot_v2.json` |

## Việc cần làm

1. Với mỗi notebook, gọi `GET /kernels/output` để lấy danh sách file và `url`. Tải từng file trong bảng trên vào `reports/kaggle/<slug>/`.
2. Ghi phiên bản notebook và commit repo mà notebook đã clone. Commit nằm ở cột `commit` trong CSV, hoặc trong log của notebook.
3. Viết `reports/kaggle/README.md`: bảng gồm notebook, phiên bản, commit, ngày chạy, file đã tải.
4. **Chỉ đọc** các CSV để đếm số dòng theo (dataset, model, seed). Không tính trung bình, không vẽ hình; việc tổng hợp làm ở GĐ H trên Kaggle. Có thể đếm bằng `wsl ... python3 -I` với module `csv` có sẵn.

## Đầu ra

- `reports/kaggle/<slug>/*.csv|json` cho 4 notebook.
- `reports/kaggle/README.md`.
- Một đoạn "còn thiếu" ghi vào dòng **Kết quả** của task 4.1 và 5.3, ví dụ: "NF-ToN-IoT-v3: đủ logreg, random_forest × 3 seed; thiếu xgboost, lightgbm, mlp".

## Tiêu chí đạt

1. Mỗi file đã tải có kích thước khớp với kích thước Kaggle báo; không tải file nào của `ckpt/`.
2. Biết chính xác NF-ToN-IoT-v3 đã có đủ 5 baseline × 3 seed và 3 GNN × 3 seed trên val chưa.
3. Cột `commit` trong các CSV trỏ tới commit có thật trong repo (`git cat-file -t <hash>`).

## Sau khi xong

- Cập nhật bảng theo dõi trong [../LO_TRINH_KAGGLE.md](../LO_TRINH_KAGGLE.md), dòng A.
- Commit: `chore(4.1, 5.3): kéo kết quả notebook 05, 07, 09, 10 từ Kaggle`.
- Phần thiếu của NF-ToN-IoT-v3 chuyển sang GĐ D (baseline) và GĐ E (GNN).

## Lưu ý

- Kết quả của CIC và NF-ToN-IoT-v2 vẫn tải về nhưng không còn bắt buộc (quyết định D8); chỉ dùng làm phụ lục nếu có.
- Không dùng số trong `reports/tables/prelim_split_v1/` vì đó là cách chia tập cũ.
