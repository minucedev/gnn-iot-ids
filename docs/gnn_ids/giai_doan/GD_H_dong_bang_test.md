# GĐ H. Đóng băng, chạy test một lần, tổng hợp

| Ngày | Task | Máy | Cần xong trước | Notebook |
|---|---|---|---|---|
| 16/11 | 6.1, 6.2, 6.3, 5.6 | Kaggle GPU (test), CPU (tổng hợp) | G | `17_final_test`, `18_aggregate` |

Đọc trước: [README.md](README.md), task 6.1–6.3, 5.6 trong [../TASKS.md](../TASKS.md), `docs/gnn_ids/freeze_list.md` do GĐ G tạo.

## Mục tiêu

Có bảng kết quả cuối, trung thực, trên tập test: hybrid so với mọi baseline ML, DL và GNN đơn, trên 2 bộ, kèm kiểm định thống kê. Mọi bảng và hình trong báo cáo đều sinh ra ở giai đoạn này.

## Việc cần làm

### 1. Đóng băng (local)

- Kiểm tra `freeze_list.md` đủ: mô hình, cấu hình, ngưỡng, commit.
- `git tag freeze-v1 && git push origin freeze-v1`. **Sau tag này không chỉnh mô hình theo kết quả test.** Nếu buộc phải sửa thì ghi rõ "sau test" trong báo cáo.

### 2. Notebook `17_final_test` (GPU)

1. Clone repo **tại tag `freeze-v1`**.
2. Mọi mô hình trong `freeze_list.md`, cả 2 bộ, 3 seed: chạy trên test **một lần**, dùng ngưỡng trong `thresholds.json`, ghi `split=test` qua `src/evaluate.py`.
3. Lưu dự đoán từng node (`pred_<bộ>_<model>_s<seed>.parquet`, chỉ để trên Kaggle) cho McNemar và phân tích lỗi.
4. Thử chéo (task 5.6): train trên NF-ToN-IoT-v3 rồi test trên NF-BoT-IoT-v3, và ngược lại; chỉ tính các lớp có ở cả hai bộ; cột `dataset` ghi dạng `A→B`.
5. Ghi `run_info.json`.

### 3. Notebook `18_aggregate` (CPU)

Đầu vào là output của các notebook 05, 07, 09, 10, 13, 14, 15, 16, 17 (qua `kernelDataSources`). Sinh ra:
1. `results.csv` tổng, đúng 20 cột của `src/evaluate.py`.
2. `summary_val.csv`, `summary_test.csv`: mean ± std theo (bộ, mô hình).
3. `mcnemar.csv`: hybrid so với baseline tốt nhất (ML, DL, GNN đơn), mỗi bộ, mỗi seed.
4. `targets.csv`: mỗi chỉ tiêu ở KE_HOACH.md mục 3, đạt hay không đạt.
5. Hình `gd6_*.png`: cột F1-macro và FPR có vạch 4,08% và thanh sai số; confusion matrix của hybrid; ROC và PR; F1 theo số tham số; F1 theo độ trễ; đường huấn luyện.
6. `error_analysis.json`: ≥ 3 nhóm lỗi của hybrid, mỗi nhóm có ví dụ.

## Đầu ra

- **Kéo về** `reports/tables/`: `results.csv`, `summary_val.csv`, `summary_test.csv`, `mcnemar.csv`, `targets.csv`.
- **Kéo về** `reports/figures/`: `gd6_*.png`.
- **Kéo về** `reports/kaggle/gnn-iot-ids-17-final-test/` và `.../18-aggregate/`: `run_info.json`, `error_analysis.json`.

## Tiêu chí đạt

1. Tag `freeze-v1` tạo trước khi chạy test; cột `commit` của các dòng test trùng commit của tag.
2. Test chạy đúng một lần cho mỗi (bộ, mô hình, seed).
3. Có kết luận rõ: hybrid có cao hơn baseline tốt nhất không (chênh lệch so với 2 lần std, p-value McNemar < 0,05). **Không đạt thì ghi đúng như vậy.**
4. Mọi số và hình trong báo cáo truy được về file đã kéo về.

## Sau khi xong

- Cập nhật dòng **Kết quả** của task 6.1, 6.2, 6.3, 5.6 và bảng theo dõi dòng H.
- Commit: `chore(6.1-6.3): kết quả test và bảng, hình tổng hợp từ Kaggle`.
