# GĐ F. Hybrid GAT + LightGBM

| Ngày | Task | Máy | Cần xong trước | Notebook |
|---|---|---|---|---|
| 14–15/11 | 5.7 | Kaggle GPU T4 | A, E | `15_hybrid` (slug `gnn-iot-ids-15-hybrid`) |

Đọc trước: [README.md](README.md), task 5.7 trong [../TASKS.md](../TASKS.md), quyết định D9 trong KE_HOACH.md mục 7. **Kiểm tra giờ GPU còn lại** trước khi chạy.

## Mục tiêu

Xây mô hình chính theo yêu cầu "mô hình hybrid" của thầy. GAT học ngữ cảnh hàng xóm, LightGBM tận dụng đặc trưng dạng bảng.

```
đồ thị cửa sổ ──► GAT (2 lớp, 4 head × 16) ──► embedding 64 chiều / node ─┐
                                                                         ├─► ghép 82 cột ──► LightGBM ──► 5 lớp
18 đặc trưng node (đã chuẩn hóa) ────────────────────────────────────────┘
```

## Đầu vào

- Đồ thị của 2 bộ (output `04-build-graphs`, `11-build-graphs-bot`).
- Cấu hình GAT của task 5.3; tham khảo checkpoint của `07-train-gnn` và `14-train-gnn-bot`.

## Việc cần làm

### 1. Viết code (local, đẩy lên GitHub)

- `src/hybrid.py`:
  - `extract_embeddings(model, graphs)`: trả vector ẩn của lớp GAT cuối cho từng node, giữ đúng thứ tự node;
  - `oof_embeddings(graphs_train, k=5)`: chia train thành 5 nhóm khối thời gian; với mỗi nhóm, huấn luyện GAT trên 4 nhóm còn lại rồi lấy embedding cho nhóm đó;
  - `build_features(emb, x)`: ghép embedding với 18 đặc trưng;
  - `fit_lgbm(...)`: tìm cấu hình trên val, ≤ 20 lần thử.
- `scripts/run_hybrid.py <data_dir> <bộ> <results.csv> <search.json> [oof=1]`: ghi kết quả qua `src/evaluate.py` với `model=hybrid_gat_lgbm` (bản có out-of-fold) và `model=hybrid_gat_lgbm_nooof` (bản không có).
- `tests/test_hybrid.py`: dữ liệu giả nhỏ; số cột = 18 + kích thước embedding; thứ tự node giữ nguyên; chạy 2 lần cùng seed cho cùng dự đoán.

### 2. Notebook `15_hybrid`

1. Clone, cài đặt, `pytest -q`.
2. Mỗi bộ, mỗi seed (42, 43, 44): embedding out-of-fold cho train, GAT trên toàn bộ train cho val; LightGBM; ghi kết quả val.
3. Chạy thêm bản không có out-of-fold để so sánh.
4. Lưu mô hình (GAT + LightGBM) vào `models_hybrid/` trên Kaggle cho GĐ H.
5. Ghi `run_info.json` (giờ GPU).

## Đầu ra

- **Trên Kaggle:** `models_hybrid/<bộ>_s<seed>/`.
- **Kéo về** `reports/kaggle/gnn-iot-ids-15-hybrid/`: `results_hybrid_<bộ>.csv`, `hybrid_search_<bộ>.json`, `run_info.json`.

## Tiêu chí đạt (task 5.7)

1. Không rò rỉ: GAT và LightGBM chỉ học trên train; chọn cấu hình chỉ trên val; dùng scaler của task 1.6.
2. Có cả bản out-of-fold và bản không out-of-fold.
3. Test `test_hybrid.py` qua trên Kaggle.
4. Ghi "kích thước" hybrid (số tham số GAT + số lá LightGBM + kích thước file) và độ trễ mỗi cửa sổ gồm cả 2 bước.
5. Đủ 3 seed × 2 bộ.

## Sau khi xong

- Cập nhật dòng **Kết quả** của task 5.7 và bảng theo dõi dòng F.
- Commit code trước khi chạy: `feat(5.7): hybrid GAT + LightGBM với embedding out-of-fold`. Commit kết quả sau: `chore(5.7): kết quả hybrid từ Kaggle`.

## Rủi ro

- Embedding train "quá sạch" làm LightGBM học vẹt: dấu hiệu là train gần 100% nhưng val tụt. Đây là lý do bắt buộc có out-of-fold.
- Out-of-fold tốn khoảng 5 lần thời gian huấn luyện GAT. Nếu giờ GPU không đủ thì báo Phúc trước, không tự bỏ.
