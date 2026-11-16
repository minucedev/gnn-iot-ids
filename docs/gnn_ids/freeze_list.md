# Danh sách mô hình đề xuất đóng băng (đầu vào của GĐ H)

> **ĐÃ ĐÓNG BĂNG (16/11/2026): tag `freeze-v1` = commit `abcb5f84a9dcf0adeb61800aa03c71fb591c2f9a` (`abcb5f8`).** Phúc đồng ý đóng băng đúng danh sách ở bảng 2 (gồm (c) `lgbm_neighbor_mean`). Commit này chứa cả mã chạy test (`scripts/final_test.py`, `scripts/aggregate_h.py`) đã chạy thử trên val (17a, 18a). Từ đây không sửa mô hình, cấu hình hay ngưỡng theo kết quả test; mọi thay đổi sau test phải ghi "sửa sau test".

Lập ngày 16/11/2026 cuối GĐ G. **Phúc đã duyệt danh sách này ngày 16/11/2026 (nhắn "duyệt"; giữ cấu hình chính, báo cả ngưỡng argmax và ngưỡng hiệu chỉnh).** Sau đó, cùng ngày, Phúc **điều chỉnh 2 điểm** (xem [mục 5](#5-điều-chỉnh-sau-khi-duyệt-16112026)): đưa phương án (c) vào danh sách chạy test, và báo cáo cả hybrid chính lẫn hybrid nhẹ trên NF-ToN-IoT-v3. Cùng ngày đã thêm (c) `lgbm_neighbor_mean` vào bảng 2 (ngưỡng tính trên val ở notebook 16h, commit `e5ee654`). Chưa tạo tag `freeze-v1`, chưa chạy test. Mọi số dưới đây là trên **val**, trung bình 3 seed (42, 43, 44). Nguồn: [reports/kaggle/gnn-iot-ids-16c-aggregate/](../../reports/kaggle/gnn-iot-ids-16c-aggregate/) (`thresholds.json`, `pareto_hybrid_<bộ>.csv`, `light_choice_<bộ>.json`, `ablation_*.csv`).

## 1. Hai mô hình chính

| Vai trò | NF-ToN-IoT-v3 (bộ chính) | NF-BoT-IoT-v3 |
|---|---|---|
| **Hybrid chính** `hybrid_gat_lgbm_oofavg_il` | 5 GAT nhóm + LightGBM (n=300, l=63) · F1-macro 0,9326 ± 0,0009 · FPR 4,94% · **144.715 tham số, 9,77 MB** (không đạt ngưỡng nhẹ) | 5 GAT nhóm + LightGBM (n=100, l=15) · F1-macro 0,9357 ± 0,0019 · FPR 20,45% · **62.435 tham số, 777 KB** (đạt ngưỡng nhẹ) |
| **Hybrid nhẹ** `hybrid_light` | **2 GAT nhóm (gat_fold0, gat_fold1) + LightGBM (n=100, l=15)** · F1-macro 0,9181 ± 0,0035 (**−1,45 điểm %** so với bản chính) · FPR 6,68% (**+1,73 điểm %**) · **30.934 tham số, 986 KB** (đạt cả hai ngưỡng) · p95 6,9 ms (bản chính 27,6 ms) | **Trùng hybrid chính** (đã đạt ≤ 100K tham số và ≤ 1 MB; không cần bản nhẹ riêng) |

Quy tắc chọn bản nhẹ: trong các điểm Pareto có `n_params ≤ 100.000` và `size_kb ≤ 1024` (kích thước là max trên 3 seed), lấy điểm có F1-macro trung bình cao nhất (`light_choice_<bộ>.json`, khóa `light_best_f1`). Ở ToN có 37/50 điểm đạt cả hai ngưỡng. Bản **nhỏ nhất** đạt ngưỡng (khóa `light_smallest`): ToN 1 GAT + LightGBM (n=25, l=7), 12.592 tham số, 172 KB, F1-macro 0,9002 (−3,24 điểm %), FPR 8,30%; BoT 1 GAT toàn train + LightGBM (n=25, l=7), 12.112 tham số, 126 KB, F1-macro 0,9286 (−0,71 điểm %), FPR 22,62%. Nếu Phúc muốn bản nhỏ nhất làm "hybrid nhẹ" thì đổi dòng tương ứng ở bảng 2.

## 2. Bảng đóng băng

Ngưỡng là xác suất "tấn công" = 1 − P(Normal), nhỏ nhất sao cho FPR val ≤ 4,08%; điểm > ngưỡng thì lớp tấn công có xác suất cao nhất, ngược lại Normal. Ngưỡng theo từng seed ở dạng "42 / 43 / 44"; đầy đủ (kèm F1, FPR, recall trước và sau, cờ khả thi) trong [thresholds.json](../../reports/kaggle/gnn-iot-ids-16c-aggregate/thresholds.json).

Cột "commit" là commit repo mà notebook sinh ngưỡng (hoặc mô hình) đã clone: `ad92f6a` (16a, 16d, 16e, 16g), `e91a3dd` (15e), `8f13f79` (15f), `e5ee654` (16h, ngưỡng của (c)). **Commit đóng băng thật là commit gắn tag `freeze-v1`, chưa có.**

### NF-ToN-IoT-v3

| Mô hình | Cấu hình | File mô hình trên Kaggle (output của notebook) | Ngưỡng (42 / 43 / 44) | Commit |
|---|---|---|---|---|
| logreg | C = 1,0, class_weight balanced | không lưu; huấn luyện lại, đối chiếu F1 lệch 0 | 0,9621 / 0,9621 / 0,9621 | ad92f6a (16e) |
| random_forest | n = 200, depth = None | không lưu (như trên) | 0,8767 / 0,8761 / 0,8765 | ad92f6a |
| xgboost | n = 300, depth = 8, lr 0,1 | không lưu; lệch F1 tối đa 0,0003 | 0,8743 / 0,8743 / 0,8743 | ad92f6a |
| lightgbm | n = 300, leaves = 63, lr 0,1 | không lưu | 0,8738 / 0,8735 / 0,8740 | ad92f6a |
| mlp | (64, 64), early stopping | không lưu | 0,6920 / 0,7224 / 0,6452 | ad92f6a |
| gcn | hidden 16, heads 4 (rộng 64), 2 lớp | `ckpt/nf_ton_iot_v3_gcn_s<seed>.pt` (07) | 0,9704 / 0,9672 / 0,9713 | ad92f6a |
| sage | như gcn | `ckpt/nf_ton_iot_v3_sage_s<seed>.pt` (07) | 0,9154 / 0,9318 / 0,8832 | ad92f6a |
| gat | hidden 16, heads 4, 2 lớp, edge_dim 3 (11.717 tham số, 50,6 KB) | `ckpt/nf_ton_iot_v3_gat_s<seed>.pt` (07) | 0,9676 / 0,9749 / 0,9623 | ad92f6a |
| **hybrid chính** | 5 GAT nhóm + LightGBM n=300, l=63; chia xen kẽ 5 × 20 khối (D13) | `models_hybrid/nf_ton_iot_v3_s<seed>/gat_fold0..4.pt`, `lgbm_oofavg_il.pkl`, `meta_oofavg_il.json` (15f) | 0,7233 / 0,7245 / 0,7201 | 8f13f79 (15f), ngưỡng ad92f6a (16d) |
| **hybrid nhẹ** | 2 GAT nhóm (`gat_fold0`, `gat_fold1` của 15f) + LightGBM n=100, l=15 | GAT: như hybrid chính; LightGBM: `models_light/nf_ton_iot_v3_s<seed>/lgbm_n100_l15.pkl` (16d) | 0,8474 / 0,9070 / 0,8896 | ad92f6a (16d) |
| **lgbm_neighbor_mean** (c) | LightGBM n = 300, leaves = 63, lr 0,1 trên 36 cột: 18 đặc trưng node + trung bình 18 đặc trưng của node kề (`hybrid.neighbor_mean`, tính tay, không học); F1-macro val 0,9342 ± 0,0003, lệch so với `ablation_cd_nf_ton_iot_v3.csv` đúng 0 | huấn luyện lại (F1 val phải khớp); bản đã huấn luyện: `models_c/nf_ton_iot_v3_s<seed>/lgbm_neighbor_mean.pkl` (16h) | 0,8166 / 0,8238 / 0,8143 | e5ee654 (16h) |

### NF-BoT-IoT-v3

| Mô hình | Cấu hình | File mô hình trên Kaggle | Ngưỡng (42 / 43 / 44) | Commit |
|---|---|---|---|---|
| logreg | C = 10,0 | không lưu; lệch F1 0 | 0,7570 / 0,7570 / 0,7570 | ad92f6a (16a) |
| random_forest | n = 100, depth = 12 | không lưu | 0,6692 / 0,6644 / 0,6692 | ad92f6a |
| xgboost | n = 200, depth = 6 | không lưu | 0,6707 / 0,6707 / 0,6707 | ad92f6a |
| lightgbm | n = 200, leaves = 31 | không lưu | 0,6703 / 0,6703 / 0,6703 | ad92f6a |
| mlp | (64, 64) | không lưu | 0,8671 / 0,7909 / 0,8088 | ad92f6a |
| gcn | như ToN | `ckpt/nf_bot_iot_v3_gcn_s<seed>.pt` (14) | 0,7393 / 0,7810 / 0,6750 | ad92f6a |
| sage | như ToN | `ckpt/nf_bot_iot_v3_sage_s<seed>.pt` (14) | 0,7302 / 0,7252 / 0,7125 | ad92f6a |
| gat | như ToN (11.587 tham số, 50,1 KB) | `ckpt/nf_bot_iot_v3_gat_s<seed>.pt` (14) | 0,7694 / 0,7152 / 0,7572 | ad92f6a |
| **hybrid chính = hybrid nhẹ** | 5 GAT nhóm + LightGBM n=100, l=15 | `models_hybrid/nf_bot_iot_v3_s<seed>/gat_fold0..4.pt`, `lgbm_oofavg_il.pkl` (15e) | 0,7234 / 0,7416 / 0,7388 | e91a3dd (15e), ngưỡng ad92f6a (16a) |
| **lgbm_neighbor_mean** (c) | LightGBM n = 100, leaves = 15, lr 0,1 trên 36 cột (như ToN); F1-macro val 0,9335, lệch so với ablation đúng 0 | huấn luyện lại; bản đã huấn luyện: `models_c/nf_bot_iot_v3_s<seed>/lgbm_neighbor_mean.pkl` (16h) | 0,6997 / 0,6997 / 0,6997 (3 seed trùng nhau vì LightGBM không có thành phần ngẫu nhiên ở bộ này) | e5ee654 (16h) |

Dòng baseline (logreg, random_forest, xgboost, lightgbm, mlp): GĐ H phải huấn luyện lại bằng đúng cấu hình ở bảng (lấy từ `baseline_search_<bộ>.json` của GĐ D), cùng seed; việc dựng lại đã đối chiếu và cho cùng F1 val (`repro_f1_diff` trong `thresholds_base_gnn_<bộ>.json`), nhưng GĐ H nên kiểm lại F1 val trước khi áp ngưỡng.

## 3. Không đóng băng (có lý do)

- **Bản cải tiến GAT** (focal loss, residual + LayerNorm, cả hai): **không bản nào được công nhận theo task 5.4** (xem dòng Kết quả của 5.4). Không đưa vào hybrid và không chạy test như mô hình độc lập; ngưỡng vẫn được ghi trong `thresholds.json` (`gat_focal`, `gat_resln`, `gat_focal_resln`, kèm `accepted_by_5_4 = false`) phòng khi Phúc muốn báo cáo chúng.
- **LightGBM chỉ embedding (d)**: chỉ là phương án ablation, không có ngưỡng, không chạy test. (c) `lgbm_neighbor_mean` **đã được đưa vào bảng 2** theo mục 5.1 (không còn nằm ngoài danh sách).

## 4. Điểm cần Phúc quyết trước khi đóng băng

1. **Ngưỡng 4,08% không đi cùng recall ≥ 95% ở bất kỳ mô hình nào.** Sau hiệu chỉnh FPR val về ≤ 4,08%, recall tấn công cao nhất ở ToN là hybrid chính 91,9% (hybrid nhẹ 90,2%, SAGE 91,7%, mọi mô hình khác thấp hơn), ở BoT chỉ 29–38% (F1-macro rơi từ 0,93 xuống khoảng 0,65). Không có ngưỡng nào đạt đồng thời FPR ≤ 4,08% và recall ≥ 95% (`meets_recall_target` = false mọi nơi). Chỉ tiêu mục 3 của KE_HOACH sẽ không đạt trên cả hai bộ nếu đo bằng ngưỡng này; BoT là do dữ liệu (xem phân tích 15a). Cần quyết: đóng băng ngưỡng này, hay đóng băng ngưỡng argmax mặc định cho chỉ tiêu F1 và dùng ngưỡng hiệu chỉnh cho chỉ tiêu FPR (báo cáo cả hai).
2. **Hybrid ToN chưa đạt ngưỡng nhẹ, chỉ bản nhẹ đạt.** Cần quyết dùng bản nhẹ (F1-macro 0,9181) làm mô hình báo cáo "Lightweight" và bản chính (0,9326) như mô hình tham chiếu, hay ngược lại. Bản nhẹ vẫn hơn GraphSAGE (0,9146) về F1 nhưng chỉ chênh +0,0035, nhỏ hơn 2 lần std (std của bản nhẹ 0,0035, tức ngưỡng 0,007), nên **không** còn "vượt hơn 2 std" so với mô hình tốt nhất.
3. **Đặc trưng hàng xóm tính tay ngang hybrid.** LightGBM + trung bình hàng xóm (c) đạt F1-macro 0,9342 ± 0,0003 trên ToN, không thấp hơn hybrid đầy đủ 0,9326 ± 0,0009, FPR 5,00% so với 4,94%, và chỉ cần 36 đặc trưng, không cần GAT (BoT: 0,9335 so với 0,9357, cũng ngang). Nghĩa là phần lớn lợi thế của hybrid so với baseline đến từ thông tin hàng xóm chứ không phải từ embedding GAT học được. Cần quyết cách trình bày trong báo cáo, và có đưa (c) vào danh sách chạy test như một baseline bổ sung hay không (đã quyết ở mục 5.1: có, đã thêm vào bảng 2).
4. **Cấu hình chính của GAT chưa phải tối ưu theo số head.** 8 head cho F1-macro 0,9153 so với 0,9065 của 4 head (hơn 0,0089, lớn hơn 2 lần std), với 39.813 tham số (vẫn ≤ 100K). Bốn yếu tố còn lại (số lớp, độ rộng, đặc trưng cạnh, cửa sổ 5 giây) không có cấu hình nào hơn cấu hình chính quá 2 std. Đổi sang 8 head nghĩa là phải huấn luyện lại hybrid (khoảng 0,7 giờ GPU cho ToN); mình **không đổi** vì hybrid D13 đã dựng trên cấu hình chính.
5. **Đặc trưng cạnh không cải thiện**: bỏ đặc trưng cạnh cho F1-macro 0,9076 so với 0,9065 (trong sai số), ít hơn 384 tham số. Ảnh hưởng tới câu chữ trong báo cáo (task 5.3 mô tả GAT "có đặc trưng cạnh").

## 5. Điều chỉnh sau khi duyệt (16/11/2026)

Phúc chọn qua chat ngày 16/11/2026, trước khi tạo tag `freeze-v1`:

1. **Đưa phương án (c) vào danh sách chạy test** như một baseline bổ sung, trên cả 2 bộ: `lgbm_neighbor_mean` (LightGBM với 18 đặc trưng node + trung bình 18 đặc trưng của các node kề, tính tay, không học; cấu hình đúng như bản đã chạy trong ablation ở GĐ G, file kết quả `ablation_hybrid_<bộ>.csv`).
   - **Lý do:** trên val, (c) ngang hoặc nhỉnh hơn hybrid (ToN 0,9342 ± 0,0003 so với 0,9326 ± 0,0009; BoT 0,9335 so với 0,9357). Đây là đối thủ mạnh nhất nhóm đã biết; không có số test của (c) thì minh chứng "hybrid hơn baseline" thiếu trung thực. Cấu hình của (c) đã chốt trên val nên thêm trước tag là hợp lệ.
   - **Đã làm (16/11/2026, notebook `gnn-iot-ids-16h-c-threshold` bản 1, CPU, commit `e5ee654`, 502 s, 0 giờ GPU; [reports/kaggle/gnn-iot-ids-16h-c-threshold/](../../reports/kaggle/gnn-iot-ids-16h-c-threshold/)):** huấn luyện lại (c) trên train, 2 bộ, 3 seed; F1-macro val khớp ablation GĐ G đúng 0 (`repro_f1_diff_max_abs` = 0,0 ở cả hai bộ); ngưỡng theo quy tắc mục 2 (FPR val ≤ 4,08%): ToN 0,8166 / 0,8238 / 0,8143 (sau hiệu chỉnh F1-macro 0,9392 / 0,9389 / 0,9394, FPR 4,08%, recall 93,8–94,1%), BoT 0,6997 ×3 (F1-macro 0,6461, FPR 4,04%, recall 36,7%). Không seed nào đạt recall ≥ 95% sau hiệu chỉnh (như mọi mô hình khác). Không đọc tập test.
   - **Việc phải làm trước khi tag** (đầu GĐ H, CPU; đã làm xong, xem dòng trên): tính ngưỡng hiệu chỉnh của (c) trên **val** theo đúng quy tắc ở mục 2 (FPR val ≤ 4,08%, 3 seed), thêm dòng `lgbm_neighbor_mean` vào bảng 2 của cả 2 bộ, kèm file mô hình hoặc cách huấn luyện lại, ngưỡng và commit. Không được nhìn tập test ở bước này.
2. **Cả hai bản hybrid đều đứng tên trong báo cáo** trên NF-ToN-IoT-v3: hybrid chính (mô hình mạnh nhất, 144.715 tham số, 9,77 MB, không đạt ngưỡng nhẹ) và hybrid nhẹ (bản triển khai, 30.934 tham số, 986 KB). Báo cáo nói rõ đánh đổi: bản nhẹ giảm 1,45 điểm % F1-macro, tăng 1,73 điểm % FPR, nhanh hơn khoảng 4 lần. **Bản gợi ý triển khai do AHP quyết định ở GĐ I.**
3. **Cách trình bày minh chứng trong báo cáo** (theo kết quả ablation): thông tin **cấu trúc đồ thị** (hàng xóm) là thứ giúp vượt các baseline ML, DL và GNN đơn; hybrid GAT + LightGBM là một cách khai thác thông tin đó; tổng hợp hàng xóm bằng tay (c) là cách rẻ hơn và cho kết quả tương đương. Không viết rằng embedding GAT tốt hơn tổng hợp tay nếu số test không cho thấy điều đó.
