# Lộ trình thực hiện trên Kaggle (12/11 – 19/11/2026)

Lập ngày 12/11/2026 theo yêu cầu bổ sung của thầy ([KE_HOACH.md mục 0](KE_HOACH.md#0-yêu-cầu-bổ-sung-của-thầy-12112026)) và quyết định D8–D12. **File này là lịch hiện hành**, thay cho mục 5b và 5c của KE_HOACH.md. Chi tiết từng giai đoạn nằm trong thư mục [giai_doan/](giai_doan/README.md); tiêu chí từng task ở [TASKS.md](TASKS.md).

## 1. Nguyên tắc

1. **Mọi xử lý dữ liệu, huấn luyện, đánh giá, tổng hợp số liệu đều chạy trên Kaggle** (tài khoản `minucelowcode`, GPU T4 khi cần). Máy cá nhân **không** train, **không** xử lý dữ liệu, **không** tải file dữ liệu lớn (`.csv` gốc, `.parquet`, `.pt`, checkpoint).
2. Máy cá nhân chỉ dùng để: sửa code và tài liệu, đẩy lên GitHub, điều khiển Kaggle qua API (đẩy notebook, xem trạng thái), **tải về các file kết quả nhỏ** (`.csv`, `.json`, `.png`) để lưu vào repo.
3. Số liệu trong báo cáo chỉ lấy từ file kết quả do Kaggle sinh ra; bảng và hình tổng hợp cũng do một notebook Kaggle sinh ra (GĐ H).
4. Quy ước cũ vẫn giữ: chọn cấu hình chỉ trên val; test chạy **một lần** sau khi đóng băng; 3 seed (42, 43, 44); commit thẳng lên `main`.

Lý do thêm: từ 12/11 máy Windows chặn `python.exe` của `.venv` và `kaggle.exe` (Application Control), nên local cũng không chạy được Python của dự án.

## 2. Quy trình chuẩn cho mỗi notebook

```
sửa code trong repo ──► push GitHub (main)
        │
        ▼
notebook Kaggle: git clone repo ──► cài requirements ──► chạy scripts/… ──► ghi /kaggle/working/
        │                                           (đầu vào: dataset Kaggle hoặc output
        │                                            của notebook trước qua kernel_sources)
        ▼
local: gọi Kaggle API xem trạng thái ──► tải file nhỏ về reports/kaggle/<tên-notebook>/ ──► commit
```

Chi tiết cách viết notebook và cách gọi Kaggle API bằng `curl`: [giai_doan/README.md](giai_doan/README.md).

## 3. Ngân sách Kaggle

- Quota GPU 30 giờ mỗi tuần. **Tài khoản `minucelowcode` đang dùng chung cho một dự án khác** (các notebook `gd6`–`gd12` chạy từ 05/11 đến 12/11), nên phải kiểm tra số giờ GPU còn lại trên trang Kaggle trước GĐ E.
- Notebook không cần GPU (dựng đồ thị, EDA, baseline cây, tổng hợp, AHP) chạy bằng CPU, không tốn quota GPU.
- Mỗi phiên chạy có giới hạn thời gian (notebook 05 từng bị dừng giữa chừng), nên tách việc lớn thành nhiều notebook nhỏ.
- **Ước lượng GPU** cho cả lộ trình: khoảng 6–10 giờ (GĐ E, F, G, H). Đây là ước lượng; con số thật ghi vào `run_info.json`. Nếu tổng thực tế vượt 25 giờ thì báo Phúc trước khi chạy tiếp.

## 4. Hiện trạng trên Kaggle (kiểm tra ngày 12/11)

| Notebook | Trạng thái | Output có sẵn (chưa kéo về repo) |
|---|---|---|
| `04-build-graphs` | Xong | Đồ thị đã chia tập cho NF-ToN-IoT-v3, CIC, v2 (`processed/<bộ>/`) |
| `05-baselines` | Bị dừng giữa chừng | `results_baselines_nf_ton_iot_v3.csv`, `baseline_search_nf_ton_iot_v3.json`, một phần CIC, `verify_*.json` |
| `07-train-gnn` | Xong | `results_gnn_{nf_ton_iot_v3, nf_cicids2018_v3, nf_ton_iot_v2}.csv`, 27 checkpoint |
| `09-baselines-cic` | Xong | Baseline CIC còn thiếu (XGBoost, LightGBM, MLP) |
| `10-baselines-v2` | Xong | Baseline NF-ToN-IoT-v2 |

## 5. Các giai đoạn

| GĐ | Ngày | Việc | Notebook Kaggle | Máy | Cần xong trước | Task |
|---|---|---|---|---|---|---|
| [A](giai_doan/GD_A_thu_hoi_ket_qua.md) | 12/11 | Thu hồi kết quả đã chạy | — | — | — | 4.1, 5.2, 5.3 |
| [B](giai_doan/GD_B_do_thi_bot_iot.md) | 12–13/11 | Dựng đồ thị NF-BoT-IoT-v3 | `11-build-graphs-bot` | CPU | — | 1.8 |
| [C](giai_doan/GD_C_eda.md) | 13/11 | EDA và trực quan hóa đồ thị | `12-eda` | CPU | B | 2.1–2.3 |
| [D](giai_doan/GD_D_baseline.md) | 13/11 | Baseline ML và MLP | `13-baselines` | CPU | A, B | 4.1 |
| [E](giai_doan/GD_E_gnn_bot_iot.md) | 14/11 | GNN đơn trên NF-BoT-IoT-v3 | `14-train-gnn-bot` | GPU | B | 5.2, 5.3 |
| [F](giai_doan/GD_F_hybrid.md) | 14–15/11 | Hybrid GAT + LightGBM | `15-hybrid` | GPU | A, E | 5.7 |
| [G](giai_doan/GD_G_ablation.md) | 15/11 | Ablation, bản cải tiến GAT, hiệu chỉnh ngưỡng | `16-ablation` | GPU | D, F | 5.4, 5.5, 5.8 |
| [H](giai_doan/GD_H_dong_bang_test.md) | 16/11 | **Đóng băng**, chạy test một lần, tổng hợp | `17-final-test`, `18-aggregate` | GPU, CPU | G | 6.1–6.3, 5.6 |
| [I](giai_doan/GD_I_ahp_realtime.md) | 17/11 | AHP và mô phỏng thời gian thực | `19-ahp`, `20-realtime` | CPU | H | 7.1, 7.2, 8.1–8.4 |
| [J](giai_doan/GD_J_bao_cao.md) | 18–19/11 | Báo cáo, kiểm tra tái lập, nộp | `21-reproduce` | CPU | H, I | 3.1–3.4, 9.1–9.3 |

C và D chạy song song được sau B; E chạy song song với C, D. **Mốc cứng: 16/11** đóng băng mô hình bằng tag `freeze-v1` trước khi xem tập test.

## 6. Bảng theo dõi

Cập nhật sau mỗi lần chạy. Giờ GPU lấy từ `run_info.json`.

| GĐ | Notebook | Phiên bản | Commit repo | Trạng thái | Giờ GPU | File đã kéo về |
|---|---|---|---|---|---|---|
| A | 05, 07, 09, 10 | 05: v2 (bị dừng), 07: v2, 09: v1, 10: v1 | 05, 07: `9d7761a`; 09, 10: `4d98421` | Xong (12/11) | 0 | 12 file `.csv`/`.json` trong `reports/kaggle/<slug>/`, không tải `ckpt/`; chi tiết ở [reports/kaggle/README.md](../../reports/kaggle/README.md) |
| B | 11-build-graphs-bot | v2 (v1 lỗi test import) | `47dccee` | Xong (12/11) | 0 (CPU, 726 s, RAM đỉnh 12,25 GB) | `summary.json`, `graph_stats.csv`, `split_search_nf_bot_iot_v3.json`, `verify_nf_bot_iot_v3.json`, `run_info.json`, `data_check_nf_bot_iot_v3.json`; `processed/nf_bot_iot_v3/` ở lại trên Kaggle |
| C | 12-eda | v4 (v1 chạy thử, v2–v3 sửa hình) | `d2e4d1a` | Xong (13/11), 2.2 và 2.3 có ghi chú | 0 (CPU, 575 s, RAM đỉnh 4,23 GB) | 26 hình `gd2_*.png` vào `reports/figures/`; `eda_summary.json`, `run_info.json`, `gd2_<bộ>_flow_stats.csv`, `gd2_homophily.csv` trong `reports/kaggle/gnn-iot-ids-12-eda/` |
| D | 13-baselines | v1 | `93c6625` | Xong (chạy 12/11, tiêu chí 6 được Phúc chấp nhận 14/11) | 0 (CPU, 293 s, RAM đỉnh 0,85 GB) | `results_baselines_nf_bot_iot_v3.csv`, `baseline_search_nf_bot_iot_v3.json`, `run_info.json` |
| E | 14-train-gnn-bot | v1 | `ac05c06` | Chạy xong (12/11); 5.2, 5.3 để `Đang làm`: FPR val 20,6–22,3% > 4,08%, chưa có log tinh chỉnh, bảng tham số theo lớp, trọng số attention | 0 (CPU 4 lõi, 424,5 s tổng, 276,6 s huấn luyện, RAM đỉnh 0,66 GB; không dùng GPU) | `results_gnn_nf_bot_iot_v3.csv`, `curves_nf_bot_iot_v3.csv`, `run_info.json` trong `reports/kaggle/gnn-iot-ids-14-train-gnn-bot/`; 9 `ckpt/*.pt` ở lại trên Kaggle cho GĐ F |
| F | 15a-fpr-analysis, 15-hybrid, 15b-hybrid-oofavg, 15c-compare-check, 15d-hybrid-stack, 15e-hybrid-il, 15f-hybrid-il-ton, 15t-hybrid-tests | 15a, 15, 15b, 15d, 15e, 15f: v1; 15c: v4 | 15a: `e1cc0fb`; 15: `603d929`; 15b: `f63efc1`; 15d: `2d2187d`; 15e: `e91a3dd`; 15f: `8f13f79`; 15c: `2ad93d8` | **Xong (15/11), 5.7 `Xong` có ghi chú:** hybrid out-of-fold xen kẽ **vượt** mô hình tốt nhất ở ToN (F1-macro 0,9326 so với GraphSAGE 0,9146, hơn 2 std; FPR 4,94% so với MLP 5,29%, không hơn 2 std) và **ngang** ở BoT (0,9357 so với LightGBM 0,9388). Bản out-of-fold đúng đặc tả (5 nhóm liên tiếp) thất bại vì nhóm cuối là nơi duy nhất có một số lớp. Kích thước ToN 9,7 MB và 144 nghìn tham số vượt ngưỡng nhẹ. FPR: ToN do node chỉ nhận flow, thí nghiệm 15h cho thấy đổi nhãn chỉ thu hẹp phạm vi đánh giá (FPR 7,9% → 1,1% trên node khởi tạo, huấn luyện lại không thêm gì) nên **không đổi nhãn**, GĐ H báo cáo thêm theo tầng; BoT do dữ liệu. Số baseline ToN mount trên Kaggle khác bản repo, dùng bản repo (15c) | 15a, 15c, 15d, 15e, 15t: 0 (CPU); **15: 0,952; 15b: 0,826; 15f: 0,693; tổng 2,471 giờ GPU** (T4) | `reports/kaggle/gnn-iot-ids-{15a-fpr-analysis,15-hybrid,15b-hybrid-oofavg,15c-compare-check,15d-hybrid-stack,15e-hybrid-il,15f-hybrid-il-ton}/`; `models_hybrid/` ở lại trên Kaggle (15, 15b, 15e, 15f) |
| G | 16-ablation (GPU), 16f-windows-gat (GPU), 16a-ablation-bot, 16b-windows, 16c-aggregate, 16d-ablation-ton, 16e-thresholds-ton, 16g-tuning-ton (CPU), 16t-tests | 16-ablation: v3 (v1, v2 hỏng ở pytest); các notebook còn lại v1 | 16a, 16d, 16e, 16g: `ad92f6a`; 16b: `ab16ed6`; 16f: `5169a1d`; 16-ablation: `5031ab5`; 16c: `260cb42` | **Xong (16/11), 5.2, 5.3, 5.4, 5.5, 5.8 `Xong`.** Bản cải tiến GAT (focal, residual + LayerNorm) không được công nhận theo 5.4. Hybrid nhẹ ToN: 2 GAT + LightGBM (n=100, l=15), 30.934 tham số, 986 KB, F1-macro 0,9181 (−1,45 điểm %), FPR 6,68% (+1,73 điểm %); BoT bản đầy đủ đã đạt ngưỡng nhẹ. Embedding GAT không hơn đặc trưng hàng xóm tính tay (c: ToN 0,9342, e: 0,9326). Sau hiệu chỉnh FPR ≤ 4,08%, không mô hình nào đạt recall ≥ 95% (ToN tốt nhất 91,9%, BoT 29–38%). Danh sách đóng băng đề xuất ở [freeze_list.md](freeze_list.md), chờ Phúc duyệt, chưa tạo tag | **1,621 giờ GPU** (16f 0,476; 16-ablation bản 3 1,067; hai lần hỏng ở pytest 0,039 + 0,039); 16a, 16b, 16c, 16d, 16e, 16g, 16t: 0 (CPU). Tổng từ đầu dự án: 2,471 + 1,621 = 4,09 giờ | `reports/kaggle/gnn-iot-ids-{16-ablation,16a-ablation-bot,16b-windows,16c-aggregate,16d-ablation-ton,16e-thresholds-ton,16f-windows-gat,16g-tuning-ton,16t-tests}/`; ckpt, `models_light/`, đồ thị W = 1, 2, 10 giây ở lại trên Kaggle |
| H | 16h-c-threshold, 17a-dryrun-val, 17-final-test, 18a-aggregate-dryrun, 18-aggregate | mỗi notebook bản 1 | 16h: `e5ee654`; 17a: `9cae412`; 18a: `b42111c`; **17 (test): `abcb5f8` = tag `freeze-v1`**; 18: `main` sau tag | **Xong (16/11), 6.1, 6.2, 6.3, 5.6 `Xong` có ghi chú.** Tag `freeze-v1` trên `abcb5f8` tạo trước khi chạy test. Test chạy đúng một lần cho 63 mục (bộ, mô hình, seed), hai chế độ ngưỡng, kèm thử chéo và báo cáo theo tầng. ToN test: hybrid chính F1-macro 0,9472 (argmax) / 0,9677 (ngưỡng), FPR 4,20% / 1,88%, recall 97,7% / 96,7%; hybrid nhẹ 0,9341 / 0,9588, FPR 5,69% / 2,27%. Hybrid vượt ML, MLP, GNN đơn nhưng **không vượt rõ (c)** (−0,0005 argmax, +0,0025 ngưỡng); hybrid nhẹ thấp hơn (c); BoT: hybrid thua LightGBM ở argmax. Chỉ tiêu "hybrid > baseline tốt nhất" **không đạt**; hybrid nhẹ đạt mọi chỉ tiêu còn lại ở ngưỡng hiệu chỉnh. Thử chéo giữa hai bộ rất kém (F1 0,18–0,38) | **0 giờ GPU** (16h: 502 s CPU; 17a: 2.450 s; 17: 2.668 s; 18a và 18: ~270 s, đều CPU). Tổng từ đầu dự án vẫn **4,09 giờ** | `reports/kaggle/gnn-iot-ids-{16h-c-threshold,17a-dryrun-val,17-final-test,18-aggregate}/`; `reports/tables/` (results, summary_*, mcnemar, conclusions, targets); `reports/figures/gd6_*.png`; `pred_*.parquet` ở lại trên Kaggle |
| I | 19-ahp, 20-realtime | mỗi notebook bản chạy đủ (20: bản 1 là chạy thử 300 cửa sổ) | 19: `9bdc7b7`; 20: `196064a` | **Xong (16/11), 7.1 và 7.2 Xong, 8.2 và 8.3 Xong, 8.4 Xong có ghi chú, 8.1 Đang làm lúc đó, Xong có ghi chú ở GĐ J.** AHP: CR 0,0098; trọng số FPR 0,431, Recall 0,204, F1-macro 0,189, Kích thước 0,109, Độ trễ 0,067; gợi ý Hybrid chính (0,831), hạng nhất không đổi khi ±20% từng trọng số, nhưng Hybrid chính không đạt ngưỡng nhẹ (144.560 tham số, 9,7 MB) và hạng nhất đổi sang MLP nếu chuẩn hóa theo tỉ lệ. Thời gian thực (toàn tập test, 11.449 cửa sổ): p95 136 ms (dựng đồ thị 108, suy luận 30); online khớp offline 100%, đặc trưng sai khác 0,0; cảnh báo giả 15,1 mỗi giờ; phát hiện median 5,1 s, max 234 s nhưng 14/27 đợt không phát hiện; RAM tăng 11,4% ở lần chạy đầu (8.1 chưa đạt lúc đó; đã sửa ở GĐ J, xem dòng J) | 0 (CPU: 19 là 179 s; 20 là 3.433 s). Tổng từ đầu dự án vẫn **4,09 giờ** | `reports/kaggle/gnn-iot-ids-{19-ahp,20-realtime}/`; `reports/tables/{ahp_result,latency,realtime_eval}.json`; `reports/figures/gd7_*.png`, `gd8_*.png` |
| J | 19b-ahp-constrained, 20-realtime (bản 4), 20a-undetected, 20b, 20c, 20d, 21-reproduce | 19b: v3; 20: v4 (v2 và v3 hỏng ở pytest, dừng sớm); 20a, 20b, 20c, 20d: v1; 21: v2 | 19b: `0fa57c1`; 20: `0fa57c1`; 20a: `8e3372e`; 20b, 20c: `3857402`; 20d: `fc6b153`; 21: `d9d3c77` | **Xong (16/11), 8.1, 9.1, 9.2, 9.3 ghi Kết quả; 9.2 và 9.3 `Xong` có ghi chú.** AHP có ràng buộc (loại 4 phương án > 100.000 tham số hoặc > 1 MB): gợi ý MLP (0,612), recall 93,86% < 95%, kịch bản mong manh; không ràng buộc vẫn hybrid chính. 14/27 đợt không phát hiện: 12 bị gán Normal, 2 dưới ngưỡng; 6 đợt chỉ 1 cửa sổ; 10 đợt lớp Khac. RAM: nguyên nhân là danh sách dict tích lũy của bộ đo trong `detector.py`; sửa xong, chạy lại: **+9,6% (đạt sát ngưỡng)**, p95 end-to-end 121 ms, online khớp offline 100%. Tái lập: `aggregate_h.py` chạy lại từ output đã lưu, lệch F1-macro 0,0 (đạt). Báo cáo DOCX, slide PPTX, README đã cập nhật. Chưa làm: video demo, tập dượt, tag `final` (chờ Phúc xác nhận) | 0 (CPU: 19b 196 s; 20 3.220 s; 20a 289 s; 20b 861 s; 20c 2.155 s; 20d 527 s; 21 155 s). Tổng từ đầu dự án vẫn **4,09 giờ** | `reports/kaggle/gnn-iot-ids-{19b-ahp-constrained,20-realtime,20a-undetected,20b-ram-diag-parts,20c-ram-diag-stream,20d-ram-diag-hybrid,21-reproduce}/`; `reports/tables/{ahp_constrained,undetected_episodes,latency,realtime_eval}.json`, `alerts_sample.csv`; `report/BAO_CAO.docx`, `report/SLIDE.pptx` |
