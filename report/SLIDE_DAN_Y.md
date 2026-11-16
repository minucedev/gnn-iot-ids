# Dàn ý slide bảo vệ (18 slide)

Nhóm: Nguyễn Minh Phúc, Lê Trung Phong, Văn Bá Phạm Tấn. Đề tài: *Ứng dụng mạng Lightweight GNN gợi ý phát hiện xâm nhập IoT thời gian thực*.

Quy ước: mỗi slide có tiêu đề, 3–4 ý và hình dùng (đường dẫn tương đối từ `report/`). Số liệu lấy từ [BAO_CAO.md](BAO_CAO.md) (ghi nguồn ở đó). Các slide đánh dấu **[chờ GĐ I]** chỉ điền được sau khi AHP và mô phỏng thời gian thực xong. Thời lượng gợi ý: khoảng 1 phút mỗi slide.

---

### Slide 1. Trang bìa

- Tên đề tài, nhóm, giảng viên, ngày bảo vệ.
- Một câu mô tả: phát hiện IP đang tấn công trong mạng IoT từ luồng NetFlow, bằng GNN nhẹ.

**Hình:** không (hoặc logo của trường).

### Slide 2. Bài toán và ràng buộc

- Gateway IoT thu NetFlow liên tục; cần phát hiện IP đang tấn công (DDoS, botnet, truy cập trái phép) trong dưới 1 giây.
- Ràng buộc "nhẹ": ≤ 100.000 tham số, ≤ 1 MB, p95 ≤ 1 s trên CPU 1 luồng.
- Đầu ra: nhãn 5 nhóm cho từng node (IP) trong cửa sổ 5 giây; suy ra nhị phân để đo FPR.
- Ngoài phạm vi: web, mạng thật; triển khai là mô phỏng phát lại flow.

**Hình:** `../reports/figures/gd2_chung_quy_trinh.png` (flow → cửa sổ → đồ thị).

### Slide 3. Yêu cầu của thầy (12/11) và cách nhóm đáp ứng

- 5 yêu cầu: ≥ 2 bộ dữ liệu có NF-ToN; hybrid kèm lý do; Kaggle 30 giờ + T4; EDA; minh chứng hybrid hơn baseline.
- Mỗi yêu cầu ứng với một chương của báo cáo (Phụ lục A).
- Mức đáp ứng trung thực: 4 yêu cầu đạt; yêu cầu 5 đạt một phần.

**Hình:** bảng đối chiếu (Phụ lục A, dạng bảng 5 dòng).

### Slide 4. Cơ sở: NetFlow và GNN

- NetFlow là định dạng; các bộ NF-* là bản chuyển đổi từ bộ dữ liệu gói tin gốc.
- Node = IP, cạnh = flow; GNN gom thông tin hàng xóm (GCN, GraphSAGE, GAT).
- GAT: attention nhiều head; dùng GATv2 có đặc trưng cạnh.
- Vì sao quan tâm FPR: lưu lượng bình thường áp đảo nên cảnh báo giả dễ làm hệ thống vô dụng.

**Hình:** công thức attention của GAT (chép từ mục 2.3 của báo cáo, vẽ lại thành một hình).

### Slide 5. Vì sao chọn mô hình hybrid GAT + LightGBM

- Cây mạnh trên đặc trưng dạng bảng nhưng không thấy cấu trúc mạng; GNN thấy cấu trúc nhưng yếu hơn trên đặc trưng bảng.
- Ghép: embedding GAT (64 chiều) + 18 đặc trưng gốc → LightGBM (82 cột).
- LightGBM cho xác suất dễ hiệu chỉnh ngưỡng quyết định.
- Đây là giả thuyết thiết kế; slide 13 đối chiếu giả thuyết với số liệu.

**Hình:** sơ đồ hybrid (sơ đồ mermaid ở mục 4.4 của báo cáo, vẽ lại).

### Slide 6. Khảo sát 8 bộ và lý do chọn 2 bộ

- 8 bộ: NF-ToN-IoT v2/v3, NF-BoT-IoT v2/v3, ToN-IoT, BoT-IoT, NF-CSE-CIC-IDS2018-v3, NF-UNSW-NB15-v3.
- Chọn **NF-ToN-IoT-v3** (bộ chính): timestamp thật, phát hành 2025, đủ 3 nhóm tấn công, đúng yêu cầu "NF-ToN".
- Chọn **NF-BoT-IoT-v3** (bộ 2): dữ liệu IoT, có timestamp; hạn chế: 20 IP nguồn, Normal 0,3%, không có Botnet_Backdoor.
- Loại các bộ không có timestamp, hoặc không phải IoT, hoặc quá nặng.

**Hình:** bảng 3.1 của báo cáo (rút gọn còn 5 cột).

### Slide 7. EDA mức flow

- NF-ToN-IoT-v3: 27,5 triệu flow, Normal 61%, tấn công theo từng giai đoạn trong 7 ngày UTC.
- NF-BoT-IoT-v3: 16,9 triệu flow, gần như toàn tấn công.
- Đặc trưng có thông tin tương hỗ cao nhất: `IN_BYTES`, `SRC_TO_DST_AVG_THROUGHPUT`.
- EDA chỉ mô tả, không dùng để chọn cấu hình mô hình.

**Hình:** `../reports/figures/gd2_nf_ton_iot_v3_flows_over_time.png`, `../reports/figures/gd2_nf_ton_iot_v3_mutual_information.png`.

### Slide 8. EDA mức đồ thị và homophily

- Đồ thị rất nhỏ: trung vị 11 node (ToN), 5 node (BoT) mỗi cửa sổ.
- Edge homophily 0,67 (ToN) và 0,86 (BoT), cao hơn mức ngẫu nhiên (0,47 và 0,50).
- Nhưng 33% (ToN) và 14% (BoT) cạnh nối hai lớp khác nhau nên hàng xóm không thay được đặc trưng của node.
- Bậc vào/ra không tách lớp mạnh; fan-in/fan-out chỉ thấy ở một số cửa sổ.

**Hình:** `../reports/figures/gd2_nf_ton_iot_v3_homophily.png`, `../reports/figures/gd2_nf_ton_iot_v3_windows_overview.png`.

### Slide 9. Dựng đồ thị và chia tập theo khối

- Cửa sổ 5 giây; 18 đặc trưng node, 3 đặc trưng cạnh; không dùng IP và cổng.
- Chia `blocked`: 10 khối, theo số node, gap 1 cửa sổ, tỉ lệ 70/15/15.
- Lý do: chia thuần theo thời gian làm val và test thiếu hẳn lớp DoS_DDoS và Truy_cap_trai_phep.
- Hạn chế: đo tổng quát hóa sang các đoạn thời gian xen kẽ, không phải dự đoán tương lai.

**Hình:** bảng 3.7 và 3.8 của báo cáo (số node mỗi lớp theo tập).

### Slide 10. Mô hình và quy trình thực nghiệm

- Baseline: LR, RF, XGBoost, LightGBM, MLP. GNN đơn: GCN, GraphSAGE, GAT (11.717 tham số).
- Cùng cách chia, 3 seed, tinh chỉnh chỉ trên val; test chạy **một lần** sau khi đóng băng (tag `freeze-v1`).
- Môi trường: Kaggle Tesla T4, tổng ≈ 4,09 giờ GPU (hạn mức 30 giờ/tuần).
- Hai chế độ báo cáo: argmax và ngưỡng hiệu chỉnh trên val (FPR val ≤ 4,08%).

**Hình:** bảng 4.2 (đếm tham số GAT), bảng 4.5 (giờ GPU).

### Slide 11. Thiết kế hybrid ban đầu thất bại và bản sửa

- Out-of-fold theo 5 nhóm khối liên tiếp: F1-macro val chỉ 0,647 (ToN) và 0,714 (BoT).
- Nguyên nhân: nhóm cuối là nơi duy nhất có một số lớp, GAT thiếu nhóm cuối chưa từng thấy lớp đó.
- Sửa: out-of-fold xen kẽ 5 × 20 khối → F1-macro val 0,9326 (ToN) và 0,9357 (BoT).
- Giá phải trả: 5 GAT, hybrid chính ToN 144.518 tham số và 9,75 MB (vượt ngưỡng nhẹ).

**Hình:** bảng 4.3 (các biến thể hybrid trên val).

### Slide 12. Kết quả test trên bộ chính NF-ToN-IoT-v3

- Ngưỡng hiệu chỉnh: hybrid chính F1-macro **0,9677** (cao nhất), FPR 1,88%, recall 96,71%.
- Argmax: (c) 0,9477 nhỉnh hơn hybrid 0,9472 (ngang nhau); MLP có FPR thấp nhất (3,53%).
- Hybrid nhẹ (30.934 tham số, 986 KB): F1-macro 0,9588, FPR 2,27%, recall 95,65% ở ngưỡng.
- ToN test dễ hơn val; không so trực tiếp điểm giữa hai tập.

**Hình:** `../reports/figures/gd6_bar_f1_fpr_nf_ton_iot_v3.png`.

### Slide 13. Hybrid có hơn baseline không? (McNemar và ablation)

- Hybrid chính vượt ML tốt nhất ở cả hai chế độ; vượt MLP và (c) chỉ ở ngưỡng; so với GraphSAGE nhỉnh +0,0078 nhưng chưa đủ 2 lần std (0,0083).
- Chỉ tiêu "hybrid vượt mọi baseline" **không đạt**.
- Ablation: (c) LightGBM + trung bình đặc trưng hàng xóm tính tay ngang hybrid (val ToN 0,9342 so với 0,9326). Lợi thế đến từ **thông tin hàng xóm**, không chứng minh được embedding GAT tốt hơn.
- Giả thuyết ở slide 5 đúng một phần.

**Hình:** bảng 5.4 (McNemar), `../reports/kaggle/gnn-iot-ids-16c-aggregate/gd5_ablation_hybrid_nf_ton_iot_v3.png`.

### Slide 14. Kết quả trên NF-BoT-IoT-v3 và vì sao FPR ≈ 18%

- Argmax: **LightGBM 0,9291 tốt hơn hybrid 0,9164** (McNemar nghiêng về LightGBM ở 3/3 seed).
- FPR ≈ 18% ở mọi mô hình: các IP máy tấn công cũng sinh flow Normal có đặc trưng giống lúc tấn công; 100% FP thuộc IP cũng từng là node tấn công.
- Ngưỡng 4,08% kéo recall hybrid xuống 47,5%.
- Nguyên nhân là dữ liệu, không phải nhãn: mọi cách gán nhãn thay thế đều làm FPR tệ hơn.

**Hình:** `../reports/figures/gd6_bar_f1_fpr_nf_bot_iot_v3.png`, `../reports/kaggle/gnn-iot-ids-15a-fpr-analysis/gd5_fpr_nf_bot_iot_v3_ip.png`.

### Slide 15. Độ nhẹ (Pareto), theo tầng node và thử chéo

- Hybrid nhẹ: nhỏ hơn 4,7 lần tham số và 10 lần kích thước, giảm F1-macro test 0,9–1,3 điểm %; GraphSAGE và GAT đơn cũng đạt mọi chỉ tiêu ở ngưỡng.
- Theo tầng (ToN, argmax): FPR hybrid 1,03% ở node có khởi tạo flow, 8,15% ở node chỉ nhận (nạn nhân).
- Thử chéo giữa hai bộ rất kém: F1-macro 0,15–0,38.
- Không đổi cách gán nhãn: giảm FPR chỉ do thu hẹp phạm vi đánh giá.

**Hình:** `../reports/kaggle/gnn-iot-ids-16c-aggregate/gd5_pareto_nf_ton_iot_v3.png`, `../reports/figures/gd6_tiers_nf_ton_iot_v3.png`, `../reports/figures/gd6_cross_dataset.png`.

### Slide 16. Phân tích lỗi

- ToN (hybrid chính, seed 42): 4.565 node sai (3,7%); 70,2% là cảnh báo giả ở node chỉ nhận flow.
- 18,2% là tấn công bị bỏ sót, chủ yếu lớp Khac (626 node → Normal); lớp Khac không được nhận ra (0/666) nhưng không tính vào F1-macro.
- BoT: 922 node sai (8,6%); 46% là FP ở node khởi tạo (IP .147–.150), 43% là bỏ sót.
- Ví dụ cụ thể (IP, cửa sổ, nhãn, dự đoán) có trong báo cáo.

**Hình:** `../reports/figures/gd6_confusion_hybrid_nf_ton_iot_v3.png`, `../reports/figures/gd6_attention_nf_ton_iot_v3.png`.

### Slide 17. AHP gợi ý mô hình triển khai và mô phỏng thời gian thực **[chờ GĐ I]**

- AHP: cây tiêu chí (F1-macro, FPR, recall, độ trễ, tham số), ma trận so sánh cặp, CR < 0,10, xếp hạng, độ nhạy.
- Mô hình được gợi ý và điều kiện áp dụng (ví dụ "nếu ưu tiên độ trễ thì chọn …").
- Mô phỏng phát lại flow: độ trễ p50/p95/p99, thông lượng, RAM, số cảnh báo giả mỗi giờ, thời gian phát hiện từng đợt.
- Demo: chạy `20_realtime` hoặc chiếu `alerts_sample.csv`.

**Hình:** (điền sau GĐ I) biểu đồ trọng số AHP, biểu đồ độ nhạy, phân bố độ trễ.

### Slide 18. Kết luận, hạn chế, hướng phát triển

- Đạt: accuracy, recall, F1-macro, độ ổn định, độ trễ (sơ bộ), 2 bộ dữ liệu; FPR đạt ở ngưỡng; hybrid nhẹ đạt mọi chỉ tiêu hiệu năng và độ nhẹ ở ngưỡng.
- Chưa đạt: "hybrid vượt mọi baseline"; hybrid chính vượt ngưỡng tham số và kích thước; FPR argmax 4,20%.
- Hạn chế: lợi thế chủ yếu từ thông tin hàng xóm; BoT bị chặn bởi dữ liệu; chia xen kẽ; thử chéo kém; AHP và thời gian thực chưa có.
- Hướng phát triển: phân loại mức flow, đặc trưng nhiều cửa sổ, ngưỡng theo tầng, thích nghi miền, làm nhẹ hơn.

**Hình:** bảng đối chiếu chỉ tiêu (bảng 6.1).

---

## Câu hỏi dự kiến và ý trả lời

| Câu hỏi | Ý trả lời (có nguồn trong báo cáo) |
|---|---|
| Vì sao GNN? | Thông tin cấu trúc (ai liên lạc với ai) có ích: LightGBM + hàng xóm vượt LightGBM thuần (ToN val 0,9342 so với 0,8984); mục 5.3 |
| Hybrid có thật sự hơn không? | Hơn ML tốt nhất, nhưng không hơn mọi baseline; (c) ngang; mục 5.2, 5.3 |
| Vì sao bộ dữ liệu này? | Có timestamp thật, mới (2025), đúng yêu cầu NF-ToN; mục 3.1 |
| Vì sao FPR BoT 18%? | Dữ liệu: IP tấn công cũng sinh flow Normal giống hệt; mục 5.8 |
| Vì sao chia theo khối? | Chia thuần thời gian làm val và test thiếu lớp; mục 3.6 |
| Vì sao AHP? | Chọn mô hình triển khai theo nhiều tiêu chí; **[chờ GĐ I]** |
| Giả định thời gian thực? | Replay flow, CPU 1 luồng; độ trễ hiện mới đo sơ bộ; mục 5.9 |
