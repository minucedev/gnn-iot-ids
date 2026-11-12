# Kế hoạch đề tài: Lightweight GNN phát hiện xâm nhập IoT thời gian thực

Hướng làm theo pipeline Phong đã dựng ở Tuần 1: NetFlow, IP = node, flow = cạnh, phân loại node theo cửa sổ.
Danh sách task chi tiết: [TASKS.md](TASKS.md).

> **Cần đối chiếu với kế hoạch gốc [docs/plan/00_TONG_QUAN.md](../plan/00_TONG_QUAN.md).** Kế hoạch gốc kéo dài 3 tuần và kết thúc 8/11/2026. File này ban đầu giả định 9 tuần (Tuần 2 bắt đầu 02/11/2026) và bổ sung các yêu cầu trong mô tả của thầy (AHP, Kaggle/Colab, ≥ 2 bộ dữ liệu thêm).
>
> **Hạn nộp thật: 15/11/2026** (Phúc xác nhận 03/11/2026; lúc đầu ghi nhầm 8/11, đã sửa trong ngày). Lịch 9 tuần và cột "Tuần" trong [TASKS.md](TASKS.md) không còn dùng được; dùng [lịch ở mục 5b](#5b-lịch-đến-hạn-nộp-15112026). **Không cắt phạm vi** so với mục 2 và 3.
>
> Nhóm: Nguyễn Minh Phúc (nhóm trưởng), Lê Trung Phong, Văn Bá Phạm Tấn. **Phúc thực hiện toàn bộ task**; Phong đã làm phần Tuần 1 (khảo sát dataset, pipeline v1).
>
> **Cập nhật 12/11/2026:** thầy đưa yêu cầu mới (xem [mục 0](#0-yêu-cầu-bổ-sung-của-thầy-12112026)). Hạn nộp **tạm tính 19/11/2026** (Phúc: "cứ cho là 1 tuần nữa"; chưa có ngày chính thức). Lịch hiện hành là [mục 5c](#5c-lịch-12111911-theo-yêu-cầu-mới); mục 5b chỉ còn để tham khảo.

## Mục lục

0. [Yêu cầu bổ sung của thầy (12/11/2026)](#0-yêu-cầu-bổ-sung-của-thầy-12112026)
1. [Đề bài toán](#1-đề-bài-toán)
2. [Kết quả cuối cùng phải nộp](#2-kết-quả-cuối-cùng-phải-nộp)
3. [Chỉ tiêu nghiệm thu](#3-chỉ-tiêu-nghiệm-thu)
4. [Phân công](#4-phân-công)
5. [Lộ trình theo giai đoạn](#5-lộ-trình-theo-giai-đoạn)
6. [Quy ước bắt buộc](#6-quy-ước-bắt-buộc)
7. [Log quyết định](#7-log-quyết-định)
8. [Rủi ro và phương án dự phòng](#8-rủi-ro-và-phương-án-dự-phòng)

## 0. Yêu cầu bổ sung của thầy (12/11/2026)

Thầy gửi danh sách sau. Cột giữa là cách nhóm đáp ứng, Phúc chốt qua chat ngày 12/11/2026 (quyết định D8–D11 ở mục 7).

| Thầy yêu cầu | Nhóm đáp ứng | Task |
|---|---|---|
| Dataset: NF-ToN, chọn ít nhất 2 bộ dữ liệu | **NF-ToN-IoT-v3** (bộ chính) + **NF-BoT-IoT-v3** (bộ thứ 2, dữ liệu IoT; Tấn đã làm notebook tiền xử lý trên Kaggle `phamta/dss-cky-bot`). NF-CSE-CIC-IDS2018-v3 và NF-ToN-IoT-v2 không còn bắt buộc | 1.8, 5.6 |
| Mô hình hybrid? Lý do chọn mô hình | **Hybrid GAT + LightGBM**: GAT học embedding ngữ cảnh hàng xóm của mỗi node, ghép với 18 đặc trưng gốc, LightGBM phân loại. Lý do: số liệu sơ bộ cho thấy mô hình cây mạnh trên đặc trưng dạng bảng (accuracy khoảng 93%) còn GNN nắm cấu trúc nhưng FPR cao (6–11%); ghép để mỗi bên bù điểm yếu bên kia | 5.7, 3.2 |
| Môi trường: Kaggle, hỗ trợ 30 giờ + T4 | Mọi huấn luyện chạy trên Kaggle T4; ghi thời gian chạy từng notebook để tổng không vượt 30 giờ GPU mỗi tuần | 0.6 |
| Xử lý EDA | EDA mức flow và mức đồ thị cho cả 2 bộ; dùng lại phần EDA của Tấn cho NF-BoT-IoT-v3 | 2.1–2.3 |
| Minh chứng hybrid cao hơn các baseline (ML, DL, …) | Cùng cách chia tập, 3 seed, mean ± std; baseline ML (LR, RF, XGBoost, LightGBM), DL (MLP), GNN đơn (GCN, GraphSAGE, GAT); kiểm định McNemar; ablation tháo từng phần của hybrid | 4.1, 5.2, 5.3, 5.8, 6.1–6.3 |

## 1. Đề bài toán

**Bối cảnh.** Gateway của mạng IoT thu được luồng NetFlow liên tục. Cần một mô hình GNN đủ nhẹ để chạy ngay trên gateway, phát hiện IP đang tấn công (DDoS, botnet, truy cập trái phép) trong vòng dưới 1 giây và gợi ý mô hình nên triển khai.

**Đầu vào.** Dãy flow $F = \{f_1, f_2, \dots\}$, mỗi flow $f = (t, \text{src\_ip}, \text{dst\_ip}, \text{proto}, \text{ports}, \text{bytes}, \text{pkts}, \text{duration}, \text{flags})$.

**Dựng đồ thị.** Chia trục thời gian thành cửa sổ dài $W$, bước $S$ (mặc định $W = S = 5$ giây; nếu bộ dữ liệu không có timestamp thì 1 cửa sổ = 1.000 flow liên tiếp). Cửa sổ $k$ cho đồ thị có hướng $G_k = (V_k, E_k)$:

- $V_k$: các IP xuất hiện trong cửa sổ. Mỗi node có vector 18 đặc trưng $x_v$ (số flow ra/vào, byte, gói, số đích/cổng khác nhau, tỉ lệ TCP/UDP/ICMP…).
- $E_k$: cạnh $u \to v$ nếu có ít nhất 1 flow từ $u$ đến $v$, kèm 3 đặc trưng cạnh (số flow, byte, gói).
- Nhãn node $y_v$: nhãn chiếm đa số trong các flow do $v$ khởi tạo (nếu $v$ chỉ nhận flow thì lấy đa số flow nhận).

**Nhiệm vụ.** Học hàm $f_\theta(G_k) \rightarrow \hat{y}_v$ cho mọi $v \in V_k$:

- **Bài toán chính (đa lớp, 5 lớp):** Normal, DoS_DDoS, Botnet_Backdoor, Truy_cap_trai_phep, Khac.
- **Bài toán phụ (nhị phân):** Normal và Tấn công. Dùng để tính FPR và recall tấn công.

**Ràng buộc "nhẹ" và "thời gian thực".** $f_\theta$ có ≤ 100.000 tham số và file ≤ 1 MB (kế hoạch gốc đặt mục tiêu < 50K). Độ trễ từ lúc đóng cửa sổ đến lúc có dự đoán: p95 ≤ 1 giây trên CPU 1 luồng.

**"Gợi ý" bằng AHP.** Dùng AHP xếp hạng các mô hình đã huấn luyện (GNN và baseline) theo nhiều tiêu chí (F1, FPR, độ trễ, tham số…) để **gợi ý mô hình triển khai trên gateway**. Cách dùng AHP cần thầy xác nhận (quyết định D6).

**Ngoài phạm vi.** Không làm web. Không triển khai trên mạng thật; triển khai là **mô phỏng phát lại (replay) flow theo thời gian thực**.

## 2. Kết quả cuối cùng phải nộp

1. **Repo GitHub** chạy lại được: `src/build_graph.py`, `src/stream_features.py`, `src/models/`, `src/evaluate.py`, `src/ahp.py` và các notebook chạy "Run All" trên Kaggle.
2. **Mô hình hybrid GAT + LightGBM** (mô hình chính từ 12/11/2026), gồm phần GAT nhẹ multi-head và phiên bản cải tiến của GAT, đã huấn luyện, có số tham số và kích thước file.
3. **Bảng so sánh**: hybrid với ≥ 4 baseline ML, MLP và ≥ 3 biến thể GNN trên 2 bộ dữ liệu (NF-ToN-IoT-v3, NF-BoT-IoT-v3), mỗi số là trung bình ± độ lệch chuẩn của 3 seed; kèm bảng ablation của hybrid.
4. **Bộ hình**: EDA, trực quan hóa đồ thị, confusion matrix, ROC/PR, đường huấn luyện, F1 theo lớp, F1 theo tham số, F1 theo độ trễ.
5. **Báo cáo AHP**: cây phân cấp, ma trận so sánh cặp (CR < 0,1), xếp hạng, phân tích độ nhạy, mô hình được gợi ý.
6. **Demo mô phỏng thời gian thực** (script dòng lệnh): phát lại flow, xuất cảnh báo theo cửa sổ, kèm báo cáo độ trễ.
7. **Báo cáo + slide.**

## 3. Chỉ tiêu nghiệm thu

Đo trên tập test của bộ chính, lấy trung bình 3 seed.

| Chỉ tiêu | Ngưỡng | Cách đo |
|---|---|---|
| Accuracy đa lớp (mức node) | ≥ 91% | Trên tập test, chỉ chạy 1 lần ở GĐ6 |
| FPR nhị phân | ≤ 4,08% | FP / (FP + TN), Normal là lớp âm |
| Recall tấn công (nhị phân) | ≥ 95% | TP / (TP + FN) |
| F1-macro trên 4 lớp chính | ≥ 0,80 | Không tính lớp Khac nếu chốt theo D5 |
| Độ ổn định | std accuracy ≤ 1,5 điểm % | 3 seed: 42, 43, 44 |
| Tổng tham số | ≤ 100.000 | `sum(p.numel() for p in model.parameters())` |
| Kích thước file mô hình | ≤ 1 MB | `state_dict` lưu bằng `torch.save` |
| Độ trễ end-to-end | p95 ≤ 1 s mỗi cửa sổ | CPU 1 luồng, ≥ 1.000 cửa sổ, ghi rõ cấu hình máy |
| So với baseline | **Hybrid** > baseline tốt nhất (ML, DL, GNN đơn) về F1-macro, chênh lệch > 2 lần std của 3 seed; nếu không đạt thì báo cáo trung thực kèm phân tích nguyên nhân | Kèm kiểm định McNemar (p-value < 0,05) trên tập test |
| Số bộ dữ liệu | ≥ 2: NF-ToN-IoT-v3 và NF-BoT-IoT-v3 | Yêu cầu của thầy ngày 12/11/2026 (trước đó là ≥ 3) |
| Tái lập | Chạy lại trên Kaggle lệch ≤ 0,5 điểm % | So với `results.csv` |

Chỉ tiêu 91% và 4,08% lấy từ mô tả đề tài. Đây là **ngưỡng tối thiểu**. Các ngưỡng còn lại (100K tham số, F1-macro 0,80, recall 95%) đề xuất ban đầu, **Phúc chốt ngày 03/11/2026 (task 0.1)**: giữ nguyên các ngưỡng trong bảng; tham số ≤ 100.000 là ngưỡng nghiệm thu, < 50.000 (mục tiêu của kế hoạch gốc) chỉ là mong muốn, không ràng buộc. Có làm AHP (D6). Không gửi thầy và không gửi Phong, Tấn để xác nhận lại.

## 4. Phân công

| Thành viên | Vai trò |
|---|---|
| **Nguyễn Minh Phúc** (nhóm trưởng) | Thực hiện toàn bộ task từ GĐ0 đến GĐ9 và ra mọi quyết định D1–D7 |
| Lê Trung Phong | Đã làm Tuần 1: phiếu dataset và pipeline v1 ([docs/dataset_cards/](../dataset_cards/), [notebooks/phong_tuan1/](../../notebooks/phong_tuan1/)). Nguồn hỏi khi cần dữ liệu đã xử lý hoặc chi tiết pipeline |
| Văn Bá Phạm Tấn | Đọc góp ý; nên tham gia chấm ma trận AHP (task 7.1) |

Quy tắc kiểm tra: vì một người làm hết nên không có review chéo. Một task chỉ được chuyển sang "Xong" khi đã tự kiểm **từng** tiêu chí và ghi bằng chứng (đường dẫn file, con số, hash) vào dòng **Kết quả** của task trong [TASKS.md](TASKS.md).

## 5. Lộ trình theo giai đoạn

Phải đạt hết cổng nghiệm thu của một giai đoạn mới sang giai đoạn sau.

| GĐ | Tuần | Mục tiêu | Đầu vào | Đầu ra | Cổng nghiệm thu |
|---|---|---|---|---|---|
| GĐ0 Chốt đề & hạ tầng | 2 | Thống nhất bài toán, quyết định mở, repo, Kaggle, module đánh giá | Mô tả của thầy, notebook Tuần 1 của Phong | File này đã duyệt, log quyết định, repo, `build_graph.py`, `evaluate.py` | D1–D7 đã chốt; Run All trên Kaggle không lỗi; `evaluate.py` qua test với mô hình giả |
| GĐ1 Dữ liệu | 1 (xong một phần), 2–3 | ≥ 3 bộ dữ liệu sạch, có đồ thị, chia tập đúng | File CSV gốc, phiếu dataset Tuần 1 | `clean_*.parquet`, `graphs_*.pt`, `split.json`, `scaler.pkl`, bảng lý do chọn bộ | Mọi lớp chính có mặt ở cả 3 tập; kiểm tra rò rỉ và tái lập đều đạt |
| GĐ2 EDA & trực quan hóa | 3 | Hiểu dữ liệu, chứng minh cấu trúc đồ thị có ích | Đồ thị và dữ liệu sạch của GĐ1 | Notebook EDA, ≥ 12 hình có nhận xét, chỉ số homophily | Mỗi hình có nhận xét 1–2 câu; có ≥ 3 cửa sổ mẫu được vẽ |
| GĐ3 Lý thuyết | 2–4 (song song) | Trả lời "tại sao chọn, còn mô hình nào khác" | Tài liệu, bài báo | Bản nháp chương lý thuyết, bảng tổng quan nghiên cứu | ≥ 15 tài liệu, ≥ 6 tài liệu từ 2022 trở lại; bảng so sánh ≥ 6 hướng mô hình |
| GĐ4 Baseline ML | 4 | Mốc so sánh không dùng đồ thị | `split.json`, đặc trưng node, `evaluate.py` | Kết quả ≥ 4 baseline trong `results.csv` (chỉ trên val) | Tinh chỉnh chỉ trên val; 3 seed; có số tham số và độ trễ |
| GĐ5 Lightweight GNN | 4–6 | GAT multi-head nhẹ + bản cải tiến | Đồ thị PyG, `evaluate.py`, kết quả baseline | Mô hình đã huấn luyện, bảng ablation, cấu hình đóng băng | ≤ 100K tham số; vượt chỉ tiêu trên val; ablation đủ 5 chiều |
| GĐ6 Đánh giá tổng hợp | 7 | Chạy test 1 lần, so sánh, phân tích lỗi | Các mô hình đóng băng (có commit hash) | Bảng tổng hợp, bộ biểu đồ, phân tích lỗi, McNemar | Mọi số trong báo cáo truy được về `results.csv` |
| GĐ7 AHP | 7 | Gợi ý mô hình triển khai | Bảng tổng hợp GĐ6 | `ahp.py`, ma trận, xếp hạng, độ nhạy | CR < 0,10 cho mọi ma trận; xếp hạng tái lập được |
| GĐ8 Mô phỏng thời gian thực | 6–8 | Chứng minh chạy được dưới 1 giây | Mô hình được AHP gợi ý, `build_graph.py` | `stream_features.py`, `detector.py`, log cảnh báo, báo cáo độ trễ | Đặc trưng online khớp offline; p95 ≤ 1 s; dự đoán online khớp offline ≥ 99,9% |
| GĐ9 Báo cáo & bảo vệ | 8–9 | Hoàn thiện sản phẩm nộp | Tất cả đầu ra ở trên | Báo cáo, slide, README, video demo | Review chéo xong; người ngoài clone repo chạy lại được |

## 5b. Lịch đến hạn nộp 15/11/2026

Còn 12 ngày kể từ 03/11. Giữ nguyên phạm vi ở mục 2 và 3 (≥ 3 bộ dữ liệu, ablation đủ 5 chiều, bộ hình đầy đủ, ≥ 15 tài liệu). Task 1.3 đã bỏ theo quyết định của Phúc (v3 có timestamp thật). Lịch do Claude đề xuất, Phúc duyệt qua chat 03/11/2026.

| Ngày | Việc | Task |
|---|---|---|
| 03–04/11 | Dựng đồ thị v3 và v2; baseline ML; vòng huấn luyện PyG; chọn và dựng bộ thứ 3; từ điển dữ liệu | 1.5, 1.6, 0.6, 4.1, 5.1, 1.4, 1.7 |
| 05–06/11 | EDA đủ ≥ 12 hình, trực quan hóa đồ thị; viết lý thuyết IDS và GNN; GCN, GraphSAGE | 2.1–2.3, 3.1, 3.2, 5.2 |
| 07–08/11 | GAT nhẹ multi-head; bản cải tiến; baseline mức flow nếu còn chỗ | 5.3, 5.4, 4.2 |
| 09–10/11 | Ablation đủ 5 chiều (gồm cả kích thước cửa sổ); bộ phụ và tổng quát hóa giữa các bộ; `stream_features.py` | 5.5, 5.6, 8.1 |
| 11–12/11 | **Đóng băng mô hình bằng commit hash**, chạy test một lần với 3 seed; bảng và hình; McNemar; AHP; `detector.py` và đo độ trễ | 6.1–6.3, 7.1, 7.2, 8.2, 8.3 |
| 13–14/11 | Đánh giá triển khai; lý thuyết baseline/AHP và tổng quan nghiên cứu; so sánh với nghiên cứu khác; viết báo cáo; slide, video demo | 8.4, 3.3, 3.4, 6.4, 9.1, 9.2, 9.3 |
| 15/11 | Chạy lại từ repo sạch, kiểm tra README, nộp | — |

**Điểm nghẽn cần để mắt:**
- Ngày 11/11 là mốc cứng: sau khi đóng băng mô hình không được chỉnh theo kết quả test (quy ước 6.2). Ablation và cải tiến phải xong trước đó.
- Báo cáo (9.2) viết dồn vào 2 ngày cuối là rủi ro lớn nhất; phần không phụ thuộc kết quả (lý thuyết, dữ liệu, phương pháp) nên viết từ 05/11.
- Chưa có xác nhận của thầy về D6 (AHP dùng để chọn mô hình triển khai).

## 5c. Lịch 12/11–19/11 theo yêu cầu mới

Lập ngày 12/11/2026, hạn tạm tính 19/11. Thứ tự ưu tiên theo danh sách của thầy: dữ liệu và EDA → baseline → hybrid và minh chứng → AHP và thời gian thực → báo cáo. Kết quả cũ trong `reports/tables/prelim_split_v1/` không dùng cho báo cáo; mọi mô hình chạy lại trên cách chia đã chốt ở D3.

| Ngày | Việc | Task |
|---|---|---|
| 12–13/11 | Dựng đồ thị NF-BoT-IoT-v3 (tích hợp notebook của Tấn); EDA 2 bộ; chạy baseline ML và MLP trên NF-ToN-IoT-v3 và NF-BoT-IoT-v3 (chỉ val) | 1.8, 2.1–2.3, 4.1 |
| 14/11 | GCN, GraphSAGE, GAT trên 2 bộ; viết `src/hybrid.py` và chạy hybrid trên val | 5.2, 5.3, 5.7 |
| 15/11 | Ablation hybrid, hiệu chỉnh ngưỡng quyết định để giảm FPR; ablation GAT (head, lớp, độ rộng, đặc trưng cạnh, cửa sổ) | 5.8, 5.4, 5.5 |
| 16/11 | **Đóng băng mô hình bằng commit hash**, chạy test một lần 3 seed; bảng, hình, McNemar; thử chéo giữa 2 bộ | 6.1–6.3, 5.6 |
| 17/11 | AHP; `stream_features.py`, `detector.py`, đo độ trễ | 7.1, 7.2, 8.1–8.3 |
| 18/11 | Đánh giá triển khai; lý thuyết; viết báo cáo, slide | 8.4, 3.1–3.4, 9.1–9.3 |
| 19/11 | Chạy lại từ repo sạch, kiểm tra README, nộp | — |

**Ngân sách Kaggle (30 giờ GPU T4 mỗi tuần):** ghi thời gian chạy của từng notebook vào dòng **Kết quả** của task. Baseline cây chạy trên CPU của Kaggle, không tính vào giờ GPU. Nếu tổng giờ GPU dự kiến vượt 25 giờ thì báo Phúc trước khi chạy tiếp.

**Điểm nghẽn:** từ 03/11 đến 12/11 chưa có kết quả nào trên cách chia tập mới, nên 12–13/11 phải chạy được baseline. Mốc cứng là 16/11 (đóng băng mô hình trước khi xem tập test).

## 6. Quy ước bắt buộc

1. **Seed:** `RANDOM_STATE = 42` khi phát triển. Kết quả cuối chạy 3 seed 42, 43, 44 và báo cáo trung bình ± độ lệch chuẩn.
2. **Tập test chỉ dùng một lần** ở GĐ6, sau khi đã đóng băng mô hình bằng commit hash. Mọi tinh chỉnh làm trên val.
3. **Chống rò rỉ:** không dùng IP hay port làm đặc trưng; scaler chỉ fit trên train; mọi tập dùng chung một `split.json`; giữa các tập có khoảng trống (gap) cửa sổ.
4. **Lấy mẫu khi thiếu RAM:** lấy mẫu **theo cửa sổ**, không lấy mẫu flow ngẫu nhiên (lấy mẫu flow sẽ phá cấu trúc đồ thị).
5. **Mọi kết quả** ghi vào `reports/tables/results.csv` qua `evaluate.py`. Không chép số bằng tay.
6. **Cột của `results.csv`:** `dataset, model, task, seed, split, accuracy, balanced_acc, precision_macro, recall_macro, f1_macro, f1_per_class, fpr, recall_attack, roc_auc, n_params, size_kb, train_time_s, infer_ms_per_window, author, commit`.
7. **Hình** lưu ở `reports/figures/gd<số>_<tên>.png`, dpi ≥ 150, có tiêu đề, nhãn trục và đơn vị.
8. **Môi trường:** huấn luyện trên Kaggle (ưu tiên vì RAM khoảng 30 GB), Colab, hoặc máy cá nhân có GPU. Một notebook chỉ được chuyển sang "Chờ review" khi đã chạy "Run All" không lỗi trên Kaggle.
9. **Git:** commit và push thẳng lên `main`, không tạo nhánh, không PR, không bật bảo vệ nhánh (Phúc quyết định 02/11/2026). Commit ghi mã task ở đầu, ví dụ `feat(0.4): ...`. Tự kiểm đủ tiêu chí rồi mới chuyển task sang "Xong".
10. **Trạng thái task:** Chưa làm → Đang làm → Chờ review → (Cần sửa →) Xong. Cập nhật trực tiếp trong [TASKS.md](TASKS.md).

## 7. Log quyết định

Chốt trong GĐ0. Muốn đổi quyết định đã chốt thì thêm dòng mới, không sửa đè.

| Mã | Câu hỏi | Lựa chọn | Đề xuất | Đã chốt (lựa chọn, lý do, người duyệt, ngày) |
|---|---|---|---|---|
| D1 | Bộ NetFlow chính | NF-ToN-IoT-v3 (nếu xác minh được là có `FLOW_START_MILLISECONDS`) / NF-ToN-IoT-v2 (cửa sổ 1.000 flow) | v3, vì có timestamp thật và đáp ứng "tính thời sự". Nếu không có v3 thì dùng v2 |  **NF-ToN-IoT-v3.** Task 1.2 xác minh có `FLOW_START_MILLISECONDS` thật, file sắp theo thời gian 100%, số dòng khớp UQ (27.520.260). Phúc (theo đề xuất, qua chat), 03/11/2026 |
| D2 | Kích thước cửa sổ | 1 / 2 / 5 / 10 giây | 5 giây làm mặc định, các mức còn lại đưa vào ablation |  **5 giây** (W = S). Trên v3: 66.722 cửa sổ, 812.318 node; số node mỗi cửa sổ p50 = 11, p95 = 19, p99 = 68, tối đa 269, nên đồ thị rất nhỏ. Bỏ ablation theo W do hạn nộp 8/11. Phúc (theo đề xuất, qua chat), 03/11/2026 |
| D3 | Cách chia tập | `chronological` / `blocked` (10 khối) | `blocked`, vì chia `chronological` làm Botnet_Backdoor có 0 node ở train. Lưu ý kế hoạch gốc cấm đổi cách chia "cho dễ", nên cần nhóm trưởng duyệt kèm số liệu |  **`blocked`, 10 khối, mỗi ranh giới giữa hai tập cách 1 cửa sổ (`gap = 1`).** Chia `chronological` làm val và test không có node DoS_DDoS và Truy_cap_trai_phep nào (xem bảng bên dưới), nên không tính được F1 trên 4 lớp chính; `blocked` có đủ 5 lớp ở cả 3 tập. Đây không phải đổi cách chia "cho dễ" mà vì chia thời gian thuần túy không đo được. Hạn chế phải ghi vào báo cáo: các tấn công diễn ra theo ngày, nên `blocked` đo khả năng tổng quát hóa sang các đoạn thời gian xen kẽ, không phải dự đoán hoàn toàn trong tương lai. Phúc (theo đề xuất, qua chat), 03/11/2026 |
| D4 | Mức phân loại | Node / Edge (flow) / Cả hai | Node là chính (đúng pipeline hiện tại); edge là tùy chọn nếu còn thời gian |  **Node** là chính; edge bỏ do hạn nộp. Phúc (theo đề xuất, qua chat), 03/11/2026 |
| D5 | Lớp Khac (mitm, ransomware ≈ 0,06%) | Giữ / Gộp vào lớp khác / Loại | Giữ 5 lớp nhưng tính chỉ tiêu F1-macro trên 4 lớp chính; báo cáo lớp Khac riêng |  **Giữ 5 lớp**; F1-macro tính trên 4 lớp chính, báo cáo Khac riêng. Khac có 2.398 node (0,3%): train 393, val 1.164, test 841 (chia `blocked`). Phúc (theo đề xuất, qua chat), 03/11/2026 |
| D6 | AHP dùng để làm gì | Chọn mô hình triển khai / Xếp hạng cảnh báo | Chọn mô hình triển khai. Hỏi thầy xác nhận |  **Chọn mô hình triển khai.** Chưa có xác nhận của thầy (Phúc quyết định không hỏi thầy); ghi là rủi ro ở mục 5b. Phúc (theo đề xuất, qua chat), 03/11/2026 |
| D7 | Máy đo độ trễ | Kaggle CPU / laptop / Docker giới hạn CPU-RAM / Raspberry Pi | Docker giới hạn CPU-RAM như kế hoạch gốc; nếu không kịp thì laptop giới hạn 1 luồng |  **Laptop i7-12800H giới hạn 1 luồng** (`torch.set_num_threads(1)`), ghi rõ cấu hình máy. Docker giới hạn CPU-RAM chỉ làm nếu còn thời gian, đây chính là phương án dự phòng trong đề xuất ban đầu do hạn nộp 8/11. Phúc (theo đề xuất, qua chat), 03/11/2026 |

**Điều chỉnh 03/11/2026** (hạn nộp thật là 15/11, không phải 8/11; các ô D2, D4, D7 ở trên được viết khi còn tưởng hạn là 8/11, phần lý do đó **không còn hiệu lực**):

| Mã | Điều chỉnh |
|---|---|
| D2 | Khôi phục ablation theo kích thước cửa sổ (1 / 2 / 5 / 10 giây); 5 giây vẫn là mặc định |
| D4 | Giữ đề xuất gốc: node là chính, edge là tùy chọn nếu còn thời gian (task 4.2) |
| D7 | Giữ đề xuất gốc: Docker giới hạn CPU-RAM; laptop 1 luồng là phương án dự phòng nếu không kịp |

**Điều chỉnh D3 (03/11/2026)**, sau khi thử 12 cấu hình chia trên từng bộ ([reports/tables/split_search/](../../reports/tables/split_search/), sinh bởi [notebooks/08_split_search.ipynb](../../notebooks/08_split_search.ipynb)). Cách chia phải đạt cả hai tiêu chí của task 1.5: mỗi lớp chính có ≥ 100 node ở train và ≥ 30 ở val, test; tỉ lệ node train/val/test trong 70/15/15 ± 3 điểm %. Chọn **số khối ít nhất** đạt cả hai, chia theo **số node** (không theo số cửa sổ), gap = 1 cho mọi bộ:

| Bộ | Cấu hình | Lý do |
|---|---|---|
| NF-ToN-IoT-v3 (chính) | `blocked`, 10 khối, theo node | Giữ nguyên D3; theo số cửa sổ thì tỉ lệ node lệch (73,2/12,1/14,7) |
| NF-CSE-CIC-IDS2018-v3 | `blocked`, **20 khối**, theo node | 10 khối làm val và test không có node DoS_DDoS nào |
| NF-ToN-IoT-v2 (phụ) | `blocked`, **200 khối**, theo node | Ít khối hơn thì val hoặc test thiếu Botnet_Backdoor (v2 chỉ có 16.809 flow backdoor) |

Hạn chế phải ghi vào báo cáo: nhiều khối làm ranh giới train/val/test sát nhau hơn, nên điểm val và test của v2 (200 khối) có thể lạc quan hơn một chút. Việc chọn chỉ dựa trên đếm nhãn, không dùng kết quả mô hình. Bảng bên dưới là số liệu của cấu hình cũ (10 khối, theo số cửa sổ, gap 0), giữ lại làm bằng chứng cho lý do điều chỉnh.

**Bằng chứng D3: số node mỗi lớp ở từng tập** (NF-ToN-IoT-v3, W = S = 5 s, gap = 0; dữ liệu trong [reports/tables/gd0_split_stats_nf_ton_iot_v3_w5.json](../../reports/tables/gd0_split_stats_nf_ton_iot_v3_w5.json), sinh bởi [notebooks/03_split_stats_nf_ton_iot_v3.ipynb](../../notebooks/03_split_stats_nf_ton_iot_v3.ipynb) trên Kaggle). Số liệu lấy với gap = 0; gap = 1 chỉ bỏ thêm vài cửa sổ ở ranh giới.

| Cách chia | Tập | Cửa sổ | Node | Normal | DoS_DDoS | Botnet_Backdoor | Truy_cap_trai_phep | Khac |
|---|---|---|---|---|---|---|---|---|
| chronological | train | 46.705 | 700.439 | 456.291 | 106.573 | 197 | 136.570 | 808 |
| chronological | val | 10.008 | 65.350 | 43.544 | **0** | 21.806 | **0** | **0** |
| chronological | test | 10.009 | 46.529 | 30.467 | **0** | 14.472 | **0** | 1.590 |
| blocked | train | 46.702 | 594.433 | 380.154 | 73.351 | 27.227 | 113.308 | 393 |
| blocked | val | 10.010 | 98.722 | 63.504 | 15.492 | 5.013 | 13.549 | 1.164 |
| blocked | test | 10.010 | 119.163 | 86.644 | 17.730 | 4.235 | 9.713 | 841 |

Làm sạch: 27.520.260 flow, bỏ 1.858.626 bản ghi trùng còn 25.661.634. RAM đỉnh 18,9 GB trên Kaggle.

**Quyết định mới ngày 12/11/2026** (theo yêu cầu bổ sung của thầy ở [mục 0](#0-yêu-cầu-bổ-sung-của-thầy-12112026); Claude đề xuất, Phúc chọn qua chat):

| Mã | Câu hỏi | Đã chốt |
|---|---|---|
| D8 | Bộ dữ liệu thứ 2 | **NF-BoT-IoT-v3** (dữ liệu IoT; dựng đồ thị bằng `src/build_graph.py`, tận dụng EDA trong notebook của Tấn). Bỏ NF-CSE-CIC-IDS2018-v3 khỏi phạm vi bắt buộc vì không phải IoT và tốn giờ Kaggle (baseline từng vượt 4 giờ). NF-ToN-IoT-v2 chỉ dùng nếu còn thời gian. Hạn chế đã biết của NF-BoT-IoT-v3: không có lớp Botnet_Backdoor riêng, 20 IP nguồn, Benign 0,3% số flow |
| D9 | Kiến trúc mô hình hybrid | **GAT + LightGBM** (phương án B là GAT + GRU, không chọn vì cần nhiều code mới). Chi tiết và cách chứng minh ở task 5.7, 5.8 |
| D10 | AHP và mô phỏng thời gian thực | Vẫn làm, nhưng **sau** khi có kết quả hybrid và baseline |
| D11 | Hạn nộp | **Tạm tính 19/11/2026** (Phúc: "cứ cho là 1 tuần nữa"). Cập nhật khi có ngày chính thức |

## 8. Rủi ro và phương án dự phòng

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Không có hoặc không tải được NetFlow v3 | Task 1.2 không tìm được nguồn chính thức | Dùng NF-ToN-IoT-v2 với giả định thứ tự dòng = thời gian, kiểm chứng bằng ToN-IoT có `ts` (task 1.3) |
| Botnet_Backdoor quá ít node | < 100 node ở train sau khi chia | Dùng class weight hoặc focal loss, báo cáo recall riêng cho lớp này |
| GNN không hơn baseline cây | Trên val, F1 GNN < F1 XGBoost | Vẫn hợp lệ: phân tích nguyên nhân (homophily, đặc trưng), nhấn mạnh tham số và độ trễ; AHP quyết định |
| Dữ liệu quá dễ (≈ 99%) | Mọi mô hình > 99% | Báo cáo F1-macro, lớp hiếm, thử tổng quát hóa giữa các bộ |
| Tràn RAM trên Kaggle/Colab | Kernel chết | Dùng file parquet, đọc theo cột, lấy mẫu theo cửa sổ |
| Không kịp hạn nộp (kế hoạch gốc kết thúc 8/11) | Cuối Tuần 2 chưa xong GĐ1 | Cắt các task "Nên có" và "Tùy chọn", giảm còn 1 bộ phụ, bỏ ablation W |
| Hybrid không hơn baseline | Trên val, F1-macro của hybrid không vượt LightGBM quá 2 lần std | Không chỉnh theo tập test. Thử thêm embedding out-of-fold (task 5.7) và hiệu chỉnh ngưỡng trên val; nếu vẫn không đạt thì báo cáo trung thực bảng ablation |
| Embedding của tập train "quá sạch" | Hybrid gần 100% trên train nhưng tụt mạnh trên val | GAT đã thấy nhãn của node train nên embedding train có thể làm LightGBM học vẹt; dùng embedding out-of-fold theo khối thời gian (task 5.7) |
| NF-BoT-IoT-v3 quá dễ hoặc quá ít Normal | Mọi mô hình ≈ 99,9%, FPR đo trên rất ít node Normal | Báo cáo số node Normal ở tập test cạnh FPR; dùng NF-ToN-IoT-v3 làm bộ chính để kết luận |
| Vượt 30 giờ GPU Kaggle mỗi tuần | Tổng giờ GPU ghi trong TASKS.md > 25 giờ | Chạy baseline cây trên CPU; giảm số lần thử siêu tham số; báo Phúc trước khi chạy tiếp |
