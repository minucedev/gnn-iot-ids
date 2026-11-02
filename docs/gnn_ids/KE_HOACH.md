# Kế hoạch đề tài: Lightweight GNN phát hiện xâm nhập IoT thời gian thực

Hướng làm theo pipeline Phong đã dựng ở Tuần 1: NetFlow, IP = node, flow = cạnh, phân loại node theo cửa sổ.
Danh sách task chi tiết: [TASKS.md](TASKS.md).

> **Cần đối chiếu với kế hoạch gốc [docs/plan/00_TONG_QUAN.md](../plan/00_TONG_QUAN.md).** Kế hoạch gốc kéo dài 3 tuần và kết thúc 8/11/2026. File này giả định 9 tuần (Tuần 2 bắt đầu 02/11/2026) và bổ sung các yêu cầu trong mô tả của thầy (AHP, Kaggle/Colab, ≥ 2 bộ dữ liệu thêm). Khi biết hạn nộp thật, chỉnh lại cột "Tuần" trong [TASKS.md](TASKS.md).
>
> Nhóm: Nguyễn Minh Phúc (nhóm trưởng), Lê Trung Phong, Văn Bá Phạm Tấn. **Phúc thực hiện toàn bộ task**; Phong đã làm phần Tuần 1 (khảo sát dataset, pipeline v1).

## Mục lục

1. [Đề bài toán](#1-đề-bài-toán)
2. [Kết quả cuối cùng phải nộp](#2-kết-quả-cuối-cùng-phải-nộp)
3. [Chỉ tiêu nghiệm thu](#3-chỉ-tiêu-nghiệm-thu)
4. [Phân công](#4-phân-công)
5. [Lộ trình theo giai đoạn](#5-lộ-trình-theo-giai-đoạn)
6. [Quy ước bắt buộc](#6-quy-ước-bắt-buộc)
7. [Log quyết định](#7-log-quyết-định)
8. [Rủi ro và phương án dự phòng](#8-rủi-ro-và-phương-án-dự-phòng)

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
2. **Mô hình GAT nhẹ multi-head** (mô hình chính) và ≥ 1 phiên bản cải tiến, đã huấn luyện, có số tham số và kích thước file.
3. **Bảng so sánh**: ≥ 4 baseline ML + ≥ 3 biến thể GNN trên ≥ 3 bộ dữ liệu, mỗi số là trung bình ± độ lệch chuẩn của 3 seed.
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
| So với baseline | GNN chính ≥ baseline tốt nhất về F1-macro, **hoặc** có phân tích nguyên nhân | Kèm kiểm định McNemar (p-value) |
| Số bộ dữ liệu | ≥ 3 (1 NetFlow chính + ≥ 2 bộ khác) | Theo yêu cầu của thầy |
| Tái lập | Chạy lại trên Kaggle lệch ≤ 0,5 điểm % | So với `results.csv` |

Chỉ tiêu 91% và 4,08% lấy từ mô tả đề tài. Đây là **ngưỡng tối thiểu**. Các ngưỡng còn lại (100K tham số, F1-macro 0,80, recall 95%) là đề xuất, cần chốt ở task 0.1.

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
| D1 | Bộ NetFlow chính | NF-ToN-IoT-v3 (nếu xác minh được là có `FLOW_START_MILLISECONDS`) / NF-ToN-IoT-v2 (cửa sổ 1.000 flow) | v3, vì có timestamp thật và đáp ứng "tính thời sự". Nếu không có v3 thì dùng v2 | |
| D2 | Kích thước cửa sổ | 1 / 2 / 5 / 10 giây | 5 giây làm mặc định, các mức còn lại đưa vào ablation | |
| D3 | Cách chia tập | `chronological` / `blocked` (10 khối) | `blocked`, vì chia `chronological` làm Botnet_Backdoor có 0 node ở train. Lưu ý kế hoạch gốc cấm đổi cách chia "cho dễ", nên cần nhóm trưởng duyệt kèm số liệu | |
| D4 | Mức phân loại | Node / Edge (flow) / Cả hai | Node là chính (đúng pipeline hiện tại); edge là tùy chọn nếu còn thời gian | |
| D5 | Lớp Khac (mitm, ransomware ≈ 0,06%) | Giữ / Gộp vào lớp khác / Loại | Giữ 5 lớp nhưng tính chỉ tiêu F1-macro trên 4 lớp chính; báo cáo lớp Khac riêng | |
| D6 | AHP dùng để làm gì | Chọn mô hình triển khai / Xếp hạng cảnh báo | Chọn mô hình triển khai. Hỏi thầy xác nhận | |
| D7 | Máy đo độ trễ | Kaggle CPU / laptop / Docker giới hạn CPU-RAM / Raspberry Pi | Docker giới hạn CPU-RAM như kế hoạch gốc; nếu không kịp thì laptop giới hạn 1 luồng | |

## 8. Rủi ro và phương án dự phòng

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Không có hoặc không tải được NetFlow v3 | Task 1.2 không tìm được nguồn chính thức | Dùng NF-ToN-IoT-v2 với giả định thứ tự dòng = thời gian, kiểm chứng bằng ToN-IoT có `ts` (task 1.3) |
| Botnet_Backdoor quá ít node | < 100 node ở train sau khi chia | Dùng class weight hoặc focal loss, báo cáo recall riêng cho lớp này |
| GNN không hơn baseline cây | Trên val, F1 GNN < F1 XGBoost | Vẫn hợp lệ: phân tích nguyên nhân (homophily, đặc trưng), nhấn mạnh tham số và độ trễ; AHP quyết định |
| Dữ liệu quá dễ (≈ 99%) | Mọi mô hình > 99% | Báo cáo F1-macro, lớp hiếm, thử tổng quát hóa giữa các bộ |
| Tràn RAM trên Kaggle/Colab | Kernel chết | Dùng file parquet, đọc theo cột, lấy mẫu theo cửa sổ |
| Không kịp hạn nộp (kế hoạch gốc kết thúc 8/11) | Cuối Tuần 2 chưa xong GĐ1 | Cắt các task "Nên có" và "Tùy chọn", giảm còn 1 bộ phụ, bỏ ablation W |
