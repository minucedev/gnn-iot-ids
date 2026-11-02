# TỔNG QUAN ĐỀ TÀI VÀ KẾ HOẠCH 3 TUẦN

**Đề tài:** Ứng dụng mạng Lightweight GNN gợi ý phát hiện xâm nhập IoT thời gian thực
**Nhóm:** Phúc (nhóm trưởng), Phong, Tấn
**Thời gian:** 3 tuần (giả định bắt đầu 19/10/2026; đổi ngày nếu hạn khác)

| File | Nội dung |
|---|---|
| `00_TONG_QUAN.md` | File này: đề tài, mục tiêu, chuẩn bị, tiêu chí dataset, mốc thời gian |
| `01_PHUC.md` | Việc của Phúc (trưởng nhóm, mô hình, Docker) |
| `02_PHONG.md` | Việc của Phong (dataset, đồ thị, baseline) |
| `03_TAN.md` | Việc của Tấn (dataset, làm nhẹ mô hình, đo hiệu năng, demo) |

---

## 1. Đề tài

### 1.1 Bài toán
Thiết bị IoT yếu về tài nguyên và thường bị tấn công: DDoS, botnet (kiểu Mirai), truy cập trái phép. IDS truyền thống phân loại từng flow riêng lẻ nên bỏ qua **quan hệ giữa các thiết bị**.

### 1.2 Ý tưởng
- Biểu diễn lưu lượng IoT thành **đồ thị**: thiết bị (IP) = node, giao tiếp = edge.
- Chia lưu lượng thành **cửa sổ trượt ngắn** (1–5 giây), mỗi cửa sổ là một đồ thị con.
- Dùng **Graph Attention Network (GAT)** nhiều head, thu gọn thành bản nhẹ để phân loại node (bình thường / loại tấn công).
- Chạy thời gian thực trên thiết bị biên (mô phỏng bằng Docker giới hạn CPU/RAM).

### 1.3 Vì sao GAT
- Tấn công có cấu trúc trên đồ thị: DDoS (nhiều node cùng hướng về một đích), botnet (một node C&C nối nhiều node), quét cổng (một node nối rất nhiều đích).
- Attention tự học láng giềng nào quan trọng; multi-head nhìn nhiều kiểu quan hệ cùng lúc; trọng số attention còn giải thích được.

## 2. Mục tiêu

### 2.1 Chỉ tiêu bắt buộc (theo đề)

| Chỉ tiêu | Mục tiêu |
|---|---|
| Test accuracy | >= 91% |
| False Positive Rate | <= 4,08% |
| Độ trễ suy luận trên thiết bị (mô phỏng) | < 1 giây |

### 2.2 Chỉ tiêu bổ sung (để báo cáo thuyết phục)
- Precision, Recall, F1 (macro), ma trận nhầm lẫn, ROC-AUC, detection rate theo từng lớp tấn công.
- Số tham số (mục tiêu < 50K), kích thước mô hình (< 1 MB), RAM, CPU.
- Bảng so sánh với baseline: Random Forest, MLP, GCN, GraphSAGE, (GAT lớn nếu còn thời gian).
- Ablation: số head, số lớp, kích thước cửa sổ, có/không lượng tử hóa.
- Hình attention của một cuộc tấn công (tính giải thích được).

### 2.3 Ngoài phạm vi
- Không triển khai trên mạng thật; không huấn luyện online/liên tục; không phát hiện tấn công mới hoàn toàn (zero-day) ngoài phần thảo luận.

## 3. Định nghĩa dùng chung (bắt buộc thống nhất)

- **Độ trễ:** tính từ lúc cửa sổ đóng đến lúc có dự đoán (gồm trích đặc trưng + dựng đồ thị + suy luận). Không tính thời gian thu thập cửa sổ. Báo cáo tách riêng từng thành phần.
- **Chia dữ liệu:** chia **theo thời gian** (train < val < test), không random theo flow, không để cửa sổ chồng lấp giữa các tập. Dùng chung 1 file `split.json`.
- **FPR:** FP / (FP + TN), tính cho lớp "bình thường" so với "mọi loại tấn công gộp".
- **Random seed:** cố định `42` cho mọi thí nghiệm.
- **Đo hiệu năng:** warm-up 20 lần, đo >= 200 lần, báo trung bình và p95.

## 4. Chuẩn bị chung (làm trong ngày 1 của tuần 1)

### 4.1 Công cụ
- [ ] Mỗi người cài Git, Python 3.10 hoặc 3.11, VS Code.
- [ ] Phúc: cài Docker Desktop (bật WSL2), kiểm tra `docker run hello-world` chạy được.
- [ ] Tạo repo GitHub chung, mời Phong và Tấn.
- [ ] Mỗi người cài thư viện: PyTorch, PyTorch Geometric, scikit-learn, pandas, pyarrow, numpy, matplotlib, onnx, onnxruntime, jupyter. Phúc chốt phiên bản và ghi vào `requirements.txt`.
- [ ] Chọn công cụ viết báo cáo chung (Word hoặc Overleaf) và tạo file chung.
- [ ] Tạo nhóm chat và lịch họp (xem mục 8).

### 4.2 Cấu trúc repo thống nhất
```
DSS/
  plan/              # các file md này
  data/
    raw/             # dữ liệu gốc (KHÔNG commit, ghi link tải vào README)
    processed/       # parquet sạch, đồ thị .pt, split.json, scaler.pkl
  src/
    build_graph.py
    stream_features.py
    model_gat.py
    train.py
    baselines.py
    realtime_pipeline.py
    benchmark.py
  notebooks/         # 01_eda.ipynb, ...
  docker/            # Dockerfile, docker-compose.yml
  results/           # bảng, hình, log (có tên file rõ ràng)
  report/            # chương báo cáo (md/docx)
  dataset_cards/     # phiếu đánh giá dataset
  requirements.txt
```
- [ ] Phúc tạo repo theo cấu trúc trên và `.gitignore` (loại `data/raw/`, file lớn).

### 4.3 Quy ước làm việc
- Mỗi người làm trên nhánh riêng (`phong/...`, `tan/...`, `phuc/...`), gộp vào `main` qua pull request, Phúc duyệt.
- Mọi đầu ra nằm trong repo đúng thư mục; không nộp qua tin nhắn.
- Tên file kết quả: `results/<nguoi>_<noi_dung>_<ngay>.csv/png`.

## 5. Tiêu chí chọn dataset

### 5.1 Điều kiện bắt buộc (không đạt thì loại ngay)

| # | Điều kiện | Bằng chứng cần nộp |
|---|---|---|
| G1 | Có địa chỉ nguồn và đích (IP hoặc MAC), hoặc dựng lại được từ PCAP trong thời gian hợp lý | Ảnh/ghi chú 5 dòng đầu có cột IP |
| G2 | Có timestamp để chia cửa sổ | Ảnh/ghi chú cột thời gian |
| G3 | Có nhãn (ít nhất bình thường/tấn công, tốt nhất cả loại tấn công) | Bảng giá trị nhãn và số mẫu |
| G4 | Có >= 3 nhóm tấn công: DDoS/DoS, botnet (hoặc backdoor/C&C), truy cập trái phép (brute force/scanning/injection) | Bảng ánh xạ nhãn gốc sang 3 nhóm |
| G5 | Tải công khai, giấy phép cho phép dùng học thuật | Link tải và trang license |
| G6 | Xử lý được trên laptop nhóm trong tối đa 1 ngày | Dung lượng, thời gian đọc thử |

### 5.2 Thang điểm (cho dataset đã qua G1–G6), tổng 100

| Tiêu chí | Điểm | Cách cho điểm |
|---|---|---|
| Khả năng dựng đồ thị | 30 | IP rõ, >= ~50 IP khác nhau, mỗi cửa sổ có nhiều node tấn công và nạn nhân: 25–30. Chỉ 2–5 IP: 5–10 |
| Độ phủ tấn công | 20 | Đủ 3 nhóm, mỗi nhóm >= 1.000 mẫu: 16–20; mỗi nhóm ít hơn: 8–15 |
| Tính IoT thực tế | 15 | Thiết bị IoT thật: 12–15. Mô phỏng/hỗn hợp: 6–11. Không phải IoT: 0–5 |
| Cân bằng lớp | 10 | Lớp bình thường >= 10% và lớp nhỏ nhất >= 0,5%: 8–10. Mất cân bằng nặng hơn: 2–7 |
| Chi phí xử lý | 10 | < 5 GB và có CSV sẵn: 8–10. Phải xử lý PCAP: 2–5 |
| Khả năng so sánh | 10 | >= 3 bài báo dùng cho GNN/IDS: 8–10. Rất ít bài: 2–5 |
| Độ khó hợp lý | 5 | Các bài báo đạt 85–97%: 4–5. Model đơn giản đã đạt 99,9% (quá dễ): 1–2 |

### 5.3 Quy tắc chọn
- Dataset chính: điểm cao nhất và >= 70/100 (hạ xuống 60 nếu không bộ nào đạt, ghi rõ hạn chế).
- Dataset phụ: điểm cao thứ hai, chỉ dùng kiểm tra tổng quát hóa nếu còn thời gian. Không bắt buộc.
- Nếu chênh nhau < 5 điểm: chọn bộ chi phí xử lý thấp hơn.
- Phong và Tấn chỉ nộp phiếu; Phúc quyết định ở buổi họp ngày 3.

### 5.4 Phân chia khảo sát

| Người | Dataset |
|---|---|
| Phong | ToN-IoT (và bản NF-ToN-IoT-v2), Bot-IoT (và bản NF-BoT-IoT-v2) |
| Tấn | CIC-IoT-2023, Edge-IIoTset |

Thêm bộ khác (ví dụ UNSW-NB15) nếu thấy đáng xem. Lưu ý các đặc điểm trên là hiểu biết ban đầu, **phải tự kiểm tra lại trên dữ liệu thật và bài báo gốc**.

## 6. Mốc thời gian tổng

| Tuần | Trọng tâm | Điều kiện sang tuần sau (đầu ra bắt buộc) |
|---|---|---|
| **Tuần 1** (19–25/10) | Dataset | 4 phiếu dataset + biên bản chọn; bộ đồ thị `.pt` + `split.json` + `build_graph.py`; EDA; đặc trưng + `scaler.pkl`; GAT chạy được trên đồ thị thật |
| **Tuần 2** (26/10–1/11) | Docker và huấn luyện | Container chạy pipeline thời gian thực; kết quả GAT + 4 baseline cùng bảng; mô hình đã làm nhẹ (ONNX) + `benchmark.py`; một số đo độ trễ sơ bộ |
| **Tuần 3** (2–8/11) | Điều chỉnh và hoàn thiện | Đạt/giải trình 91% Acc và 4,08% FPR; ablation; hình attention; đo hiệu năng cuối; demo; báo cáo hoàn chỉnh; slide |

Nội dung báo cáo chốt vào **ngày 5 của tuần 3**; ngày 6–7 chỉ sửa lỗi, đọc chéo và làm slide.

## 7. Phân bổ chương báo cáo

| Chương | Nội dung | Phụ trách |
|---|---|---|
| 1 | Giới thiệu | Phong |
| 2 | Cơ sở lý thuyết (IDS, GNN, GAT, kỹ thuật làm nhẹ) | Phong (GNN/GAT/IDS), Tấn (kỹ thuật làm nhẹ) |
| 3 | Khảo sát liên quan | Phúc |
| 4 | Dataset (phân tích và lựa chọn) | Phong, Tấn |
| 5 | Phương pháp (đồ thị, mô hình, huấn luyện) | Phong (dựng đồ thị), Phúc (mô hình) |
| 6 | Triển khai thời gian thực (Docker, làm nhẹ, pipeline) | Phúc (Docker/pipeline), Tấn (làm nhẹ) |
| 7 | Thực nghiệm và đánh giá | Phúc (độ chính xác), Tấn (hiệu năng) |
| 8 | Thảo luận và kết luận | Tấn, Phúc duyệt |

## 8. Họp và theo dõi

- [ ] Họp 15 phút vào **ngày 3 và ngày 6** hằng tuần (thêm buổi chốt dataset ngày 3 tuần 1: 30 phút).
- Mỗi người nói 3 điều: đã xong gì (đầu ra ở đâu trong repo), sắp làm gì, đang kẹt gì.
- Tích checkbox trong file md của mình khi xong việc, đẩy lên repo.

## 9. Rủi ro và phương án dự phòng

| Rủi ro | Phương án |
|---|---|
| Không bộ nào đạt 70 điểm | Hạ ngưỡng xuống 60, chọn bộ ít rủi ro nhất, ghi hạn chế |
| Chậm đồ thị đến cuối tuần 1 | Dùng bản NetFlow (NF-ToN-IoT-v2): có sẵn IP và dạng bảng |
| Không đạt 91% / 4,08% | Chỉnh ngưỡng và cửa sổ trước; nếu vẫn không, báo cáo kết quả thực kèm phân tích nguyên nhân. **Không** đổi cách chia dữ liệu cho dễ |
| Không có thiết bị thật | Dùng Docker giả lập, ghi rõ là mô phỏng |
| Tuần 3 chật | Viết sớm các phần không phụ thuộc kết quả (lý thuyết, dataset, phương pháp) từ tuần 2 |
| Dùng random split làm accuracy ảo | Bắt buộc chia theo thời gian, kiểm tra không có cửa sổ chồng lấp giữa các tập |

---

## Phụ lục A: Mẫu phiếu đánh giá dataset (lưu ở `dataset_cards/<ten_dataset>.md`)

```
# Phiếu dataset: <tên>
Người khảo sát: ...   Ngày: ...
Link tải: ...   License: ...   Bài báo gốc: ...
Dung lượng: ...   Số mẫu: ...   Thời gian đọc thử: ...

## Điều kiện bắt buộc
| # | Đạt? | Bằng chứng (ảnh/dòng mẫu) |
| G1 | | |
| G2 | | |
| G3 | | |
| G4 | | |
| G5 | | |
| G6 | | |

## Ánh xạ nhãn
| Nhãn gốc | Số mẫu | Nhóm (DDoS/Botnet/Truy cập trái phép/Bình thường/Khác) |

## Thống kê
Số IP nguồn khác nhau: ...   Số IP đích khác nhau: ...
Tỉ lệ bình thường: ...%   Lớp nhỏ nhất: ... (...%)

## Chấm điểm
| Tiêu chí | Điểm | Lý do (1 dòng) |
| Khả năng dựng đồ thị (30) | | |
| Độ phủ tấn công (20) | | |
| Tính IoT thực tế (15) | | |
| Cân bằng lớp (10) | | |
| Chi phí xử lý (10) | | |
| Khả năng so sánh (10) | | |
| Độ khó hợp lý (5) | | |
| **Tổng (100)** | | |

## Kết luận
Nên chọn? (Có/Không/Làm dataset phụ) — Lý do: ...
Rủi ro: ...
```
