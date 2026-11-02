# PHÚC — Nhóm trưởng, Mô hình GAT, Docker

**Vai trò:** quyết định dataset, xây dựng và huấn luyện Lightweight GAT, dựng Docker và pipeline thời gian thực, điều chỉnh để đạt chỉ tiêu, duyệt và gộp báo cáo.
**Người phối hợp:** Phong (dữ liệu, baseline), Tấn (làm nhẹ, đo hiệu năng).
Xem định nghĩa chung (độ trễ, chia dữ liệu, seed) ở `00_TONG_QUAN.md` mục 3.

---

## TUẦN 1 — Dataset (19–25/10)

### Ngày 1: Thiết lập
- [ ] Cài Docker Desktop (WSL2), kiểm tra `docker run hello-world`
- [ ] Tạo repo GitHub theo cấu trúc ở `00_TONG_QUAN.md` mục 4.2, thêm `.gitignore`, mời Phong và Tấn
- [ ] Chốt phiên bản thư viện, ghi vào `requirements.txt`
- [ ] Gửi mẫu phiếu dataset (Phụ lục A) cho Phong và Tấn, nhắc hạn nộp: hết ngày 2
- [ ] Đặt lịch họp cố định (ngày 3 và ngày 6 mỗi tuần)
- [ ] Đọc bài GAT gốc (Veličković và cộng sự, 2018)

### Ngày 2–3: Tổng quan tài liệu
- [ ] Đọc 6–8 bài GNN-IDS cho IoT (ví dụ E-GraphSAGE, Anomal-E và các bài khác nhóm tìm được)
- [ ] Lập bảng tóm tắt: bài, dataset, phương pháp, kết quả, hạn chế (lưu `report/03_khao_sat.md`)
- [ ] Bắt đầu danh mục tài liệu tham khảo (`report/references.md`)
- [ ] **Ngày 3:** họp 30 phút chốt dataset. Đọc 4 phiếu, kiểm tra bảng điểm theo tiêu chí mục 5 của tổng quan
- [ ] Ghi biên bản chọn dataset (`dataset_cards/QUYET_DINH.md`): dataset chính, dataset phụ, người phụ trách

### Ngày 4–5: Khung mô hình
- [ ] Viết `src/model_gat.py`: 2 lớp GATConv, lớp 1 gồm 4 head x 16–32 chiều (concat), lớp 2 gồm 1–2 head, lớp phân loại, dropout, ELU
- [ ] Viết `src/train.py`: nạp đồ thị, huấn luyện, early stopping theo validation, lưu checkpoint tốt nhất, ghi log
- [ ] Hỗ trợ trọng số lớp / focal loss cho mất cân bằng
- [ ] Thử trên đồ thị giả (random) để chắc code chạy được
- [ ] Đếm tham số, ghi vào log (mục tiêu < 50K)

### Ngày 6–7: Chạy trên dữ liệu thật
- [ ] Nhận đồ thị `.pt`, `split.json` từ Phong; kiểm tra nhanh (số node, edge, nhãn, không có cửa sổ chồng giữa train/val/test)
- [ ] Chạy GAT bản đầu trên đồ thị thật, ghi Acc, FPR, F1 ban đầu vào `results/phuc_gat_v0.csv`
- [ ] Họp ngày 6: kiểm tra điều kiện sang tuần 2 (checklist cuối mục này)

### Điều kiện sang tuần 2
- [ ] Có biên bản chọn dataset
- [ ] Có bộ đồ thị, `split.json`, `build_graph.py`
- [ ] GAT chạy được trên đồ thị thật và có số liệu bản đầu

---

## TUẦN 2 — Docker và huấn luyện (26/10–1/11)

### Docker (ưu tiên đầu tuần)
- [ ] Viết `docker/Dockerfile` (Python, PyTorch, PyG, ONNX Runtime; image gọn)
- [ ] Viết `docker/docker-compose.yml` với 2 profile: `train` (không giới hạn) và `edge` (giới hạn CPU/RAM)
- [ ] Cấu hình giả lập thiết bị yếu cho profile `edge`: `--cpus` và `--memory` tương đương Raspberry Pi 4 (ví dụ 4 nhân, 2–4 GB RAM). Ghi rõ: đây là **mô phỏng**
- [ ] Kiểm tra: Phong và Tấn chạy được container bằng 1 lệnh trên máy của họ
- [ ] Viết `docker/README.md`: cách build, chạy train, chạy edge

### Huấn luyện GAT bản chính
- [ ] Tinh chỉnh siêu tham số cơ bản: số head, chiều ẩn, dropout, learning rate (ghi bảng thử nghiệm `results/phuc_hparam.csv`)
- [ ] Chạy với ít nhất 3 seed, báo trung bình và độ lệch chuẩn
- [ ] Lưu `best_gat.pt` và giao cho Tấn (thông báo định dạng input/output để Tấn xuất ONNX)
- [ ] Đánh giá đầy đủ: Acc, Precision, Recall, F1 macro, FPR, ma trận nhầm lẫn, ROC-AUC

### Pipeline thời gian thực
- [ ] Nhận `stream_features.py` từ Phong
- [ ] Viết `src/realtime_pipeline.py`: đọc dữ liệu replay theo timestamp, gom cửa sổ, trích đặc trưng, dựng đồ thị, suy luận, in cảnh báo
- [ ] Đóng gói chạy trong container profile `edge`
- [ ] Thêm log thời gian từng bước (trích đặc trưng, dựng đồ thị, suy luận) để Tấn đo
- [ ] Có thể dùng mô hình tạm trong lúc chờ `best_gat.pt`

### Điều phối
- [ ] Thống nhất với Tấn phương pháp đo (định nghĩa độ trễ, warm-up, số lần đo)
- [ ] Duyệt pull request của Phong (baseline) và Tấn (ONNX, benchmark)
- [ ] Họp ngày 3 và 6

### Điều kiện sang tuần 3
- [ ] Container chạy pipeline thời gian thực
- [ ] Bảng kết quả GAT và 4 baseline trên cùng cách chia
- [ ] Có số đo độ trễ sơ bộ trong container

---

## TUẦN 3 — Điều chỉnh và hoàn thiện (2–8/11)

### Điều chỉnh để đạt chỉ tiêu (ưu tiên đầu tuần)
- [ ] So sánh kết quả với mục tiêu: Acc >= 91%, FPR <= 4,08%
- [ ] Nếu chưa đạt, thử lần lượt: ngưỡng quyết định, kích thước cửa sổ, xử lý mất cân bằng, đặc trưng. **Không** đổi cách chia dữ liệu để kết quả đẹp hơn
- [ ] Nếu vẫn không đạt: ghi kết quả thực và phân tích nguyên nhân cho báo cáo
- [ ] Kiểm tra lại mô hình đã làm nhẹ (ONNX) vẫn giữ Acc / FPR

### Ablation và giải thích
- [ ] Ablation: số head (1, 2, 4, 8), số lớp (1, 2, 3), kích thước cửa sổ (1s, 2s, 5s), có/không lượng tử hóa (phối hợp Tấn)
- [ ] Trực quan hóa trọng số attention của một cuộc tấn công DDoS (hình `results/attention_ddos.png`)
- [ ] Bảng so sánh cuối với baseline

### Báo cáo
- [ ] Viết chương 3 (khảo sát liên quan), chương 5 phần mô hình, chương 6 phần Docker/pipeline, chương 7 phần độ chính xác
- [ ] **Ngày 5:** nhận bản nháp của Phong và Tấn, đọc và ghi nhận xét
- [ ] Gộp báo cáo, thống nhất văn phong, kiểm tra trích dẫn, đánh số hình/bảng
- [ ] Ngày 6–7: sửa lỗi, đọc chéo, làm slide cùng nhóm

### Hoàn tất
- [ ] Kiểm tra repo chạy được từ đầu (clone, docker, tái lập kết quả chính)
- [ ] Xem demo của Tấn, góp ý
- [ ] Nộp báo cáo và slide

---

## Đầu ra cuối cùng của Phúc
- [ ] `src/model_gat.py`, `src/train.py`, `src/realtime_pipeline.py`
- [ ] `docker/` đầy đủ và README
- [ ] `best_gat.pt`, bảng siêu tham số, bảng ablation, hình attention
- [ ] Biên bản chọn dataset
- [ ] Chương 3, phần mô hình của chương 5, phần Docker của chương 6, phần độ chính xác của chương 7
- [ ] Báo cáo tổng hợp hoàn chỉnh
