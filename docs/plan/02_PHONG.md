# PHONG — Dataset, đồ thị, baseline

**Vai trò:** khảo sát và đánh giá dataset (ToN-IoT, Bot-IoT), dựng đồ thị từ dữ liệu, chạy baseline, viết các chương lý thuyết và dataset.
**Người phối hợp:** Tấn (cùng làm dataset), Phúc (nhóm trưởng, mô hình, Docker).
Tiêu chí chọn dataset và mẫu phiếu: `00_TONG_QUAN.md` mục 5 và Phụ lục A.

---

## TUẦN 1 — Dataset (19–25/10)

### Ngày 1: Chuẩn bị và tải dữ liệu
- [ ] Cài đặt môi trường (Python, PyTorch, PyG, pandas, pyarrow, jupyter), clone repo
- [ ] Tải **ToN-IoT** (ưu tiên bảng mạng/network, kèm bản **NF-ToN-IoT-v2** nếu có)
- [ ] Tải **Bot-IoT** (kèm **NF-BoT-IoT-v2** nếu có)
- [ ] Đặt dữ liệu ở `data/raw/` (không commit); ghi link tải vào `data/README.md`
- [ ] Kiểm tra G1–G3 (IP, timestamp, nhãn) cho từng bộ, chụp 5 dòng đầu làm bằng chứng

### Ngày 2: Khảo sát sâu
- [ ] Kiểm tra G4 (ánh xạ nhãn gốc sang DDoS / botnet / truy cập trái phép)
- [ ] Kiểm tra G5 (license) và G6 (dung lượng, thời gian đọc thử)
- [ ] Tính: số IP nguồn/đích khác nhau, số mẫu từng lớp, tỉ lệ lớp bình thường
- [ ] Tìm số bài báo GNN/IDS dùng mỗi bộ (cho tiêu chí "khả năng so sánh") và mức chính xác họ đạt (cho tiêu chí "độ khó")
- [ ] Chấm điểm 7 tiêu chí

### Ngày 2 (hạn nộp): Nộp phiếu
- [ ] Viết `dataset_cards/ton_iot.md` (và bản NF-ToN-IoT-v2 nếu khảo sát riêng)
- [ ] Viết `dataset_cards/bot_iot.md` (và bản NF-BoT-IoT-v2 nếu khảo sát riêng)
- [ ] Đẩy lên repo, báo Phúc

### Ngày 3: Họp chốt dataset
- [ ] Tham gia buổi họp 30 phút, trình bày kết quả 2 bộ
- [ ] Ghi nhận quyết định: dataset chính, dataset phụ, **ai phụ trách** (người khảo sát bộ được chọn là người phụ trách chính, người còn lại hỗ trợ)

### Ngày 4–5: Xử lý dataset chính
*(Nếu dataset chính là của Tấn: Phong chuyển sang vai hỗ trợ, đổi việc A3–A4 bên dưới với Tấn nhưng đầu ra vẫn thuộc trách nhiệm Phong.)*
- [ ] **Làm sạch:** bỏ giá trị thiếu/vô cực, loại bản ghi trùng lặp, thống nhất kiểu dữ liệu, xuất `data/processed/clean.parquet`
- [ ] **Chọn đặc trưng:** 15–20 đặc trưng (số flow vào/ra, tổng byte và gói, số đích và cổng khác nhau, tỉ lệ cờ SYN/ACK/RST, thời gian giữa gói, tỉ lệ giao thức). Ghi lý do chọn vào `results/phong_features.md`
- [ ] **Chuẩn hóa:** fit scaler chỉ trên tập train, lưu `data/processed/scaler.pkl`
- [ ] **Viết `src/build_graph.py`:**
  - [ ] Chia cửa sổ trượt (tham số: độ dài 1–5 giây, bước trượt)
  - [ ] Node = IP, edge = có giao tiếp trong cửa sổ, đặc trưng node = đặc trưng đã gộp
  - [ ] Nhãn node = lớp tấn công chiếm đa số trong cửa sổ (quy tắc ghi rõ trong code)
  - [ ] Lưu từng đồ thị dạng PyG `.pt`
- [ ] Chia train/val/test **theo thời gian**, không chồng cửa sổ, lưu `data/processed/split.json`

### Ngày 6–7: Hoàn thiện và bàn giao
- [ ] Chạy `build_graph.py` trên toàn bộ dữ liệu, xuất bộ đồ thị đầy đủ
- [ ] Thống kê: số đồ thị, số node và edge trung bình, phân bố nhãn từng tập (ghi vào `results/phong_graph_stats.csv`)
- [ ] Kiểm tra rò rỉ: không có IP/cửa sổ/timestamp trùng giữa các tập chia (hoặc ghi rõ nếu IP trùng là bình thường)
- [ ] Bàn giao cho Phúc: bộ đồ thị, `split.json`, `scaler.pkl`, `build_graph.py`
- [ ] Viết bản nháp chương 4 (phần dataset) và phần dựng đồ thị của chương 5
- [ ] Họp ngày 6

### Điều kiện sang tuần 2
- [ ] 2 phiếu dataset đã nộp
- [ ] Bộ đồ thị + `split.json` + `scaler.pkl` + `build_graph.py` đã bàn giao
- [ ] `build_graph.py` chạy lại được từ đầu và cho cùng kết quả (seed cố định)

---

## TUẦN 2 — Baseline và module thời gian thực (26/10–1/11)

### Baseline
- [ ] Viết `src/baselines.py` dùng **cùng bộ đồ thị và `split.json`** với GAT
- [ ] **Random Forest** (trên đặc trưng node, không dùng cấu trúc đồ thị)
- [ ] **MLP** (cùng đầu vào)
- [ ] **GCN** (2 lớp, PyG)
- [ ] **GraphSAGE** (2 lớp, PyG)
- [ ] Cố định seed, chạy 3 seed, báo trung bình và độ lệch chuẩn
- [ ] Đo Acc, Precision, Recall, F1 macro, FPR, ROC-AUC, tham số và thời gian suy luận
- [ ] Xuất bảng `results/phong_baselines.csv`

### Module trích đặc trưng theo luồng
- [ ] Viết `src/stream_features.py`: nhận luồng flow theo thời gian, gom cửa sổ, tạo đặc trưng node và edge **giống hệt** `build_graph.py` (dùng lại hàm chung, không viết lại logic)
- [ ] Kiểm tra: chạy offline và chạy streaming trên cùng dữ liệu phải ra cùng đồ thị
- [ ] Viết ví dụ sử dụng ngắn trong docstring/README
- [ ] Bàn giao cho Phúc dùng trong `realtime_pipeline.py`

### Báo cáo (viết sớm các phần không phụ thuộc kết quả)
- [ ] Chương 2: IDS, GNN, GAT (multi-head attention, công thức attention)
- [ ] Chương 4: hoàn thiện (bảng so sánh dataset, lập luận chọn, thống kê dataset chính)

### Điều phối
- [ ] Hỗ trợ Phúc khi Docker cần chạy baseline hoặc build đồ thị trong container
- [ ] Họp ngày 3 và 6

### Điều kiện sang tuần 3
- [ ] Bảng baseline hoàn chỉnh
- [ ] `stream_features.py` chạy được trong container của Phúc

---

## TUẦN 3 — Hoàn thiện (2–8/11)

### Kết quả
- [ ] Chạy lại baseline nếu Phúc đổi cửa sổ/đặc trưng (cùng seed, cùng split)
- [ ] Hoàn thiện bảng so sánh: baseline và GAT (phối hợp Phúc)
- [ ] Phân tích lỗi: lớp tấn công nào bị nhầm nhiều nhất với bình thường (đầu vào cho phần thảo luận)

### Báo cáo
- [ ] Chương 1: Giới thiệu (bối cảnh, bài toán, mục tiêu, đóng góp, bố cục)
- [ ] Chương 2: hoàn thiện (phần IDS, GNN, GAT)
- [ ] Chương 4: hoàn thiện
- [ ] Chương 5 phần dựng đồ thị: cửa sổ, node/edge, đặc trưng, nhãn, chia dữ liệu
- [ ] **Ngày 5:** nộp bản nháp các chương cho Phúc
- [ ] Ngày 6–7: sửa theo góp ý, đọc chéo chương của Phúc và Tấn
- [ ] Làm phần slide về dữ liệu và dựng đồ thị

### Hoàn tất
- [ ] Kiểm tra `build_graph.py` và `baselines.py` chạy được từ repo sạch trong container

---

## Đầu ra cuối cùng của Phong
- [ ] `dataset_cards/ton_iot.md`, `dataset_cards/bot_iot.md`
- [ ] `src/build_graph.py`, `src/stream_features.py`, `src/baselines.py`
- [ ] Bộ đồ thị, `split.json`, `scaler.pkl`, `results/phong_graph_stats.csv`
- [ ] `results/phong_baselines.csv`
- [ ] Chương 1, 2, 4 và phần dựng đồ thị của chương 5
- [ ] Slide phần dữ liệu
