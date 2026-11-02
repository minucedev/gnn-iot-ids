# TẤN — Dataset, làm nhẹ mô hình, đo hiệu năng, demo

**Vai trò:** khảo sát và đánh giá dataset (CIC-IoT-2023, Edge-IIoTset), hỗ trợ Phong xử lý dataset, sau đó làm nhẹ mô hình, đo hiệu năng và làm demo.
**Người phối hợp:** Phong (cùng làm dataset), Phúc (nhóm trưởng, mô hình, Docker).
Tiêu chí chọn dataset và mẫu phiếu: `00_TONG_QUAN.md` mục 5 và Phụ lục A.

---

## TUẦN 1 — Dataset (19–25/10)

### Ngày 1: Chuẩn bị và tải dữ liệu
- [ ] Cài đặt môi trường (Python, PyTorch, PyG, pandas, pyarrow, jupyter), clone repo
- [ ] Tải **CIC-IoT-2023** (bắt đầu với bản CSV đã xử lý; kiểm tra có IP hay không; ghi nhận PCAP lớn đến mức nào)
- [ ] Tải **Edge-IIoTset** (bản CSV)
- [ ] Đặt dữ liệu ở `data/raw/` (không commit); ghi link tải vào `data/README.md`
- [ ] Kiểm tra G1–G3 (IP, timestamp, nhãn) cho từng bộ, chụp 5 dòng đầu làm bằng chứng

### Ngày 2: Khảo sát sâu
- [ ] Kiểm tra G4 (ánh xạ nhãn gốc sang DDoS / botnet / truy cập trái phép)
- [ ] Kiểm tra G5 (license) và G6 (dung lượng, thời gian đọc thử). Nếu bộ nào chỉ có IP trong PCAP, ước lượng thời gian trích xuất rồi ghi vào tiêu chí "chi phí xử lý"
- [ ] Tính: số IP nguồn/đích khác nhau, số mẫu từng lớp, tỉ lệ lớp bình thường
- [ ] Tìm số bài báo GNN/IDS dùng mỗi bộ và mức chính xác họ đạt
- [ ] Chấm điểm 7 tiêu chí
- [ ] Nộp phiếu: `dataset_cards/cic_iot_2023.md`, `dataset_cards/edge_iiotset.md` (hạn: hết ngày 2)

### Ngày 3: Họp chốt dataset
- [ ] Tham gia buổi họp 30 phút, trình bày kết quả 2 bộ
- [ ] Ghi nhận quyết định: dataset chính, dataset phụ, ai phụ trách

### Ngày 4–5: Hỗ trợ xử lý dataset chính
*(Người phụ trách chính là người khảo sát dataset được chọn. Nếu dataset chính của Phong: Tấn hỗ trợ như bên dưới. Nếu của Tấn: Tấn đổi vai với Phong, xem `02_PHONG.md` ngày 4–5, phần việc đồ thị.)*
- [ ] **EDA:** `notebooks/01_eda.ipynb`: phân bố lớp, giá trị thiếu, phân bố từng đặc trưng chính, biểu đồ thời gian theo lớp, số IP theo thời gian
- [ ] **Xử lý mất cân bằng lớp:** thử class weight và/hoặc focal loss và/hoặc lấy mẫu có kiểm soát (chỉ áp dụng trên tập train). Viết hàm trong `src/imbalance.py`, ghi bảng so sánh có/không xử lý (`results/tan_imbalance.csv`)
- [ ] **Hàm chia train/val/test theo thời gian:** viết kiểm tra rò rỉ (`src/check_split.py`): không chồng cửa sổ, không trùng timestamp giữa các tập

### Ngày 6–7: Kiểm thử và viết
- [ ] Chạy `check_split.py` trên bộ đồ thị Phong xuất, báo kết quả cho Phong và Phúc
- [ ] Kiểm thử `build_graph.py`: kiểm tra ngẫu nhiên 20 đồ thị (đối chiếu với dữ liệu gốc: node, edge, nhãn có đúng)
- [ ] Viết bản nháp phần tiền xử lý và mất cân bằng cho chương 4
- [ ] Đọc trước tài liệu về làm nhẹ mô hình (xem tuần 2)
- [ ] Họp ngày 6

### Điều kiện sang tuần 2
- [ ] 2 phiếu dataset đã nộp
- [ ] `01_eda.ipynb`, `imbalance.py`, `check_split.py` có trong repo
- [ ] Báo cáo kiểm thử bộ đồ thị (đúng/sai, lỗi nếu có)

---

## TUẦN 2 — Làm nhẹ mô hình và đo hiệu năng (26/10–1/11)

### Khảo sát kỹ thuật làm nhẹ
- [ ] Đọc và tóm tắt: lượng tử hóa (int8, FP16), pruning, giảm head/chiều ẩn, neighbor sampling, xuất ONNX
- [ ] Bảng so sánh: kỹ thuật, lợi ích, rủi ro mất độ chính xác, độ khó áp dụng (`report/02_lam_nhe.md`)
- [ ] Chọn 2–3 kỹ thuật áp dụng (đề xuất: FP16/int8 + pruning nhẹ + ONNX)

### Làm nhẹ mô hình
- [ ] Nhận `best_gat.pt` và `model_gat.py` từ Phúc (tuần 2, giữa tuần; trước đó dùng mô hình tạm để chuẩn bị mã)
- [ ] Xuất ONNX bản FP32 (`gat.onnx`); kiểm tra kết quả ONNX khớp PyTorch (sai số nhỏ)
- [ ] Lượng tử hóa int8 hoặc FP16 → `gat_quant.onnx`
- [ ] Thử pruning (nếu phù hợp), ghi lại ảnh hưởng đến độ chính xác
- [ ] Bảng trước và sau: kích thước file, số tham số, Acc, FPR, F1 (`results/tan_quant.csv`)
- [ ] Nếu GAT khó xuất ONNX vì thao tác thưa/attention: ghi vấn đề, báo Phúc sớm để chuyển sang cách cài đặt tương thích (ma trận kề thưa viết bằng PyTorch thuần)

### Script đo hiệu năng
- [ ] Thống nhất với Phúc phương pháp đo (độ trễ tính từ lúc cửa sổ đóng; tách trích đặc trưng, dựng đồ thị, suy luận)
- [ ] Viết `src/benchmark.py`: warm-up 20 lần, đo >= 200 lần, ghi trung bình, p95, RAM đỉnh, CPU
- [ ] Chạy trong container profile `edge` của Phúc
- [ ] Ghi số đo sơ bộ vào `results/tan_latency_v0.csv`

### Báo cáo (viết sớm)
- [ ] Chương 2 phần kỹ thuật làm nhẹ mô hình
- [ ] Hoàn thiện nháp phần tiền xử lý / mất cân bằng của chương 4

### Điều phối
- [ ] Hỗ trợ Phong nếu cần xử lý dataset phụ
- [ ] Họp ngày 3 và 6

### Điều kiện sang tuần 3
- [ ] `gat_quant.onnx` + bảng trước/sau
- [ ] `benchmark.py` chạy được trong container
- [ ] Một số đo độ trễ sơ bộ

---

## TUẦN 3 — Đo cuối, demo, báo cáo (2–8/11)

### Đo hiệu năng cuối
- [ ] Nếu Phúc đổi mô hình hoặc cửa sổ: xuất lại ONNX, lượng tử hóa lại, kiểm tra Acc / FPR
- [ ] Đo lần cuối trong container, cấu hình cố định (ghi `--cpus`, `--memory`, phiên bản)
- [ ] Bảng cuối `results/tan_latency_final.csv`: độ trễ trung bình, p95 (tách từng bước), RAM, CPU, kích thước mô hình
- [ ] So sánh với mục tiêu < 1 giây; nếu không đạt, phân tích bước nào chậm nhất và thử tối ưu (giảm cửa sổ, giảm số node tối đa, cache)
- [ ] Ablation lượng tử hóa: có/không (phối hợp Phúc)

### Demo
- [ ] Chọn hình thức: video 2–3 phút hoặc dashboard (ví dụ terminal hiển thị cảnh báo hoặc trang web đơn giản)
- [ ] Demo kịch bản: dòng lưu lượng bình thường → bắt đầu DDoS → hệ thống cảnh báo trong dưới 1 giây → hiển thị node tấn công
- [ ] Chạy demo trong container profile `edge`
- [ ] Quay video, kiểm tra hình và tiếng; lưu ở `report/demo/`

### Báo cáo
- [ ] Chương 6 phần làm nhẹ và đo hiệu năng
- [ ] Chương 7 phần hiệu năng (bảng độ trễ, RAM, so sánh)
- [ ] Chương 8: thảo luận (hạn chế, tấn công thích nghi, drift, hướng phát triển) và kết luận
- [ ] **Ngày 5:** nộp bản nháp cho Phúc
- [ ] Ngày 6–7: sửa theo góp ý, đọc chéo
- [ ] Slide phần triển khai, hiệu năng, demo

### Hoàn tất
- [ ] Kiểm tra `benchmark.py` và demo chạy được từ repo sạch

---

## Đầu ra cuối cùng của Tấn
- [ ] `dataset_cards/cic_iot_2023.md`, `dataset_cards/edge_iiotset.md`
- [ ] `notebooks/01_eda.ipynb`, `src/imbalance.py`, `src/check_split.py`
- [ ] `gat.onnx`, `gat_quant.onnx`, `results/tan_quant.csv`
- [ ] `src/benchmark.py`, `results/tan_latency_final.csv`
- [ ] Demo (video hoặc dashboard)
- [ ] Chương 2 (phần làm nhẹ), 6 (phần làm nhẹ), 7 (phần hiệu năng), 8; phần tiền xử lý của chương 4
- [ ] Slide phần triển khai và demo
