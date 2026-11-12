# Danh sách task

Kế hoạch tổng: [KE_HOACH.md](KE_HOACH.md). Trạng thái: `Chưa làm` → `Đang làm` → `Chờ review` → (`Cần sửa` →) `Xong`.
Phúc làm toàn bộ task. Một task chỉ được chuyển sang `Xong` khi đã tự kiểm **từng** tiêu chí kiểm tra và ghi bằng chứng (đường dẫn file, con số, hash) vào dòng **Kết quả** ngay dưới task. Khi đổi trạng thái thì sửa cả bảng tổng hợp và dòng "Trạng thái" trong mục chi tiết.

Ưu tiên: **B** = Bắt buộc, **N** = Nên có.

## Bảng tổng hợp

| Mã | Task | Giai đoạn | Phụ trách | Kiểm tra | Tuần | Ưu tiên | Trạng thái |
|---|---|---|---|---|---|---|---|
| [0.1](#01-chốt-phát-biểu-bài-toán-và-chỉ-tiêu-nghiệm-thu) | Chốt phát biểu bài toán và chỉ tiêu | GĐ0 | Phúc | Tự kiểm | 2 | B | **Xong** |
| [0.2](#02-chốt-7-quyết-định-mở-d1d7) | Chốt quyết định D1–D7 | GĐ0 | Phúc | Tự kiểm | 2 | B | **Xong** |
| [0.3](#03-repo-github-cấu-trúc-thư-mục-requirements) | Repo GitHub, cấu trúc, requirements | GĐ0 | Phúc | Tự kiểm | 2 | B | **Xong** |
| [0.4](#04-tách-code-notebook-thành-srcbuild_graphpy) | Tách `src/build_graph.py` | GĐ0 | Phúc | Tự kiểm | 2 | B | **Xong** |
| [0.5](#05-module-đánh-giá-chung-srcevaluatepy) | Module `src/evaluate.py` | GĐ0 | Phúc | Tự kiểm | 2 | B | **Xong** |
| [0.6](#06-chuyển-pipeline-lên-kaggle) | Chuyển pipeline lên Kaggle | GĐ0 | Phúc | Tự kiểm | 2 | B | Chưa làm |
| [1.0](#10-tuần-1-pipeline-v1) | [Tuần 1] Pipeline v1 | GĐ1 | Phúc | Tự kiểm | 1 | B | **Cần sửa** |
| [1.1](#11-tuần-1-khảo-sát-4-bộ-dữ-liệu) | [Tuần 1] Khảo sát 4 bộ dữ liệu | GĐ1 | Phúc | Tự kiểm | 1 | B | **Xong** |
| [1.2](#12-xác-minh-và-tải-netflow-v3) | Xác minh và tải NetFlow v3 | GĐ1 | Phúc | Tự kiểm | 2 | B | **Xong** |
| [1.3](#13-tải-ton-iot-có-ts-và-kiểm-chứng-giả-định-thứ-tự-dòng) | ToN-IoT có `ts`, kiểm chứng thứ tự dòng | GĐ1 | Phúc | Tự kiểm | 2 | B | **Bỏ** |
| [1.4](#14-chốt-danh-sách--3-bộ-dữ-liệu) | Chốt bộ dữ liệu (≥ 2 từ 12/11) | GĐ1 | Phúc | Tự kiểm | 3 | B | Chưa làm |
| [1.5](#15-sửa-chiến-lược-chia-tập) | Sửa chiến lược chia tập | GĐ1 | Phúc | Tự kiểm | 3 | B | Chưa làm |
| [1.6](#16-dựng-lại-đồ-thị-cho-mọi-bộ-đã-chốt) | Dựng lại đồ thị mọi bộ | GĐ1 | Phúc | Tự kiểm | 3 | B | Chưa làm |
| [1.7](#17-từ-điển-dữ-liệu) | Từ điển dữ liệu | GĐ1 | Phúc | Tự kiểm | 3 | B | Chưa làm |
| [1.8](#18-dựng-đồ-thị-nf-bot-iot-v3-bộ-thứ-2) | Dựng đồ thị NF-BoT-IoT-v3 (bộ thứ 2) | GĐ1 | Phúc | Tự kiểm | 12–13/11 | B | **Xong** |
| [2.1](#21-eda-mức-flow) | EDA mức flow | GĐ2 | Phúc | Tự kiểm | 3 | B | Chưa làm |
| [2.2](#22-eda-mức-đồ-thị) | EDA mức đồ thị, homophily | GĐ2 | Phúc | Tự kiểm | 3 | B | Chưa làm |
| [2.3](#23-trực-quan-hóa-đồ-thị-các-cửa-sổ-mẫu) | Trực quan hóa đồ thị | GĐ2 | Phúc | Tự kiểm | 3 | B | Chưa làm |
| [3.1](#31-lý-thuyết-ids-cho-iot-các-loại-tấn-công-netflow) | Lý thuyết IDS, tấn công, NetFlow | GĐ3 | Phúc | Tự kiểm | 3 | B | Chưa làm |
| [3.2](#32-lý-thuyết-gnn-tại-sao-chọn-và-còn-mô-hình-nào-khác) | Lý thuyết GNN | GĐ3 | Phúc | Tự kiểm | 3 | B | Chưa làm |
| [3.3](#33-lý-thuyết-baseline-ml-chỉ-số-đánh-giá-và-ahp) | Lý thuyết baseline, chỉ số, AHP | GĐ3 | Phúc | Tự kiểm | 4 | B | Chưa làm |
| [3.4](#34-bảng-tổng-quan-nghiên-cứu-liên-quan) | Tổng quan nghiên cứu liên quan | GĐ3 | Phúc | Tự kiểm | 4 | B | Chưa làm |
| [4.1](#41-baseline-ml-trên-đặc-trưng-node) | Baseline ML trên đặc trưng node | GĐ4 | Phúc | Tự kiểm | 4 | B | Chưa làm |
| [4.2](#42-baseline-mức-flow) | Baseline mức flow | GĐ4 | Phúc | Tự kiểm | 4 | N | Chưa làm |
| [5.1](#51-dataloader-pyg-và-vòng-huấn-luyện-chung) | Dataloader PyG, vòng huấn luyện | GĐ5 | Phúc | Tự kiểm | 4 | B | Chưa làm |
| [5.2](#52-gnn-cơ-sở-gcn-và-graphsage) | GCN và GraphSAGE | GĐ5 | Phúc | Tự kiểm | 5 | B | Đang làm |
| [5.3](#53-mô-hình-chính-gat-nhẹ-multi-head) | GAT nhẹ multi-head | GĐ5 | Phúc | Tự kiểm | 5 | B | Đang làm |
| [5.4](#54-phiên-bản-cải-tiến-của-gat-nhẹ) | Bản cải tiến của GAT | GĐ5 | Phúc | Tự kiểm | 6 | B | Chưa làm |
| [5.5](#55-ablation) | Ablation | GĐ5 | Phúc | Tự kiểm | 6 | B | Chưa làm |
| [5.6](#56-các-bộ-còn-lại-và-tổng-quát-hóa-giữa-các-bộ) | Các bộ còn lại, tổng quát hóa | GĐ5 | Phúc | Tự kiểm | 6 | B | Chưa làm |
| [5.7](#57-mô-hình-hybrid-gat--lightgbm) | **Hybrid GAT + LightGBM** | GĐ5 | Phúc | Tự kiểm | 14/11 | B | Chưa làm |
| [5.8](#58-ablation-hybrid-và-minh-chứng-hơn-baseline) | Ablation hybrid, minh chứng hơn baseline | GĐ5 | Phúc | Tự kiểm | 15/11 | B | Chưa làm |
| [6.1](#61-đóng-băng-mô-hình-và-chạy-test-một-lần) | Đóng băng, chạy test 1 lần | GĐ6 | Phúc | Tự kiểm | 7 | B | Chưa làm |
| [6.2](#62-bảng-so-sánh-tổng-hợp-và-bộ-biểu-đồ) | Bảng tổng hợp, biểu đồ | GĐ6 | Phúc | Tự kiểm | 7 | B | Chưa làm |
| [6.3](#63-kiểm-định-thống-kê-và-phân-tích-lỗi) | Kiểm định, phân tích lỗi | GĐ6 | Phúc | Tự kiểm | 7 | B | Chưa làm |
| [6.4](#64-so-sánh-với-các-nghiên-cứu-liên-quan) | So sánh với nghiên cứu khác | GĐ6 | Phúc | Tự kiểm | 7 | N | Chưa làm |
| [7.1](#71-cây-phân-cấp-ahp-và-ma-trận-so-sánh-cặp) | Cây AHP, ma trận so sánh cặp | GĐ7 | Phúc | Tự kiểm | 7 | B | Chưa làm |
| [7.2](#72-tính-ahp-xếp-hạng-và-phân-tích-độ-nhạy) | Tính AHP, xếp hạng, độ nhạy | GĐ7 | Phúc | Tự kiểm | 7 | B | Chưa làm |
| [8.1](#81-srcstream_featurespy) | `src/stream_features.py` | GĐ8 | Phúc | Tự kiểm | 6 | B | Chưa làm |
| [8.2](#82-detectorpy-và-log-cảnh-báo) | `detector.py`, log cảnh báo | GĐ8 | Phúc | Tự kiểm | 7 | B | Chưa làm |
| [8.3](#83-đo-hiệu-năng-triển-khai) | Đo hiệu năng triển khai | GĐ8 | Phúc | Tự kiểm | 7 | B | Chưa làm |
| [8.4](#84-đánh-giá-kết-quả-triển-khai) | Đánh giá kết quả triển khai | GĐ8 | Phúc | Tự kiểm | 8 | B | Chưa làm |
| [9.1](#91-dàn-ý-báo-cáo-và-phân-công-chương) | Dàn ý báo cáo | GĐ9 | Phúc | Tự kiểm | 4 | B | Chưa làm |
| [9.2](#92-viết-báo-cáo-hoàn-chỉnh) | Viết báo cáo | GĐ9 | Phúc | Tự kiểm | 8 | B | Chưa làm |
| [9.3](#93-slide-video-demo-và-readme) | Slide, video demo, README | GĐ9 | Phúc | Tự kiểm | 9 | B | Chưa làm |

---

## GĐ0 Chốt đề & hạ tầng

### 0.1 Chốt phát biểu bài toán và chỉ tiêu nghiệm thu

- **Trạng thái:** **Xong** · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 2 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Báo cáo
- **Đầu vào:** Mô tả đề tài của thầy; mục 1–3 của [KE_HOACH.md](KE_HOACH.md); [docs/plan/00_TONG_QUAN.md](../plan/00_TONG_QUAN.md)
- **Đầu ra:** Mục 1–3 của KE_HOACH.md đã được cả 3 xác nhận; tin nhắn hoặc email gửi thầy kèm phản hồi (lưu link)
- **Tiêu chí kiểm tra:**
  1. Phúc (nhóm trưởng) chốt, ghi ngày dưới mục 3 của KE_HOACH.md; gửi Phong và Tấn đọc.
  2. Mọi chỉ tiêu có con số và cách đo cụ thể.
  3. Các thuật ngữ node, cửa sổ, FPR, độ trễ đều có định nghĩa.
  4. Có phản hồi của thầy về phạm vi, hoặc ghi rõ ngày đã gửi mà chưa có phản hồi.
  5. Đã thống nhất với kế hoạch gốc trong [docs/plan/](../plan/): hạn nộp, ngưỡng tham số (50K hay 100K), có làm AHP không.
- **Phụ thuộc:** —
- **Kết quả:** (03/11/2026) Chốt trong [KE_HOACH.md](KE_HOACH.md) mục 3 và mục 5b.
  1. **Đạt, điều chỉnh theo Phúc.** Phúc chốt mục 1–3 ngày 03/11/2026 (ghi dưới mục 3). Phúc quyết định **không gửi Phong và Tấn**, nên không có xác nhận của hai người.
  2. **Đạt.** Mọi chỉ tiêu ở bảng mục 3 đều có ngưỡng và cách đo. Các ngưỡng đề xuất (100K tham số, F1-macro 0,80, recall 95%) được giữ nguyên.
  3. **Đạt.** Node, cửa sổ, nhãn node, nhiệm vụ ở mục 1; FPR và độ trễ ở mục 1 và bảng mục 3.
  4. **Bỏ theo quyết định của Phúc.** Phúc quyết định tự chốt, không hỏi thầy; chưa có phản hồi nào của thầy về phạm vi. Rủi ro: D6 (AHP dùng để chọn mô hình triển khai) chưa được thầy xác nhận, ghi ở mục 5b.
  5. **Đạt.** Hạn nộp **15/11/2026** (lúc đầu ghi nhầm 8/11, Phúc sửa ngày 03/11/2026); ngưỡng tham số **≤ 100.000** (< 50.000 chỉ là mục tiêu mong muốn); **có làm AHP**. Cột "Tuần" của 9 tuần được thay bằng [lịch 12 ngày](KE_HOACH.md#5b-lịch-đến-hạn-nộp-15112026); không cắt phạm vi.

### 0.2 Chốt 7 quyết định mở D1–D7

- **Trạng thái:** **Xong** · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 2 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Bộ dữ liệu, Thuật toán & đánh giá
- **Đầu vào:** Log quyết định (mục 7 của KE_HOACH.md); kết quả task 1.2 (task 1.3 đã bỏ)
- **Đầu ra:** Cột "Đã chốt" của D1–D7 được điền: lựa chọn, lý do, người duyệt, ngày
- **Tiêu chí kiểm tra:**
  1. Hết Tuần 2 không còn quyết định nào để trống.
  2. Mỗi quyết định có lý do dựa trên số liệu (ví dụ D3 phải kèm bảng số node mỗi lớp trong từng tập).
  3. Muốn đổi quyết định đã chốt thì phải ghi dòng mới, không sửa đè.
- **Phụ thuộc:** 1.2
- **Kết quả:** (03/11/2026) Cột "Đã chốt" của D1–D7 điền ở [KE_HOACH.md mục 7](KE_HOACH.md#7-log-quyết-định). Phúc duyệt các đề xuất qua chat; không hỏi thầy.
  1. **Đạt.** D1–D7 đều đã điền, không còn ô trống.
  2. **Đạt.** D1: số liệu task 1.2. D2: 66.722 cửa sổ, node mỗi cửa sổ p50 11, p95 19, p99 68. D3: bảng số node mỗi lớp trong train/val/test của cả hai cách chia (chronological làm val và test có 0 node ở DoS_DDoS và Truy_cap_trai_phep). D5: số node lớp Khac theo từng tập. D4, D6, D7 là quyết định phương pháp, không có số liệu; D6 chưa có xác nhận của thầy.
  3. **Đạt.** Mới chốt lần đầu nên chưa có quyết định bị đổi.
  - **Đổi so với đề xuất ban đầu:** D3 thêm `gap = 1`. D7 ban đầu ghi laptop 1 luồng do tưởng hạn là 8/11; đã điều chỉnh về Docker (xem "Điều chỉnh 03/11/2026" ở KE_HOACH mục 7). Chi tiết D3: chronological không phải 0 node Botnet_Backdoor ở train (197 node) mà là val và test thiếu hẳn DoS_DDoS và Truy_cap_trai_phep, nên lập luận của đề xuất cũ cần đọc theo bảng mới.

### 0.3 Repo GitHub, cấu trúc thư mục, requirements

- **Trạng thái:** **Xong** · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 2 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Môi trường
- **Đầu vào:** Notebook Tuần 1 của Phong; cấu trúc repo ở [docs/plan/00_TONG_QUAN.md](../plan/00_TONG_QUAN.md) mục 4.2. Nếu nhóm trưởng đã tạo repo theo `plan/` thì clone repo đó, không tạo repo mới.
- **Đầu ra:** Repo có `data/`, `notebooks/`, `src/`, `models/`, `reports/{figures,tables}/`, `docs/`; `requirements.txt`; `.gitignore`; README hướng dẫn cài đặt
- **Tiêu chí kiểm tra:**
  1. Clone mới rồi `pip install -r requirements.txt` chạy được trên Kaggle và trên 1 máy cá nhân.
  2. Mọi thư viện ghi đúng phiên bản (`==`), gồm torch và torch_geometric.
  3. `.gitignore` loại `data/`, `models/*.pt`, checkpoint.
  4. ~~Bật bảo vệ nhánh `main`, chỉ merge qua PR.~~ Bỏ: Phúc quyết định commit thẳng lên `main` (02/11/2026).
- **Phụ thuộc:** —
- **Kết quả:** (02/11/2026, nhánh `main` của repo `minucedev/gnn-iot-ids`; chi tiết ở [docs/notes_env.md](../notes_env.md). Mục 1 và 6 ứng với tiêu chí 1 ở trên, mục 2–3 ứng với tiêu chí 2–3, mục 4 là kiểm tra link mới thêm, mục 5 ứng với tiêu chí 4.)
  1. **Đạt (máy cá nhân).** Xóa `.venv` rồi cài lại đúng theo README (uv, Python 3.11.17, `torch==2.14.1` từ index `cu126`, sau đó `-r requirements.txt`): cài mất 34,1 s nhờ cache của uv. Lần đầu tải mới mất khoảng 39 phút (torch 2,4 GB ở khoảng 1,7 MB/s). `.venv\Scripts\python.exe -I scripts\check_env.py` thoát mã 0: `cuda.is_available: True`, RTX A1000; GATv2Conv (heads=4, edge_dim=3) cho out (50, 64), 2752 tham số trên CPU và GPU, chênh lệch CPU/GPU 3,58e-07; parquet ghi/đọc khớp. Kernel Jupyter `gnn-iot-ids` đã đăng ký.
  2. **Đạt.** `requirements.txt` có 20 gói cấp cao nhất, gói nào cũng ghi `==`, gồm `torch==2.14.1` (bản thật 2.14.1+cu126, cách cài ghi trong README) và `torch-geometric==2.8.1`. `uv pip check`: 148 gói tương thích.
  3. **Đạt.** `git status` sạch sau commit cuối. `git check-ignore -v` xác nhận `data/raw/x.csv` (`data/raw/**`), `models/a.pt` (`models/*.pt`), `.venv` (`.venv/`), `data/processed/g.parquet` (`*.parquet`) đều bị bỏ qua; `data/raw/.gitkeep`, `data/processed/.gitkeep` không bị bỏ qua. File lớn hơn 50 MB bị chặn bởi `.githooks/pre-commit`: thử với file 51 MB thì hook thoát 1, không có file lớn thì thoát 0.
  4. **Đạt.** `scripts/check_links.py` quét mọi file `.md` trong repo: 0 link tương đối hỏng (link trong `docs/gnn_ids/` đã đổi từ `../../plan/`, `../Phong/` sang `../plan/`, `../dataset_cards/`, `../../notebooks/phong_tuan1/`).
  5. **Bỏ theo quyết định của Phúc (02/11/2026).** Không bật bảo vệ nhánh `main`, không dùng nhánh hay PR; mọi commit push thẳng lên `main` của `minucedev/gnn-iot-ids`.
  6. **Đạt (Kaggle).** [notebooks/00_kaggle_env_check.ipynb](../../notebooks/00_kaggle_env_check.ipynb) đẩy bằng Kaggle API (`minucelowcode/gnn-iot-ids-00-env-check`, GPU T4), bản 2 clone `main` tại `c58965a`: `pip install -r requirements.txt` (bỏ qua torch, jupyter, ipykernel) mất 82 s, `check_env.py` thoát mã 0 với Python 3.13.15, torch 2.11.0+cu128 có sẵn, Tesla T4. Bản 1 lỗi cú pháp trong notebook, đã sửa. Chi tiết ở [docs/notes_env.md](../notes_env.md#kaggle).

### 0.4 Tách code notebook thành src/build_graph.py

- **Trạng thái:** **Xong** · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 2 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** EDA & tiền xử lý
- **Đầu vào:** [phong_tuan1_dataset_graph.ipynb](../../notebooks/phong_tuan1/phong_tuan1_dataset_graph.ipynb) (các hàm `load_*`, `clean_flows`, `assign_windows`, `node_edge_tables`, `to_pyg`, `build_dataset`)
- **Đầu ra:** `src/build_graph.py` (có docstring); `tests/test_build_graph.py`; notebook gọi module thay cho code nhúng
- **Tiêu chí kiểm tra:**
  1. Hash đồ thị dựng bằng module trùng với hash của notebook cũ (`cb8842d60f9a3a5a` cho NF-ToN-IoT-v2).
  2. Có test với 1 cửa sổ 6–10 flow tự tính tay: số node, số cạnh, 18 đặc trưng và nhãn đều khớp.
  3. Thêm bộ mới chỉ cần viết một hàm adapter.
- **Phụ thuộc:** 0.3
- **Kết quả:** (03/11/2026) [src/build_graph.py](../../src/build_graph.py), [tests/test_build_graph.py](../../tests/test_build_graph.py), [scripts/check_graph_hash.py](../../scripts/check_graph_hash.py). `pytest`: 16 passed (cả `test_evaluate.py`).
  1. **Đạt.** Notebook [02_check_build_graph_hash.ipynb](../../notebooks/02_check_build_graph_hash.ipynb) chạy trên Kaggle (`minucelowcode/gnn-iot-ids-02-check-build-graph-hash`, bản 2, clone `main` tại `dbd9591`) với NF-ToN-IoT-v2 (`athena21/netflow-datasets-v2`, file 2.676.491.000 byte, trùng dung lượng file Phong dùng): **hash `cb8842d60f9a3a5a` trùng notebook Tuần 1**, đầy đủ `cb8842d60f9a3a5a379397a6fcccff7cdbf344e4dc0017d5feccf59ea15874d2`; 16.941 đồ thị (train 11.858 / val 2.541 / test 2.542 cửa sổ); số dòng sau làm sạch 16.871.353, khớp notebook. Chạy với torch 2.11.0 và pandas 3.1.0 của Kaggle, trong khi Phong dùng pandas 3.0.6, nên kết quả không phụ thuộc phiên bản này. Bản 1 lỗi vì `pip` đè numpy trong kernel đang chạy; đã chuyển sang chạy trong tiến trình con.
  2. **Đạt.** `test_hand_computed_window`: 1 cửa sổ 7 flow (1 flow tự vòng bị bỏ) tự tính tay; 4 node, 4 cạnh, đủ 18 đặc trưng node, 3 đặc trưng cạnh và nhãn (gồm cả trường hợp hòa nhãn và node chỉ có flow vào) đều khớp đến `rtol=1e-12`.
  3. **Đạt.** `test_new_dataset_needs_only_adapter`: bộ NF v3 chạy qua `load_clean("nf_ton_iot_v3", ...)` chỉ nhờ adapter `load_nf_v3` (ts thật = `FLOW_START_MILLISECONDS` / 1000), phần dựng đồ thị dùng chung. Mới thử trên dữ liệu giả nhỏ, **chưa chạy trên NF-ToN-IoT-v3 thật** (việc của task 1.6).
  - **Chưa làm:** notebook Tuần 1 của Phong vẫn giữ code nhúng (cố ý, để giữ nguyên output); chưa có notebook nào được viết lại để gọi module ngoài notebook 02. Thêm `GROUP_MAP` cho tên lớp của CIC (ví dụ `DoS_attacks-Hulk`) để task 1.4/1.6 quyết định.

### 0.5 Module đánh giá chung src/evaluate.py

- **Trạng thái:** **Xong** · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 2 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Thuật toán & đánh giá
- **Đầu vào:** Danh sách chỉ tiêu (mục 3) và các cột `results.csv` (quy ước 6) trong KE_HOACH.md
- **Đầu ra:** `evaluate.py` gồm: `compute_metrics()`, `log_result()` ghi vào `results.csv`, `plot_confusion()`, `plot_roc_pr()`, `count_params()`, `measure_latency()`
- **Tiêu chí kiểm tra:**
  1. Chạy thử với `DummyClassifier` và 1 bộ `y_true`/`y_pred` tự tạo: accuracy, FPR, recall tấn công, F1-macro khớp với số tính tay.
  2. FPR luôn tính trên bài toán nhị phân, Normal là lớp âm.
  3. `results.csv` có đúng 20 cột theo quy ước, không thêm hay bớt cột.
  4. `measure_latency` khởi động nóng (warm-up) 20 lần rồi đo ≥ 200 lần, trả về trung bình, p50, p95, p99 (theo [docs/plan/00_TONG_QUAN.md](../plan/00_TONG_QUAN.md) mục 3).
- **Phụ thuộc:** 0.3
- **Kết quả:** (02/11/2026) [src/evaluate.py](../../src/evaluate.py), [tests/test_evaluate.py](../../tests/test_evaluate.py). `.venv\Scripts\python.exe -m pytest -q`: **10 passed**. Có thêm `model_size_kb()` (cột `size_kb`) và `git_commit()` (cột `commit`, gắn `-dirty` nếu còn thay đổi chưa commit). `roc_auc` là AUC nhị phân (tấn công so với Normal, điểm = 1 − P(Normal)).
  1. **Đạt.** Bộ tính tay 12 node, 5 lớp (`test_hand_computed_metrics`): accuracy 9/12, FPR 1/5, recall tấn công 6/7, F1-macro = (4/5 + 4/7 + 2/3 + 1 + 1)/5 = 0,8076, đều khớp. Tham số `macro_labels=[0,1,2,3]` cho F1-macro trên 4 lớp chính (D5) = 0,7595. `DummyClassifier(most_frequent)` trên 70% Normal: accuracy 0,70, FPR 0, recall tấn công 0.
  2. **Đạt.** FPR và recall tấn công tính trên nhãn gộp (Normal = 0 là âm). Nhầm giữa hai lớp tấn công không bị tính là FP (`test_fpr_is_binary_normal_negative`); không có Normal thì FPR = NaN.
  3. **Đạt.** `RESULT_COLUMNS` có đúng 20 cột theo quy ước 6; ghi 2 dòng thì mọi dòng đều 20 cột; header sai hoặc `split` sai thì `log_result` báo lỗi.
  4. **Đạt.** `measure_latency` gọi hàm 20 + 200 lần (đếm được 220 lần gọi), trả về `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms`; `n_runs < 200` thì báo lỗi; `cuda_sync=True` đồng bộ GPU sau mỗi lần đo.

### 0.6 Chuyển pipeline lên Kaggle

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 2 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Môi trường
- **Đầu vào:** `clean_*.parquet`, `graphs_*.pt`, `split.json`, `scaler.pkl`; `src/`
- **Đầu ra:** Kaggle Dataset riêng tư chứa dữ liệu đã xử lý; notebook mẫu đọc dữ liệu và gọi `src/`
- **Tiêu chí kiểm tra:**
  1. Run All trên Kaggle không lỗi, có ghi lại thời gian chạy và lượng RAM tối đa.
  2. Hash đồ thị đọc trên Kaggle trùng với bản local.
  3. Một phiên Kaggle mới chạy được chỉ bằng cách làm theo README, không cần thao tác tay ngoài README.
- **Phụ thuộc:** 0.4

## GĐ1 Dữ liệu

### 1.0 [Tuần 1] Pipeline v1

Làm sạch, 18 đặc trưng, dựng đồ thị, kiểm tra rò rỉ.

- **Trạng thái:** **Cần sửa** · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 1 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** EDA & tiền xử lý
- **Đầu vào:** NF-ToN-IoT-v2, BoT-IoT đầy đủ
- **Đầu ra:** `clean_*.parquet`, `graphs_*.pt` (16.941 và 12.726 đồ thị), `split.json`, `scaler.pkl`, `phong_features.md`
- **Tiêu chí kiểm tra:** Đã đạt kiểm tra rò rỉ và tái lập. **Cần sửa:** chia `chronological` làm Botnet_Backdoor có 0 node ở train (NF-ToN-IoT-v2) và DoS_DDoS có 0 node ở train (BoT-IoT). Sửa ở task 1.5.
- **Phụ thuộc:** —

### 1.1 [Tuần 1] Khảo sát 4 bộ dữ liệu

Phiếu dataset và chấm điểm.

- **Trạng thái:** **Xong** · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 1 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Bộ dữ liệu
- **Đầu vào:** NF-ToN-IoT-v2, NF-BoT-IoT-v2, ToN-IoT (bản rút gọn), BoT-IoT đầy đủ
- **Đầu ra:** 4 phiếu trong [docs/dataset_cards/](../dataset_cards/), bảng chấm 7 tiêu chí, `data/README.md`
- **Tiêu chí kiểm tra:** Đã đạt: có G1–G6 kèm bằng chứng 5 dòng đầu, có kiểm tra thứ tự dòng. Còn lại: thông tin license và bài báo ghi "cần xác minh", chuyển sang task 1.4.
- **Phụ thuộc:** —

### 1.2 Xác minh và tải NetFlow v3

Đáp ứng yêu cầu "tính thời sự" của thầy. Hỏi Phong trước xem đã tải bản v3 chưa.

- **Trạng thái:** **Xong** · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 2 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Bộ dữ liệu
- **Đầu vào:** Trang NIDS datasets của University of Queensland; bài báo giới thiệu NetFlow v3
- **Đầu ra:** Phiếu dataset cho NF-ToN-IoT-v3 (và NF-UNSW-NB15-v3 hoặc NF-CSE-CIC-IDS2018-v3 nếu có); ghi chú năm phát hành và khác biệt so với v2
- **Tiêu chí kiểm tra:**
  1. Link là nguồn chính thức, không lấy bản đăng lại không rõ gốc.
  2. Kiểm tra bằng code: có cột timestamp (`FLOW_START_MILLISECONDS` hoặc tương đương), in 5 dòng làm bằng chứng.
  3. Kiểm tra thứ tự thời gian: tỉ lệ dòng có ts tăng dần, khoảng thời gian bao phủ.
  4. Có trích dẫn đầy đủ bài báo gốc.
  5. Nếu không có v3 thì ghi rõ đã tìm ở đâu, và D1 chuyển sang phương án v2.
- **Phụ thuộc:** —
- **Kết quả:** (02/11/2026) Phiếu [nf_ton_iot_v3.md](../dataset_cards/nf_ton_iot_v3.md) và [netflow_v3_others.md](../dataset_cards/netflow_v3_others.md). Notebook [01_verify_netflow_v3.ipynb](../../notebooks/01_verify_netflow_v3.ipynb) chạy trên Kaggle (`minucelowcode/gnn-iot-ids-01-verify-netflow-v3`); output `reports/tables/gd1_*_check.json`. Toàn bộ xử lý trên Kaggle, không tải dữ liệu về máy. Chưa hỏi được Phong đã tải v3 chưa.
  1. **Đạt, có ghi chú.** Nguồn chính thức: trang UQ NIDS datasets và bản ghi UQ eSpace DOI 10.48610/44d7c5e (Open Access, "Permitted Re-Use with Commercial Use Restriction"). Link tải UQ (`api.rdm.uq.edu.au`) trả về 401 khi chưa đăng nhập, nên dùng bản Kaggle `athena21/netflow-v3-datasets`. Bản này có nguồn rõ ràng (ghi nguồn là trang UQ, khai báo không chỉnh sửa) và đã kiểm bằng code: NF-ToN-IoT-v3 khớp đúng 27.520.260 / 16.792.214 / 10.728.046 của UQ, CIC khớp cả 3 số, BoT và UNSW chỉ lệch đúng ở 2 chỗ trang UQ đánh máy nhầm. Chưa so được SHA-256 với file của UQ.
  2. **Đạt.** NF-ToN-IoT-v3 có 55 cột, gồm `FLOW_START_MILLISECONDS` và `FLOW_END_MILLISECONDS` (Unix ms), không null; 5 dòng đầu in trong phiếu.
  3. **Đạt.** NF-ToN-IoT-v3: 100% dòng có ts không giảm (0 bước lùi), 44,7% tăng ngặt; bao phủ 2019-04-23 14:09 → 2019-04-29 14:45 UTC (144,6 h). Bộ CIC đạt 99,99999%, BoT 99,76%, UNSW 85,8% (hai bộ sau phải sắp lại).
  4. **Đạt.** Trích dẫn: dữ liệu (Luay et al. 2025, DOI 10.48610/44d7c5e); bài báo IEEE Access vol. 14, pp. 66899–66913, 2026, DOI 10.1109/ACCESS.2026.3688204 (đã kiểm trên Crossref); preprint arXiv:2503.04404; ToN-IoT gốc Moustafa 2021, DOI 10.1016/j.scs.2021.102994.
  5. Không áp dụng: có v3. **Đề xuất D1 = NF-ToN-IoT-v3**, Phúc chốt ở task 0.2. Lưu ý cho D3: Backdoor chỉ có ngày 28–29/04, ransomware 28/04, mitm 29/04.

### 1.3 Tải ToN-IoT có ts và kiểm chứng giả định thứ tự dòng

Bản `Processed_Network_dataset`.

- **Trạng thái:** **Bỏ** · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 2 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Bộ dữ liệu
- **Đầu vào:** Trang ToN-IoT của UNSW
- **Đầu ra:** Phiếu dataset ToN-IoT network (bản gốc); đoạn kết luận giả định "thứ tự dòng NF-ToN-IoT-v2 = thời gian" đúng hay sai
- **Tiêu chí kiểm tra:**
  1. Xác nhận được cột `ts` tồn tại và đủ 3 nhóm tấn công.
  2. So thứ tự các pha nhãn theo `ts` với thứ tự pha theo số dòng của NF-v2, kèm bảng hoặc hình.
  3. Kết luận có số liệu, không chỉ nhận xét bằng mắt.
- **Phụ thuộc:** —
- **Kết quả:** (03/11/2026) **Bỏ theo quyết định của Phúc.** Task này sinh ra để kiểm chứng giả định "thứ tự dòng NF-ToN-IoT-v2 = thời gian" khi v2 không có timestamp. Task 1.2 đã xác minh NF-ToN-IoT-v3 có `FLOW_START_MILLISECONDS` thật, file sắp theo thời gian 100% ([nf_ton_iot_v3.md](../dataset_cards/nf_ton_iot_v3.md)), và D1 đề xuất dùng v3, nên giả định đó không còn cần. Nếu sau này phải quay lại v2 (ví dụ D1 đổi) thì mở lại task này. Task 0.2 và 1.4 không còn phụ thuộc 1.3.

### 1.4 Chốt danh sách ≥ 3 bộ dữ liệu

Kèm bảng "Phân tích lựa chọn bộ dữ liệu".

> **Cập nhật 12/11/2026:** thầy yêu cầu NF-ToN cùng **ít nhất 2 bộ**; nhóm chốt NF-ToN-IoT-v3 + NF-BoT-IoT-v3 (D8). Tiêu chí 1 bên dưới đổi thành "1 bộ NF-ToN chính và ≥ 1 bộ khác". Bảng lựa chọn vẫn liệt kê đủ các bộ đã khảo sát, ghi rõ lý do bỏ CIC (không phải IoT, tốn giờ Kaggle).

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 3 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Bộ dữ liệu, Báo cáo
- **Đầu vào:** Phiếu của 1.1, 1.2
- **Đầu ra:** `docs/dataset_selection.md`: bảng so sánh mọi bộ đã khảo sát, gồm năm, phiên bản, điểm 7 tiêu chí, quyết định chọn hay loại
- **Tiêu chí kiểm tra:**
  1. Có 1 bộ NetFlow chính và ≥ 2 bộ khác, đúng yêu cầu của thầy.
  2. Mỗi bộ bị loại có lý do định lượng.
  3. Có mục "tính thời sự": năm phát hành và lần dùng gần nhất trong bài báo.
  4. License và bài báo gốc đã được xác minh, không còn ghi "cần xác minh".
- **Phụ thuộc:** 1.2

### 1.5 Sửa chiến lược chia tập

Để mọi lớp chính có ở train, val và test.

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 3 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** EDA & tiền xử lý
- **Đầu vào:** Hàm `time_split` (mode `blocked`), quyết định D3
- **Đầu ra:** `split.json` mới cho từng bộ; bảng số node mỗi lớp × mỗi tập
- **Tiêu chí kiểm tra:**
  1. Mỗi lớp chính có ≥ 100 node ở train và ≥ 30 node ở val và ở test. Nếu không đạt thì ghi rõ lớp nào và chỉnh D5.
  2. Giữa các tập có gap ≥ W/S − 1 cửa sổ ở mọi ranh giới khối.
  3. Mọi kiểm tra rò rỉ cũ vẫn đạt.
  4. Tỉ lệ tổng số node train/val/test nằm trong 70/15/15 ± 3 điểm %.
- **Phụ thuộc:** 0.2, 0.4

### 1.6 Dựng lại đồ thị cho mọi bộ đã chốt

Cửa sổ theo giây nếu bộ có `ts`.

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 3 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** EDA & tiền xử lý
- **Đầu vào:** Dữ liệu gốc các bộ đã chốt ở 1.4; `build_graph.py`; `split.json` ở 1.5
- **Đầu ra:** Với mỗi bộ: `clean_<bộ>.parquet`, `graphs_<bộ>_w<W>s<S>.pt`, `split.json`, `scaler.pkl`, `graph_stats.csv`. Upload lên Kaggle Dataset
- **Tiêu chí kiểm tra:**
  1. Kiểm tra rò rỉ (5 mục) và tái lập (hash) đạt cho từng bộ.
  2. IP và port không nằm trong vector đặc trưng.
  3. Không có NaN hoặc inf trong `x` và `edge_attr`.
  4. Có bảng log số dòng trước và sau từng bước làm sạch.
  5. Dựng thêm sẵn các mức W cho ablation D2: 1, 2, 5, 10 giây.
- **Phụ thuộc:** 1.4, 1.5

### 1.7 Từ điển dữ liệu

18 đặc trưng node và 3 đặc trưng cạnh.

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 3 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** EDA & tiền xử lý, Báo cáo
- **Đầu vào:** `phong_features.md`, `build_graph.py`
- **Đầu ra:** `docs/data_dictionary.md`: tên, công thức, đơn vị, có log-transform hay không, ý nghĩa với từng loại tấn công
- **Tiêu chí kiểm tra:**
  1. Mọi đặc trưng có công thức toán.
  2. Người review tự tính lại 3 đặc trưng trên 1 cửa sổ và ra kết quả khớp code.
  3. Ánh xạ nhãn gốc sang 5 nhóm được ghi đầy đủ cho từng bộ.
- **Phụ thuộc:** 1.6

### 1.8 Dựng đồ thị NF-BoT-IoT-v3 (bộ thứ 2)

Thêm ngày 12/11/2026 theo yêu cầu "NF-ToN, ít nhất 2 bộ" của thầy (quyết định D8). Tấn đã có notebook tiền xử lý bộ này trên Kaggle (`phamta/dss-cky-bot`, dữ liệu `seyhed/nf-bot-iot-v3`). Notebook đó gắn nhãn cho cạnh, dùng cửa sổ 10 giây, lấy mẫu ngẫu nhiên 300.000 flow mỗi lớp và chia thời gian liên tục, nên **không dùng trực tiếp đầu ra của nó**; chỉ dùng lại phần đọc file theo đoạn và phần EDA.

- **Trạng thái:** Xong · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 12–13/11 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Bộ dữ liệu, EDA & tiền xử lý
- **Đầu vào:** NF-BoT-IoT-v3 trên Kaggle (`athena21/netflow-v3-datasets` như bộ chính, hoặc `seyhed/nf-bot-iot-v3` sau khi đối chiếu số dòng); adapter `load_nf_v3` trong `src/build_graph.py`; notebook của Tấn
- **Đầu ra:** `reports/tables/graphs_v2/nf_bot_iot_v3/` (summary.json, graph_stats.csv); đồ thị `.pt`, `split.json`, `scaler.pkl` trên Kaggle Dataset; cấu hình chia tập ghi vào KE_HOACH.md mục 7 (bảng điều chỉnh D3)
- **Tiêu chí kiểm tra:**
  1. Số dòng và số flow từng nhãn khớp số liệu UQ công bố (16.933.808 flow; Benign 51.989; trang UQ ghi nhầm tổng là 16.993.808, xem `docs/dataset_cards/netflow_v3_others.md`).
  2. Ánh xạ nhãn: DoS, DDoS → DoS_DDoS; Reconnaissance, Theft → Truy_cap_trai_phep; Benign → Normal. Ghi rõ bộ này **không có lớp Botnet_Backdoor**, nên F1-macro tính trên các lớp có mặt.
  3. Cửa sổ 5 giây, chia theo khối (blocked, cân theo số node, gap 1); số khối chọn bằng `scripts/split_search.py` như các bộ khác. Mỗi lớp có mặt có ≥ 100 node ở train và ≥ 30 ở val, test; nếu Normal không đạt thì ghi rõ số node.
  4. Không lấy mẫu flow ngẫu nhiên (quy ước 6.4). Nếu thiếu RAM thì lấy mẫu theo cửa sổ và ghi lại tỉ lệ.
  5. Kiểm tra rò rỉ, NaN và tái lập (`scripts/verify_graphs.py`) đều đạt.
- **Phụ thuộc:** 0.4, 1.5
- **Kết quả:** (12/11/2026) Notebook `gnn-iot-ids-11-build-graphs-bot` (Kaggle CPU 4 nhân, bản 2, clone `main` tại `47dccee`), chạy 726 s, RAM đỉnh 12,25 GB. Bản 1 dừng ở bước pytest vì `test_split.py` import `tests.test_build_graph` bị gói `tests` khác của Kaggle che mất (test cũ `test_split_search_script_runs` cũng lỗi như vậy); đã sửa ở `47dccee`. Cấu hình chọn: `blocked`, **10 khối**, theo node, gap 1, W = S = 5 s. File kéo về: [reports/tables/graphs_v2/nf_bot_iot_v3/](../../reports/tables/graphs_v2/nf_bot_iot_v3/), [split_search_nf_bot_iot_v3.json](../../reports/tables/split_search/split_search_nf_bot_iot_v3.json), [reports/kaggle/gnn-iot-ids-11-build-graphs-bot/](../../reports/kaggle/gnn-iot-ids-11-build-graphs-bot/). `.pt`, `split.json`, `scaler.pkl` ở lại trong output notebook trên Kaggle.
  1. **Đạt.** [data_check_nf_bot_iot_v3.json](../../reports/kaggle/gnn-iot-ids-11-build-graphs-bot/data_check_nf_bot_iot_v3.json): sha256 `8bde1f6f…9ea` và 3.817.527.532 byte khớp; 16.933.808 dòng; DoS 8.034.190, DDoS 7.150.882, Reconnaissance 1.695.132, Theft 1.615, Benign 51.989 đều khớp.
  2. **Đạt.** Lớp có mặt: Normal, DoS_DDoS, Truy_cap_trai_phep; `absent_classes` = Botnet_Backdoor, Khac (ghi trong `split_search_nf_bot_iot_v3.json` và trong chi tiết kiểm tra của `summary.json`). Số node (train/val/test): Normal 15.578 / 2.918 / 2.784; DoS_DDoS 2.855 / 996 / 910; Truy_cap_trai_phep 31.387 / 6.712 / 6.971; tổng 49.820 / 10.626 / 10.665 (70,1 / 14,9 / 15,0 %), 9.720 / 2.145 / 2.072 đồ thị.
  3. **Đạt.** Mọi lớp có mặt vượt 100 / 30 / 30; Normal có 15.578 / 2.918 / 2.784 node, không phải hạ ngưỡng. 10 khối là số nhỏ nhất đạt (split_search thử 10, 20, 50, 100, 200; cả 5 đều đạt). Chia `chronological` thì DoS_DDoS bị thiếu (mục `chronological` trong JSON), đây là lý do phải dùng `blocked`.
  4. **Đạt.** Không lấy mẫu flow. Nhật ký làm sạch: 16.933.808 → 16.933.807 dòng (chỉ bỏ 1 bản ghi trùng). 13.937 đồ thị, trung bình 5,1 node và 4,2 cạnh mỗi cửa sổ.
  5. **Đạt.** `build_dataset_cli.py` và `verify_graphs.py` thoát mã 0, 10/10 kiểm tra đạt (không trùng cửa sổ, gap ≥ 1, không NaN/inf, hình dạng, scaler chỉ fit trên train, tỉ lệ node). Hash lần 1 = lần 2 = `3aa51017b6ec0089` (đầy đủ `3aa51017b6ec00899458a7564fc2dbab16ae04c51c50cfcf7ac0b648fa968bab`).
  6. **Đạt (Kaggle).** `pytest -q`: 42 passed (gồm test mới trong `test_checks.py`, `test_build_graph.py`, `test_split.py`).
  Ghi chú: hành vi "chỉ xét lớp có mặt" thực ra đã có sẵn vì `encode_labels` chỉ đánh mã cho nhóm có mặt; thay đổi ở `src/checks.py` và `scripts/split_search.py` chỉ gom logic ngưỡng vào một chỗ, ghi `absent_classes` và khóa hành vi bằng test.

## GĐ2 EDA & trực quan hóa

### 2.1 EDA mức flow

Làm cho từng bộ dữ liệu.

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 3 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** EDA & tiền xử lý
- **Đầu vào:** `clean_<bộ>.parquet`
- **Đầu ra:** `notebooks/02_eda_flow.ipynb`; ≥ 6 hình mỗi bộ: phân bố lớp, boxplot log đặc trưng theo lớp, heatmap tương quan, mutual information, phân bố theo thời gian, giao thức và cổng
- **Tiêu chí kiểm tra:**
  1. Mỗi hình có tiêu đề, nhãn trục, đơn vị và 1–2 câu nhận xét ngay bên dưới.
  2. Có bảng tóm tắt thống kê (min, median, max, % thiếu) cho mọi đặc trưng số.
  3. Chỉ ra ≥ 3 đặc trưng phân biệt lớp tốt nhất kèm số liệu MI.
  4. Run All trên Kaggle.
- **Phụ thuộc:** 1.6

### 2.2 EDA mức đồ thị

Kích thước, bậc, homophily.

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 3 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** EDA & tiền xử lý
- **Đầu vào:** `graphs_<bộ>.pt`
- **Đầu ra:** `notebooks/02_eda_graph.ipynb`; hình phân bố số node và cạnh mỗi cửa sổ, bậc vào/ra theo lớp; bảng homophily
- **Tiêu chí kiểm tra:**
  1. Có edge homophily và class-insensitive homophily cho từng bộ.
  2. Có nhận xét homophily cao hay thấp nghĩa là gì cho GNN (dùng làm luận cứ ở chương lý thuyết và phân tích lỗi).
  3. Có so sánh bậc vào/ra của lớp DDoS với Normal để chứng minh dấu hiệu fan-in và fan-out.
- **Phụ thuộc:** 1.6

### 2.3 Trực quan hóa đồ thị các cửa sổ mẫu

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 3 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** EDA & tiền xử lý, Báo cáo
- **Đầu vào:** `graphs_<bộ>.pt`
- **Đầu ra:** ≥ 3 hình đồ thị (networkx hoặc pyvis): cửa sổ bình thường, DDoS, scan hoặc backdoor; node tô màu theo nhãn, kích thước theo bậc
- **Tiêu chí kiểm tra:**
  1. Mỗi hình ≤ 300 node để còn đọc được, có chú thích màu.
  2. Có cùng layout seed để so sánh được.
  3. Nhìn thấy rõ cấu trúc đặc trưng: hình sao fan-in của DDoS, fan-out của scan; có chú thích chỉ vào các cấu trúc này.
  4. Có thêm 1 hình biểu diễn quy trình flow → cửa sổ → đồ thị để đưa vào báo cáo.
- **Phụ thuộc:** 1.6

## GĐ3 Lý thuyết

### 3.1 Lý thuyết: IDS cho IoT, các loại tấn công, NetFlow

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 3 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Lý thuyết, Báo cáo
- **Đầu vào:** Bài báo gốc của các bộ dữ liệu, tài liệu NetFlow v9/IPFIX
- **Đầu ra:** Bản nháp 4–6 trang: IDS (dựa trên chữ ký và dựa trên bất thường), đặc thù IoT, DDoS, botnet, truy cập trái phép, NetFlow và các trường đặc trưng
- **Tiêu chí kiểm tra:**
  1. ≥ 6 tài liệu tham khảo có DOI hoặc arXiv.
  2. Mỗi loại tấn công có mô tả hành vi mạng, nối với đặc trưng node trong data_dictionary.
  3. Không chép nguyên văn; ghi nguồn cho mọi hình mượn.
- **Phụ thuộc:** —

### 3.2 Lý thuyết GNN: tại sao chọn và còn mô hình nào khác

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 3 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Lý thuyết, Mô hình, Báo cáo
- **Đầu vào:** Bài báo GCN, GraphSAGE, GAT, GATv2, E-GraphSAGE, E-ResGAT; tài liệu về lightweight và quantization
- **Đầu ra:** Bản nháp 6–8 trang: message passing, GCN, GraphSAGE, GAT (công thức attention, multi-head), GATv2, đặc trưng cạnh; kỹ thuật làm nhẹ mô hình; bảng so sánh các hướng mô hình
- **Tiêu chí kiểm tra:**
  1. Có công thức của GAT và multi-head attention, giải thích từng ký hiệu.
  2. Bảng so sánh ≥ 6 hướng (ML cây, MLP, CNN/LSTM, Transformer, GCN/GraphSAGE, GAT) về ưu điểm, nhược điểm, khi nào dùng.
  3. Mục "tại sao chọn GAT nhẹ" lập luận bằng ≥ 3 lý do có dẫn chứng.
  4. ≥ 8 tài liệu, ≥ 4 tài liệu từ 2022 trở lại.
- **Phụ thuộc:** —

### 3.3 Lý thuyết baseline ML, chỉ số đánh giá và AHP

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 4 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Lý thuyết, AHP, Báo cáo
- **Đầu vào:** Tài liệu về RF, XGBoost, LightGBM, MLP; Saaty (1980) về AHP
- **Đầu ra:** Bản nháp 4–6 trang: các baseline; định nghĩa chỉ số (accuracy, precision, recall, F1-macro, FPR, ROC-AUC, McNemar); AHP (thang 1–9, vector riêng, CI, CR, RI)
- **Tiêu chí kiểm tra:**
  1. Mỗi chỉ số có công thức và giải thích vì sao cần cho IDS (đặc biệt FPR và F1-macro khi lệch lớp).
  2. AHP có ví dụ tính tay 3×3 kèm CR.
  3. ≥ 5 tài liệu.
- **Phụ thuộc:** —

### 3.4 Bảng tổng quan nghiên cứu liên quan

Làm mốc so sánh.

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 4 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Lý thuyết, Báo cáo
- **Đầu vào:** Bài báo GNN-IDS dùng các bộ NF-*, ToN-IoT, BoT-IoT
- **Đầu ra:** Bảng ≥ 8 bài: năm, bộ dữ liệu, mô hình, mức phân loại (node hay edge), chỉ số, số tham số, độ trễ (nếu có)
- **Tiêu chí kiểm tra:**
  1. Mỗi dòng có link DOI hoặc arXiv và số liệu lấy đúng từ bảng trong bài (ghi số trang hoặc số bảng).
  2. Ghi rõ bài nào khác mức phân loại với nhóm (node so với flow) để tránh so sánh sai.
  3. ≥ 4 bài từ 2023 trở lại.
- **Phụ thuộc:** —

## GĐ4 Baseline ML

### 4.1 Baseline ML trên đặc trưng node

LR, RF, XGBoost, LightGBM, MLP.

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 4 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Baseline, Thuật toán & đánh giá
- **Đầu vào:** `graphs_<bộ>.pt` (chỉ lấy `x` và `y`, bỏ cạnh), `split.json`, `evaluate.py`
- **Đầu ra:** `notebooks/04_baseline.ipynb`; `models/baseline_*`; dòng kết quả `split=val` trong `results.csv`; file log không gian tìm kiếm siêu tham số
- **Tiêu chí kiểm tra:**
  1. Dùng đúng `split.json` chung, không tự chia lại.
  2. Tinh chỉnh chỉ trên val, ≤ 30 lần thử mỗi mô hình (Optuna hoặc grid), lưu log.
  3. 3 seed cho cấu hình tốt nhất.
  4. Có số tham số hoặc kích thước mô hình và độ trễ mỗi cửa sổ.
  5. Chưa chạy trên test, test để dành cho 6.1.
  6. Làm cả đa lớp và nhị phân.
- **Phụ thuộc:** 0.5, 1.6
- **Kết quả:** (12/11/2026, GĐ A, chưa đổi trạng thái vì còn NF-BoT-IoT-v3) Kéo về [reports/kaggle/](../../reports/kaggle/README.md): từ notebook 05 (bị dừng giữa chừng, bản 2, commit `9d7761a`): `results_baselines_nf_ton_iot_v3.csv` (6412 B), `baseline_search_nf_ton_iot_v3.json`, `results_baselines_nf_cicids2018_v3.csv`, `verify_nf_ton_iot_v3.json`, `verify_nf_cicids2018_v3.json`; từ 09 (bản 1, commit `4d98421`): `results_baselines_nf_cicids2018_v3.csv`, `baseline_search_nf_cicids2018_v3.json`; từ 10 (bản 1, commit `4d98421`): `results_baselines_nf_ton_iot_v2.csv`, `baseline_search_nf_ton_iot_v2.json`. Cả 12 file khớp kích thước Kaggle báo; cả hai commit có thật trong repo. Chỉ đếm, chưa tính trung bình.
  - **NF-ToN-IoT-v3, `split=val`, `task=multiclass`: 15 dòng = logreg, random_forest, xgboost, lightgbm, mlp × seed 42, 43, 44 (đủ 5 baseline × 3 seed, không trùng dòng)**, đều từ notebook 05. Log tìm siêu tham số có 3/3/3/3/2 lần thử (≤ 30). `verify_nf_ton_iot_v3.json`: 10/10 kiểm tra đạt.
  - **Còn thiếu:** `task=binary` (tiêu chí 6); NF-BoT-IoT-v3 (chưa chạy; làm ở GĐ D). Test chưa chạy (đúng tiêu chí 5).
  - Phụ lục (D8): NF-ToN-IoT-v2 đủ 5 × 3 seed (notebook 10); NF-CICIDS2018-v3 đủ 5 × 3 seed khi ghép 05 (logreg, random_forest; xgboost chỉ 42, 43) và 09 (xgboost, lightgbm, mlp × 3 seed).
  - **(12/11/2026, GĐ D) NF-BoT-IoT-v3** từ notebook [13_baselines](../../notebooks/13_baselines.ipynb) (Kaggle CPU, v1, commit `93c6625`, `complete`, 293 s, RAM đỉnh 0,85 GB; pytest 42 passed). Kéo về [reports/kaggle/gnn-iot-ids-13-baselines/](../../reports/kaggle/gnn-iot-ids-13-baselines/): `results_baselines_nf_bot_iot_v3.csv` (5302 B), `baseline_search_nf_bot_iot_v3.json` (1887 B), `run_info.json` (852 B). Chỉ đếm, chưa tính trung bình.
    1. **Đạt.** Đọc `split.json` của `processed/nf_bot_iot_v3` (output notebook 11), `classes` = Normal, DoS_DDoS, Truy_cap_trai_phep; không chia lại. `run_baselines.py` không cần sửa: số lớp lấy từ `split.json`, F1-macro bỏ lớp không có trong nhãn thật (`MAIN4` có mã 3 nhưng bộ này chỉ có mã 0, 1, 2).
    2. **Đạt.** `baseline_search_nf_bot_iot_v3.json`: 3/3/3/3/2 lần thử (logreg, random_forest, xgboost, lightgbm, mlp) ≤ 30, chọn trên val seed 42.
    3. **Đạt.** 15 dòng = 5 mô hình × seed 42, 43, 44, mỗi (model, seed) đúng 1 dòng.
    4. **Đạt.** Cột `n_params`, `size_kb`, `infer_ms_per_window` có ở cả 15 dòng.
    5. **Đạt.** 15/15 dòng `split=val`, `task=multiclass`; không có dòng test.
    6. **Chờ Phúc chốt.** `fpr`, `recall_attack`, `roc_auc` đã là chỉ số nhị phân suy ra từ dự đoán đa lớp (Normal là lớp âm, xem `src/evaluate.py`); chưa có dòng `task=binary` riêng. Đề xuất coi là đạt; chưa huấn luyện thêm.
  - **Còn thiếu:** chỉ còn tiêu chí 6 (cả 2 bộ). Chưa đổi trạng thái sang Xong.

### 4.2 Baseline mức flow

RF trên đặc trưng NetFlow gốc, để đối chiếu với các bài báo.

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 4 · **Ưu tiên:** Nên có
- **Mục đề tài:** Baseline
- **Đầu vào:** `clean_<bộ>.parquet`, cùng ranh giới thời gian với `split.json`
- **Đầu ra:** Kết quả RF và XGBoost phân loại flow, đưa vào `results.csv` với `task=flow_multi` hoặc `flow_binary`
- **Tiêu chí kiểm tra:**
  1. Mỗi flow được gán đúng tập theo ranh giới thời gian của `split.json`.
  2. Bỏ cột IP, port, timestamp khỏi đặc trưng.
  3. Ghi rõ trong báo cáo đây là bài toán khác (flow so với node), không so trực tiếp với kết quả node.
- **Phụ thuộc:** 4.1

## GĐ5 Lightweight GNN

### 5.1 Dataloader PyG và vòng huấn luyện chung

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 4 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Mô hình, Môi trường
- **Đầu vào:** `graphs_<bộ>.pt`, `split.json`, `scaler.pkl`, `evaluate.py`
- **Đầu ra:** `src/train_gnn.py`: DataLoader theo batch đồ thị, class weight, early stopping theo F1-macro trên val, lưu checkpoint tốt nhất, log đường học
- **Tiêu chí kiểm tra:**
  1. Sanity check: overfit 20 đồ thị đạt ≥ 99% accuracy trên train.
  2. Chạy 2 lần cùng seed cho cùng loss tới 1e-6 (bật deterministic).
  3. Chạy được trên GPU (Kaggle hoặc máy cá nhân), có ghi thời gian mỗi epoch.
  4. Config lưu thành file YAML hoặc JSON đi kèm checkpoint.
- **Phụ thuộc:** 0.5, 1.6

### 5.2 GNN cơ sở: GCN và GraphSAGE

2 lớp.

- **Trạng thái:** Đang làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 5 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Mô hình, Thuật toán & đánh giá
- **Đầu vào:** `train_gnn.py`
- **Đầu ra:** `src/models/gcn.py`, `sage.py`; kết quả `split=val` trong `results.csv`
- **Tiêu chí kiểm tra:**
  1. ≤ 100K tham số mỗi mô hình.
  2. 3 seed.
  3. Tinh chỉnh ≤ 20 lần thử trên val, có lưu log.
  4. Có đường loss và F1 trên val theo epoch.
- **Phụ thuộc:** 5.1
- **Kết quả:** (12/11/2026) gcn, sage × 3 seed, `split=val`: NF-ToN-IoT-v3 đủ 6 dòng (GĐ A), NF-BoT-IoT-v3 đủ 6 dòng (GĐ E); chi tiết số tham số, đường học và lưu ý ở dòng Kết quả của [5.3](#53-mô-hình-chính-gat-nhẹ-multi-head). Tiêu chí 1, 2 đạt cho cả hai bộ; tiêu chí 4 đạt cho NF-BoT-IoT-v3, chưa có cho NF-ToN-IoT-v3; tiêu chí 3 chưa làm (chưa tinh chỉnh, chưa có log).

### 5.3 Mô hình chính: GAT nhẹ multi-head

Có đặc trưng cạnh.

- **Trạng thái:** Đang làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 5 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Mô hình, Thuật toán & đánh giá
- **Đầu vào:** `train_gnn.py`; kết quả 5.2 và 4.1
- **Đầu ra:** `src/models/gat_lite.py` (GATv2Conv, `edge_dim=3`, 2 lớp, 4 head, hidden ≤ 32); checkpoint; bảng đếm tham số theo từng lớp
- **Tiêu chí kiểm tra:**
  1. Tổng tham số ≤ 100K, file ≤ 1 MB, có bảng đếm theo từng lớp.
  2. Trên val đạt accuracy ≥ 91%, FPR ≤ 4,08%, recall tấn công ≥ 95%. Nếu không đạt thì phải có phân tích trước khi sang 5.4.
  3. 3 seed, std accuracy ≤ 1,5 điểm %.
  4. Lưu được trọng số attention để giải thích ở 6.3.
- **Phụ thuộc:** 5.2
- **Kết quả:** (12/11/2026, GĐ A, chưa đổi trạng thái vì còn NF-BoT-IoT-v3) Kéo về [reports/kaggle/gnn-iot-ids-07-train-gnn/](../../reports/kaggle/gnn-iot-ids-07-train-gnn/): `results_gnn_nf_ton_iot_v3.csv` (3870 B), `results_gnn_nf_cicids2018_v3.csv`, `results_gnn_nf_ton_iot_v2.csv`; không tải `ckpt/` (27 file `.pt`). Notebook `gnn-iot-ids-07-train-gnn` bản 2, GPU T4, clone commit `9d7761a` (có thật trong repo). Chi tiết ở [reports/kaggle/README.md](../../reports/kaggle/README.md). Chỉ đếm, chưa tính trung bình.
  - **NF-ToN-IoT-v3, `split=val`, `task=multiclass`: 9 dòng = gcn, sage, gat × seed 42, 43, 44 (đủ 3 GNN × 3 seed, không trùng dòng).** Phần này cũng là kết quả của 5.2 (gcn, sage).
  - **Còn thiếu cho 5.3 / 5.2 (cập nhật 12/11, GĐ E):** `task=binary`; log tìm siêu tham số GNN (cả hai bộ chạy cấu hình mặc định, chưa tinh chỉnh); bảng đếm tham số theo lớp; trọng số attention; đường loss/F1 theo epoch của NF-ToN-IoT-v3 (output của 07 chỉ có 3 CSV và `ckpt/`; NF-BoT-IoT-v3 đã có, xem dưới). Ngưỡng 91% / 4,08% / 95% đối chiếu ở mục NF-BoT-IoT-v3 dưới đây; NF-ToN-IoT-v3 chưa đối chiếu.
  - **NF-BoT-IoT-v3 (12/11/2026, GĐ E, notebook `gnn-iot-ids-14-train-gnn-bot` bản 1, Kaggle CPU 4 lõi, clone commit `ac05c06`, pytest 47 passed):** `split=val`, `task=multiclass`: **9 dòng = gcn, sage, gat × seed 42, 43, 44, mỗi (model, seed) đúng 1 dòng**, không có split khác. Kéo về [reports/kaggle/gnn-iot-ids-14-train-gnn-bot/](../../reports/kaggle/gnn-iot-ids-14-train-gnn-bot/): `results_gnn_nf_bot_iot_v3.csv` (3219 B), `curves_nf_bot_iot_v3.csv` (19,9 KB: loss train, loss val, F1-macro val theo epoch, 9 lần chạy × 19–40 epoch, cột `is_best`), `run_info.json`; 9 file `ckpt/nf_bot_iot_v3_<model>_s<seed>.pt` ở lại trên Kaggle cho GĐ F. Chạy CPU 276,6 s huấn luyện (tổng notebook 424,5 s), RAM đỉnh 0,66 GB, 0 giờ GPU.
    - Số tham số: gcn 5.571, sage 10.819, **gat 11.587** (≤ 100.000), file gat 50,1 KB (≤ 1 MB); dữ liệu 3 lớp nên nhỏ hơn mức 11.700 ước tính cho 5 lớp.
    - Std accuracy giữa 3 seed (ddof = 1, chỉ để đối chiếu ngưỡng): gcn 0,405 điểm %, sage 0,052, gat 0,054, **đều ≤ 1,5**.
    - **Không đạt ngưỡng FPR của 5.3:** FPR trên val của cả 9 dòng nằm trong 20,6–22,3% (ngưỡng ≤ 4,08%); accuracy 91,4–93,2% và recall tấn công 96,6–98,5% thì đạt. F1 lớp Normal của gat s42 là 0,847 (DoS_DDoS 0,964, Truy_cap_trai_phep 0,945). Theo tiêu chí 2 của 5.3 cần có phân tích trước khi sang 5.4; chưa làm. F1-macro trung bình đúng 3 lớp có mặt (mã 0, 1, 2).
    - Lưu ý: gcn seed 42 và 44 chạy hết 40 epoch (không dừng sớm vì patience), có thể chưa hội tụ; các mô hình khác dừng ở epoch 19–26.
    - Đổi tên `MAIN4` → `MACRO_CODES`, `val_f1_macro4` → `val_f1_macro` ([src/train.py](../../src/train.py)), cách tính không đổi; `run_gnn.py` sửa lỗi không nhận `models=gcn,sage,gat`. Test mới [tests/test_train.py](../../tests/test_train.py).
    - **Trạng thái 5.2, 5.3 để `Đang làm`, chưa `Xong`:** đã đủ 9 dòng cho cả hai bộ nhưng còn thiếu tiêu chí (5.2: log tinh chỉnh ≤ 20 lần thử; 5.3: FPR, bảng tham số theo lớp, trọng số attention).
  - CIC và NF-ToN-IoT-v2 đủ gcn, sage, gat × 3 seed, chỉ làm phụ lục (D8).

### 5.4 Phiên bản cải tiến của GAT nhẹ

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 6 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Mô hình, Thuật toán & đánh giá
- **Đầu vào:** `gat_lite.py`, kết quả 5.3
- **Đầu ra:** Ít nhất 1 bản cải tiến, chọn trong: focal loss hoặc class-balanced loss; residual + LayerNorm; đặc trưng thời gian (ghép embedding cửa sổ trước); quantization INT8 hoặc pruning
- **Tiêu chí kiểm tra:** Bản cải tiến chỉ được công nhận khi đạt một trong hai điều kiện trên val:
  1. Tăng F1-macro hoặc recall lớp Botnet_Backdoor lớn hơn 2 lần std của 3 seed; hoặc
  2. Giảm ≥ 30% tham số hoặc độ trễ mà F1-macro giảm ≤ 1 điểm %.

  Nếu không đạt thì vẫn báo cáo là thử nghiệm thất bại.
- **Phụ thuộc:** 5.3

### 5.5 Ablation

Số head, số lớp, hidden, đặc trưng cạnh, kích thước cửa sổ.

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 6 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Thuật toán & đánh giá
- **Đầu vào:** `gat_lite.py`; các đồ thị W = 1, 2, 5, 10 giây từ 1.6
- **Đầu ra:** Bảng ablation và 5 biểu đồ: head ∈ {1, 2, 4, 8}, lớp ∈ {1, 2, 3}, hidden ∈ {8, 16, 32, 64}, có/không `edge_attr`, W ∈ {1, 2, 5, 10} giây
- **Tiêu chí kiểm tra:**
  1. Đổi đúng 1 yếu tố mỗi lần, các yếu tố khác giữ ở cấu hình chính.
  2. Chỉ dùng val, không dùng test.
  3. Mỗi điểm có 3 seed và thanh sai số.
  4. Mỗi biểu đồ có cả F1-macro và số tham số hoặc độ trễ (2 trục).
  5. Có kết luận cấu hình chính đã hợp lý chưa.
- **Phụ thuộc:** 5.3, 1.6

### 5.6 Các bộ còn lại và tổng quát hóa giữa các bộ

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 6 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Thuật toán & đánh giá, Bộ dữ liệu
- **Đầu vào:** Cấu hình chính đã đóng băng; đồ thị của các bộ phụ
- **Đầu ra:** Kết quả trên từng bộ phụ (train và test trên cùng bộ); ma trận train trên bộ A, test trên bộ B (chỉ dùng các lớp chung)
- **Tiêu chí kiểm tra:**
  1. Mọi bộ dùng chung 18 đặc trưng và cùng ánh xạ nhãn.
  2. Phần chéo giữa các bộ chỉ tính trên các lớp có ở cả hai bộ, có ghi rõ.
  3. Kết quả ghi vào `results.csv` với cột `dataset` dạng "A→B".
- **Phụ thuộc:** 5.3

### 5.7 Mô hình hybrid GAT + LightGBM

Thêm ngày 12/11/2026 theo yêu cầu "mô hình hybrid, lý do chọn" của thầy (quyết định D9). Đây là **mô hình chính** từ ngày này.

**Kiến trúc.** (1) Huấn luyện GAT nhẹ (cấu hình của task 5.3) để phân loại node. (2) Cố định GAT, lấy vector ẩn của lớp GAT cuối (64 chiều) làm embedding của mỗi node. (3) Ghép embedding với 18 đặc trưng gốc (đã chuẩn hóa) thành 82 cột. (4) LightGBM phân loại 82 cột đó.

**Lý do chọn** (ghi vào báo cáo): số liệu sơ bộ cho thấy mô hình cây mạnh trên đặc trưng dạng bảng nhưng không thấy hàng xóm; GNN thấy cấu trúc mạng nhưng FPR cao. Embedding GAT mang thông tin cấu trúc mà 18 đặc trưng không có; LightGBM tận dụng tốt đặc trưng dạng bảng và cho xác suất dễ hiệu chỉnh ngưỡng.

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 14/11 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Mô hình, Thuật toán & đánh giá
- **Đầu vào:** GAT đã huấn luyện (5.3) cho mỗi bộ và mỗi seed; đồ thị và `split.json` (1.6, 1.8); `src/evaluate.py`
- **Đầu ra:** `src/hybrid.py` (trích embedding, ghép đặc trưng, huấn luyện LightGBM); `scripts/run_hybrid.py`; notebook Kaggle; dòng kết quả `model=hybrid_gat_lgbm` (split=val) trong `results.csv`; `tests/test_hybrid.py`
- **Tiêu chí kiểm tra:**
  1. **Không rò rỉ:** GAT và LightGBM chỉ học trên train; tinh chỉnh chỉ trên val; scaler dùng lại của task 1.6.
  2. **Embedding out-of-fold cho tập train:** GAT đã thấy nhãn của node train, nên embedding train có thể "quá sạch". Chia train thành K = 5 nhóm khối thời gian; với mỗi nhóm, GAT huấn luyện trên 4 nhóm còn lại rồi sinh embedding cho nhóm đó. Val và test dùng embedding của GAT huấn luyện trên toàn bộ train. Báo cáo cả bản không dùng out-of-fold để so sánh.
  3. Test với dữ liệu giả nhỏ: số cột đầu vào LightGBM = 18 + kích thước embedding; thứ tự node giữ nguyên sau khi ghép; chạy 2 lần cùng seed cho cùng dự đoán.
  4. Đếm "kích thước" của hybrid: số tham số GAT + số lá của LightGBM, kèm kích thước file; độ trễ mỗi cửa sổ gồm cả bước GAT và bước LightGBM.
  5. 3 seed (42, 43, 44) trên cả 2 bộ; ghi thời gian chạy GPU vào dòng Kết quả.
- **Phụ thuộc:** 5.3, 1.8

### 5.8 Ablation hybrid và minh chứng hơn baseline

Thêm ngày 12/11/2026 theo yêu cầu "cần minh chứng hybrid cao hơn các mô hình baseline (ML, DL, …)" của thầy.

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 15/11 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Thuật toán & đánh giá, Baseline
- **Đầu vào:** Kết quả 4.1 (LR, RF, XGBoost, LightGBM, MLP), 5.2–5.3 (GCN, GraphSAGE, GAT), 5.7 (hybrid)
- **Đầu ra:** Bảng ablation 5 dòng trên val của cả 2 bộ:
  - (a) LightGBM chỉ 18 đặc trưng;
  - (b) GAT đơn;
  - (c) LightGBM + đặc trưng hàng xóm tính tay (trung bình 18 đặc trưng của các node kề, không học), để chứng minh embedding học được có ích hơn tổng hợp thủ công;
  - (d) LightGBM chỉ embedding;
  - (e) Hybrid đầy đủ.

  Kèm ngưỡng quyết định đã hiệu chỉnh trên val để FPR ≤ 4,08%, và bảng so sánh hybrid với mọi baseline ML, DL, GNN đơn.
- **Tiêu chí kiểm tra:**
  1. Mọi mô hình dùng cùng `split.json`, cùng 3 seed, cùng `evaluate.py`.
  2. Ngưỡng quyết định (xác suất "tấn công" tối thiểu) chỉ chọn trên val, rồi giữ nguyên khi chạy test ở 6.1.
  3. Hybrid được coi là "cao hơn" khi F1-macro vượt baseline tốt nhất **hơn 2 lần std** trên val; ở 6.1 kèm McNemar p < 0,05 trên test. Không đạt thì ghi rõ ở dòng Kết quả và trong báo cáo, không chỉnh theo tập test.
  4. Có biểu đồ cột F1-macro và FPR cho mọi mô hình, có thanh sai số, và vạch ngưỡng 4,08%.
- **Phụ thuộc:** 4.1, 5.2, 5.3, 5.7

## GĐ6 Đánh giá tổng hợp

### 6.1 Đóng băng mô hình và chạy test một lần

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 7 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Thuật toán & đánh giá
- **Đầu vào:** Checkpoint và config của mọi mô hình (4.1, 5.2–5.6)
- **Đầu ra:** Tag git `freeze-v1`; dòng kết quả `split=test` cho mọi mô hình, mọi bộ, 3 seed
- **Tiêu chí kiểm tra:**
  1. Tag git được tạo trước khi chạy test; cột `commit` trong `results.csv` trùng với tag.
  2. Test chạy đúng 1 lần, có log thời điểm chạy.
  3. Không chỉnh mô hình sau khi đã xem kết quả test; muốn chỉnh thì phải ghi rõ trong báo cáo là "sau test".
  4. Kiểm tra lại toàn bộ chỉ tiêu ở mục 3 của KE_HOACH.md, ghi đạt hoặc không đạt cho từng chỉ tiêu.
- **Phụ thuộc:** 4.1, 5.4, 5.5, 5.6

### 6.2 Bảng so sánh tổng hợp và bộ biểu đồ

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 7 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Thuật toán & đánh giá, Báo cáo
- **Đầu vào:** `results.csv` (`split=test`)
- **Đầu ra:** `scripts/make_figures.py`; bảng tổng hợp mean ± std; confusion matrix, ROC và PR, F1 theo lớp, F1 theo tham số, F1 theo độ trễ, đường huấn luyện, biểu đồ thời gian huấn luyện
- **Tiêu chí kiểm tra:**
  1. Mọi hình và bảng sinh từ `results.csv` bằng script, chạy lại cho kết quả y hệt.
  2. Mọi số ghi mean ± std của 3 seed.
  3. Có ít nhất 8 loại biểu đồ.
  4. Bảng có đủ: accuracy, F1-macro, FPR, recall tấn công, AUC, tham số, kích thước, độ trễ.
- **Phụ thuộc:** 6.1

### 6.3 Kiểm định thống kê và phân tích lỗi

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 7 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Thuật toán & đánh giá
- **Đầu vào:** Dự đoán trên test của GNN chính và baseline tốt nhất; trọng số attention
- **Đầu ra:** McNemar test (GNN chính và baseline tốt nhất); phân tích các node sai: lớp nào nhầm với lớp nào, đặc điểm IP và cửa sổ sai; ví dụ attention trên 2 cửa sổ
- **Tiêu chí kiểm tra:**
  1. Có p-value và kết luận ở mức α = 0,05.
  2. Phân tích ≥ 3 nhóm lỗi, mỗi nhóm có ví dụ cụ thể.
  3. Nếu GNN không hơn baseline thì có phần giải thích nguyên nhân, liên hệ với homophily ở 2.2.
- **Phụ thuộc:** 6.1

### 6.4 So sánh với các nghiên cứu liên quan

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 7 · **Ưu tiên:** Nên có
- **Mục đề tài:** Thuật toán & đánh giá, Báo cáo
- **Đầu vào:** Bảng 3.4, bảng 6.2
- **Đầu ra:** Bảng đặt kết quả của nhóm cạnh các bài báo; đoạn bình luận
- **Tiêu chí kiểm tra:**
  1. Chỉ so trực tiếp khi cùng bộ dữ liệu và cùng mức phân loại; các trường hợp khác đánh dấu "không so trực tiếp".
  2. Không khẳng định "tốt hơn" khi chênh lệch nhỏ hơn std.
- **Phụ thuộc:** 6.2, 3.4

## GĐ7 AHP

### 7.1 Cây phân cấp AHP và ma trận so sánh cặp

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 7 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** AHP
- **Đầu vào:** Quyết định D6; lý thuyết 3.3
- **Đầu ra:** Cây: mục tiêu "chọn mô hình triển khai trên gateway IoT" → 5 tiêu chí (F1-macro, FPR, recall tấn công, độ trễ, số tham số hoặc bộ nhớ) → các phương án (mọi mô hình ở 6.1); ma trận 5×5
- **Tiêu chí kiểm tra:**
  1. Phúc chấm theo thang Saaty 1–9. Nên nhờ thêm Phong và Tấn chấm độc lập (chỉ mất khoảng 10 phút), rồi gộp bằng trung bình nhân.
  2. CR của từng ma trận cá nhân và của ma trận gộp đều < 0,10. Nếu vượt thì chấm lại, có ghi log.
  3. Có lý do bằng lời cho từng cặp so sánh.
- **Phụ thuộc:** 0.2

### 7.2 Tính AHP, xếp hạng và phân tích độ nhạy

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 7 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** AHP, Thuật toán & đánh giá
- **Đầu vào:** Ma trận 7.1; `results.csv` (`split=test`)
- **Đầu ra:** `src/ahp.py`; bảng trọng số tiêu chí; điểm từng phương án; xếp hạng; biểu đồ độ nhạy; mô hình được gợi ý
- **Tiêu chí kiểm tra:**
  1. Điểm phương án theo từng tiêu chí lấy từ số liệu thật, có chuẩn hóa và đảo chiều cho các tiêu chí càng nhỏ càng tốt (FPR, độ trễ, tham số).
  2. Kiểm tra lại `ahp.py` bằng ví dụ 3×3 ở 3.3.
  3. Độ nhạy: thay đổi từng trọng số ±20% và chỉ ra điểm làm thứ hạng đảo chiều.
  4. Kết luận kèm điều kiện, ví dụ "nếu ưu tiên độ trễ thì chọn …".
- **Phụ thuộc:** 7.1, 6.1

## GĐ8 Mô phỏng thời gian thực

### 8.1 src/stream_features.py

Gom cửa sổ online từ luồng flow.

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 6 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Triển khai
- **Đầu vào:** `build_graph.py`; file flow của tập test
- **Đầu ra:** `stream_features.py`: đọc flow theo thứ tự thời gian (replay theo tốc độ thật hoặc tăng tốc ×N), đóng cửa sổ, dựng đồ thị bằng đúng hàm offline
- **Tiêu chí kiểm tra:**
  1. Với cùng 1 cửa sổ, đặc trưng online và offline có max |sai khác| < 1e-6.
  2. Bộ nhớ không tăng theo thời gian chạy: chạy ≥ 30 phút dữ liệu, RAM ổn định ±10%.
  3. Dùng scaler đã fit trên train, không fit lại.
- **Phụ thuộc:** 0.4

### 8.2 detector.py và log cảnh báo

Bộ phát hiện thời gian thực.

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 7 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Triển khai
- **Đầu vào:** `stream_features.py`; mô hình được AHP gợi ý (dùng tạm GAT chính nếu 7.2 chưa xong)
- **Đầu ra:** `detector.py` (CLI): nhận file flow, xuất `alerts.csv` gồm thời gian, cửa sổ, IP, lớp dự đoán, xác suất, thời gian xử lý
- **Tiêu chí kiểm tra:**
  1. Một lệnh duy nhất chạy được demo, có hướng dẫn trong README.
  2. Dự đoán online khớp dự đoán offline trên cùng các cửa sổ ≥ 99,9%.
  3. Chạy CPU, không cần GPU.
- **Phụ thuộc:** 8.1, 5.3

### 8.3 Đo hiệu năng triển khai

Độ trễ, thông lượng, RAM.

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 7 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Triển khai, Thuật toán & đánh giá
- **Đầu vào:** `detector.py`; quyết định D7
- **Đầu ra:** Báo cáo hiệu năng: độ trễ p50/p95/p99 (tách riêng phần dựng đồ thị và phần suy luận), số flow/giây, RAM tối đa, cấu hình máy
- **Tiêu chí kiểm tra:**
  1. Đo trên ≥ 1.000 cửa sổ, có warm-up, `torch.set_num_threads(1)`.
  2. p95 end-to-end ≤ 1 s.
  3. Ghi đầy đủ CPU, RAM, hệ điều hành, phiên bản thư viện (và giới hạn CPU/RAM của Docker nếu dùng Docker).
  4. Có biểu đồ phân bố độ trễ và độ trễ theo số node của cửa sổ.
- **Phụ thuộc:** 8.2

### 8.4 Đánh giá kết quả triển khai

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 8 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Triển khai, Thuật toán & đánh giá
- **Đầu vào:** `alerts.csv`, nhãn thật, báo cáo 8.3
- **Đầu ra:** Mục "Đánh giá kết quả triển khai": số cảnh báo giả mỗi giờ, thời gian phát hiện (từ lúc đợt tấn công bắt đầu đến cảnh báo đầu tiên) cho từng đợt, so với chỉ tiêu
- **Tiêu chí kiểm tra:**
  1. Thời gian phát hiện tính cho mọi đợt tấn công trong tập test, có median và max.
  2. Số cảnh báo giả mỗi giờ tính trên các đoạn chỉ có Normal.
  3. Có bảng đạt hoặc không đạt cho từng chỉ tiêu triển khai.
- **Phụ thuộc:** 8.3

## GĐ9 Báo cáo & bảo vệ

### 9.1 Dàn ý báo cáo và phân công chương

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 4 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Báo cáo
- **Đầu vào:** KE_HOACH.md, các bản nháp lý thuyết, phân bổ chương ở [docs/plan/00_TONG_QUAN.md](../plan/00_TONG_QUAN.md) mục 7
- **Đầu ra:** Dàn ý: Ch1 Tổng quan; Ch2 Lý thuyết; Ch3 Dữ liệu, EDA, dựng đồ thị; Ch4 Mô hình và thực nghiệm; Ch5 Đánh giá, AHP, triển khai; Ch6 Kết luận
- **Tiêu chí kiểm tra:**
  1. Mỗi mục của mô tả đề tài ứng với ít nhất 1 mục trong dàn ý (có bảng ánh xạ).
  2. Mỗi chương ghi rõ các task và file trong repo cung cấp số liệu cho chương đó.
- **Phụ thuộc:** 0.1

### 9.2 Viết báo cáo hoàn chỉnh

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 8 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Báo cáo
- **Đầu vào:** Mọi đầu ra của GĐ1–GĐ8
- **Đầu ra:** Báo cáo (docx hoặc pdf) theo dàn ý 9.1
- **Tiêu chí kiểm tra:**
  1. Mọi con số trong báo cáo truy được về `results.csv` hoặc một file trong repo.
  2. Mọi hình có số thứ tự, chú thích và được nhắc đến trong văn bản.
  3. Tài liệu tham khảo theo 1 chuẩn trích dẫn thống nhất.
  4. Có mục hạn chế, ghi trung thực các chỉ tiêu không đạt.
- **Phụ thuộc:** 6.2, 7.2, 8.4

### 9.3 Slide, video demo và README

- **Trạng thái:** Chưa làm · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 9 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Báo cáo, Triển khai
- **Đầu vào:** Báo cáo 9.2, `detector.py`
- **Đầu ra:** Slide 15–20 trang; video demo 2–3 phút chạy `detector.py`; README từ cài đặt đến tái tạo bảng kết quả
- **Tiêu chí kiểm tra:**
  1. Một người ngoài nhóm clone repo và làm theo README tái tạo được bảng kết quả (lệch ≤ 0,5 điểm %).
  2. Tập dượt thuyết trình ≥ 1 lần, đúng thời lượng quy định.
  3. Chuẩn bị sẵn câu trả lời cho: vì sao GNN, vì sao bộ dữ liệu này, vì sao AHP, giả định thời gian.
- **Phụ thuộc:** 9.2
