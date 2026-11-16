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
| [2.1](#21-eda-mức-flow) | EDA mức flow | GĐ2 | Phúc | Tự kiểm | 3 | B | **Xong** |
| [2.2](#22-eda-mức-đồ-thị) | EDA mức đồ thị, homophily | GĐ2 | Phúc | Tự kiểm | 3 | B | **Xong, có ghi chú** (bậc không chứng minh fan-in ở mức node) |
| [2.3](#23-trực-quan-hóa-đồ-thị-các-cửa-sổ-mẫu) | Trực quan hóa đồ thị | GĐ2 | Phúc | Tự kiểm | 3 | B | **Xong, có ghi chú** (scan BoT fan-out yếu) |
| [3.1](#31-lý-thuyết-ids-cho-iot-các-loại-tấn-công-netflow) | Lý thuyết IDS, tấn công, NetFlow | GĐ3 | Phúc | Tự kiểm | 3 | B | Chưa làm |
| [3.2](#32-lý-thuyết-gnn-tại-sao-chọn-và-còn-mô-hình-nào-khác) | Lý thuyết GNN | GĐ3 | Phúc | Tự kiểm | 3 | B | Chưa làm |
| [3.3](#33-lý-thuyết-baseline-ml-chỉ-số-đánh-giá-và-ahp) | Lý thuyết baseline, chỉ số, AHP | GĐ3 | Phúc | Tự kiểm | 4 | B | Chưa làm |
| [3.4](#34-bảng-tổng-quan-nghiên-cứu-liên-quan) | Tổng quan nghiên cứu liên quan | GĐ3 | Phúc | Tự kiểm | 4 | B | Chưa làm |
| [4.1](#41-baseline-ml-trên-đặc-trưng-node) | Baseline ML trên đặc trưng node | GĐ4 | Phúc | Tự kiểm | 4 | B | **Xong** |
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

- **Trạng thái:** Xong · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 3 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** EDA & tiền xử lý
- **Đầu vào:** `clean_<bộ>.parquet`
- **Đầu ra:** `notebooks/02_eda_flow.ipynb`; ≥ 6 hình mỗi bộ: phân bố lớp, boxplot log đặc trưng theo lớp, heatmap tương quan, mutual information, phân bố theo thời gian, giao thức và cổng
- **Tiêu chí kiểm tra:**
  1. Mỗi hình có tiêu đề, nhãn trục, đơn vị và 1–2 câu nhận xét ngay bên dưới.
  2. Có bảng tóm tắt thống kê (min, median, max, % thiếu) cho mọi đặc trưng số.
  3. Chỉ ra ≥ 3 đặc trưng phân biệt lớp tốt nhất kèm số liệu MI.
  4. Run All trên Kaggle.
- **Phụ thuộc:** 1.6
- **Kết quả:** (13/11/2026) Notebook `gnn-iot-ids-12-eda` (Kaggle CPU 4 nhân, bản 4, clone `main` tại `d2e4d1a`), chạy 575 s, RAM đỉnh 4,23 GB, `pytest tests/test_eda.py` 5 passed. Bản 1 là chạy thử (`--smoke`); bản 2 và 3 chạy đủ rồi sửa hình sau khi mở xem (trục thời gian BoT, layout cửa sổ, nhãn trục đè nhau). Code: [scripts/eda.py](../../scripts/eda.py). Hình: `reports/figures/gd2_<bộ>_*.png`. Số liệu và nhận xét: [reports/kaggle/gnn-iot-ids-12-eda/](../../reports/kaggle/gnn-iot-ids-12-eda/) (`eda_summary.json`, `run_info.json`, `gd2_<bộ>_flow_stats.csv`). Mỗi bộ có 5 hình mức flow (phân bố nhãn gốc và 5 lớp, số flow theo thời gian, boxplot log(1 + x) của 8 đặc trưng theo lớp, heatmap Spearman, mutual information) và 1 bảng thống kê. CSV gốc được đọc theo khối bằng pyarrow hai lượt: số nhãn, số flow theo phút, min, max, % thiếu tính trên toàn bộ file (27.520.260 và 16.933.808 flow); median và tương quan lấy từ mẫu đều 2 %; boxplot và MI lấy từ mẫu cân bằng theo nhãn gốc (tối đa 30.000 và 5.000 flow mỗi nhãn). EDA chỉ mô tả, không dùng để chọn cấu hình mô hình.
  1. **Đạt, có ghi chú.** 26 hình đều dpi 160, có tiêu đề, nhãn trục kèm đơn vị (script tự kiểm: `axis_label_problems` rỗng cho mọi hình). Đã mở xem 13 hình của cả hai bộ cùng sơ đồ quy trình để kiểm chữ có đè không (ba vòng sửa, bản 4 không còn đè); các hình còn lại chưa mở. Nhận xét 1–2 câu nằm trong trường `comment` của `eda_summary.json` (sinh từ chính số liệu), không in dưới hình.
  2. **Đạt.** [gd2_nf_ton_iot_v3_flow_stats.csv](../../reports/kaggle/gnn-iot-ids-12-eda/gd2_nf_ton_iot_v3_flow_stats.csv) và [gd2_nf_bot_iot_v3_flow_stats.csv](../../reports/kaggle/gnn-iot-ids-12-eda/gd2_nf_bot_iot_v3_flow_stats.csv): 49 đặc trưng số mỗi bộ, có min, median, max, % thiếu. ToN không thiếu giá trị nào; BoT có 2 đặc trưng thiếu 6,213 % (`SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_SECOND_BYTES`); không đặc trưng nào có giá trị âm.
  3. **Đạt.** MI với nhãn 5 lớp. NF-ToN-IoT-v3: `IN_BYTES` 1,106, `SRC_TO_DST_AVG_THROUGHPUT` 0,972, `OUT_BYTES` 0,954 nat. NF-BoT-IoT-v3: `IN_BYTES` 0,914, `SRC_TO_DST_AVG_THROUGHPUT` 0,892, `LONGEST_FLOW_PKT` 0,846 nat.
  4. **Đạt (Kaggle).** Notebook chạy hết; ô kiểm tra cuối (≥ 12 hình, dpi ≥ 150, < 5 MB, đủ nhận xét, đủ nhãn trục) báo `problems` rỗng.
  Ghi chú: mục "Đầu ra" ghi ≥ 6 hình mỗi bộ gồm cả hình giao thức và cổng. Theo yêu cầu GĐ C chỉ làm 5 hình mức flow cộng bảng thống kê, nên **không có hình riêng cho giao thức và cổng** (`L4_DST_PORT` có trong boxplot và MI). Cần hình này thì thêm một hàm vào `scripts/eda.py`. Đầu vào là CSV gốc `athena21/netflow-v3-datasets`, không phải `clean_<bộ>.parquet`, nên số flow lớn hơn số sau làm sạch (ToN 27,5 triệu so với 25,7 triệu).

### 2.2 EDA mức đồ thị

Kích thước, bậc, homophily.

- **Trạng thái:** Xong · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 3 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** EDA & tiền xử lý
- **Đầu vào:** `graphs_<bộ>.pt`
- **Đầu ra:** `notebooks/02_eda_graph.ipynb`; hình phân bố số node và cạnh mỗi cửa sổ, bậc vào/ra theo lớp; bảng homophily
- **Tiêu chí kiểm tra:**
  1. Có edge homophily và class-insensitive homophily cho từng bộ.
  2. Có nhận xét homophily cao hay thấp nghĩa là gì cho GNN (dùng làm luận cứ ở chương lý thuyết và phân tích lỗi).
  3. Có so sánh bậc vào/ra của lớp DDoS với Normal để chứng minh dấu hiệu fan-in và fan-out.
- **Phụ thuộc:** 1.6
- **Kết quả:** (13/11/2026) Cùng notebook `gnn-iot-ids-12-eda` bản 4 (commit `d2e4d1a`) như task 2.1. Hình: `gd2_<bộ>_graph_size.png`, `gd2_<bộ>_degree_by_class.png`, `gd2_<bộ>_homophily.png`; bảng: [gd2_homophily.csv](../../reports/kaggle/gnn-iot-ids-12-eda/gd2_homophily.csv) và mục `homophily` trong `eda_summary.json` (có cả số theo train, val, test và ma trận trộn lớp). Tính trên toàn bộ đồ thị đã dựng (ToN 66.693, BoT 13.937 cửa sổ). Kích thước: ToN median 11 node và 11 cạnh mỗi cửa sổ, p99 68 và 79, lớn nhất 269 node; BoT median 5 và 4, p99 11 và 11, lớn nhất 39 node.
  1. **Đạt.** NF-ToN-IoT-v3: edge homophily **0,671** (mức ngẫu nhiên 0,474), class-insensitive **0,555**, adjusted 0,427. NF-BoT-IoT-v3: edge **0,861** (ngẫu nhiên 0,496), class-insensitive **0,750**, adjusted 0,726. Ba tập train, val, test lệch nhau không quá 0,07 (ToN class-insensitive 0,593 / 0,522 / 0,593; BoT 0,733 / 0,791 / 0,771).
  2. **Đạt.** Ý nghĩa với GNN (câu ghi trong `eda_summary.json`): phần lớn cạnh nối node cùng lớp, cao hơn rõ rệt mức ngẫu nhiên, nên gom thông điệp láng giềng có thể giúp GNN làm mượt nhãn; nhưng 32,9 % cạnh ở ToN và 13,9 % ở BoT vẫn nối hai lớp khác nhau (chủ yếu tấn công ↔ Normal), nên láng giềng không thay được đặc trưng riêng của từng node. Đây là luận cứ cho hybrid giữ đặc trưng node, và để đọc kết quả GNN so với baseline ở GĐ F–G. Lớp Botnet_Backdoor ở ToN có h_k thấp nhất (0,476, dưới cả mức Normal 0,725).
  3. **Đã so sánh, nhưng kết quả không chứng minh fan-in/fan-out ở mức node.** Bậc là số cặp src→dst trong cửa sổ. NF-ToN-IoT-v3: bậc vào trung bình DoS_DDoS 1,52 so với Normal 1,08 (gấp 1,40 lần; 1,34 % so với 1,12 % node có bậc ≥ 5); bậc ra trung bình Truy_cap_trai_phep 1,24 so với 1,13 (gấp 1,09 lần; 8,33 % so với 2,77 % node có bậc ≥ 5, lớn nhất 229). NF-BoT-IoT-v3: bậc vào DoS_DDoS 0,62 so với Normal 1,02 (chỉ bằng 0,61 lần); bậc ra Truy_cap_trai_phep 0,87 so với 0,76 (1,14 lần). Tức fan-in và fan-out chỉ xuất hiện ở một số ít cửa sổ, nhìn thấy ở hình cửa sổ mẫu (task 2.3), chứ không làm lệch phân bố bậc của cả lớp. Bậc không nên được coi là đặc trưng tách lớp mạnh. Cần nói đúng như vậy trong báo cáo.

### 2.3 Trực quan hóa đồ thị các cửa sổ mẫu

- **Trạng thái:** Xong · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 3 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** EDA & tiền xử lý, Báo cáo
- **Đầu vào:** `graphs_<bộ>.pt`
- **Đầu ra:** ≥ 3 hình đồ thị (networkx hoặc pyvis): cửa sổ bình thường, DDoS, scan hoặc backdoor; node tô màu theo nhãn, kích thước theo bậc
- **Tiêu chí kiểm tra:**
  1. Mỗi hình ≤ 300 node để còn đọc được, có chú thích màu.
  2. Có cùng layout seed để so sánh được.
  3. Nhìn thấy rõ cấu trúc đặc trưng: hình sao fan-in của DDoS, fan-out của scan; có chú thích chỉ vào các cấu trúc này.
  4. Có thêm 1 hình biểu diễn quy trình flow → cửa sổ → đồ thị để đưa vào báo cáo.
- **Phụ thuộc:** 1.6
- **Kết quả:** (13/11/2026) Cùng notebook `gnn-iot-ids-12-eda` bản 4 (commit `d2e4d1a`). Hình: `gd2_<bộ>_window_<loại>.png` (ToN: normal, ddos, scan, backdoor; BoT: normal, ddos, scan), `gd2_<bộ>_windows_overview.png` (các cửa sổ cạnh nhau) và `gd2_chung_quy_trinh.png`. Cửa sổ lấy từ tập train, chọn bằng quy tắc cố định ghi trong `eda_summary.json` (trường `criterion`): Normal gần median kích thước; DDoS và scan là cửa sổ ≤ 300 node có hub lớn nhất (ghi thứ hạng so với các cửa sổ cùng loại).
  1. **Đạt.** Lớn nhất 237 node (ToN, cửa sổ scan); chú thích màu theo lớp ở chân hình.
  2. **Đạt.** Mọi hình dùng seed 42 (`spring_layout` theo từng thành phần liên thông, thành phần lớn ở giữa), cùng bảng màu, cùng công thức kích thước node ∝ √(bậc vào + bậc ra).
  3. **Đạt một phần.** Có chú thích mũi tên chỉ vào node trung tâm. ToN: fan-in rõ (DDoS: 13 IP cùng gửi tới 1 IP, 39 node) và fan-out rất rõ (scan: 1 IP gửi tới 229 IP, 237 node). BoT: fan-in DDoS rõ (8 IP → 1 IP, 10 node) nhưng fan-out của scan yếu (1 IP → 5 IP, 10 node). Lưu ý: cửa sổ Normal cũng có hình sao (một node nối tới 100 % node còn lại, ở cả hai bộ), nên hình dạng một mình không phân biệt được; khác biệt nằm ở màu lớp và quy mô. Cửa sổ DDoS và scan là các ví dụ cực đoan (hub lớn nhất), không phải cửa sổ điển hình.
  4. **Đạt.** `gd2_chung_quy_trinh.png` (flow → cửa sổ 5 s → đồ thị), có số liệu thật của hai bộ ở chân hình (ví dụ flow trong hình là minh họa).

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

- **Trạng thái:** Xong · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 4 · **Ưu tiên:** Bắt buộc
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
    6. **Đạt (Phúc chấp nhận 14/11/2026).** `fpr`, `recall_attack`, `roc_auc` đã là chỉ số nhị phân suy ra từ dự đoán đa lớp (Normal là lớp âm, xem `src/evaluate.py`), có ở mọi dòng của cả 2 bộ. Không huấn luyện mô hình nhị phân riêng, không có dòng `task=binary`; nếu cần precision hoặc F1 của lớp "tấn công" gộp thì tính lại từ dự đoán ở GĐ H.
  - **Đủ cả 2 bộ (NF-ToN-IoT-v3 từ notebook 05, NF-BoT-IoT-v3 từ notebook 13), 6/6 tiêu chí đạt → Xong.** Test chưa chạy (để GĐ H).

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

- **Trạng thái:** Xong · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 5 · **Ưu tiên:** Bắt buộc
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
- **Kết quả GĐ G (16/11/2026, GĐ G, bù tiêu chí 3 và 4):** tiêu chí 1, 2 như trên; **tiêu chí 3 đạt:** log tinh chỉnh trên val, 3 seed, chỉ val, mỗi mô hình ≤ 20 lần thử (GCN 6, GraphSAGE 6: độ rộng 32, 64, 128 × 2, 3 lớp; GAT 10, xem 5.5), cả hai bộ, trong `tuning_gnn.csv` ([reports/kaggle/gnn-iot-ids-16c-aggregate/](../../reports/kaggle/gnn-iot-ids-16c-aggregate/)); chạy ở notebook `gnn-iot-ids-16g-tuning-ton` (ToN, CPU, 7244 s, commit `ad92f6a`, pytest 110 passed), `gnn-iot-ids-16a-ablation-bot` (BoT, CPU, 2900 s) và `gnn-iot-ids-16-ablation` (GAT ToN, GPU). Cấu hình mặc định gần tốt nhất: F1-macro val tốt nhất ToN GCN 0,8797 (hidden 16, 3 lớp; mặc định 0,8726), GraphSAGE 0,9152 (hidden 32, 3 lớp, 71.173 tham số; mặc định 0,9146); BoT GCN 0,9147 (mặc định), GraphSAGE 0,9353 (hidden 32, 3 lớp). **Tiêu chí 4 đạt cho ToN:** đường loss train, loss val, F1-macro val theo epoch của mọi lần chạy (kể cả cấu hình mặc định) ở `curves_grid_nf_ton_iot_v3.csv` trong [16g](../../reports/kaggle/gnn-iot-ids-16g-tuning-ton/) và [16](../../reports/kaggle/gnn-iot-ids-16-ablation/). **Cả bốn tiêu chí đạt trên cả hai bộ → `Xong`.**

### 5.3 Mô hình chính: GAT nhẹ multi-head

Có đặc trưng cạnh.

- **Trạng thái:** Xong (có ghi chú: FPR val và recall tấn công không đạt ngưỡng, có phân tích) · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 5 · **Ưu tiên:** Bắt buộc
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
  - **Phân tích FPR (14/11/2026, GĐ F phần 1, tiêu chí 2; notebook `gnn-iot-ids-15a-fpr-analysis` bản 1, Kaggle CPU, clone commit `e1cc0fb`, pytest 9 passed, 656 s, RAM đỉnh 19,61 GB, 0 giờ GPU).** Kéo về [reports/kaggle/gnn-iot-ids-15a-fpr-analysis/](../../reports/kaggle/gnn-iot-ids-15a-fpr-analysis/): `fpr_analysis_<bộ>.json`, 8 hình `gd5_fpr_<bộ>_{ip,purity,centroid,relabel}.png`, `run_info.json`. Mô hình: LightGBM (cấu hình tốt nhất của baseline_search, seed 42, huấn luyện lại trên train) và GAT (checkpoint seed 42), chỉ train và val, không đọc test. Nhãn dựng lại từ flow gốc khớp nhãn đồ thị 100% ở cả hai bộ. Mọi số dưới đây là trên val.
    - **NF-BoT-IoT-v3** (10.626 node val, 2.918 Normal; FPR LightGBM 19,67%, GAT 20,84%): (1) FP dồn vào 7 IP trong 169 IP có node Normal; 4 IP 192.168.100.147–150 mỗi IP chiếm khoảng 20% số FP với 62–66% node Normal của chính IP đó bị báo nhầm, thêm 192.168.100.3 (23–25% node Normal của IP bị báo nhầm) và .7 (6–10%); **100% FP thuộc IP cũng là node tấn công ở cửa sổ khác** (chỉ 4,7% IP có node Normal thuộc loại này); toàn bộ train+val chỉ có 8 IP làm node tấn công trong 249 IP. (2) Nhãn rất thuần: 98,4% (LightGBM) / 96,5% (GAT) node FP có mọi flow đều Normal; độ thuần của nhãn đa số 99,6%; 15% node FP lấy nhãn từ flow nhận (TN: 44%). Không phải nhiễu nhãn. (3) 83–84% node FP có tâm lớp gần nhất là Truy_cap_trai_phep (chỉ 16% gần Normal), trong khi 71% node Normal đoán đúng gần tâm Normal. (4) Mọi cách gán thay thế đều làm FPR **tệ hơn** (phản thực tế trên dự đoán đã có): đa số mọi flow 21,3%, chỉ flow khởi tạo 26,8%, có ≥ 1 flow tấn công 21,2% (LightGBM; GAT 22,1 / 28,7 / 22,1%). **Kết luận: FPR ~20% của BoT do dữ liệu, không do cách gán nhãn:** các IP của máy tấn công cũng sinh ra phần flow Normal (Benign chỉ 0,3% số flow) có đặc trưng cấp node giống hệt đặc trưng lúc tấn công, nên đặc trưng 18 chiều không tách được; mọi mô hình kẹt cùng một mức. Không đề xuất đổi nhãn cho BoT.
    - **NF-ToN-IoT-v3** (121.599 node val, 88.461 Normal; FPR LightGBM 7,92%, GAT 7,31%): (1) FP rải rộng hơn (190 và 215 IP có FP), top-1 IP chiếm 14% (LightGBM) / 18% (GAT), top-3 33% / 40%; 86% (LightGBM) / 90% (GAT) FP thuộc IP cũng là node tấn công ở cửa sổ khác (chỉ 2,5% IP có node Normal thuộc loại này). (2) **Node FP phần lớn là node chỉ nhận flow, không khởi tạo flow nào: 92,5% (LightGBM) / 80,2% (GAT) node FP có nhãn lấy từ flow nhận, so với 41,7% / 43,0% của node Normal đoán đúng**; nhãn thuần theo nghĩa 77% / 72% node FP có mọi flow đều Normal. (3) Node FP gần tâm Khac (53% / 39%) và Botnet_Backdoor (24% / 30%), chỉ 4,9% / 7,3% gần tâm Normal (TN: 51% gần Normal). (4) Phản thực tế: nếu **chỉ gán nhãn node có flow khởi tạo và bỏ node chỉ nhận khỏi đánh giá**, FPR của dự đoán đã có giảm từ 7,92% xuống 1,10% (LightGBM) và từ 7,31% xuống 2,67% (GAT), recall tấn công 94,4% / 94,7% (trước 95,5% / 96,1%), nhưng bỏ 59.273 trong 121.599 node val (48,7%). Hai cách còn lại không giúp: đa số mọi flow FPR 10,6% / 9,8% kèm recall tụt còn 78–79%; có ≥ 1 flow tấn công FPR 7,4% / 6,3% kèm recall tụt còn 69–70%.
    - **Đề xuất cần Phúc quyết (chưa làm gì):** với ToN, nguyên nhân chính của FPR là node chỉ nhận flow (không khởi tạo) bị đoán là tấn công; chỉ gán nhãn node khởi tạo sẽ đưa FPR xuống 1–3% nhưng đổi định nghĩa nhiệm vụ (bỏ gần một nửa số node) và kéo theo phải dựng lại đồ thị, chạy lại mọi mô hình ở cả hai bộ. Đây chỉ là con số phản thực tế, chưa huấn luyện lại. Cách gán nhãn của dữ liệu **chưa bị đổi**; BoT không đổi được gì bằng cách này.
  - **Thí nghiệm đổi nhãn "chỉ node có khởi tạo flow" (15/11/2026, notebook `gnn-iot-ids-15h-label-pilot` bản 1, CPU, commit `2c600be`, 267 s, 0 giờ GPU; [reports/kaggle/gnn-iot-ids-15h-label-pilot/](../../reports/kaggle/gnn-iot-ids-15h-label-pilot/)):** huấn luyện lại LightGBM (cấu hình tốt nhất của baseline_search, seed 42) chỉ trên node train có khởi tạo flow và đánh giá trên node val có khởi tạo; mặt nạ suy từ đặc trưng `n_out`, không đổi dữ liệu. **NF-ToN-IoT-v3** (val: 62.326/121.599 node có khởi tạo = 51,3%, 59.273 node chỉ nhận bị bỏ; trong số 33.138 node tấn công của val chỉ 14.285 có khởi tạo, 18.853 là node chỉ nhận bị gán nhãn tấn công từ flow nhận): LightGBM cũ (huấn luyện trên mọi node) đánh giá trên cùng node có khởi tạo đã cho F1-macro 0,9674, FPR **1,10%**, recall 94,4%; LightGBM huấn luyện lại chỉ trên node có khởi tạo cho F1-macro 0,9679, FPR **1,01%**, recall 94,8%. Trên mọi node val, LightGBM cũ có F1-macro 0,8984, FPR 7,92%. **NF-BoT-IoT-v3** (75,5% node val có khởi tạo): LightGBM cũ trên node có khởi tạo FPR 26,8%, huấn luyện lại 27,0% (trên mọi node: 19,67%), tức đổi nhãn làm FPR xấu đi. **Kết luận: FPR giảm ở ToN là do thu hẹp phạm vi đánh giá, không phải do mô hình tốt hơn.** Huấn luyện lại chỉ trên node có khởi tạo gần như không đổi gì (FPR 1,10% → 1,01%); mô hình cũ đã phân loại tốt các node có khởi tạo, chỗ yếu là các node chỉ nhận (nạn nhân, máy chủ). Đổi nhãn nghĩa là hệ thống ngừng phát hiện 57% node tấn công của val (18.853/33.138, các node nạn nhân) và đổi định nghĩa bài toán, kéo theo dựng lại đồ thị và chạy lại mọi mô hình ở cả hai bộ (khoảng 1–1,5 giờ GPU nữa) mà BoT còn xấu đi. **Quyết định (Claude đề xuất, theo yêu cầu chi phí thấp của Phúc): KHÔNG đổi cách gán nhãn.** Thay vào đó, GĐ H báo cáo thêm số liệu theo tầng (node có khởi tạo và node chỉ nhận) tính từ chính dự đoán đã có, không huấn luyện lại, để nêu rõ FPR 1% ở node khởi tạo và điểm yếu ở node chỉ nhận.
  - CIC và NF-ToN-IoT-v2 đủ gcn, sage, gat × 3 seed, chỉ làm phụ lục (D8).
  - **Phần còn thiếu đã bù (16/11/2026, GĐ G, notebook `gnn-iot-ids-16d-ablation-ton` và `gnn-iot-ids-16a-ablation-bot`, CPU, commit `ad92f6a`):**
    - **Tiêu chí 1 đạt:** bảng đếm tham số theo từng lớp ([16d](../../reports/kaggle/gnn-iot-ids-16d-ablation-ton/): `gat_param_table_nf_ton_iot_v3.csv`, `gat_param_summary_nf_ton_iot_v3.json`; BoT ở [16a](../../reports/kaggle/gnn-iot-ids-16a-ablation-bot/)). ToN **11.717 tham số** = lớp GAT 1 (`convs.0`) 2.752 + lớp GAT 2 (`convs.1`) 8.640 + đầu phân loại (`head`) 325; mỗi lớp GAT gồm `lin_l`, `lin_r` (hai ma trận chiếu), `lin_edge` (đặc trưng cạnh, 3 → 64), `att` và `bias`. File 50,6 KB (≤ 1 MB). BoT 11.587 tham số, 50,1 KB.
    - **Tiêu chí 2:** val ToN (3 seed) accuracy 92,18% (đạt ≥ 91%), FPR 8,44% (**không đạt ≤ 4,08%**), recall tấn công 95,98% (đạt ≥ 95%); BoT accuracy 93,1%, FPR 21,03% (không đạt), recall 98,49%. Phân tích FPR đã có từ GĐ F (notebook 15a, ở trên), nên điều kiện "phải có phân tích trước khi sang 5.4" đạt. Sau hiệu chỉnh ngưỡng trên val (task 5.8), FPR về 4,08% nhưng recall tụt còn 89,0% (ToN) và 34,9% (BoT).
    - **Tiêu chí 3 đạt:** std accuracy giữa 3 seed ToN 0,726 điểm %, BoT 0,054 điểm % (≤ 1,5).
    - **Tiêu chí 4 đạt:** trọng số attention của 2 cửa sổ val mẫu, GAT seed 42, `attention_samples_nf_ton_iot_v3.json` (21,5 KB): cửa sổ 311226701 (15 node, 20 cạnh, mọi node DoS_DDoS, đoán đúng 15/15) và cửa sổ 311267371 (12 node, 13 cạnh, 10 node Truy_cap_trai_phep và 2 node Normal, đoán đúng 12/12). Mỗi cửa sổ lưu IP, nhãn, dự đoán và với từng lớp GAT cạnh (gồm cạnh ngược và vòng lặp tự thân đúng như `forward`) cùng hệ số attention từng head (tổng theo node nhận bằng 1). Nhóm "Truy_cap_trai_phep" gồm dò quét (scanning) theo ánh xạ nhãn, nhưng đồ thị chỉ lưu nhãn nhóm nên chưa phân biệt dò quét với tấn công cùng nhóm; GĐ H vẽ hình từ file này. BoT có file tương ứng ở 16a.
    - **Trạng thái `Xong` có ghi chú:** mọi tiêu chí kiểm tra được, tiêu chí 2 chỉ đạt về accuracy và recall (FPR không đạt, có phân tích).

### 5.4 Phiên bản cải tiến của GAT nhẹ

- **Trạng thái:** Xong (có ghi chú: không bản nào được công nhận, báo cáo là thử nghiệm thất bại) · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 6 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Mô hình, Thuật toán & đánh giá
- **Đầu vào:** `gat_lite.py`, kết quả 5.3
- **Đầu ra:** Ít nhất 1 bản cải tiến, chọn trong: focal loss hoặc class-balanced loss; residual + LayerNorm; đặc trưng thời gian (ghép embedding cửa sổ trước); quantization INT8 hoặc pruning
- **Tiêu chí kiểm tra:** Bản cải tiến chỉ được công nhận khi đạt một trong hai điều kiện trên val:
  1. Tăng F1-macro hoặc recall lớp Botnet_Backdoor lớn hơn 2 lần std của 3 seed; hoặc
  2. Giảm ≥ 30% tham số hoặc độ trễ mà F1-macro giảm ≤ 1 điểm %.

  Nếu không đạt thì vẫn báo cáo là thử nghiệm thất bại.
- **Phụ thuộc:** 5.3
- **Kết quả (16/11/2026, GĐ G, notebook `gnn-iot-ids-16-ablation` bản 3, Kaggle GPU Tesla T4, commit `5031ab5`, pytest 109 passed (bỏ qua 1 test tái lập hai lần chạy không ổn định trên GPU; test đó pass trên CPU ở 16t, 110 passed), 3841 s = 1,067 giờ GPU, RAM đỉnh 2,02 GB; [reports/kaggle/gnn-iot-ids-16c-aggregate/](../../reports/kaggle/gnn-iot-ids-16c-aggregate/): `improved_gat.csv`):** trên NF-ToN-IoT-v3, val, 3 seed, so với GAT cấu hình chính (F1-macro 0,9065 ± 0,0019, recall Botnet_Backdoor 0,9994, 11.717 tham số). Đã thử 3 bản (đều chỉ đổi một yếu tố hoặc cặp yếu tố so với cấu hình chính), chưa thử đặc trưng thời gian và quantization/pruning (task cho phép chọn một số trong danh sách):

  | Bản | F1-macro | Δ so với chính | Recall Botnet_Backdoor | Tham số | Tiêu chí 1 (> 2 std) | Tiêu chí 2 (giảm ≥ 30% tham số hoặc độ trễ, F1 giảm ≤ 1 điểm %) |
  |---|---|---|---|---|---|---|
  | focal loss (γ = 2) | 0,8940 ± 0,0007 | −0,0125 | 0,9994 (−0,0001) | 11.717 | không | không (không giảm tham số) |
  | residual + LayerNorm | 0,9113 ± 0,0027 | +0,0048 | 0,9987 (−0,0007) | 13.125 (+12%) | **không**: +0,0048 < 2 × 0,0027 = 0,0054 | không (tham số tăng) |
  | focal + residual + LayerNorm | 0,8992 ± 0,0063 | −0,0073 | 0,9989 (−0,0006) | 13.125 | không | không |

  **Kết luận: không bản nào đạt tiêu chí của 5.4 → không công nhận, báo cáo là thử nghiệm thất bại; không đưa vào hybrid** (không cần báo Phúc để quyết đưa vào hybrid). Residual + LayerNorm gần đạt (tăng F1-macro 0,0048, chưa tới 2 lần std 0,0054 với std = max của hai bên, quy ước `compare_val.py`). Recall Botnet_Backdoor đã gần 1 nên không còn chỗ để tăng. Quan sát ngoài tiêu chí (không tính là cải tiến theo 5.4): một số cấu hình của ablation 5.5 thỏa điều kiện "giảm ≥ 30% tham số mà F1-macro giảm ≤ 1 điểm %" (1 lớp: −74% tham số, F1 −0,0023; hidden 8: −67%, −0,0050; 2 head: −67%, −0,0092), nhưng đó là cấu hình ablation chứ không phải bản cải tiến đề xuất, và chưa được kiểm trong hybrid; nêu ra để Phúc quyết có coi là bản GAT nhẹ hơn hay không. Ngưỡng của 3 bản vẫn được ghi trong `thresholds.json` với `accepted_by_5_4 = false`.

### 5.5 Ablation

Số head, số lớp, hidden, đặc trưng cạnh, kích thước cửa sổ.

- **Trạng thái:** Xong · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 6 · **Ưu tiên:** Bắt buộc
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
- **Kết quả (16/11/2026, GĐ G; [reports/kaggle/gnn-iot-ids-16c-aggregate/](../../reports/kaggle/gnn-iot-ids-16c-aggregate/): `ablation_gat.csv`, `ablation_gat_conclusion.json`, 5 hình `gd5_ablation_gat_{heads,layers,hidden,edge,window}.png`):** NF-ToN-IoT-v3, val, 3 seed, đổi một yếu tố mỗi lần, mỗi hình có F1-macro (± std) ở trên và số tham số (hoặc độ trễ p95 với cửa sổ) ở dưới. Lưới GAT chạy ở `gnn-iot-ids-16-ablation` (GPU, 10 cấu hình × 3 seed, 1,067 giờ GPU); các đồ thị W = 1, 2, 10 giây dựng ở `gnn-iot-ids-16b-windows` (CPU, 1421 s, RAM đỉnh 18,79 GB, mọi kiểm tra đạt, [reports/kaggle/gnn-iot-ids-16b-windows/](../../reports/kaggle/gnn-iot-ids-16b-windows/)) và huấn luyện ở `gnn-iot-ids-16f-windows-gat` (GPU, 1713 s = 0,476 giờ). Cách chia của W = 1, 2, 10: `blocked`, 10 khối, theo node, gap 1, như W = 5; số node val khác nhau giữa các W nên F1 giữa các W so sánh được ở mức gần đúng.

  | Yếu tố | Giá trị → F1-macro val (số tham số) |
  |---|---|
  | head | 1: 0,8537 (1.397) · 2: 0,8973 (3.813) · **4 (chính): 0,9065 ± 0,0019 (11.717)** · 8: **0,9153 ± 0,0036 (39.813)** |
  | số lớp | 1: 0,9042 (3.077) · **2 (chính): 0,9065** · 3: 0,9034 ± 0,0089 (20.357) |
  | độ rộng mỗi head | 8: 0,9015 (3.813) · **16 (chính): 0,9065** · 32: 0,9078 (39.813) · 64: 0,9074 (145.157, vượt 100K) |
  | đặc trưng cạnh | **có (chính): 0,9065 (11.717)** · không: 0,9076 ± 0,0067 (11.333) |
  | cửa sổ (giây) | 1: 0,9011 · 2: 0,8985 · **5 (chính): 0,9065** · 10: 0,8866 ± 0,0078 (cùng 11.717 tham số; p95 1,9–2,1 ms) |

  **Kết luận về cấu hình chính (hidden 16, 4 head, 2 lớp, có đặc trưng cạnh, W = 5 s):** hợp lý ở số lớp (3 lớp không tốt hơn, 1 lớp kém 0,0023 nhưng ít hơn 74% tham số), độ rộng (32 và 64 chỉ hơn trong sai số), cửa sổ (5 giây tốt nhất trong 4 mức; W = 2 và 1 thấp hơn, W = 10 thấp nhất) và đặc trưng cạnh (có hay không đều trong sai số, nên đặc trưng cạnh **không** được dữ liệu ủng hộ rõ rệt). **Chưa tối ưu ở số head:** 8 head cao hơn 4 head 0,0089 (lớn hơn 2 lần std) với 39.813 tham số (vẫn ≤ 100K). Mình không đổi cấu hình chính vì hybrid (D13) và mọi kết quả khác đã dựng trên nó; đây là điểm Phúc quyết (xem `freeze_list.md` mục 4).

### 5.6 Các bộ còn lại và tổng quát hóa giữa các bộ

- **Trạng thái:** Xong (có ghi chú: tổng quát hóa giữa hai bộ rất kém) · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 6 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Thuật toán & đánh giá, Bộ dữ liệu
- **Đầu vào:** Cấu hình chính đã đóng băng; đồ thị của các bộ phụ
- **Đầu ra:** Kết quả trên từng bộ phụ (train và test trên cùng bộ); ma trận train trên bộ A, test trên bộ B (chỉ dùng các lớp chung)
- **Tiêu chí kiểm tra:**
  1. Mọi bộ dùng chung 18 đặc trưng và cùng ánh xạ nhãn.
  2. Phần chéo giữa các bộ chỉ tính trên các lớp có ở cả hai bộ, có ghi rõ.
  3. Kết quả ghi vào `results.csv` với cột `dataset` dạng "A→B".
- **Phụ thuộc:** 5.3
- **Kết quả:** (16/11/2026) Cùng notebook 17 (xem 6.1): hybrid chính, gat, lightgbm, lgbm_neighbor_mean huấn luyện trên bộ A, đánh giá trên **tập test** bộ B, 3 seed, hai chế độ ngưỡng; ghi trong `results_test_<bộ nguồn>.csv` với cột `dataset` = `nf_ton_iot_v3→nf_bot_iot_v3` hoặc `nf_bot_iot_v3→nf_ton_iot_v3` (24 dòng mỗi chiều). **Tiêu chí 1:** 18 đặc trưng và ánh xạ nhãn dùng chung ở mọi bộ (`build_graph.FEATURES`, `GROUP_MAP`); vì scaler fit riêng từng bộ, đặc trưng node của bộ B được đổi về không gian chuẩn hóa của bộ A bằng `inverse_transform` rồi `transform` (đặc trưng cạnh là log1p, không chuẩn hóa). **Tiêu chí 2:** chỉ tính các lớp chung Normal, DoS_DDoS, Truy_cap_trai_phep: bỏ node B thuộc lớp khác (BoT → ToN bỏ Botnet_Backdoor và Khac), và khi dự đoán chỉ xét các cột lớp chung (argmax trên 3 cột; ngưỡng của mô hình A trên điểm 1 − P(Normal) rồi chọn lớp tấn công chung mạnh nhất); ghi trong docstring `final_test.py` và `README.md` kết quả. **Tiêu chí 3 đạt.** Kết quả (argmax, F1-macro 3 lớp chung / FPR / recall; so với cùng bộ: ToN 0,92–0,95, BoT 0,89–0,93): **ToN → BoT** hybrid chính 0,175 / 0,1% / 12,8%, GAT 0,258 / 3,1% / 28,2%, LightGBM 0,221 / 6,7% / 20,9%, (c) 0,197 / 1,5% / 13,2%; **BoT → ToN** hybrid chính 0,377 / 29,8% / 77,6%, GAT 0,363, LightGBM 0,371, (c) 0,362. **Kết luận: mọi mô hình tổng quát hóa kém giữa hai bộ** (F1-macro 0,18–0,38), hybrid không tốt hơn (ToN→BoT hybrid thấp nhất; BoT→ToN cao nhất nhưng chỉ nhỉnh 0,006–0,015, trong sai số đo). Nguyên nhân hợp lý: hai bộ khác nhau về hạ tầng, loại tấn công và phân bố đặc trưng (BoT chỉ có 20 IP nguồn). Hình `gd6_cross_dataset.png`. Không tune gì theo kết quả này.

### 5.7 Mô hình hybrid GAT + LightGBM

Thêm ngày 12/11/2026 theo yêu cầu "mô hình hybrid, lý do chọn" của thầy (quyết định D9). Đây là **mô hình chính** từ ngày này.

**Kiến trúc.** (1) Huấn luyện GAT nhẹ (cấu hình của task 5.3) để phân loại node. (2) Cố định GAT, lấy vector ẩn của lớp GAT cuối (64 chiều) làm embedding của mỗi node. (3) Ghép embedding với 18 đặc trưng gốc (đã chuẩn hóa) thành 82 cột. (4) LightGBM phân loại 82 cột đó.

**Lý do chọn** (ghi vào báo cáo): số liệu sơ bộ cho thấy mô hình cây mạnh trên đặc trưng dạng bảng nhưng không thấy hàng xóm; GNN thấy cấu trúc mạng nhưng FPR cao. Embedding GAT mang thông tin cấu trúc mà 18 đặc trưng không có; LightGBM tận dụng tốt đặc trưng dạng bảng và cho xác suất dễ hiệu chỉnh ngưỡng.

- **Trạng thái:** Xong (có ghi chú: cách chia out-of-fold khác đặc tả, xem Kết quả) · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 14/11 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Mô hình, Thuật toán & đánh giá
- **Đầu vào:** GAT đã huấn luyện (5.3) cho mỗi bộ và mỗi seed; đồ thị và `split.json` (1.6, 1.8); `src/evaluate.py`
- **Đầu ra:** `src/hybrid.py` (trích embedding, ghép đặc trưng, huấn luyện LightGBM); `scripts/run_hybrid.py`; notebook Kaggle; dòng kết quả `model=hybrid_gat_lgbm` (split=val) trong `results.csv`; `tests/test_hybrid.py`
- **Tiêu chí kiểm tra:**
  1. **Không rò rỉ:** GAT và LightGBM chỉ học trên train; tinh chỉnh chỉ trên val; scaler dùng lại của task 1.6.
  2. **Embedding out-of-fold cho tập train:** GAT đã thấy nhãn của node train, nên embedding train có thể "quá sạch". Chia train thành K = 5 nhóm khối thời gian; với mỗi nhóm, GAT huấn luyện trên 4 nhóm còn lại rồi sinh embedding cho nhóm đó. Val và test dùng embedding của GAT huấn luyện trên toàn bộ train. Báo cáo cả bản không dùng out-of-fold để so sánh. **Cập nhật 15/11/2026 (quyết định D13): thiết kế chính thức là out-of-fold xen kẽ (5 × 20 khối, khối i vào nhóm i mod 5), val và test dùng trung bình xác suất của LightGBM qua embedding của 5 GAT nhóm; bản "5 nhóm liên tiếp, val dùng GAT toàn train" ở trên là bản đặc tả gốc, đã chạy và thất bại, vẫn báo cáo kèm theo.**
  3. Test với dữ liệu giả nhỏ: số cột đầu vào LightGBM = 18 + kích thước embedding; thứ tự node giữ nguyên sau khi ghép; chạy 2 lần cùng seed cho cùng dự đoán.
  4. Đếm "kích thước" của hybrid: số tham số GAT + số lá của LightGBM, kèm kích thước file; độ trễ mỗi cửa sổ gồm cả bước GAT và bước LightGBM.
  5. 3 seed (42, 43, 44) trên cả 2 bộ; ghi thời gian chạy GPU vào dòng Kết quả.
- **Phụ thuộc:** 5.3, 1.8
- **Kết quả:** (14–15/11/2026, GĐ F) Notebook `gnn-iot-ids-15-hybrid` bản 1, Kaggle **GPU Tesla T4**, clone commit `603d929`, pytest 76 passed. Tổng 3427 s = **0,952 giờ GPU** (ToN: out-of-fold 2067 s + không out-of-fold 825 s; BoT: 370 s + 47 s), RAM đỉnh 2,78 GB. Kéo về [reports/kaggle/gnn-iot-ids-15-hybrid/](../../reports/kaggle/gnn-iot-ids-15-hybrid/): `results_hybrid_<bộ>.csv`, `hybrid_search_<bộ>.json`, `compare_val_<bộ>.csv`, `run_info.json`; `models_hybrid/` (6 thư mục `<bộ>_s<seed>/`) ở lại trên Kaggle cho GĐ G, H. Chỉ `split=val`, không đọc test; cấu hình LightGBM chọn trên val ở seed 42 (3 lần thử, cùng không gian với baseline).
  - **Tiêu chí kỹ thuật 1–5 đạt:** không rò rỉ (GAT, LightGBM chỉ học trên train; val chỉ chọn epoch GAT và cấu hình LightGBM); có cả `hybrid_gat_lgbm` (out-of-fold, K = 5) và `hybrid_gat_lgbm_nooof`, mỗi (mô hình, seed) đúng 1 dòng, 3 seed × 2 bộ; `test_hybrid.py` qua trên Kaggle; có số tham số (GAT + lá LightGBM), kích thước file, độ trễ p95 CPU 1 luồng gồm cả hai bước (2,9–5,7 ms mỗi cửa sổ).
  - **Hybrid KHÔNG vượt mô hình tốt nhất, ở cả hai bộ** (mean ± std 3 seed, val; cột `*_beats_2std` đều `False`):
    - NF-ToN-IoT-v3: hybrid out-of-fold F1-macro **0,647 ± 0,008**, FPR 6,36%, recall 84,6%; không out-of-fold F1 **0,912 ± 0,035** (seed 43 chỉ 0,871), FPR 7,24%, recall 95,1%. Tốt nhất trong `compare_val`: MLP F1 0,938 (xem lưu ý số baseline bên dưới); GAT đơn 0,908.
    - NF-BoT-IoT-v3: out-of-fold F1 **0,714 ± 0,171** (3 seed: 0,882 / 0,721 / 0,540), FPR 24,2%, recall 89,4%; không out-of-fold F1 **0,935 ± 0,0003**, FPR 19,66% ± 0,05, recall 98,2%. Tốt nhất: LightGBM F1 0,9388, FPR 19,67%. Không out-of-fold thấp hơn LightGBM 0,0037 F1 và FPR bằng nhau (hơn 0,0001), tức **ngang, không hơn**; hơn GAT đơn 0,9320 một chút.
  - **Out-of-fold làm hybrid TỆ HƠN rất nhiều, ngược với kỳ vọng.** Chẩn đoán trong `hybrid_search`: out-of-fold train F1 0,978 nhưng val 0,64–0,65 (ToN); BoT train 0,93 nhưng val 0,54–0,88. Giải thích khả dĩ (**chưa kiểm chứng**): mỗi nhóm out-of-fold lấy embedding từ một GAT khác (khởi tạo và dữ liệu khác), các không gian embedding không căn chỉnh với nhau, LightGBM học từ hỗn hợp 5 không gian rồi bị đánh giá trên embedding của một GAT thứ sáu (huấn luyện trên toàn bộ train). Bản không out-of-fold không bị lệch này nên giữ được F1 gần GAT. Bản không out-of-fold có train F1 0,996 ở ToN nhưng val vẫn 0,93 (seed 42), nên "học vẹt" không phải vấn đề lớn như dự kiến ở mục Rủi ro.
  - **Lưu ý số baseline trong `compare_val_nf_ton_iot_v3.csv`:** các mô hình baseline ToN ở đó (LightGBM 0,935, MLP 0,938, XGBoost 0,933) **cao hơn** số trong `results_baselines_nf_ton_iot_v3.csv` đã lưu ở repo (LightGBM 0,898, MLP 0,910, XGBoost 0,897), dù notebook 05 bản 2 trên Kaggle cho file cùng kích thước 6412 B với bản repo; số GNN thì khớp. Chưa tìm ra nguyên nhân (nghi file do Kaggle mount từ 05 khác bản tải qua API). Kết luận "không vượt" không đổi theo cả hai bộ số (so với bản repo, tốt nhất là GraphSAGE 0,9146 và hybrid không out-of-fold 0,9116 ± 0,035), nhưng **không dùng bảng ToN trong `compare_val` của notebook 15.** Đã kiểm ở notebook `gnn-iot-ids-15c-compare-check` (15/11, CPU, commit `724ff83`, 3,4 s; [reports/kaggle/gnn-iot-ids-15c-compare-check/](../../reports/kaggle/gnn-iot-ids-15c-compare-check/)): file baseline ToN do Kaggle mount (md5 `cb86eb84`) **khác về nội dung** bản trong repo (md5 `a76a6d9b`; LightGBM 0,9351 so với 0,8984, MLP 0,9381 so với 0,9100, XGBoost 0,9331 so với 0,8971, random forest 0,9351 so với 0,8988, logreg 0,7385 so với 0,6956); bản repo khớp log `baseline_search` (0,8983) nên **dùng bản repo**; nguyên nhân bản mount khác (nghi mount một phiên bản khác của notebook 05) chưa rõ. Các file còn lại (GNN ToN, baseline và GNN BoT) chỉ khác ký tự xuống dòng, số giống hệt. `compare_val_<bộ>.csv` dựng lại từ file trong repo: ToN, tốt nhất F1 là GraphSAGE 0,9146 ± 0,0008 và FPR thấp nhất là MLP 5,29%; hybrid không out-of-fold 0,9116 ± 0,0353 không vượt; BoT không đổi so với bảng cũ.
  - **Thử sửa out-of-fold bằng ensemble (15/11/2026, notebook `gnn-iot-ids-15b-hybrid-oofavg` bản 1, GPU T4, commit `f63efc1`, pytest 80 passed, 2974 s = 0,826 giờ GPU, RAM đỉnh 3,36 GB; [reports/kaggle/gnn-iot-ids-15b-hybrid-oofavg/](../../reports/kaggle/gnn-iot-ids-15b-hybrid-oofavg/)):** `model=hybrid_gat_lgbm_oofavg`: val được dự đoán bằng trung bình xác suất qua 5 GAT nhóm (mỗi GAT nhóm cho một embedding val). **Không sửa được:** ToN F1 **0,6725 ± 0,0041**, FPR 4,45%, recall 77,1%; BoT F1 **0,7426 ± 0,1518**, FPR 23,5%, recall 95,8%. Bảng dựng lại từ file trong repo ([15c](../../reports/kaggle/gnn-iot-ids-15c-compare-check/), `compare_val_<bộ>.csv`): hybrid out-of-fold (cả hai dạng) không vượt mô hình tốt nhất ở cả hai bộ (ToN: tốt nhất F1 GraphSAGE 0,9146, FPR thấp nhất MLP 5,29%; BoT: LightGBM 0,9388 và 19,67%). Cờ `fpr_beats_2std = True` của oofavg ở ToN (FPR 4,45% so với 5,29%) **không có ý nghĩa**: recall chỉ 77% và F1 0,67, tức mô hình đoán Normal quá nhiều.
  - **Chẩn đoán (cùng GAT nhóm và cùng LightGBM, trong `hybrid_search_oofavg_<bộ>.json`):** F1 val theo từng góc nhìn (mỗi góc = embedding val từ một GAT nhóm) rất khác nhau: ToN seed 42 là 0,66 / 0,66 / 0,67 / 0,65 và **0,908** cho góc từ GAT nhóm cuối; BoT seed 42 là 0,56 / 0,83 / 0,77 / 0,50 / **0,935**. Chỉ góc cuối (GAT huấn luyện trên 4 nhóm đầu của train) cho F1 ngang LightGBM; trung bình các góc không cứu được. F1 train 0,996 (ToN) nhưng val 0,67. **Giả thuyết cập nhật, chưa kiểm chứng:** mỗi GAT nhóm có một "dấu vân tay" embedding riêng, LightGBM học được "embedding này đến từ GAT nào", mà nhóm khối thời gian liên tiếp tương ứng với phân bố lớp khác nhau, nên đây là lối tắt thay vì tín hiệu thật; trên val, kết quả phụ thuộc vào góc nhìn đang đo. Cách làm chuẩn cho stacking là dùng đầu ra có ngữ nghĩa cố định giữa các GAT (xác suất 5 lớp, không phải embedding 64 chiều), nhưng đó là thay đổi thiết kế so với đặc tả 5.7 (embedding 64 chiều); **chưa làm, cần Phúc quyết**.
  - **Stacking bằng xác suất lớp (15/11/2026, notebook `gnn-iot-ids-15d-hybrid-stack` bản 1, CPU, commit `2d2187d`, 848 s, 0 giờ GPU; dùng lại GAT nhóm của 15b, không huấn luyện lại; [reports/kaggle/gnn-iot-ids-15d-hybrid-stack/](../../reports/kaggle/gnn-iot-ids-15d-hybrid-stack/)):** `hybrid_gat_lgbm_stack`: ToN F1 0,778 ± 0,091, FPR 7,7%; BoT F1 0,883 ± 0,018, FPR 19,1%. Không vượt, vẫn tệ hơn bản không out-of-fold.
  - **NGUYÊN NHÂN đã xác định (thay cho giả thuyết "dấu vân tay embedding" ở trên), từ trường `classes_missing_in_fit` của `hybrid_search_oofavg_<bộ>.json`:** 5 nhóm out-of-fold liên tiếp theo thời gian không hợp với dữ liệu vì mỗi loại tấn công chỉ xuất hiện ở một giai đoạn. Nhóm cuối của train là nơi duy nhất có DoS_DDoS (BoT) và Botnet_Backdoor, Khac (ToN); GAT huấn luyện trên 4 nhóm đầu chưa từng thấy các lớp đó (F1 val riêng của GAT này 0,57 ở BoT và 0,67 ở ToN, bốn GAT còn lại 0,89–0,94), nên xác suất và embedding out-of-fold của nhóm cuối vô nghĩa và làm LightGBM học sai. Đây là lỗi của cách chia theo đặc tả 5.7 (nhóm khối thời gian liên tiếp) trên dữ liệu không dừng, không phải lỗi của ý tưởng hybrid.
  - **Sửa bằng out-of-fold chia xen kẽ (15/11/2026, notebook `gnn-iot-ids-15e-hybrid-il` bản 1, Kaggle CPU, commit `e91a3dd`, pytest 87 passed, 669 s, RAM đỉnh 0,76 GB, 0 giờ GPU; [reports/kaggle/gnn-iot-ids-15e-hybrid-il/](../../reports/kaggle/gnn-iot-ids-15e-hybrid-il/)):** train cắt thành 5 × 20 khối liên tiếp theo thời gian, khối i thuộc nhóm i mod 5 (`fold_groups(..., interleave=20)`); không GAT nhóm nào thiếu lớp (`classes_missing_in_fit` rỗng ở mọi nhóm, mọi seed). **Mới chạy NF-BoT-IoT-v3, chưa chạy ToN** (cần GPU, chờ Phúc). BoT, val, 3 seed: `hybrid_gat_lgbm_oofavg_il` (embedding 64 chiều, val = trung bình 5 góc) F1 **0,9357 ± 0,0019**, FPR 20,45% ± 0,25, recall 98,7% (so với 0,714 ± 0,171 khi chia liên tiếp); `hybrid_gat_lgbm_stack_il` (xác suất lớp) F1 0,890 ± 0,081 (seed 42, 43, 44: 0,934 / 0,797 / 0,939; seed 43 sụp), FPR 17,1%, recall 88,2%. **So với mô hình tốt nhất ở BoT (LightGBM F1 0,9388, FPR 19,67%): không vượt** (oofavg_il kém 0,0031 F1 và 0,78 điểm % FPR; `compare_val` ghi `beats_2std = False` cho cả hai), tức hybrid xen kẽ ngang LightGBM chứ không hơn. Mỗi mô hình hybrid này dùng 5 GAT (62.435 tham số, 776 KB), độ trễ p95 8,7 ms mỗi cửa sổ (LightGBM 1,5 ms). Còn một góc nhìn trong 5 góc cho F1 val thấp (0,68–0,75) dù GAT tương ứng riêng lẻ đạt 0,93: chưa giải thích.
  - **Bảng so sánh dựng lại chỉ từ file trong repo:** notebook `gnn-iot-ids-15c-compare-check` bản 3 (commit `240849b`), `compare_val_<bộ>.csv` trong [reports/kaggle/gnn-iot-ids-15c-compare-check/](../../reports/kaggle/gnn-iot-ids-15c-compare-check/) gồm mọi biến thể hybrid. Dùng bảng này, không dùng `compare_val` của notebook 15.
  - **ToN với out-of-fold xen kẽ (15/11/2026, notebook `gnn-iot-ids-15f-hybrid-il-ton` bản 1, Kaggle GPU Tesla T4, commit `8f13f79`, pytest 87 passed, 2494 s = 0,693 giờ GPU, RAM đỉnh 3,39 GB; [reports/kaggle/gnn-iot-ids-15f-hybrid-il-ton/](../../reports/kaggle/gnn-iot-ids-15f-hybrid-il-ton/)):** `hybrid_gat_lgbm_oofavg_il`, val, 3 seed: F1-macro **0,9326 ± 0,0009**, FPR **4,94% ± 0,01**, recall tấn công 93,6% (so với 0,647 khi chia liên tiếp). **Vượt mô hình tốt nhất về F1-macro:** GraphSAGE 0,9146 ± 0,0008, chênh +0,018, lớn hơn 2 lần std (`f1_macro_beats_2std = True` trong `compare_val`); hơn MLP 0,9100 và GAT đơn 0,9079. **Không vượt về FPR:** MLP 5,29% ± 0,26, chênh 0,35 điểm % chưa tới 2 std (`fpr_beats_2std = False`), và MLP có recall thấp hơn (90,3% so với 93,6%). Recall của hybrid thấp hơn GraphSAGE (96,5%) và LightGBM (95,5%). Mỗi GAT nhóm vẫn thiếu lớp Khac ở nhóm cuối (`classes_missing_in_fit = [Khac]` mọi seed; Khac không tính vào F1-macro theo D5).
  - **Kích thước và độ trễ của hybrid xen kẽ (dùng 5 GAT nhóm + một LightGBM):** ToN 144.278–144.715 tham số (GAT + lá LightGBM), file 9,7 MB, p95 19,5–20,1 ms mỗi cửa sổ; BoT 62.435 tham số, 776 KB, p95 8,7 ms. **Vượt ngưỡng "nhẹ" ≤ 100.000 tham số và ≤ 1 MB ở ToN** (phần lớn là file LightGBM; LightGBM baseline ToN cũng 10 MB); độ trễ vẫn ≪ 1 s. Cần nói rõ trong báo cáo.
  - **Kết luận (so với mô hình tốt nhất, val, 3 seed, bảng dựng lại ở [15c](../../reports/kaggle/gnn-iot-ids-15c-compare-check/), notebook bản 4, commit `2ad93d8`):** ToN, hybrid xen kẽ F1-macro 0,9326 hơn GraphSAGE 0,9146 và hơn mọi baseline ML, DL (LightGBM 0,8984, MLP 0,9100); **vượt**. BoT, hybrid xen kẽ F1-macro 0,9357 so với LightGBM 0,9388, FPR 20,45% so với 19,67%; **không vượt, ngang**. Chỉ dạng embedding xen kẽ đạt mức này; stacking bằng xác suất kém ổn định (BoT 0,890 ± 0,081, ToN chưa chạy vì bản BoT đã kém). Hybrid không out-of-fold ngang mô hình tốt nhất ở BoT (0,9351) và kém GraphSAGE ở ToN (0,9116 ± 0,0353). **Bản out-of-fold đúng đặc tả (5 nhóm liên tiếp) thất bại (ToN 0,647, BoT 0,714) vì nhóm cuối là nơi duy nhất có một số lớp; bản xen kẽ là thay đổi so với đặc tả và là bản dùng cho báo cáo, cần ghi rõ.** Chưa chỉnh gì theo test. Mô hình xen kẽ ToN lưu ở `models_hybrid/nf_ton_iot_v3_s<seed>/` (output 15f) và BoT ở output 15e, cho GĐ G, H.
  - **Giờ GPU của GĐ F: 2,471 giờ** (15: 0,952; 15b: 0,826; 15f: 0,693); 15a, 15c, 15d, 15e, 15t chạy CPU.
  - **Trạng thái `Xong` có ghi chú:** tiêu chí kỹ thuật 1–5 đạt (không rò rỉ, có out-of-fold và không out-of-fold, 3 seed × 2 bộ, test qua trên Kaggle, có kích thước và độ trễ, `compare_val` nói rõ vượt hay không); ghi chú: bản đúng đặc tả thất bại và bản dùng cho báo cáo chia out-of-fold xen kẽ, hybrid vượt ở ToN nhưng chỉ ngang ở BoT, kích thước vượt ngưỡng nhẹ ở ToN.

### 5.8 Ablation hybrid và minh chứng hơn baseline

Thêm ngày 12/11/2026 theo yêu cầu "cần minh chứng hybrid cao hơn các mô hình baseline (ML, DL, …)" của thầy.

- **Trạng thái:** Xong (có ghi chú: hybrid vượt baseline ở ToN, ngang ở BoT; embedding GAT không hơn đặc trưng hàng xóm tính tay) · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 15/11 · **Ưu tiên:** Bắt buộc
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
- **Kết quả (16/11/2026, GĐ G; [reports/kaggle/gnn-iot-ids-16c-aggregate/](../../reports/kaggle/gnn-iot-ids-16c-aggregate/): `ablation_hybrid_<bộ>.csv`, `pareto_hybrid_<bộ>.csv`, `light_choice_<bộ>.json`, `compare_models_<bộ>.csv`, `thresholds.json`, hình `gd5_ablation_hybrid_<bộ>.png`, `gd5_pareto_<bộ>.png`, `gd5_compare_<bộ>.png`).** Notebook: `gnn-iot-ids-16a-ablation-bot` (BoT, CPU, 2900 s), `16d-ablation-ton` (ToN, CPU, 3747 s), `16e-thresholds-ton` (ngưỡng baseline và GNN ToN, CPU, 1548 s), tổng hợp `16c-aggregate` (CPU); tất cả clone commit `ad92f6a`, pytest 110 passed, 0 giờ GPU. Chỉ val, 3 seed. Hybrid đầy đủ dựng lại từ file đã lưu cho F1 khớp kết quả 15e/15f (ToN 0,93232 seed 42; BoT 0,93604 seed 42).
  - **Ablation 5 phương án, F1-macro val (FPR):**

    | Phương án | NF-ToN-IoT-v3 | NF-BoT-IoT-v3 |
    |---|---|---|
    | (a) LightGBM, 18 đặc trưng (GĐ D) | 0,8984 ± 0,0001 (7,94%) | 0,9388 (19,67%) |
    | (b) GAT đơn (GĐ A/E) | 0,9079 ± 0,0069 (8,44%) | 0,9320 ± 0,0009 (21,03%) |
    | (c) LightGBM + trung bình đặc trưng hàng xóm (tính tay) | **0,9342 ± 0,0003 (5,00%)** | 0,9335 (19,91%) |
    | (d) LightGBM chỉ embedding GAT | 0,9182 ± 0,0186 (5,99%) | 0,9351 ± 0,0009 (20,41%) |
    | (e) Hybrid đầy đủ (GĐ F) | 0,9326 ± 0,0009 (4,94%) | 0,9357 ± 0,0019 (20,45%) |

  - **Embedding GAT có giúp hơn đặc trưng hàng xóm tính tay không (c so với e): KHÔNG.** ToN: (e) − (c) = −0,0016 (hybrid thấp hơn một chút, trong sai số; FPR 4,94% so với 5,00%); BoT: +0,0022 (trong sai số; `e_beats_c_2std = false` cả hai bộ). Cả (c) lẫn (e) hơn xa (a) và (b) ở ToN (+0,034 và +0,025 cho (e) so với (a) và (b)), tức lợi thế của hybrid đến từ thông tin hàng xóm, không rõ phần nào do embedding học được. (d) (chỉ embedding) kém hơn (e) ở ToN (−0,0144) và không ổn định giữa seed (std 0,0186). Với tiêu chí 3 (hơn baseline > 2 std): hybrid hơn baseline tốt nhất ở ToN (5.7, không đổi); ở BoT ngang (5.7, không đổi).
  - **Đường Pareto hybrid nhẹ** (số GAT dùng khi suy luận 5, 3, 2, 1 hoặc 1 GAT huấn luyện trên toàn train, × LightGBM n_estimators {25, 50, 100} × num_leaves {7, 15, 31}, cộng cấu hình của bản chính; 50 điểm ToN, 45 điểm BoT; kích thước là max trên 3 seed). **ToN: có bản đạt cả ≤ 100.000 tham số và ≤ 1 MB (37/50 điểm).** Bản F1 cao nhất đạt ngưỡng: 2 GAT nhóm + LightGBM (n=100, l=15), 30.934 tham số, 986 KB, F1-macro 0,9181 ± 0,0035 (**−1,45 điểm %** so với bản đầy đủ 0,9326), FPR 6,68% (**+1,73 điểm %** so với 4,94%), p95 6,9 ms (bản đầy đủ 27,6 ms); **bản nhỏ nhất**: 1 GAT + LightGBM (n=25, l=7), 12.592 tham số, 172 KB, F1-macro 0,9002 (−3,24 điểm %), FPR 8,30%. Ở ToN, dùng một GAT huấn luyện trên toàn train (`gatfull`) tệ hơn dùng một GAT nhóm 0,8–3,6 điểm % và không ổn định giữa seed (nhiều điểm có thanh sai số lớn), vì LightGBM học trên embedding out-of-fold của các GAT khác; nên **không dùng `gatfull`** ở ToN (ở BoT chúng ngang nhau). Phần lớn kích thước nằm ở LightGBM (9,6 MB ở bản chính). **BoT: bản đầy đủ đã đạt ngưỡng nhẹ (62.435 tham số, 777 KB)**, không cần giảm; bản nhỏ nhất 12.112 tham số, 126 KB (F1-macro 0,9286, −0,71 điểm %, FPR +2,17 điểm %).
  - **Hiệu chỉnh ngưỡng (FPR val ≤ 4,08%, recall cao nhất), mọi mô hình sẽ chạy test, 3 seed:** `thresholds.json`, đủ baseline (5), GNN đơn (gcn, sage, gat), hybrid đầy đủ, hybrid nhẹ (và 3 bản cải tiến GAT, không công nhận), mỗi (bộ, mô hình, seed) có ngưỡng, F1-macro, FPR, recall tấn công trước và sau. Mọi mô hình đều đưa được FPR val về ≤ 4,08% (không có trường hợp không có ngưỡng khả thi), nhưng **không mô hình nào đạt recall ≥ 95% cùng lúc**: ToN, recall sau hiệu chỉnh: hybrid đầy đủ 91,9%, hybrid nhẹ 90,2%, SAGE 91,7%, XGBoost 90,0%, LightGBM 89,9%, GAT 89,0%, MLP 88,8%, random forest 87,8%, GCN 70,5%, logreg 65,9% (F1-macro hybrid đầy đủ 0,9326 → 0,9362); BoT: recall sau chỉ 29–38% mọi mô hình (F1-macro rơi 0,93 → khoảng 0,65), do dữ liệu (phân tích 15a). Baseline huấn luyện lại trên train cho F1 val khớp kết quả cũ, lệch tối đa 0,0003 (xgboost ToN).
  - **Biểu đồ cột F1-macro và FPR của mọi mô hình có vạch 4,08% và thanh sai số** ở `gd5_compare_<bộ>.png` (FPR trước và sau hiệu chỉnh).
  - **Trạng thái `Xong` có ghi chú:** đủ 5 phương án × 2 bộ, ngưỡng đủ, biểu đồ đủ; ghi chú: embedding GAT không hơn đặc trưng hàng xóm tính tay, hybrid chỉ vượt baseline ở ToN, và ngưỡng 4,08% không đi cùng recall ≥ 95%.

## GĐ6 Đánh giá tổng hợp

### 6.1 Đóng băng mô hình và chạy test một lần

- **Trạng thái:** Xong (có ghi chú: hybrid không vượt mọi nhóm baseline, xem 6.3) · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 7 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Thuật toán & đánh giá
- **Đầu vào:** Checkpoint và config của mọi mô hình (4.1, 5.2–5.6)
- **Đầu ra:** Tag git `freeze-v1`; dòng kết quả `split=test` cho mọi mô hình, mọi bộ, 3 seed
- **Tiêu chí kiểm tra:**
  1. Tag git được tạo trước khi chạy test; cột `commit` trong `results.csv` trùng với tag.
  2. Test chạy đúng 1 lần, có log thời điểm chạy.
  3. Không chỉnh mô hình sau khi đã xem kết quả test; muốn chỉnh thì phải ghi rõ trong báo cáo là "sau test".
  4. Kiểm tra lại toàn bộ chỉ tiêu ở mục 3 của KE_HOACH.md, ghi đạt hoặc không đạt cho từng chỉ tiêu.
- **Phụ thuộc:** 4.1, 5.4, 5.5, 5.6
- **Kết quả:** (16/11/2026) Tag `freeze-v1` tạo trên commit `abcb5f84a9dcf0adeb61800aa03c71fb591c2f9a` (`abcb5f8`) **trước** khi chạy test, ghi ở đầu [freeze_list.md](../freeze_list.md). Notebook `gnn-iot-ids-17-final-test` bản 1 (CPU, **0 giờ GPU**, 2.668 s, RAM đỉnh 2,6 GB, 123 test pass) clone repo **tại tag** (xác nhận commit đã clone = commit của tag trên GitHub), chạy 04:48–05:27 UTC ngày 16/11. [reports/kaggle/gnn-iot-ids-17-final-test/](../../reports/kaggle/gnn-iot-ids-17-final-test/): `results_test_<bộ>.csv`, `tiers_test_<bộ>.csv`, `ledger_test.csv` (thời điểm chạy từng mục), `val_check_test_<bộ>.json`, `run_info.json`, `README.md` (quy ước tên mô hình). Dự đoán từng node (`pred_*.parquet`) ở lại Kaggle.
  - **Tiêu chí 1 đạt:** cột `commit` của cả 126 dòng test + 48 dòng thử chéo đều là `abcb5f8` = commit của tag. **Tiêu chí 2 đạt:** 63 mục (bộ, mô hình, seed) trong sổ, không trùng; mỗi mục một lần dự đoán test, ghi hai dòng (argmax và `_thr`). Kiểm F1 val dựng lại khớp F1 val đã đóng băng trước khi đọc test: lệch tối đa **0,0** (baseline huấn luyện lại, GNN và hybrid nạp checkpoint). **Tiêu chí 3 đạt:** không sửa mô hình, cấu hình, ngưỡng sau tag. Mã chạy test (`scripts/final_test.py`, `aggregate_h.py`) nằm trong commit của tag và đã chạy thử trên val (17a, 18a) trước khi tag.
  - **Tiêu chí 4:** [reports/tables/targets.csv](../../reports/tables/targets.csv), từng chỉ tiêu × mô hình × chế độ ngưỡng. **Bộ chính NF-ToN-IoT-v3, hybrid nhẹ (30.934 tham số, 986 KB):** đạt mọi chỉ tiêu khi dùng ngưỡng hiệu chỉnh (accuracy 97,1%, FPR 2,27%, recall 95,65%, F1-macro 0,9588, std accuracy 0,07 điểm %, p95 7,3 ms); ở argmax FPR 5,69% **không đạt**. **Hybrid chính (144.518 tham số, 9,75 MB):** đạt accuracy, recall, F1, ổn định, độ trễ; FPR đạt ở ngưỡng hiệu chỉnh (1,88%) nhưng không đạt ở argmax (4,20%); **không đạt tham số và kích thước** (như đã biết). **Chỉ tiêu "hybrid > baseline tốt nhất" không đạt** (xem 6.3). Số bộ dữ liệu ≥ 2 đạt. Tái lập: sơ bộ đạt (F1 val dựng lại lệch 0; GĐ J chạy lại đầy đủ). Độ trễ chỉ đo sơ bộ (200 lần trên 1 cửa sổ trung vị, CPU Kaggle 1 luồng), chưa phải ≥ 1.000 cửa sổ. **Lưu ý:** recall ≥ 95% mà val cho là không đạt (91,9% ở ToN) thì ở test đạt (96,7%); đây là khác biệt giữa val và test của cách chia xen kẽ, không dùng để chỉnh gì. BoT: FPR 17,97% ở argmax (recall 94,8%) và recall 47,5% ở ngưỡng (FPR 1,19%), không đạt như dự báo từ val (do dữ liệu, xem 15a).

### 6.2 Bảng so sánh tổng hợp và bộ biểu đồ

- **Trạng thái:** Xong · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 7 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Thuật toán & đánh giá, Báo cáo
- **Đầu vào:** `results.csv` (`split=test`)
- **Đầu ra:** `scripts/make_figures.py`; bảng tổng hợp mean ± std; confusion matrix, ROC và PR, F1 theo lớp, F1 theo tham số, F1 theo độ trễ, đường huấn luyện, biểu đồ thời gian huấn luyện
- **Tiêu chí kiểm tra:**
  1. Mọi hình và bảng sinh từ `results.csv` bằng script, chạy lại cho kết quả y hệt.
  2. Mọi số ghi mean ± std của 3 seed.
  3. Có ít nhất 8 loại biểu đồ.
  4. Bảng có đủ: accuracy, F1-macro, FPR, recall tấn công, AUC, tham số, kích thước, độ trễ.
- **Phụ thuộc:** 6.1
- **Kết quả:** (16/11/2026) Notebook `gnn-iot-ids-18-aggregate` bản 1 (CPU, 0 giờ GPU, 268 s, 123 test pass; commit repo `main` sau tag, chỉ thêm ghi chú tag vào freeze_list). Đầu vào: output 17 (test) và 17a (val, cùng mã và mô hình). [reports/kaggle/gnn-iot-ids-18-aggregate/](../../reports/kaggle/gnn-iot-ids-18-aggregate/) (`run_info.json`, `error_analysis.json`, `aggregate_h_summary.json`). Bảng ở `reports/tables/`: [results.csv](../../reports/tables/results.csv) (20 cột, val + test; dòng val mang commit `9cae412` của lần chạy thử 17a, dòng test mang `abcb5f8`), [summary_val.csv](../../reports/tables/summary_val.csv), [summary_test.csv](../../reports/tables/summary_test.csv) (mean ± std 3 seed theo bộ, mô hình, chế độ ngưỡng; có accuracy, F1-macro, FPR, recall tấn công, AUC, tham số, kích thước, độ trễ), [summary_test_tiers.csv](../../reports/tables/summary_test_tiers.csv), [mcnemar.csv](../../reports/tables/mcnemar.csv), [conclusions.csv](../../reports/tables/conclusions.csv), [targets.csv](../../reports/tables/targets.csv). Hình `reports/figures/gd6_*.png` (23 hình, 12 loại: cột F1-macro và FPR có vạch 4,08% và thanh sai số, confusion matrix hybrid chính, ROC và PR, F1 theo số tham số (vạch 100K), theo kích thước (vạch 1 MB), theo độ trễ, F1 theo lớp, thời gian huấn luyện, báo cáo theo tầng, thử chéo, đường huấn luyện, attention). Mọi bảng và hình do `scripts/aggregate_h.py` sinh ra, chạy trên Kaggle; chạy lại cho cùng kết quả vì không có thành phần ngẫu nhiên. Script tên `aggregate_h.py` (thay cho `make_figures.py` trong đặc tả). Đường huấn luyện ToN của GCN và GraphSAGE lấy từ 16g (cấu hình mặc định hidden 16, 2 lớp), của GAT từ 16 (`gat_default`); 07 không lưu đường học cho ToN.
  - **Bảng test chính (3 seed, mean ± std):** NF-ToN-IoT-v3, argmax / ngưỡng hiệu chỉnh: hybrid chính F1 0,9472 ± 0,0007 / 0,9677 ± 0,0005, FPR 4,20% / 1,88%, recall 97,7% / 96,7%; hybrid nhẹ 0,9341 ± 0,0043 / 0,9588 ± 0,0011, FPR 5,69% / 2,27%, recall 97,8% / 95,6%; (c) 0,9477 ± 0,0003 / 0,9653 ± 0,0003; MLP 0,9422 / 0,9575; XGBoost 0,9199 / 0,9611; GraphSAGE 0,9327 / 0,9600; GAT 0,9270 / 0,9572. NF-BoT-IoT-v3: hybrid chính 0,9164 ± 0,0012 (argmax, FPR 17,97%, recall 94,8%) / 0,7084 ± 0,0123 (ngưỡng, FPR 1,19%, recall 47,5%); LightGBM 0,9291 / 0,6949; (c) 0,8900 / 0,6550.
  - **Báo cáo theo tầng (quyết định ở 5.3):** ToN, argmax, FPR theo tầng (node có khởi tạo flow / node chỉ nhận): hybrid chính **1,03% / 8,15%**, (c) 1,52% / 8,47%, LightGBM 1,75% / 14,35%, GAT 4,17% / 13,16%; recall tấn công gần như như nhau giữa hai tầng (97,6–98,4%). Nghĩa là FPR 1% ở node khởi tạo, điểm yếu ở node chỉ nhận (khớp phân tích 15h). BoT: FPR cao ở node khởi tạo (25%), thấp ở node chỉ nhận (7%), ngược ToN. [summary_test_tiers.csv](../../reports/tables/summary_test_tiers.csv), `gd6_tiers_<bộ>.png`.

### 6.3 Kiểm định thống kê và phân tích lỗi

- **Trạng thái:** Xong (có ghi chú: hybrid vượt ML, MLP và GNN đơn nhưng không vượt rõ (c); BoT thua LightGBM ở argmax) · **Phụ trách:** Phúc · **Kiểm tra:** Tự kiểm · **Tuần:** 7 · **Ưu tiên:** Bắt buộc
- **Mục đề tài:** Thuật toán & đánh giá
- **Đầu vào:** Dự đoán trên test của GNN chính và baseline tốt nhất; trọng số attention
- **Đầu ra:** McNemar test (GNN chính và baseline tốt nhất); phân tích các node sai: lớp nào nhầm với lớp nào, đặc điểm IP và cửa sổ sai; ví dụ attention trên 2 cửa sổ
- **Tiêu chí kiểm tra:**
  1. Có p-value và kết luận ở mức α = 0,05.
  2. Phân tích ≥ 3 nhóm lỗi, mỗi nhóm có ví dụ cụ thể.
  3. Nếu GNN không hơn baseline thì có phần giải thích nguyên nhân, liên hệ với homophily ở 2.2.
- **Phụ thuộc:** 6.1
- **Kết quả:** (16/11/2026) [reports/tables/mcnemar.csv](../../reports/tables/mcnemar.csv) (hybrid chính và nhẹ so với mọi mô hình đơn, mỗi bộ, mỗi seed, hai chế độ, kèm b, c, thống kê, p-value; chính xác khi b + c < 25, χ² hiệu chỉnh liên tục khi lớn hơn), [conclusions.csv](../../reports/tables/conclusions.csv), [error_analysis.json](../../reports/kaggle/gnn-iot-ids-18-aggregate/error_analysis.json), hình `gd6_attention_<bộ>.png` (attention của GAT trên 2 cửa sổ, từ trọng số lưu ở 5.3), `gd6_confusion_hybrid_<bộ>.png`. Mô hình "tốt nhất" của từng nhóm (ML, MLP, GNN đơn) **chọn theo F1-macro val** để không nhìn test: ToN ML = random_forest, GNN = sage; BoT ML = lightgbm, GNN = sage. Kiểm định tính trên đúng/sai đa lớp của từng node; các node cùng cửa sổ không hoàn toàn độc lập nên p-value rất nhỏ (< 1e-100 ở nhiều chỗ) chỉ nên đọc là "có khác biệt", không đọc độ lớn. "Vượt" = chênh F1-macro trung bình > 2 · max(std hai bên) VÀ McNemar p < 0,05 ở cả 3 seed.
  - **Kết luận NF-ToN-IoT-v3, hybrid chính** (argmax / ngưỡng hiệu chỉnh): so với ML tốt nhất (random_forest): **vượt** / **vượt** (+0,0254 / +0,0136); so với GNN đơn tốt nhất (sage): **vượt** (+0,0145) / chưa đủ (+0,0078 < 2 std 0,0083, dù McNemar 3/3 seed có ý nghĩa); so với MLP: chưa đủ (+0,0050 < 0,0079) / **vượt** (+0,0102); so với **(c)**: **không vượt ở argmax** (−0,0005; F1 ngang, McNemar nghiêng về hybrid ở độ chính xác nhưng chênh F1 nằm trong 2 std) và **vượt rất nhỏ ở ngưỡng** (+0,0025 > 2 std 0,0010, McNemar 3/3). Tóm lại: hybrid chính **không vượt mọi nhóm baseline ở cả hai chế độ**; chỉ tiêu "hybrid > baseline tốt nhất" ghi **không đạt**. Với ngưỡng hiệu chỉnh hybrid chính cao nhất bảng (0,9677) nhưng cách (c) chỉ 0,0025. Chọn nhóm theo val nên random_forest là "ML tốt nhất", dù ở ngưỡng hiệu chỉnh trên test XGBoost (0,9611) và LightGBM (0,9601) cao hơn random_forest (0,9542); hybrid chính (0,9677) vẫn cao hơn cả hai. **Hybrid nhẹ (ToN):** vượt ML tốt nhất ở cả hai chế độ, **không** vượt MLP, GNN đơn, và **thấp hơn (c)** (−0,0136 argmax, McNemar 3/3 nghiêng về (c); −0,0065 ngưỡng).
  - **Kết luận NF-BoT-IoT-v3, hybrid chính** (= hybrid nhẹ): argmax **thua LightGBM** (−0,0127, McNemar 3/3 seed nghiêng về LightGBM), ngang MLP, **vượt** (c) (+0,0264) và GNN đơn (+0,0293); ngưỡng hiệu chỉnh: vượt MLP, (c), GNN đơn, chưa đủ so với LightGBM (+0,0135 < 2 std 0,0245; recall 47,5% so với 45,2%). Không đạt "vượt mọi nhóm".
  - **Tiêu chí 3 (giải thích):** nguyên nhân hybrid không vượt (c): ablation ở 5.8 đã cho thấy phần lớn lợi thế đến từ thông tin hàng xóm; (c) lấy trung bình 18 đặc trưng của node kề bằng tay, đủ để tái tạo phần đó vì đồ thị rất nhỏ (ToN median 11 node/cửa sổ, BoT 5) và hàng xóm cùng lớp khá nhiều (homophily cạnh ToN 0,67, BoT 0,86, xem 2.2): ít cấu trúc để GAT học thêm ngoài trung bình hàng xóm. Ở BoT, LightGBM thuần đã đạt 0,9291, cao nhất bảng; có thể do dữ liệu chỉ có 20 IP nguồn và 3 lớp, nhưng chưa kiểm chứng; hybrid và (c) không thêm được gì (FPR 18% ở mọi mô hình là do nhãn và dữ liệu, xem 15a).
  - **Phân tích lỗi (hybrid chính, seed 42, ≥ 3 nhóm, mỗi nhóm có 3 ví dụ cụ thể gồm IP, cửa sổ, nhãn, dự đoán, điểm):** ToN, 6.883 node sai / 121.770 (5,7%): (1) **FP trên node chỉ nhận flow** 3.989 (58% số lỗi; IP gặp nhiều nhất là 192.168.1.193 với 947 node; đây là node nhận flow, không khởi tạo flow nào); (2) **tấn công bị bỏ sót** 2.131 (31%); (3) **lớp hiếm Khac** bị đoán sai 1.300 (18,9% số lỗi; lớp này chỉ có 841 node test); (4) FP trên node có khởi tạo flow 389 (5,7%); (5) nhầm giữa các lớp tấn công 374 (5,4%); (6) Botnet_Backdoor 2 node. BoT, 698 sai / 10.665 (6,6%): FP trên node có khởi tạo 504 (72%; IP gặp nhiều nhất là 192.168.100.149 với 119 node), FN 100, FP trên node chỉ nhận 89, nhầm giữa lớp tấn công 5. Ghi chú kèm nguyên nhân và đặc trưng trung bình của từng nhóm trong file JSON.

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
