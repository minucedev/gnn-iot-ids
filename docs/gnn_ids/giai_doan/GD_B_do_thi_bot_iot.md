# GĐ B. Dựng đồ thị NF-BoT-IoT-v3

| Ngày | Task | Máy | Cần xong trước | Notebook |
|---|---|---|---|---|
| 12–13/11 | 1.8 | Kaggle CPU | — | `11_build_graphs_bot` (slug `gnn-iot-ids-11-build-graphs-bot`) |

Đọc trước: [README.md](README.md), task 1.8 trong [../TASKS.md](../TASKS.md), phiếu `docs/dataset_cards/netflow_v3_others.md`.

## Mục tiêu

Có bộ đồ thị chính thức thứ 2 (dữ liệu IoT) theo đúng pipeline của bộ chính: cửa sổ 5 giây, node là IP, chia theo khối, đã kiểm tra rò rỉ và tái lập.

## Đầu vào

- CSV `NF-BoT-IoT-v3.csv` trong dataset Kaggle `athena21/netflow-v3-datasets`. Tấn dùng `seyhed/nf-bot-iot-v3`; đối chiếu số dòng nếu dùng bản đó.
- Notebook của Tấn `phamta/dss-cky-bot`: tham khảo phần đọc file theo đoạn. **Không** dùng phần lấy mẫu 300.000 flow mỗi lớp, cửa sổ 10 giây, chia thời gian liên tục.
- `src/build_graph.py`, `scripts/split_search.py`, `scripts/build_dataset_cli.py`, `scripts/verify_graphs.py`.

## Việc cần làm

### 1. Sửa code (local, đẩy lên GitHub trước khi chạy notebook)

1. `src/build_graph.py`: thêm `"nf_bot_iot_v3": load_nf_v3` vào `ADAPTERS`. Ánh xạ nhãn đã có sẵn: dos, ddos → DoS_DDoS; reconnaissance, theft → Truy_cap_trai_phep; benign → Normal.
2. `scripts/split_search.py` và `src/checks.py` đang bắt buộc **cả 4 lớp chính** (gồm Botnet_Backdoor) có ≥ 100 node ở train và ≥ 30 ở val, test. NF-BoT-IoT-v3 không có Botnet_Backdoor nên sẽ luôn "không đạt". Sửa để chỉ xét **các lớp có mặt trong bộ dữ liệu**, và ghi danh sách lớp vắng mặt vào JSON output. Không được làm thay đổi kết quả của 3 bộ cũ.
3. Thêm test trong `tests/` cho 2 thay đổi trên (chạy trên Kaggle ở bước 2).

### 2. Notebook `11_build_graphs_bot` (Kaggle CPU)

1. Clone repo, cài requirements, chạy `pytest -q`; dừng nếu có test lỗi.
2. Đếm số dòng và số flow từng nhãn của CSV, so với số UQ công bố.
3. `python -I scripts/split_search.py nf_bot_iot_v3 <csv> 5 5 split_search_nf_bot_iot_v3.json` để chọn số khối nhỏ nhất đạt tiêu chí.
4. `python -I scripts/build_dataset_cli.py nf_bot_iot_v3 <csv> 5 5 blocked 1 /kaggle/working/processed/nf_bot_iot_v3 blocks=<số khối> balance=1 tag=nf_bot_iot_v3_w5s5_blocked`.
5. `python -I scripts/verify_graphs.py /kaggle/working/processed/nf_bot_iot_v3 verify_nf_bot_iot_v3.json`.
6. Ghi `run_info.json`.

## Đầu ra

- **Trên Kaggle**, làm đầu vào cho GĐ C–H: `processed/nf_bot_iot_v3/` (graphs `.pt`, `split.json`, `scaler.pkl`).
- **Kéo về repo:** `reports/tables/graphs_v2/nf_bot_iot_v3/summary.json`, `graph_stats.csv`; `reports/tables/split_search/split_search_nf_bot_iot_v3.json`; `reports/kaggle/gnn-iot-ids-11-build-graphs-bot/verify_nf_bot_iot_v3.json`, `run_info.json`.
- Thêm dòng cấu hình chia của NF-BoT-IoT-v3 vào bảng "Điều chỉnh D3" trong KE_HOACH.md mục 7.

## Tiêu chí đạt (task 1.8)

1. Số dòng và số flow từng nhãn khớp UQ: 16.933.808 flow, Benign 51.989. Trang UQ ghi nhầm tổng là 16.993.808.
2. Bảng số node mỗi lớp × mỗi tập; ghi rõ không có Botnet_Backdoor.
3. Mỗi lớp có mặt có ≥ 100 node ở train và ≥ 30 ở val, test. Nếu Normal không đạt thì ghi rõ số node, không tự hạ ngưỡng.
4. Không lấy mẫu flow ngẫu nhiên. Thiếu RAM thì lấy mẫu theo cửa sổ và ghi tỉ lệ.
5. `verify_graphs.py` báo đạt mọi kiểm tra (không trùng cửa sổ giữa các tập, có gap, không NaN, tái lập hash).
6. Test cũ và test mới đều qua trên Kaggle.

## Sau khi xong

- Cập nhật dòng **Kết quả** của task 1.8 và bảng theo dõi dòng B.
- Commit: `feat(1.8): dựng đồ thị NF-BoT-IoT-v3, chia tập theo lớp có mặt`.

## Rủi ro

- Chỉ 20 IP nguồn nên mỗi cửa sổ rất ít node; Normal chỉ 0,3% số flow nên có thể quá ít node Normal. Ghi số thật, không chỉnh để đẹp.
- Đọc file 16,9 triệu dòng có thể tốn RAM; đọc theo cột bằng pyarrow như bộ chính (bộ chính có RAM đỉnh 18,9 GB).
