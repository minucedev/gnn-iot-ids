# Phiếu dataset: ToN-IoT (network)
Người khảo sát: Phong   Ngày: 2026-10-22
Link tải: https://research.unsw.edu.au/projects/toniot-datasets   License: Học thuật, bắt buộc trích dẫn (UNSW)   Bài báo gốc: Moustafa (2021), Sustainable Cities and Society
Dung lượng: 28.5 MB (`data/ToN_IoT/ToN-IoT.csv`)   Số mẫu: 211,043   Thời gian đọc thử: đọc toàn bộ CSV 1.0s

## Điều kiện bắt buộc
| # | Đạt? | Bằng chứng |
|---|---|---|
| G1 | Đạt | Cột IP: ('src_ip', 'dst_ip') |
| G2 | **Không** | Không có cột timestamp. Thứ tự dòng: **theo pha (có thể theo thời gian)** (độ dài chuỗi nhãn liên tiếp TB 21104.3; tỉ lệ lớp trội trong khối 100k dòng 0.31 so với toàn file 0.237) |
| G3 | Đạt | Cột nhãn: ['type', 'label'], 10 giá trị |
| G4 | Đạt | Nhóm tấn công có mặt: ['DoS_DDoS', 'Botnet_Backdoor', 'Truy_cap_trai_phep'] |
| G5 | Đạt | Học thuật, bắt buộc trích dẫn (UNSW) (cần xác minh trang license) |
| G6 | Đạt | 28.5 MB, 211,043 dòng; đọc toàn bộ CSV 1.0s |

5 dòng đầu (bằng chứng G1–G3):

| src_ip | dst_ip | type | label |
|---|---|---|---|
| 192.168.1.37 | 192.168.1.193 | backdoor | 1 |
| 192.168.1.193 | 192.168.1.37 | backdoor | 1 |
| 192.168.1.193 | 192.168.1.37 | backdoor | 1 |
| 192.168.1.193 | 192.168.1.37 | backdoor | 1 |
| 192.168.1.193 | 192.168.1.37 | backdoor | 1 |

## Ánh xạ nhãn
| nhan_goc | so_mau | ti_le_% | nhom |
|---|---|---|---|
| normal | 50000 | 23.692 | Normal |
| backdoor | 20000 | 9.477 | Botnet_Backdoor |
| ddos | 20000 | 9.477 | DoS_DDoS |
| dos | 20000 | 9.477 | DoS_DDoS |
| injection | 20000 | 9.477 | Truy_cap_trai_phep |
| password | 20000 | 9.477 | Truy_cap_trai_phep |
| ransomware | 20000 | 9.477 | Khac |
| scanning | 20000 | 9.477 | Truy_cap_trai_phep |
| xss | 20000 | 9.477 | Truy_cap_trai_phep |
| mitm | 1043 | 0.494 | Khac |

## Thống kê
Số IP nguồn khác nhau: 51   Số IP đích khác nhau: 753
Tỉ lệ bình thường: 23.69%   Lớp nhỏ nhất: mitm (0.494%)

## Chấm điểm
| Tiêu chí | Điểm | Lý do |
|---|---|---|
| Khả năng dựng đồ thị (30) | 25 | 51 IP nguồn, 753 IP đích (bản rút gọn); nhiều nạn nhân cho mỗi nguồn tấn công |
| Độ phủ tấn công (20) | 18 | Đủ 3 nhóm: ddos/dos; backdoor; scanning/password/injection/xss, mỗi nhóm ≥ 20.000 mẫu. Botnet chỉ có backdoor, không có C&C riêng |
| Tính IoT thực tế (15) | 11 | Testbed IoT/IIoT gồm thiết bị thật và mô phỏng (hỗn hợp) |
| Cân bằng lớp (10) | 7 | Normal 23.69%; lớp nhỏ nhất mitm (0.494%) < 0,5% |
| Chi phí xử lý (10) | 8 | CSV sẵn, bản rút gọn 29 MB; bản Processed_Network_dataset có ts lớn hơn nhiều (cần kiểm tra dung lượng) |
| Khả năng so sánh (10) | 9 | Nhiều bài GNN/IDS (chủ yếu bản NF) |
| Độ khó hợp lý (5) | 3 | Binary ≈ 99%+; multi-class khó hơn |
| **Tổng (100)** | **81** | |

## Kết luận
Nên chọn? Làm dataset phụ — cần tải bản `Processed_Network_dataset` (có `ts`) để kiểm chứng giả định thứ tự dòng của NF-ToN-IoT-v2
Rủi ro: Bản đang có thiếu `ts` (trượt G2) và xếp theo lớp; bản đầy đủ lớn hơn nhiều; mitm < 0,5%.

_Sinh tự động bởi `notebooks/phong_tuan1_dataset_graph.ipynb`; phần license/bài báo cần đối chiếu nguồn gốc._
