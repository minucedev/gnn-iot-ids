# Phiếu dataset: NF-BoT-IoT-v2
Người khảo sát: Phong   Ngày: 2026-10-22
Link tải: https://staff.itee.uq.edu.au/marius/NIDS_datasets/   License: Học thuật, trích dẫn Sarhan et al.   Bài báo gốc: Sarhan et al. (2022), MONET
Dung lượng: 5767.2 MB (`data/BoT-IoT/NF-BoT-IoT-v2.csv`)   Số mẫu: 37,763,497   Thời gian đọc thử: đọc 3 cột (IP nguồn, IP đích, nhãn) bằng pyarrow 39.6s

## Điều kiện bắt buộc
| # | Đạt? | Bằng chứng |
|---|---|---|
| G1 | Đạt | Cột IP: ('IPV4_SRC_ADDR', 'IPV4_DST_ADDR') |
| G2 | **Không** | Không có cột timestamp. Thứ tự dòng: **xáo trộn** (độ dài chuỗi nhãn liên tiếp TB 1.8; tỉ lệ lớp trội trong khối 100k dòng 0.485 so với toàn file 0.485) |
| G3 | Đạt | Cột nhãn: ['Attack', 'Label'], 5 giá trị |
| G4 | **Không** | Nhóm tấn công có mặt: ['DoS_DDoS', 'Truy_cap_trai_phep'] |
| G5 | Đạt | Học thuật, trích dẫn Sarhan et al. (cần xác minh trang license) |
| G6 | Đạt | 5767.2 MB, 37,763,497 dòng; đọc 3 cột (IP nguồn, IP đích, nhãn) bằng pyarrow 39.6s |

5 dòng đầu (bằng chứng G1–G3):

| IPV4_SRC_ADDR | IPV4_DST_ADDR | Attack | Label |
|---|---|---|---|
| 192.168.100.149 | 192.168.100.6 | DoS | 1 |
| 192.168.100.147 | 192.168.100.5 | DoS | 1 |
| 192.168.100.148 | 192.168.100.5 | DoS | 1 |
| 192.168.100.147 | 192.168.100.7 | DDoS | 1 |
| 192.168.100.150 | 192.168.100.6 | DDoS | 1 |

## Ánh xạ nhãn
| nhan_goc | so_mau | ti_le_% | nhom |
|---|---|---|---|
| DDoS | 18331847 | 48.544 | DoS_DDoS |
| DoS | 16673183 | 44.152 | DoS_DDoS |
| Reconnaissance | 2620999 | 6.941 | Truy_cap_trai_phep |
| Benign | 135037 | 0.358 | Normal |
| Theft | 2431 | 0.006 | Truy_cap_trai_phep |

## Thống kê
Số IP nguồn khác nhau: 20   Số IP đích khác nhau: 291
Tỉ lệ bình thường: 0.36%   Lớp nhỏ nhất: Theft (0.006%)

## Chấm điểm
| Tiêu chí | Điểm | Lý do |
|---|---|---|
| Khả năng dựng đồ thị (30) | 6 | Chỉ 20 IP nguồn, 291 IP đích; không có timestamp và dòng đã bị xáo trộn → không dựng được cửa sổ |
| Độ phủ tấn công (20) | 12 | DDoS/DoS và Recon rất nhiều, Theft 2.431; không có botnet/C&C riêng |
| Tính IoT thực tế (15) | 8 | Cùng testbed BoT-IoT (mô phỏng) |
| Cân bằng lớp (10) | 2 | Benign 0.36% |
| Chi phí xử lý (10) | 4 | CSV 5,767.2 MB (> 5 GB), 37,763,497 dòng |
| Khả năng so sánh (10) | 9 | E-GraphSAGE, E-ResGAT |
| Độ khó hợp lý (5) | 1 | ≈ 99,9% |
| **Tổng (100)** | **42** | |

## Kết luận
Nên chọn? Không
Rủi ro: Không có timestamp và các dòng đã bị xáo trộn (không dựng được cửa sổ); Benign 0,36%; chỉ 20 IP nguồn; quá dễ.

_Sinh tự động bởi `notebooks/phong_tuan1_dataset_graph.ipynb`; phần license/bài báo cần đối chiếu nguồn gốc._
