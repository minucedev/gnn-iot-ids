# Phiếu dataset: NF-ToN-IoT-v2
Người khảo sát: Phong   Ngày: 2026-10-22
Link tải: https://staff.itee.uq.edu.au/marius/NIDS_datasets/   License: Học thuật, trích dẫn Sarhan et al.   Bài báo gốc: Sarhan et al. (2022), MONET
Dung lượng: 2552.5 MB (`data/ToN_IoT/NF-ToN-IoT-v2.csv`)   Số mẫu: 16,940,496   Thời gian đọc thử: đọc 3 cột (IP nguồn, IP đích, nhãn) bằng pyarrow 6.4s

## Điều kiện bắt buộc
| # | Đạt? | Bằng chứng |
|---|---|---|
| G1 | Đạt | Cột IP: ('IPV4_SRC_ADDR', 'IPV4_DST_ADDR') |
| G2 | **Không** | Không có cột timestamp. Thứ tự dòng: **theo pha (có thể theo thời gian)** (độ dài chuỗi nhãn liên tiếp TB 11.3; tỉ lệ lớp trội trong khối 100k dòng 0.906 so với toàn file 0.36) |
| G3 | Đạt | Cột nhãn: ['Attack', 'Label'], 10 giá trị |
| G4 | Đạt | Nhóm tấn công có mặt: ['DoS_DDoS', 'Botnet_Backdoor', 'Truy_cap_trai_phep'] |
| G5 | Đạt | Học thuật, trích dẫn Sarhan et al. (cần xác minh trang license) |
| G6 | Đạt | 2552.5 MB, 16,940,496 dòng; đọc 3 cột (IP nguồn, IP đích, nhãn) bằng pyarrow 6.4s |

5 dòng đầu (bằng chứng G1–G3):

| IPV4_SRC_ADDR | IPV4_DST_ADDR | Attack | Label |
|---|---|---|---|
| 192.168.1.193 | 192.168.1.33 | ransomware | 1 |
| 192.168.1.193 | 192.168.1.152 | Benign | 0 |
| 192.168.1.152 | 192.168.1.193 | Benign | 0 |
| 192.168.1.169 | 239.255.255.250 | Benign | 0 |
| 192.168.1.79 | 192.168.1.255 | Benign | 0 |

## Ánh xạ nhãn
| nhan_goc | so_mau | ti_le_% | nhom |
|---|---|---|---|
| Benign | 6099469 | 36.005 | Normal |
| scanning | 3781419 | 22.322 | Truy_cap_trai_phep |
| xss | 2455020 | 14.492 | Truy_cap_trai_phep |
| ddos | 2026234 | 11.961 | DoS_DDoS |
| password | 1153323 | 6.808 | Truy_cap_trai_phep |
| dos | 712609 | 4.207 | DoS_DDoS |
| injection | 684465 | 4.04 | Truy_cap_trai_phep |
| backdoor | 16809 | 0.099 | Botnet_Backdoor |
| mitm | 7723 | 0.046 | Khac |
| ransomware | 3425 | 0.02 | Khac |

## Thống kê
Số IP nguồn khác nhau: 23079   Số IP đích khác nhau: 6868
Tỉ lệ bình thường: 36.01%   Lớp nhỏ nhất: ransomware (0.020%)

## Chấm điểm
| Tiêu chí | Điểm | Lý do |
|---|---|---|
| Khả năng dựng đồ thị (30) | 22 | 23,079 IP nguồn, 6,868 IP đích; không có timestamp, phải dùng thứ tự dòng làm thời gian (giả định) |
| Độ phủ tấn công (20) | 20 | Đủ 3 nhóm, số mẫu rất lớn (scanning 3,8M; ddos 2,0M; backdoor 16,8K) |
| Tính IoT thực tế (15) | 11 | Cùng testbed ToN-IoT (hỗn hợp) |
| Cân bằng lớp (10) | 4 | Benign 36.01%; lớp nhỏ nhất ransomware (0.020%) |
| Chi phí xử lý (10) | 8 | CSV có sẵn 2,552.5 MB (< 5 GB), 16,940,496 dòng |
| Khả năng so sánh (10) | 10 | Bộ chuẩn của E-GraphSAGE, Anomal-E, E-ResGAT |
| Độ khó hợp lý (5) | 3 | Binary ≈ 99%+ |
| **Tổng (100)** | **78** | |

## Kết luận
Nên chọn? Có — đề xuất làm dataset chính (dùng thứ tự dòng thay timestamp)
Rủi ro: Không có timestamp, thứ tự dòng = thời gian chỉ là giả định; chia theo thời gian làm backdoor/ransomware chỉ nằm ở test; Benign gồm hơn 23 nghìn IP nguồn (nhiều IP bên ngoài); FLOW_DURATION có giá trị tràn 32-bit.

_Sinh tự động bởi `notebooks/phong_tuan1_dataset_graph.ipynb`; phần license/bài báo cần đối chiếu nguồn gốc._
