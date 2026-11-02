# Phiếu dataset: BoT-IoT (đầy đủ, 74 file)
Người khảo sát: Phong   Ngày: 2026-10-23
Link tải: https://research.unsw.edu.au/projects/bot-iot-dataset (bản dùng: Kaggle `vigneshvenkateswaran/bot-iot`)   License: học thuật, bắt buộc trích dẫn (UNSW)   Bài báo gốc: Koroniotis et al. (2019), FGCS
Dung lượng: 14.0 GB (`data/BoT-IoT/kaggle_full/data_1..74.csv`)   Số mẫu: 73,370,443   Thời gian đọc thử: 27s (6 cột)

## Điều kiện bắt buộc
| # | Đạt? | Bằng chứng |
|---|---|---|
| G1 | Đạt | Cột IP: saddr, daddr |
| G2 | Đạt | Cột thời gian: stime (giây epoch, độ phân giải µs) |
| G3 | Đạt | Cột nhãn: category (5 giá trị), subcategory (11 giá trị), attack (0/1) |
| G4 | **Không** | Nhóm tấn công có mặt: ['DoS_DDoS', 'Truy_cap_trai_phep']; thiếu Botnet_Backdoor |
| G5 | Đạt | Học thuật, bắt buộc trích dẫn Koroniotis et al. (2019) (cần xác minh trang license) |
| G6 | Đạt | 14.0 GB; đọc 6 cột toàn bộ 27s; xử lý đầy đủ ≈ 15–25 phút trên laptop |

5 dòng đầu (bằng chứng G1–G3):

| pkSeqID | stime | saddr | daddr | category | subcategory |
|---|---|---|---|---|---|
| 1 | 1526344121.188091 | 192.168.100.1 | 192.168.100.3 | Normal | Normal |
| 2 | 1526344223.197482 | 192.168.100.7 | 192.168.100.4 | Normal | Normal |
| 3 | 1526344227.029374 | 192.168.100.149 | 27.124.125.250 | Normal | Normal |
| 4 | 1526344228.312317 | 192.168.100.4 | 192.168.100.7 | Normal | Normal |
| 5 | 1526344302.636337 | 192.168.100.27 | 192.168.100.1 | Normal | Normal |

## Ánh xạ nhãn
| category | subcategory | so_mau | ti_le_% | nhom |
|---|---|---|---|---|
| DoS | UDP | 20659491 | 28.1578 | DoS_DDoS |
| DDoS | TCP | 19547603 | 26.6423 | DoS_DDoS |
| DDoS | UDP | 18965106 | 25.8484 | DoS_DDoS |
| DoS | TCP | 12315997 | 16.786 | DoS_DDoS |
| Reconnaissance | Service_Scan | 1463364 | 1.9945 | Truy_cap_trai_phep |
| Reconnaissance | OS_Fingerprint | 358275 | 0.4883 | Truy_cap_trai_phep |
| DoS | HTTP | 29706 | 0.0405 | DoS_DDoS |
| DDoS | HTTP | 19771 | 0.0269 | DoS_DDoS |
| Normal | Normal | 9543 | 0.013 | Normal |
| Theft | Keylogging | 1469 | 0.002 | Truy_cap_trai_phep |
| Theft | Data_Exfiltration | 118 | 0.0002 | Truy_cap_trai_phep |

## Thống kê
Số IP nguồn khác nhau: 28   Số IP đích khác nhau: 294
Tỉ lệ bình thường: 0.0130%   Lớp nhỏ nhất: Data_Exfiltration (0.0002%)
Các file được xếp theo loại tấn công (pkSeqID liên tục), không theo thời gian → phải gộp và sắp lại theo `stime`.

## Chấm điểm
| Tiêu chí | Điểm | Lý do |
|---|---|---|
| Khả năng dựng đồ thị (30) | 10 | 28 IP nguồn, 294 IP đích; tấn công chỉ từ vài IP nội bộ (192.168.100.147–150), nạn nhân ít |
| Độ phủ tấn công (20) | 12 | DDoS/DoS hàng chục triệu, Recon 1,821,639, Theft 1,587; không có nhãn botnet/C&C/backdoor riêng (DDoS do botnet thực hiện nhưng nhãn là DDoS) |
| Tính IoT thực tế (15) | 8 | Thiết bị IoT mô phỏng (Node-RED) trên máy ảo |
| Cân bằng lớp (10) | 1 | Normal chỉ 0.0130% |
| Chi phí xử lý (10) | 4 | 14.0 GB CSV, 74 file, 73,370,443 dòng (> 5 GB) |
| Khả năng so sánh (10) | 9 | Rất phổ biến trong IDS (thường dùng bản 5%), có bản NF cho GNN |
| Độ khó hợp lý (5) | 1 | Mô hình đơn giản đã ≈ 99,9% |
| **Tổng (100)** | **45** | |

## Kết luận
Nên chọn? Không làm dataset chính (dưới 60 điểm); có thể dùng để kiểm tra tổng quát hóa.
Rủi ro: Normal cực ít (FPR đo trên rất ít node), thiếu nhóm botnet/backdoor, quá dễ, ít IP, dung lượng lớn.

_Sinh tự động bởi `notebooks/phong_tuan1_bot_iot_full.ipynb`._
