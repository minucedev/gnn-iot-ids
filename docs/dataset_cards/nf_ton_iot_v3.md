# Phiếu dataset: NF-ToN-IoT-v3
Người khảo sát: Phúc   Ngày: 2026-11-02
Link chính thức: https://staff.itee.uq.edu.au/marius/NIDS_datasets/ (bản ghi UQ eSpace https://doi.org/10.48610/44d7c5e, file tại `rdm.uq.edu.au/files/343e2e8c-6e6e-4a0c-813d-a46acea1b7f4`)
License: "Permitted Re-Use with Commercial Use Restriction", Open Access (theo API UQ eSpace); bản đăng lại trên Kaggle ghi CC-BY-NC-SA-4.0. Dùng cho học thuật được, phải trích dẫn.
Bài báo gốc: Luay et al., IEEE Access 14 (2026), trích dẫn đầy đủ ở cuối phiếu
Dung lượng: 5.302.886.266 byte (4,94 GiB), CSV   Số mẫu: 27.520.260   Thời gian đọc thử: đọc 6 cột bằng pyarrow trên Kaggle 92 s, 1,70 GiB RAM

**Nguồn dữ liệu đã kiểm:** link tải chính thức của UQ (`api.rdm.uq.edu.au`) trả về 401 nếu không đăng nhập. Vì vậy dùng bản đăng lại `athena21/netflow-v3-datasets` trên Kaggle (tác giả ghi là bản gốc, không chỉnh sửa, nguồn là trang UQ). Bản này **khớp đúng số liệu UQ công bố**: 27.520.260 dòng, 16.792.214 benign, 10.728.046 tấn công. SHA-256 file: `53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78`.

Bằng chứng: [notebooks/01_verify_netflow_v3.ipynb](../../notebooks/01_verify_netflow_v3.ipynb) chạy trên Kaggle (`minucelowcode/gnn-iot-ids-01-verify-netflow-v3`, bản 1); output ở [reports/tables/gd1_nf_ton_iot_v3_check.json](../../reports/tables/gd1_nf_ton_iot_v3_check.json). Lần chạy trước bằng notebook riêng cho ToN (commit `1840dc9`) cho cùng kết quả và cùng SHA-256. Ba bộ v3 còn lại: [netflow_v3_others.md](netflow_v3_others.md).

## Điều kiện bắt buộc
| # | Đạt? | Bằng chứng |
|---|---|---|
| G1 | Đạt | Cột IP: `IPV4_SRC_ADDR`, `IPV4_DST_ADDR` |
| G2 | **Đạt** | Có `FLOW_START_MILLISECONDS`, `FLOW_END_MILLISECONDS` (Unix ms, UTC), không có giá trị null; 100% dòng có ts không giảm (file đã sắp xếp theo thời gian) |
| G3 | Đạt | Cột nhãn: `Label` (0/1), `Attack` (10 giá trị) |
| G4 | Đạt | Có đủ DoS_DDoS, Botnet_Backdoor, Truy_cap_trai_phep |
| G5 | Đạt | Open Access, cấm dùng thương mại; trích dẫn bài báo |
| G6 | Đạt | 4,94 GiB; trên Kaggle đọc 6 cột mất 92 s và 1,70 GiB RAM. Xử lý trên Kaggle, không tải về máy |

5 dòng đầu (bằng chứng G1–G3, tiêu chí 2 của task 1.2; đủ 55 cột nằm trong file JSON):

| FLOW_START_MILLISECONDS | FLOW_END_MILLISECONDS | IPV4_SRC_ADDR | IPV4_DST_ADDR | Attack | Label |
|---|---|---|---|---|---|
| 1556028590618 | 1556028590618 | 192.168.1.32 | 192.168.1.169 | Benign | 0 |
| 1556028590618 | 1556028590621 | 192.168.1.32 | 192.168.1.194 | Benign | 0 |
| 1556028590618 | 1556028590621 | 192.168.1.32 | 192.168.1.193 | scanning | 1 |
| 1556028590624 | 1556028590628 | 192.168.1.32 | 192.168.1.194 | Benign | 0 |
| 1556028590624 | 1556028590624 | 192.168.1.32 | 192.168.1.180 | Benign | 0 |

## Thứ tự và khoảng thời gian (tiêu chí 3 của task 1.2)
| Chỉ số | Giá trị |
|---|---|
| ts nhỏ nhất (UTC) | 2019-04-23 14:09:50.618 |
| ts lớn nhất (UTC) | 2019-04-29 14:45:55.177 |
| Khoảng bao phủ | 144,6 giờ (khoảng 6 ngày, nằm trên 7 ngày lịch UTC) |
| Tỉ lệ dòng có ts ≥ dòng trước | **1,000000** (không có bước lùi nào) |
| Tỉ lệ dòng có ts > dòng trước | 0,447 (nhiều flow bắt đầu cùng 1 ms) |
| Tỉ lệ dòng có FLOW_END ≥ FLOW_START | 1,000000 |
| Thời lượng flow p50 / p99 | 0 ms / 1.680 ms |

Phân bố theo ngày (UTC), số flow:

| Ngày | Benign | Tấn công |
|---|---|---|
| 2019-04-23 | 5.580.827 | scanning 262.207 |
| 2019-04-24 | 3.108.051 | scanning 1.096.770; dos 133.307 |
| 2019-04-25 | 3.434.087 | ddos 2.487.061; injection 381.777; dos 70.149 |
| 2019-04-26 | 2.140.689 | ddos 1.654.195; password 1.032.571 |
| 2019-04-27 | 354.308 | xss 2.834.435; password 562.206 |
| 2019-04-28 | 215.980 | Backdoor 109.536; ransomware 3.971 |
| 2019-04-29 | 1.958.272 | Backdoor 93.848; mitm 6.013 |

Bảng 4 của bản arXiv ghi 10 ngày, 3 ngày đầu chỉ có benign. Dữ liệu thực tế chỉ có 7 ngày UTC và ngày đầu đã có scanning. Thứ tự các loại tấn công thì khớp (scanning → dos/ddos/injection → password → xss → backdoor/ransomware → mitm). Có thể bài báo đếm theo ngày thu PCAP gốc, chưa xác minh.

## Ánh xạ nhãn
| nhan_goc | so_mau | ti_le_% | nhom |
|---|---|---|---|
| Benign | 16.792.214 | 61,018 | Normal |
| ddos | 4.141.256 | 15,048 | DoS_DDoS |
| xss | 2.834.435 | 10,299 | Truy_cap_trai_phep |
| password | 1.594.777 | 5,795 | Truy_cap_trai_phep |
| scanning | 1.358.977 | 4,938 | Truy_cap_trai_phep |
| injection | 381.777 | 1,387 | Truy_cap_trai_phep |
| dos | 203.456 | 0,739 | DoS_DDoS |
| Backdoor | 203.384 | 0,739 | Botnet_Backdoor |
| mitm | 6.013 | 0,022 | Khac |
| ransomware | 3.971 | 0,014 | Khac |

Theo nhóm: Normal 61,018%; Truy_cap_trai_phep 22,420%; DoS_DDoS 15,787%; Botnet_Backdoor 0,739%; Khac 0,036%.
v3 viết `Backdoor` (hoa), còn v2 viết `backdoor`. `to_group` trong `src/build_graph.py` đã chuyển sang chữ thường trước khi ánh xạ, có test `test_to_group_case_insensitive`.

## Thống kê
Số IP nguồn khác nhau: 15.270   Số IP đích khác nhau: 8.777
Tỉ lệ bình thường: 61,02%   Lớp nhỏ nhất: ransomware (0,014%)

## Khác biệt so với NF-ToN-IoT-v2
| Mục | v2 | v3 |
|---|---|---|
| Năm phát hành | 2021–2022 (Sarhan et al.) | 2025 (dữ liệu), bài báo IEEE Access 2026 |
| Số đặc trưng | 43 | 53 (thêm 10 đặc trưng thời gian: `FLOW_START/END_MILLISECONDS`, IAT min/max/avg/stddev hai chiều) |
| Timestamp | Không có; phải giả định thứ tự dòng là thời gian | Có, Unix ms; file đã sắp theo thời gian |
| Số flow | 16.940.496 | 27.520.260 |
| Tỉ lệ Benign | 36,0% | 61,0% |
| backdoor | 16.809 | 203.384 |
| IP nguồn / đích | 23.079 / 6.868 | 15.270 / 8.777 |
| Dung lượng CSV | 2,55 GB | 5,30 GB |

## Chấm điểm (đề xuất, Phúc duyệt)
| Tiêu chí | Điểm | Lý do |
|---|---|---|
| Khả năng dựng đồ thị (30) | 28 | Có timestamp thật để chia cửa sổ W = 5 s; 15.270 IP nguồn, 8.777 IP đích |
| Độ phủ tấn công (20) | 20 | Đủ 3 nhóm, nhóm nhỏ nhất Backdoor có 203K mẫu |
| Tính IoT thực tế (15) | 11 | Cùng testbed ToN-IoT (hỗn hợp) |
| Cân bằng lớp (10) | 5 | Benign 61%; lớp nhỏ nhất ransomware 0,014% (< 0,5%) |
| Chi phí xử lý (10) | 7 | CSV 5,3 GB (hơi > 5 GB); xử lý trên Kaggle |
| Khả năng so sánh (10) | 5 | Bộ mới (2025), ít bài báo GNN-IDS dùng |
| Độ khó hợp lý (5) | 3 | Chưa có số; v2 binary ≈ 99% |
| **Tổng (100)** | **79** | v2 được 78 |

## Kết luận
Nên chọn? **Đề xuất dùng làm bộ chính cho D1**, vì có timestamp thật, đã xác minh bằng code, và đáp ứng "tính thời sự". Phúc chốt ở task 0.2.
Rủi ro:
- **Chia theo thời gian:** Backdoor chỉ xuất hiện ngày 28–29/04, ransomware ngày 28/04, mitm ngày 29/04. Nếu chia `chronological`, các lớp này chỉ nằm ở val/test, giống vấn đề của v2. Cần cân nhắc khi chốt D3 (`blocked`).
- **Kích thước:** 5,3 GB nên phải đọc theo cột trên Kaggle.
- **Nguồn:** tải chính thức cần đăng nhập UQ; bản Kaggle đã khớp số liệu nhưng chưa so được hash với bản UQ.
- **Đồng hồ thời gian:** 55% flow trùng ms với flow trước, nên chia cửa sổ theo `FLOW_START_MILLISECONDS` là ổn.

## Trích dẫn
- Dữ liệu: Luay, M., Layeghy, S., Sarhan, M., Hoseininoorbin, S., Moustafa, N., & Portmann, M. (2025). *NF-ToN-IoT-v3* [Data Collection]. The University of Queensland. https://doi.org/10.48610/44d7c5e
- Bài báo (bản đã đăng): M. Luay, S. Layeghy, N. Noorbin, M. Sarhan, G. Kulatilleke, N. Moustafa, M. Portmann, "Time Matters: Temporal NetFlow Features for ML-Based Network Intrusion Detection," *IEEE Access*, vol. 14, pp. 66899–66913, 2026. https://doi.org/10.1109/ACCESS.2026.3688204
- Bản preprint: M. Luay, S. Layeghy, S. Hosseininoorbin, M. Sarhan, N. Moustafa, M. Portmann, "Temporal Analysis of NetFlow Datasets for Network Intrusion Detection Systems," arXiv:2503.04404, 2025 (bản v3 ngày 28/08/2026).
- Bộ ToN-IoT gốc: N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets," *Sustainable Cities and Society*, vol. 72, 102994, 2021. https://doi.org/10.1016/j.scs.2021.102994
