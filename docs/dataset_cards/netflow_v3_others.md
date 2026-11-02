# Phiếu tóm tắt: 3 bộ NetFlow v3 còn lại (NF-CSE-CIC-IDS2018-v3, NF-BoT-IoT-v3, NF-UNSW-NB15-v3)
Người khảo sát: Phúc   Ngày: 2026-11-02
Nguồn, license, bài báo: giống [nf_ton_iot_v3.md](nf_ton_iot_v3.md) (trang UQ NIDS datasets; bản Kaggle `athena21/netflow-v3-datasets`; Luay et al., IEEE Access 2026).
Bằng chứng: [notebooks/01_verify_netflow_v3.ipynb](../../notebooks/01_verify_netflow_v3.ipynb) chạy trên Kaggle (`minucelowcode/gnn-iot-ids-01-verify-netflow-v3`, bản 1); output trong `reports/tables/gd1_<bộ>_check.json`.

Phiếu này chỉ ghi kết quả xác minh của task 1.2. Chấm điểm và chọn bộ phụ làm ở task 1.4.

## Đối chiếu với số liệu trang UQ
| Bộ | Dòng | Benign | Tấn công | Khớp trang UQ |
|---|---|---|---|---|
| NF-CSE-CIC-IDS2018-v3 | 20.115.529 | 17.514.626 | 2.600.903 | Khớp cả 3 |
| NF-BoT-IoT-v3 | 16.933.808 | 51.989 | 16.881.819 | Benign và tấn công khớp. Trang ghi tổng 16.993.808, trong khi benign + tấn công = 16.933.808 = số đếm được, nên **trang UQ đánh máy nhầm** |
| NF-UNSW-NB15-v3 | 2.365.424 | 2.237.731 | 127.693 | Dòng và benign khớp. Trang ghi tấn công 127.639, nhưng tổng từng lớp trên chính trang đó là 127.693 = số đếm được, nên **trang UQ đánh máy nhầm** |

SHA-256: CIC `242a6971cc801eae621b1fc4d966db2cd0af9cc866805f36a6fc5d0058dfbb74`, BoT `8bde1f6f1c8bc59dcb49828fb5b9d65c0b63e06d92b2b9f15b37159e923009ea`, UNSW `4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5`.

## Timestamp và thứ tự
Cả 3 bộ đều có 55 cột (53 đặc trưng + `Label`, `Attack`), gồm `FLOW_START_MILLISECONDS` và `FLOW_END_MILLISECONDS`, không có giá trị null.

| Bộ | Từ (UTC) | Đến (UTC) | Bao phủ | Dòng có ts không giảm | Số bước lùi | Bước lùi lớn nhất |
|---|---|---|---|---|---|---|
| NF-CSE-CIC-IDS2018-v3 | 2018-02-14 12:28 | 2018-03-02 20:04 | 391,6 h, 11 ngày có dữ liệu | 99,99999% | 1 | 718.755 s (khoảng 8,3 ngày; có lẽ là chỗ nối 2 file) |
| NF-BoT-IoT-v3 | 2018-05-15 00:27 | 2018-06-19 04:17 | 843,8 h, 6 ngày có dữ liệu | 99,76% | 40.736 | 1.131.522 s |
| NF-UNSW-NB15-v3 | 2015-01-22 11:49 | 2015-02-18 12:29 | 648,7 h, 3 ngày có dữ liệu | 85,8% | 335.443 | 2.305.101 s |

Hệ quả: CIC gần như đã sắp theo thời gian. BoT và UNSW **phải sắp lại theo `FLOW_START_MILLISECONDS`** trước khi chia cửa sổ (`clean_flows` đã sắp theo `ts`).

## Lớp và IP
| Bộ | IP nguồn / đích | Các lớp (số mẫu) | Ánh xạ sang 5 nhóm của đề tài |
|---|---|---|---|
| NF-CSE-CIC-IDS2018-v3 | 181.876 / 29.036 | DDOS_attack-HOIC 1.032.311; FTP-BruteForce 386.720; DDoS_attacks-LOIC-HTTP 288.589; **Bot 207.703**; SSH-Bruteforce 188.474; Infilteration 188.152; DoS_attacks-SlowHTTPTest 105.550; DoS_attacks-Hulk 100.076; DoS_attacks-GoldenEye 61.300; DoS_attacks-Slowloris 36.040; DDOS_attack-LOIC-UDP 3.450; Brute_Force_-Web 1.618; Brute_Force_-XSS 480; SQL_Injection 440 | Đủ cả 3 nhóm chính (DoS_DDoS, Botnet_Backdoor qua `Bot`, Truy_cap_trai_phep qua brute force/injection/infiltration). **Cần mở rộng `GROUP_MAP`** vì tên lớp dài, ví dụ `DoS_attacks-Hulk` hiện bị đưa vào Khac |
| NF-BoT-IoT-v3 | 20 / 291 | DoS 8.034.190; DDoS 7.150.882; Reconnaissance 1.695.132; Theft 1.615; Benign 51.989 (0,3%) | **Không có nhóm Botnet_Backdoor**; chỉ có 20 IP nguồn |
| NF-UNSW-NB15-v3 | 40 / 40 | Exploits 42.748; Fuzzers 33.816; Generic 19.651; Reconnaissance 17.074; DoS 5.980; Backdoor 4.659; Shellcode 2.381; Analysis 1.226; Worms 158 | Không phải IoT; 40 IP; nhiều lớp sẽ rơi vào Khac |

Phân bố theo ngày: CIC mỗi ngày có 1–3 loại tấn công, `Bot` chỉ có ngày 02/03, Infiltration ngày 28/02–01/03. BoT có DoS/DDoS dồn hết vào ngày 04/06. UNSW chỉ có 3 ngày. Bảng đầy đủ nằm trong các file JSON.

## Nhận định ban đầu (cho task 1.4)
- **NF-CSE-CIC-IDS2018-v3** là ứng viên bộ phụ tốt nhất trong các bản v3: có timestamp, đủ 3 nhóm tấn công, nhiều IP. Nhược điểm: không phải IoT, và `Bot` chỉ có trong 1 ngày nên chia theo thời gian sẽ khó.
- **NF-BoT-IoT-v3** là IoT nhưng có rất ít IP và thiếu nhóm Botnet_Backdoor; nếu dùng thì chỉ để kiểm tổng quát hóa cho DoS/DDoS và Recon.
- **NF-UNSW-NB15-v3**: không phải IoT, ít IP, không khuyến nghị.
