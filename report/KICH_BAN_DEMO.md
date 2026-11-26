# Kịch bản demo: phát hiện xâm nhập IoT thời gian thực

Viết ngày 27/11/2026. Dùng khi bảo vệ đề tài, sau phần trình bày slide. Thời lượng **6 phút**. Người trình bày: Phúc.

Mọi số liệu trong kịch bản lấy từ lần chạy mô phỏng thật trên toàn bộ tập test NF-ToN-IoT-v3 (notebook `20_realtime`, mô hình đã đóng băng ở tag `freeze-v1`): `reports/tables/latency.json`, `reports/tables/realtime_eval.json`, `reports/kaggle/gnn-iot-ids-20a-undetected/undetected_episodes.json`.

## 1. Thông điệp của demo

Người xem cần thấy 3 điều:
1. **Chạy được thật:** lưu lượng được phát lại theo thời gian, cứ 5 giây đóng một cửa sổ, dựng đồ thị, mô hình dự đoán, cảnh báo hiện ra.
2. **Đủ nhanh và đủ nhẹ:** mỗi cửa sổ xử lý khoảng 0,1 giây trên **1 luồng CPU**, xa ngưỡng 1 giây.
3. **Trung thực về giới hạn:** bắt tốt các đợt tấn công lớn, bỏ sót các đợt rất nhỏ và lớp hiếm.

## 2. Hình thức

| Phương án | Dùng khi | Rủi ro |
|---|---|---|
| **A. Chạy trực tiếp trên Kaggle** (notebook `22_demo`, CPU) | Chính | Cần Internet, đăng nhập Kaggle, khởi động phiên mất 1–2 phút |
| **B. Trang phát lại offline** (`report/demo/demo_replay.html`, mở bằng trình duyệt) | Dự phòng khi mất mạng hoặc Kaggle chậm | Không có; chỉ hiển thị lại kết quả đã chạy |
| **C. Video quay sẵn** (`report/demo/demo.mp4`, 3–4 phút) | Dự phòng cuối cùng | Không có |

Demo **không chạy Python trên máy trình bày** (Windows chặn Python của dự án; mọi tính toán của đề tài chạy trên Kaggle). Phương án B chỉ là trang HTML đọc file JSON đã sinh trên Kaggle.

## 3. Ba đoạn dữ liệu dùng để demo

Lấy từ tập **test** NF-ToN-IoT-v3, giờ UTC. Mỗi đoạn được cắt sẵn thành file nhỏ để demo khởi động nhanh, không phải đọc file CSV 5,3 GB.

| Đoạn | Khoảng thời gian dữ liệu | Có gì | Kết quả thật đã đo |
|---|---|---|---|
| **D1. Yên tĩnh rồi DDoS** | 24/04/2019 20:07:00 → 20:12:00 | Khoảng 2 phút bình thường, rồi đợt DDoS bắt đầu lúc **20:09:15** (đợt kéo dài 110 phút, 22 IP) | Cảnh báo đầu tiên sau **5,0 giây** |
| **D2. Dò quét** | 26/04/2019 18:29:30 → 18:33:00 | Đợt truy cập trái phép bắt đầu **18:30:45** (kéo dài 127 phút, 9 IP) | Phát hiện sau **5,1 giây** |
| **D3. Backdoor** | 28/04/2019 15:27:00 → 15:31:00 | Đợt backdoor bắt đầu **15:28:35** (kéo dài 104 phút, 5 IP) | Phát hiện sau **4,4 giây** |

Lưu ý khi dựng: kiểm tra phần trước mốc tấn công của D1 chỉ có lưu lượng bình thường (không có đợt tấn công khác); nếu có thì dời mốc bắt đầu.

## 4. Kịch bản từng cảnh

Tốc độ phát lại **×5**: 1 phút dữ liệu chiếu trong 12 giây.

### Cảnh 0. Giới thiệu màn hình (30 giây)

- **Màn hình:** notebook `22_demo` trên Kaggle, ô đầu tiên in cấu hình máy: CPU Intel Xeon 2,2 GHz, **giới hạn 1 luồng**, mô hình đang dùng, ngưỡng đã đóng băng.
- **Lời nói:** "Đây là bộ phát hiện chạy như trên gateway IoT. Lưu lượng NetFlow thật của bộ NF-ToN-IoT-v3 được phát lại theo đúng thứ tự thời gian, nhanh gấp 5 lần. Cứ mỗi 5 giây, hệ thống đóng một cửa sổ, dựng đồ thị các IP đang giao tiếp, rồi mô hình hybrid GAT + LightGBM dự đoán từng IP. Mô hình và ngưỡng đã được đóng băng từ trước khi chạy tập test."

### Cảnh 1. Mạng yên tĩnh (40 giây, đoạn D1 phần đầu)

- **Màn hình:** mỗi dòng một cửa sổ: giờ, số node, số cạnh, thời gian xử lý, số cảnh báo (bằng 0).
  ```
  [20:07:05] cửa sổ   12 node   11 cạnh   103 ms   cảnh báo: 0
  [20:07:10] cửa sổ    9 node    8 cạnh   101 ms   cảnh báo: 0
  ```
- **Lời nói:** "Khi mạng bình thường, hầu như không có cảnh báo. Trên toàn bộ các đoạn chỉ có lưu lượng bình thường của tập test, hệ thống báo nhầm khoảng 15 lần mỗi giờ, tức khoảng 0,2% số IP bình thường."

### Cảnh 2. DDoS bắt đầu (60 giây, đoạn D1 từ 20:09:15)

- **Màn hình:** dòng cảnh báo đỏ xuất hiện ngay sau mốc tấn công.
  ```
  [20:09:20] cửa sổ   41 node   38 cạnh   109 ms   ⚠ 3 cảnh báo
             ⚠ 192.168.1.39   DoS_DDoS   p = 1,00
  ```
  Sau đó hiện **hình đồ thị của cửa sổ đầu tiên có cảnh báo**: node tấn công tô đỏ, thấy rõ nhiều IP cùng dồn vào một đích (fan-in).
- **Lời nói:** "Đợt DDoS bắt đầu lúc 20 giờ 9 phút 15 giây. Hệ thống phát hiện sau 5 giây, tức ngay ở cửa sổ đầu tiên. Nhìn đồ thị: nhiều máy cùng dồn về một đích. GAT nhìn được cấu trúc này nhờ thông tin từ các node hàng xóm."

### Cảnh 3. Dò quét và backdoor (60 giây, đoạn D2 và D3)

- **Màn hình:** chạy nhanh D2 rồi D3; mỗi đoạn dừng ở cửa sổ có cảnh báo đầu tiên. Với D2, hình đồ thị cho thấy một IP chạm tới nhiều IP (fan-out).
- **Lời nói:** "Dò quét được phát hiện sau 5,1 giây, backdoor sau 4,4 giây. Trên toàn bộ tập test, cả 4 đợt DDoS đều bị phát hiện, thời gian phát hiện trung vị là 5,1 giây."

### Cảnh 4. Tốc độ và độ nhẹ (60 giây)

- **Màn hình:** hình `gd8_latency_hist.png` (phân bố độ trễ) và một bảng nhỏ:

  | | Hybrid chính | Hybrid nhẹ |
  |---|---|---|
  | Độ trễ mỗi cửa sổ (trung vị / p95) | 107 / 121 ms | 103 / 115 ms |
  | Phần dựng đồ thị / phần dự đoán (p95) | 96 / 28 ms | — |
  | Số tham số / dung lượng | 144.560 / 9,7 MB | 30.934 / 986 KB |
  | Dự đoán online khớp offline | 100% | — |

- **Lời nói:** "Ngưỡng cho phép là 1 giây mỗi cửa sổ; hệ thống chỉ dùng khoảng 0,12 giây ở mức p95, trên một luồng CPU, xử lý khoảng 1.900 flow mỗi giây. Phần lớn thời gian là dựng đồ thị, không phải mô hình. Bản hybrid nhẹ chỉ 31 nghìn tham số và dưới 1 MB, đủ nhỏ cho thiết bị biên. Kết quả chạy theo luồng khớp 100% với kết quả tính một lần trên cả tập, nên con số trong báo cáo đúng với cách hệ thống chạy thật."
- **Tùy chọn (nếu còn giờ):** chạy lại D1 với `--model hybrid_light` để thấy kết quả gần giống.

### Cảnh 5. Giới hạn (40 giây)

- **Màn hình:** bảng tóm tắt từ `undetected_episodes.json`:

  | Nhóm | Số đợt | Phát hiện | Ghi chú |
  |---|---|---|---|
  | DoS_DDoS | 4 | 4 | Khoảng 5 giây |
  | Botnet_Backdoor | 5 | 4 | Đợt bỏ sót chỉ có 11 flow tấn công |
  | Truy_cap_trai_phep | 5 | 2 | 3 đợt bỏ sót chỉ có 1–2 flow, kéo dài 1 cửa sổ |
  | Khac (mitm, ransomware) | 13 | 3 | Lớp hiếm nhất (0,04% flow), mô hình thường tin là Normal |

- **Lời nói:** "Hệ thống bắt tốt các đợt tấn công có quy mô; 13 trên 27 đợt được phát hiện. Các đợt bị bỏ sót chủ yếu rất nhỏ: 6 đợt chỉ kéo dài một cửa sổ với 1 đến 2 flow, còn mitm và ransomware là lớp có rất ít dữ liệu để học. Đây là hướng cải thiện tiếp theo."

### Cảnh 6. Kết (20 giây)

- **Lời nói:** "Tóm lại: hệ thống chạy được theo thời gian thực, phát hiện các đợt tấn công lớn trong khoảng 5 giây, mỗi cửa sổ xử lý khoảng 0,1 giây trên một luồng CPU, và có bản nhẹ dưới 1 MB. Theo AHP, nếu bắt buộc mô hình dưới 100 nghìn tham số, dưới 1 MB và đạt cả hai chỉ tiêu FPR ≤ 4,08%, recall ≥ 95% thì bản được gợi ý triển khai là hybrid nhẹ."

## 5. Chuẩn bị trước buổi bảo vệ

**Một ngày trước:**
- [ ] Chạy notebook `22_demo` trọn vẹn một lần trên Kaggle; lưu phiên bản (Save Version) để có sẵn output nếu chạy trực tiếp gặp sự cố.
- [ ] Mở `report/demo/demo_replay.html` trên máy trình bày, kiểm tra chạy được khi **tắt mạng**.
- [ ] Quay video phương án C từ phương án A hoặc B; chép `demo.mp4` vào máy trình bày.
- [ ] Tập nói theo kịch bản, bấm giờ: tổng ≤ 6 phút.

**Trước giờ trình bày 10 phút:**
- [ ] Đăng nhập Kaggle, mở notebook `22_demo`, bật phiên (CPU, Internet bật), chạy ô khởi động (nạp mô hình, khởi động nóng).
- [ ] Mở sẵn 3 tab: notebook Kaggle, `demo_replay.html`, video.
- [ ] Tắt thông báo trên máy, phóng to chữ trong notebook.

**Khi có sự cố:** Kaggle chậm hơn 1 phút thì chuyển sang tab B ngay, nói: "Để tiết kiệm thời gian, em chiếu bản phát lại kết quả của đúng lần chạy này."

## 6. Câu hỏi có thể gặp về demo

| Câu hỏi | Trả lời ngắn |
|---|---|
| Đây có phải thời gian thực thật không? | Dữ liệu là lưu lượng đã ghi, được phát lại đúng thứ tự thời gian (tăng tốc để demo). Độ trễ đo trên từng cửa sổ từ lúc đóng cửa sổ tới lúc có dự đoán, trên 1 luồng CPU, toàn bộ 11.449 cửa sổ của tập test. Chưa triển khai trên mạng thật, đã nêu trong phạm vi đề tài. |
| Sao lại mất 5 giây mới phát hiện? | Cửa sổ dài 5 giây; hệ thống chỉ quyết định khi cửa sổ đóng. Phần xử lý chỉ khoảng 0,1 giây. Muốn nhanh hơn thì dùng cửa sổ ngắn hơn (đã thử 1 và 2 giây trong ablation). |
| 14 đợt bị bỏ sót thì sao? | Chủ yếu là đợt rất nhỏ (1–2 flow, 1 cửa sổ) và lớp mitm, ransomware rất hiếm. Các đợt lớn đều bị phát hiện. Đã phân tích từng đợt trong báo cáo mục 5.11. |
| Báo nhầm 15 lần mỗi giờ có nhiều không? | Tương ứng khoảng 0,2% số IP bình thường trong các đoạn không có tấn công. Có thể giảm thêm bằng cách nâng ngưỡng, đổi lại bỏ sót nhiều hơn. |
| Sao demo dùng hybrid chính mà không dùng bản nhẹ? | AHP không ràng buộc chọn hybrid chính. Nếu chỉ bắt buộc ≤ 100 nghìn tham số và ≤ 1 MB thì AHP chọn MLP; nếu bắt buộc thêm FPR ≤ 4,08% và recall ≥ 95% (MLP không đạt recall) thì chọn hybrid nhẹ. Nguồn: `reports/kaggle/gnn-iot-ids-19b-ahp-constrained/ahp_constrained.json`. Demo chạy được cả hai bản; độ trễ gần như nhau vì phần lớn thời gian là dựng đồ thị. |

## 7. Việc cần làm để có demo

Phần mô phỏng đã có (`scripts/detector.py`, notebook `20_realtime`). Còn thiếu, làm trên Kaggle theo quy tắc của dự án (không sửa mô hình hay ngưỡng đã đóng băng):

1. Thêm vào `scripts/detector.py` tùy chọn `--start` và `--end` (giờ UTC) để chạy đúng một đoạn, và chế độ in từng cửa sổ theo định dạng ở cảnh 1–2.
2. Notebook `22a_demo_prep` (CPU): cắt 3 đoạn D1, D2, D3 từ CSV gốc thành file nhỏ; kiểm tra phần đầu D1 chỉ có lưu lượng bình thường; chạy thử detector trên từng đoạn; xuất `demo_replay.json` (từng cửa sổ: giờ, số node, cạnh, thời gian xử lý, cảnh báo, và node, cạnh của các cửa sổ có cảnh báo để vẽ).
3. Notebook `22_demo` (CPU): ô khởi động; 3 ô chạy D1, D2, D3 với `--speedup 5`; ô vẽ đồ thị cửa sổ có cảnh báo đầu tiên; ô bảng tốc độ và giới hạn.
4. `report/demo/demo_replay.html`: một trang HTML không cần Internet, đọc `demo_replay.json` nhúng sẵn, phát lại từng cửa sổ, hiện cảnh báo và vẽ đồ thị cửa sổ có cảnh báo.
5. Quay `report/demo/demo.mp4` (3–4 phút).
