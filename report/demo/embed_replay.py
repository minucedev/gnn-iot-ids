"""Nhúng demo_replay.json (do notebook 22a_demo_prep sinh trên Kaggle) vào report/demo/demo_replay.html để trang tự chứa, mở được khi không có mạng.

Chạy (chỉ dùng thư viện chuẩn, không tính toán nặng; từ thư mục gốc repo):
    python -I report/demo/embed_replay.py <đường dẫn demo_replay.json> [report/demo/demo_replay.html]
Thay nội dung thẻ <script id="replay-data" type="application/json">...</script>; chạy lại bao nhiêu lần cũng được.
"""

import json
import re
import sys
from pathlib import Path

PATTERN = re.compile(r'(<script id="replay-data" type="application/json">)(.*?)(</script>)', re.S)


def embed(json_path, html_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    for key in ("meta", "segments", "speed", "limits"):
        if key not in data:
            raise SystemExit(f"{json_path}: thiếu khóa '{key}'")
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")     # "</script>" trong dữ liệu sẽ làm hỏng trang
    html = Path(html_path).read_text(encoding="utf-8")
    if not PATTERN.search(html):
        raise SystemExit(f"{html_path}: không thấy thẻ <script id=\"replay-data\">")
    out = PATTERN.sub(lambda m: m.group(1) + blob + m.group(3), html, count=1)
    Path(html_path).write_text(out, encoding="utf-8", newline="\n")
    return len(blob)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    html = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).with_name("demo_replay.html"))
    n = embed(sys.argv[1], html)
    print(f"Đã nhúng {n / 1024:.0f} KB dữ liệu vào {html}")
