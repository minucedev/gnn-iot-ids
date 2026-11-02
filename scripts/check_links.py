"""Kiểm tra mọi link tương đối trong các file .md của repo trỏ tới file/thư mục có thật.

Chạy:  .venv\\Scripts\\python.exe -I scripts/check_links.py
Bỏ qua link http(s)/mailto. Với link có #anchor chỉ kiểm phần đường dẫn.
Thoát mã 1 nếu có link hỏng.
"""

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SKIP_DIRS = {".venv", ".git", "data", "node_modules"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    broken, checked = [], 0
    for md in sorted(ROOT.rglob("*.md")):
        if SKIP_DIRS & set(md.relative_to(ROOT).parts):
            continue
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for target in LINK.findall(line):
                if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                    continue
                path = unquote(target.split("#", 1)[0])
                dest = (md.parent / path).resolve()
                checked += 1
                ok = dest.exists()
                rel = md.relative_to(ROOT).as_posix()
                print(f"{'OK  ' if ok else 'HỎNG'} {rel}:{lineno} -> {target}")
                if not ok:
                    broken.append(f"{rel}:{lineno} -> {target}")
    print(f"\n{checked} link tương đối, {len(broken)} hỏng")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
