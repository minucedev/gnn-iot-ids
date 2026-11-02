"""Chuyển file .py dạng "percent" (ô bắt đầu bằng `# %%`, ô markdown `# %% [markdown]`)
sang .ipynb, đồng thời biên dịch thử mọi ô code để bắt lỗi cú pháp trước khi đẩy lên Kaggle.

Chạy:  .venv\\Scripts\\python.exe -I scripts/py_to_ipynb.py notebooks/x.py [notebooks/x.ipynb]
"""

import ast
import json
import sys
from pathlib import Path


def parse_cells(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = "\n".join(buf).strip("\n").splitlines()
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
        src = "\n".join(lines)
        if src.strip():
            cells.append((kind, src))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".ipynb")
    cells = parse_cells(src.read_text(encoding="utf-8"))
    nb_cells = []
    for i, (kind, text) in enumerate(cells):
        if kind == "code":
            ast.parse(text, filename=f"{src}:ô {i}")  # lỗi cú pháp thì dừng tại đây
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    nb = {
        "cells": nb_cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    n_code = sum(k == "code" for k, _ in cells)
    print(f"{dst}: {len(cells)} ô ({n_code} code, đều biên dịch OK)")


if __name__ == "__main__":
    main()
