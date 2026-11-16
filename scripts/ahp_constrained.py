"""AHP có ràng buộc cứng (GĐ J, kịch bản "sàng lọc rồi xếp hạng") từ summary_test.csv và ma trận đã duyệt docs/ahp_matrix.json.

Chạy:  python -I scripts/ahp_constrained.py <summary_test.csv> <ahp_matrix.json> <ahp_constrained.json> [thư mục hình]
Ghi:   ahp_constrained.json; nếu có thư mục hình thì thêm gd7_ahp_constrained.png.
Loại phương án có > 100.000 tham số hoặc > 1 MB (1.024 KB) trước khi chuẩn hóa và xếp hạng; không đổi ma trận, không đổi số liệu.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import ahp  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    summary_csv, matrix_json, out = sys.argv[1:4]
    res = ahp.run_constrained(matrix_json, summary_csv)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    a, b = res["scenario_a_unconstrained"], res["scenario_b_constrained"]
    print("(a) không ràng buộc:", a["ranking"], "→", a["top_label"])
    print("(b) loại:", b["excluded"])
    print("(b) xếp hạng:", b["ranking"], "→", b["top_label"], {k: round(v, 4) for k, v in b["totals"].items()})
    print("(b) không chuẩn hóa lại:", b["ranking_without_renormalization"])
    print("(b) độ bền chuẩn hóa:", {k: v["top"] for k, v in b["normalization_robustness"].items()})
    print("(b) đổi hạng 1 khi ±20% trọng số:", b["top_flips_under_20pct"])
    print("(b) có điều kiện:", {k: v["top"] for k, v in b["conditional"].items()})
    ext = res["extension_constrained_and_targets"]
    print("mở rộng (ràng buộc nhẹ + FPR/recall đạt chỉ tiêu):", ext["result"] and ext["result"]["ranking"])
    if len(sys.argv) > 4:
        sys.path.insert(0, str(ROOT / "scripts"))
        import ahp_plots

        print(ahp_plots.plot_constrained(res, Path(sys.argv[4]) / "gd7_ahp_constrained.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
