import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src import build_graph as bg

ROOT = Path(__file__).resolve().parents[1]


def test_time_split_weights_balance_nodes():
    wins = np.arange(100)
    w = np.where(wins < 80, 1, 10)                     # 20 cửa sổ cuối nặng gấp 10 lần
    plain = bg.time_split(wins, bg.SPLIT_RATIO, 0)
    bal = bg.time_split(wins, bg.SPLIT_RATIO, 0, weights=w)
    share = lambda sp, k: w[np.isin(wins, sp[k])].sum() / w.sum()
    assert abs(share(plain, "test") - 0.15) > 0.3      # chia theo cửa sổ: test chiếm quá nhiều node
    for k, r in zip(("train", "val", "test"), bg.SPLIT_RATIO):
        assert abs(share(bal, k) - r) < 0.06           # chia theo node: sát 70/15/15


def test_unit_weights_equal_plain_split():
    wins = np.arange(100)
    for mode in ("chronological", "blocked"):
        a = bg.time_split(wins, bg.SPLIT_RATIO, 1, mode=mode, blocks=5)
        b = bg.time_split(wins, bg.SPLIT_RATIO, 1, mode=mode, blocks=5, weights=np.ones(100))
        assert all(np.array_equal(a[k], b[k]) for k in a)


def test_more_blocks_put_short_episode_into_val_and_test():
    # một đợt tấn công ngắn (cửa sổ 100-119) nằm trong phần train của khối lớn, nhưng trải qua val/test khi chia nhỏ
    wins = np.arange(1000)
    episode = (wins >= 100) & (wins < 120)
    cover = lambda sp: {k: int(np.isin(wins[episode], sp[k]).sum()) for k in sp}
    coarse = cover(bg.time_split(wins, bg.SPLIT_RATIO, 0, mode="blocked", blocks=2))
    fine = cover(bg.time_split(wins, bg.SPLIT_RATIO, 0, mode="blocked", blocks=50))
    assert coarse["val"] == 0 and coarse["test"] == 0
    assert fine["val"] > 0 and fine["test"] > 0


def test_split_search_script_runs(tmp_path):
    from tests.test_build_graph import _nf_csv
    csv = tmp_path / "v3.csv"
    _nf_csv(csv, with_ts=True)
    out = tmp_path / "out.json"
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "split_search.py"), "nf_ton_iot_v3", str(csv),
                        "1", "1", str(out)], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert r.returncode == 0, r.stderr[-1500:]
    res = json.loads(out.read_text(encoding="utf-8"))
    assert 2 <= len(res["configs"]) <= 12 and {"ok_classes", "ok_ratio", "counts"} <= set(res["configs"][0])
