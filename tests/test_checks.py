import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from src import build_graph as bg
from src import checks as C

CLASSES = ["Normal", "DoS_DDoS", "Botnet_Backdoor", "Truy_cap_trai_phep"]


def _toy(n_win=200, seed=0, n_classes=4, rare=None):
    """rare = mã lớp chỉ xuất hiện ở 5 cửa sổ đầu (để thử lớp có mặt nhưng quá ít node)."""
    rng = np.random.default_rng(seed)
    graphs = []
    for w in range(n_win):
        n = int(rng.integers(5, 12))
        y = rng.integers(0, n_classes, n)
        if rare is not None and w >= 5:
            y[y == rare] = 0
        graphs.append(Data(x=torch.from_numpy(rng.normal(size=(n, 18)).astype("float32")),
                           y=torch.from_numpy(y.astype("int64")),
                           edge_index=torch.from_numpy(rng.integers(0, n, (2, 6))),
                           edge_attr=torch.from_numpy(rng.normal(size=(6, 3)).astype("float32")),
                           win=w, ips=[str(i) for i in range(n)]))
    return graphs


def _run(graphs, gap=1, mutate=None, classes=CLASSES, with_detail=False):
    wins = np.array([g.win for g in graphs])
    split = bg.time_split(wins, bg.SPLIT_RATIO, gap)
    in_train = {int(w) for w in split["train"]}
    x_train = torch.cat([g.x for g in graphs if g.win in in_train]).numpy()
    scaler = StandardScaler().fit(x_train)
    for g in graphs:
        g.x = torch.from_numpy(scaler.transform(g.x.numpy()).astype("float32"))
    if mutate:
        mutate(graphs, split)
    checks = C.check_graphs(graphs, split, scaler, classes, gap)
    if with_detail:
        return {name: (ok, detail) for name, ok, detail in checks}, split
    return {name: ok for name, ok, _ in checks}, split


def test_clean_toy_graphs_pass_all_checks():
    res, _ = _run(_toy())
    assert all(res.values()), [k for k, v in res.items() if not v]


def test_nan_is_detected():
    def poison(graphs, split):
        graphs[3].x[0, 0] = float("nan")
    res, _ = _run(_toy(), mutate=poison)
    assert not res["Không có NaN hoặc inf trong x và edge_attr"]


def test_overlapping_windows_and_missing_gap_are_detected():
    def overlap(graphs, split):
        split["val"] = np.concatenate([split["val"], split["test"][:1]])
        split["test"] = split["test"]
    res, _ = _run(_toy(), mutate=overlap)
    assert not res["Không cửa sổ trùng giữa các tập"]
    res0, _ = _run(_toy(), gap=0)                    # tách với gap = 0 nhưng kiểm với yêu cầu gap = 1
    wins = np.arange(200)
    split0 = bg.time_split(wins, bg.SPLIT_RATIO, 0)
    g = _toy()
    sc = StandardScaler().fit(torch.cat([x.x for x in g if x.win in set(split0["train"].tolist())]).numpy())
    out = {n: ok for n, ok, _ in C.check_graphs(g, split0, sc, CLASSES, 1)}
    assert not out["Gap >= 1 cửa sổ ở mọi ranh giới giữa các tập"]


def test_scaler_fit_on_all_data_is_detected():
    graphs = _toy()
    wins = np.array([g.win for g in graphs])
    split = bg.time_split(wins, bg.SPLIT_RATIO, 1)
    scaler = StandardScaler().fit(torch.cat([g.x for g in graphs]).numpy())     # fit trên mọi tập: sai
    out = {n: ok for n, ok, _ in C.check_graphs(graphs, split, scaler, CLASSES, 1)}
    assert not out["Scaler chỉ fit trên train (số mẫu = số node train)"]


CLASS_CHECK = "Mỗi lớp chính có >= 100 node train, >= 30 node val và test"
BOT_CLASSES = ["Normal", "DoS_DDoS", "Truy_cap_trai_phep"]          # NF-BoT-IoT-v3: không có Botnet_Backdoor


def test_absent_classes_lists_groups_missing_from_dataset():
    assert C.absent_classes(bg.GROUP_ORDER) == []
    assert C.absent_classes(CLASSES) == ["Khac"]
    assert C.absent_classes(BOT_CLASSES) == ["Botnet_Backdoor", "Khac"]


def test_classes_below_threshold_only_looks_at_present_main_classes():
    ok = {"train": 100, "val": 30, "test": 30}
    rows = {p: {c: ok[p] for c in BOT_CLASSES} for p in ok}
    assert C.classes_below_threshold(rows, BOT_CLASSES) == []                    # không cần có Botnet_Backdoor
    rows["val"]["DoS_DDoS"] = 29
    rows["train"]["Truy_cap_trai_phep"] = 99
    assert C.classes_below_threshold(rows, BOT_CLASSES) == ["DoS_DDoS", "Truy_cap_trai_phep"]
    rows4 = {p: {c: ok[p] for c in CLASSES} for p in ok}
    rows4["test"]["Botnet_Backdoor"] = 0                                         # bộ 4 lớp: thiếu vẫn bị bắt như cũ
    assert C.classes_below_threshold(rows4, CLASSES) == ["Botnet_Backdoor"]


def test_dataset_without_botnet_passes_class_check_and_reports_absent_class():
    res, _ = _run(_toy(n_classes=3), classes=BOT_CLASSES, with_detail=True)
    ok, detail = res[CLASS_CHECK]
    assert ok, detail
    assert "Botnet_Backdoor" in detail
    assert all(v[0] for v in res.values()), [k for k, v in res.items() if not v[0]]


def test_four_class_dataset_detail_unchanged_when_nothing_absent():
    res, _ = _run(_toy(), with_detail=True)
    assert res[CLASS_CHECK] == (True, "đủ")


def test_present_class_with_too_few_nodes_still_fails():
    res, _ = _run(_toy(rare=2), with_detail=True)                                # Botnet_Backdoor có mặt nhưng ~vài node
    ok, detail = res[CLASS_CHECK]
    assert not ok and "Botnet_Backdoor" in detail
