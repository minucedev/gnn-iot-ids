import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from src import build_graph as bg
from src import checks as C

CLASSES = ["Normal", "DoS_DDoS", "Botnet_Backdoor", "Truy_cap_trai_phep"]


def _toy(n_win=200, seed=0):
    rng = np.random.default_rng(seed)
    graphs = []
    for w in range(n_win):
        n = int(rng.integers(5, 12))
        y = rng.integers(0, 4, n)
        graphs.append(Data(x=torch.from_numpy(rng.normal(size=(n, 18)).astype("float32")),
                           y=torch.from_numpy(y.astype("int64")),
                           edge_index=torch.from_numpy(rng.integers(0, n, (2, 6))),
                           edge_attr=torch.from_numpy(rng.normal(size=(6, 3)).astype("float32")),
                           win=w, ips=[str(i) for i in range(n)]))
    return graphs


def _run(graphs, gap=1, mutate=None):
    wins = np.array([g.win for g in graphs])
    split = bg.time_split(wins, bg.SPLIT_RATIO, gap)
    in_train = {int(w) for w in split["train"]}
    x_train = torch.cat([g.x for g in graphs if g.win in in_train]).numpy()
    scaler = StandardScaler().fit(x_train)
    for g in graphs:
        g.x = torch.from_numpy(scaler.transform(g.x.numpy()).astype("float32"))
    if mutate:
        mutate(graphs, split)
    return {name: ok for name, ok, _ in C.check_graphs(graphs, split, scaler, CLASSES, gap)}, split


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
