import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src import data as D
from src import evaluate as ev
from src import hybrid as H
from src import models as M
from src import train as T

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Normal", "DoS_DDoS", "Truy_cap_trai_phep"]
CPU = torch.device("cpu")
FAST = {"epochs": 2, "patience": 1, "batch_size": 8}


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _graphs(n_graphs=10, seed=0, n=6):
    """Dữ liệu giả: node lớp k có đặc trưng quanh k, nên mô hình học được chút ít."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_graphs):
        y = torch.tensor([0, 1, 2, 0, 1, 2][:n])
        x = torch.randn(n, 18, generator=g) * 0.3 + y[:, None].float()
        out.append(Data(x=x, y=y, edge_index=torch.randint(0, n, (2, 9), generator=g),
                        edge_attr=torch.randn(9, 3, generator=g)))
    return out


def test_embed_is_hidden_layer_before_head_and_forward_unchanged():
    torch.manual_seed(0)
    g = _graphs(1)[0]
    m = M.GNN("gat", n_classes=3).eval()
    z = m.embed(g.x, g.edge_index, g.edge_attr)
    assert tuple(z.shape) == (6, 64)
    assert torch.allclose(m.head(z), m(g.x, g.edge_index, g.edge_attr))


def test_feature_columns_are_18_plus_embedding_and_x_first():
    graphs = _graphs(4)
    m = M.GNN("gat", n_classes=3).eval()
    emb = H.extract_embeddings(m, graphs, CPU)
    x, _ = D.node_arrays(graphs)
    f = H.build_features(emb, x)
    assert f.shape == (len(x), 18 + emb.shape[1]) == (len(x), 82)
    np.testing.assert_array_equal(f[:, :18], x)
    np.testing.assert_array_equal(f[:, 18:], emb)
    with pytest.raises(ValueError):
        H.build_features(emb[:-1], x)


def test_embeddings_keep_node_order_across_batches():
    graphs = _graphs(7)
    m = M.GNN("gat", n_classes=3).eval()
    per_graph = np.concatenate([m.embed(g.x, g.edge_index, g.edge_attr).detach().numpy() for g in graphs])
    batched = H.extract_embeddings(m, graphs, CPU, batch_size=3)        # nhiều lô nhỏ
    np.testing.assert_allclose(batched, per_graph, atol=1e-5)
    assert len(batched) == sum(g.num_nodes for g in graphs)


def test_fold_groups_are_contiguous_balanced_and_complete():
    graphs = _graphs(23)
    groups = H.fold_groups(graphs, 5)
    assert np.array_equal(np.concatenate(groups), np.arange(23))
    assert all(len(g) > 0 for g in groups)
    assert all(np.array_equal(g, np.arange(g[0], g[-1] + 1)) for g in groups)      # liên tiếp theo thời gian
    with pytest.raises(ValueError):
        H.fold_groups(_graphs(3), 5)


def test_oof_never_trains_on_its_own_group_and_keeps_order():
    graphs, val = _graphs(10), _graphs(4, seed=1)
    seen = []

    def spy(gs, v, classes, device, seed, model_hp, train_hp, verbose):
        seen.append({id(g) for g in gs})
        return H.train_gat(gs, v, classes, device, seed, model_hp, train_hp, verbose)

    emb, info = H.oof_embeddings(graphs, val, CLASSES, CPU, 0, k=5, train_hp=FAST, train_fn=spy)
    groups = H.fold_groups(graphs, 5)
    assert len(seen) == 5
    for f, held in enumerate(groups):
        held_ids = {id(graphs[i]) for i in held}
        assert not (held_ids & seen[f])                                  # GAT không thấy node của nhóm bị giữ lại
        assert len(seen[f]) == len(graphs) - len(held)                   # mà thấy mọi nhóm còn lại
    assert emb.shape == (sum(g.num_nodes for g in graphs), 64)
    assert [i["n_nodes_held"] for i in info] == [sum(graphs[i].num_nodes for i in g) for g in groups]


def test_same_seed_gives_same_predictions():
    graphs, val = _graphs(12), _graphs(4, seed=1)
    x, y = D.node_arrays(graphs)
    runs = []
    for _ in range(2):
        gat, _, _ = H.train_gat(graphs, val, CLASSES, CPU, 7, train_hp=FAST)
        emb = H.extract_embeddings(gat, graphs, CPU)
        lgbm = H.fit_lgbm(H.build_features(emb, x), y, {"n": 8, "l": 4}, 7, 3, n_jobs=1)
        runs.append(H.predict_proba(lgbm, H.build_features(emb, x), 3))
    np.testing.assert_allclose(runs[0], runs[1], atol=1e-6)
    assert (runs[0].argmax(1) == runs[1].argmax(1)).all()


def test_search_space_matches_baseline_lightgbm():
    base = _load_script("run_baselines")
    make, grid = base.make_models(42)["lightgbm"]
    assert H.LGBM_GRID == grid and len(H.LGBM_GRID) <= H.MAX_TRIALS
    a, b = make(grid[1]).get_params(), H.make_lgbm(grid[1], 42).get_params()
    for k in ("n_estimators", "num_leaves", "learning_rate", "random_state"):
        assert a[k] == b[k]


def test_search_picks_best_on_val_and_counts_leaves():
    graphs = _graphs(12)
    x, y = D.node_arrays(graphs)
    grid = [{"n": 3, "l": 4}, {"n": 6, "l": 8}]
    cfg, log = H.search_lgbm(x, y, x, y, CLASSES, seed=1, grid=grid)
    assert len(log) == 2 and cfg == grid[int(np.argmax([t["val_f1_macro"] for t in log]))]
    m = H.fit_lgbm(x, y, grid[1], 1, 3, n_jobs=1)
    assert 1 <= H.lgbm_n_leaves(m) <= 3 * 6 * 8                      # tối đa 3 lớp × 6 vòng × 8 lá
    with pytest.raises(ValueError):
        H.search_lgbm(x, y, x, y, CLASSES, grid=[{"n": 1, "l": 2}] * 21)


def test_predict_window_matches_batch_pipeline():
    graphs = _graphs(6)
    gat = M.GNN("gat", n_classes=3).eval()
    x, y = D.node_arrays(graphs)
    emb = H.extract_embeddings(gat, graphs, CPU)
    lgbm = H.fit_lgbm(H.build_features(emb, x), y, {"n": 5, "l": 4}, 0, 3, n_jobs=1)
    w = graphs[2]
    one = H.predict_window(gat, lgbm, w, 3)
    ref = H.predict_proba(lgbm, H.build_features(emb, x), 3)[2 * 6:3 * 6]
    np.testing.assert_allclose(one, ref, atol=1e-6)


def _make_dataset(tmp_path):
    graphs = _graphs(30, seed=3)
    torch.save(graphs, tmp_path / "g.pt")
    idx = {"train": list(range(0, 20)), "val": list(range(20, 26)), "test": list(range(26, 30))}
    meta = {"graph_file": "g.pt", "classes": CLASSES, "features": [f"f{i}" for i in range(18)],
            **{p: {"graph_idx": v} for p, v in idx.items()}}
    (tmp_path / "split.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_run_hybrid_end_to_end_both_modes_val_only_reproducible(tmp_path):
    rh = _load_script("run_hybrid")
    (tmp_path / "data").mkdir()
    data_dir = _make_dataset(tmp_path / "data")
    hp = {"train": FAST}
    grid = [{"n": 5, "l": 4}, {"n": 8, "l": 6}]
    out = {}
    for tag in ("a", "b"):                                                # 2 lần chạy độc lập, cùng seed
        res, search, models = tmp_path / f"{tag}.csv", tmp_path / f"{tag}.json", tmp_path / f"models_{tag}"
        for oof in (True, False):
            rh.run(data_dir, "fake", res, search, oof=oof, models_dir=models, k=2, seeds=(42, 43), hp=hp, grid=grid)
        out[tag] = (res, search, models)
    rows = _rows(out["a"][0])
    assert {(r["model"], r["seed"], r["split"]) for r in rows} == {
        (m, s, "val") for m in ("hybrid_gat_lgbm", "hybrid_gat_lgbm_nooof") for s in ("42", "43")}
    assert len(rows) == 4                                                 # mỗi (mô hình, seed) đúng 1 dòng, không có test
    r = rows[0]
    assert int(r["n_params"]) > ev.count_params(M.GNN("gat", n_classes=3))   # GAT + lá LightGBM
    assert float(r["size_kb"]) > 0 and float(r["infer_ms_per_window"]) > 0
    s = json.loads(out["a"][1].read_text(encoding="utf-8"))
    assert set(s) == {"hybrid_gat_lgbm", "hybrid_gat_lgbm_nooof"}
    assert len(s["hybrid_gat_lgbm"]["seeds"]["42"]["search"]) == 2        # chọn cấu hình ở seed 42, ≤ 20 lần thử
    assert "search" not in s["hybrid_gat_lgbm"]["seeds"]["43"]            # seed sau dùng lại cấu hình
    assert len(s["hybrid_gat_lgbm"]["seeds"]["42"]["folds"]) == 2
    d = out["a"][2] / "fake_s42"
    assert {p.name for p in d.iterdir()} >= {"gat.pt", "gat_meta.json", "lgbm_oof.pkl", "lgbm_nooof.pkl",
                                              "meta_oof.json", "meta_nooof.json"}
    # cùng seed cho cùng kết quả ở hai lần chạy độc lập
    ra, rb = _rows(out["a"][0]), _rows(out["b"][0])
    for x, y in zip(ra, rb):
        assert (x["model"], x["seed"]) == (y["model"], y["seed"])
        for c in ("f1_macro", "fpr", "recall_attack", "accuracy"):
            assert float(x[c]) == pytest.approx(float(y[c]), abs=1e-6)
