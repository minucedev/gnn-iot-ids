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


def test_predict_proba_views_is_mean_of_separate_calls():
    graphs = _graphs(6)
    x, y = D.node_arrays(graphs)
    lgbm = H.fit_lgbm(x, y, {"n": 5, "l": 4}, 0, 3, n_jobs=1)
    views = [x, x + 0.5, x - 0.5]
    ref = np.mean([H.predict_proba(lgbm, v, 3) for v in views], axis=0)
    np.testing.assert_allclose(H.predict_proba_views(lgbm, views, 3), ref, atol=1e-9)


def test_oof_ensemble_returns_fold_models_and_their_own_val_embeddings():
    graphs, val = _graphs(10), _graphs(4, seed=1)
    emb, models, val_embs, info = H.oof_ensemble(graphs, val, CLASSES, CPU, 0, k=5, train_hp=FAST)
    assert len(models) == len(val_embs) == len(info) == 5
    assert emb.shape == (sum(g.num_nodes for g in graphs), 64)
    for m, e in zip(models, val_embs):                                  # mỗi embedding val đến từ đúng GAT nhóm đó
        np.testing.assert_allclose(e, H.extract_embeddings(m, val, CPU), atol=1e-6)
    assert not np.allclose(val_embs[0], val_embs[1])                    # các GAT nhóm khác nhau cho không gian khác nhau
    emb2, _ = H.oof_embeddings(graphs, val, CLASSES, CPU, 0, k=5, train_hp=FAST)
    np.testing.assert_allclose(emb, emb2, atol=1e-5)                    # wrapper cũ không đổi


def test_predict_window_ensemble_matches_batch_views():
    graphs = _graphs(6)
    gats = [M.GNN("gat", n_classes=3).eval() for _ in range(3)]
    x, y = D.node_arrays(graphs)
    lgbm = H.fit_lgbm(H.build_features(H.extract_embeddings(gats[0], graphs, CPU), x), y, {"n": 5, "l": 4}, 0, 3, n_jobs=1)
    w = graphs[2]
    one = H.predict_window_ensemble(gats, lgbm, w, 3)
    views = [H.build_features(H.extract_embeddings(g, graphs, CPU), x)[12:18] for g in gats]
    np.testing.assert_allclose(one, H.predict_proba_views(lgbm, views, 3), atol=1e-6)


def test_run_hybrid_oofavg_end_to_end_reproducible(tmp_path):
    ra = _load_script("run_hybrid_oofavg")
    (tmp_path / "data").mkdir()
    data_dir = _make_dataset(tmp_path / "data")
    grid = [{"n": 5, "l": 4}, {"n": 8, "l": 6}]
    outs = []
    for tag in ("a", "b"):
        res, search = tmp_path / f"{tag}.csv", tmp_path / f"{tag}.json"
        ra.run(data_dir, "fake", res, search, models_dir=tmp_path / f"m_{tag}", k=2, seeds=(42, 43),
               hp={"train": FAST}, grid=grid)
        outs.append((res, search))
    rows = _rows(outs[0][0])
    assert {(r["model"], r["seed"], r["split"]) for r in rows} == {("hybrid_gat_lgbm_oofavg", "42", "val"),
                                                                 ("hybrid_gat_lgbm_oofavg", "43", "val")}
    one_gat = ev.count_params(M.GNN("gat", n_classes=3))
    assert int(rows[0]["n_params"]) > 2 * one_gat                       # k = 2 GAT + lá LightGBM
    s = json.loads(outs[0][1].read_text(encoding="utf-8"))["hybrid_gat_lgbm_oofavg"]["seeds"]["42"]
    assert len(s["diag"]["val_f1_macro_each_view"]) == 2 and "val_f1_macro_full_gat_embedding" in s["diag"]
    assert len(s["search"]) == 2
    for x, y in zip(rows, _rows(outs[1][0])):
        for c in ("f1_macro", "fpr", "recall_attack"):
            assert float(x[c]) == pytest.approx(float(y[c]), abs=1e-6)


def test_class_proba_rows_sum_to_one_and_keep_order():
    graphs = _graphs(5)
    m = M.GNN("gat", n_classes=3).eval()
    p = H.class_proba(m, graphs, CPU, batch_size=2)
    assert p.shape == (sum(g.num_nodes for g in graphs), 3)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)
    ref = np.concatenate([torch.softmax(m(g.x, g.edge_index, g.edge_attr), 1).detach().numpy() for g in graphs])
    np.testing.assert_allclose(p, ref, atol=1e-5)


def test_oof_class_proba_uses_the_fold_model_of_each_group():
    graphs = _graphs(10)
    models = [M.GNN("gat", n_classes=3).eval() for _ in range(5)]            # trọng số khác nhau
    p = H.oof_class_proba(graphs, models, 5, CPU)
    assert p.shape == (sum(g.num_nodes for g in graphs), 3)
    off = 0
    for f, g in enumerate(H.fold_groups(graphs, 5)):
        sub = [graphs[i] for i in g]
        n = sum(x.num_nodes for x in sub)
        np.testing.assert_allclose(p[off:off + n], H.class_proba(models[f], sub, CPU), atol=1e-6)
        off += n


def test_predict_window_stack_matches_batch_features():
    graphs = _graphs(6)
    gats = [M.GNN("gat", n_classes=3).eval() for _ in range(3)]
    x, y = D.node_arrays(graphs)
    pavg = np.mean([H.class_proba(g, graphs, CPU) for g in gats], axis=0)
    lgbm = H.fit_lgbm(H.build_features(pavg, x), y, {"n": 5, "l": 4}, 0, 3, n_jobs=1)
    ref = H.predict_proba(lgbm, H.build_features(pavg, x), 3)[12:18]
    np.testing.assert_allclose(H.predict_window_stack(gats, lgbm, graphs[2], 3), ref, atol=1e-5)
    assert H.build_features(pavg, x).shape[1] == 18 + 3


def test_run_hybrid_stack_end_to_end_uses_saved_fold_gats(tmp_path):
    ra, rs = _load_script("run_hybrid_oofavg"), _load_script("run_hybrid_stack")
    (tmp_path / "data").mkdir()
    data_dir = _make_dataset(tmp_path / "data")
    grid = [{"n": 5, "l": 4}, {"n": 8, "l": 6}]
    ra.run(data_dir, "fake", tmp_path / "oofavg.csv", tmp_path / "oofavg.json", models_dir=tmp_path / "m_fold", k=2,
           seeds=(42, 43), hp={"train": FAST}, grid=grid)
    outs = []
    for tag in ("a", "b"):                                                  # không huấn luyện GAT nên hai lần phải y hệt
        res, search = tmp_path / f"{tag}.csv", tmp_path / f"{tag}.json"
        rs.run(data_dir, "fake", res, search, tmp_path / "m_fold", full_dir=tmp_path / "m_fold",
               fold_search=tmp_path / "oofavg.json", models_dir=tmp_path / f"m_{tag}", k=2, seeds=(42, 43), grid=grid)
        outs.append((res, search))
    rows = _rows(outs[0][0])
    assert {(r["model"], r["seed"], r["split"]) for r in rows} == {("hybrid_gat_lgbm_stack", "42", "val"),
                                                                 ("hybrid_gat_lgbm_stack", "43", "val")}
    assert int(rows[0]["n_params"]) > 2 * ev.count_params(M.GNN("gat", n_classes=3))
    s = json.loads(outs[0][1].read_text(encoding="utf-8"))["hybrid_gat_lgbm_stack"]["seeds"]["42"]
    assert len(s["search"]) == 2 and len(s["diag"]["val_f1_macro_each_gat"]) == 2
    assert {p.name for p in (tmp_path / "m_a" / "fake_s42").iterdir()} >= {"gat_fold0.pt", "gat_fold1.pt", "lgbm_stack.pkl",
                                                                          "meta_stack.json"}
    for x, y in zip(rows, _rows(outs[1][0])):
        for c in ("f1_macro", "fpr", "recall_attack", "accuracy"):
            assert float(x[c]) == pytest.approx(float(y[c]), abs=1e-9)


def test_interleaved_folds_cover_everything_and_span_all_periods():
    graphs = _graphs(60)
    groups = H.fold_groups(graphs, 5, interleave=4)                          # 20 khối, khối i thuộc nhóm i mod 5
    assert np.array_equal(np.sort(np.concatenate(groups)), np.arange(60))
    assert sum(len(g) for g in groups) == 60 and len(set(np.concatenate(groups))) == 60
    for g in groups:
        assert np.array_equal(g, np.sort(g))
        assert g.min() < 60 * 0.25 and g.max() > 60 * 0.75                    # mỗi nhóm trải khắp mọi giai đoạn
        runs = np.split(g, np.flatnonzero(np.diff(g) > 1) + 1)
        assert len(runs) >= 4 and all(len(r) >= 2 for r in runs)              # nhưng vẫn gồm các khối liên tiếp
    contiguous = H.fold_groups(graphs, 5)
    assert all(np.array_equal(g, np.arange(g[0], g[-1] + 1)) for g in contiguous)   # interleave=0 không đổi


def test_interleaved_oof_keeps_train_order_and_never_trains_on_own_group():
    graphs, val = _graphs(20), _graphs(4, seed=1)
    seen = []

    def spy(gs, v, classes, device, seed, model_hp, train_hp, verbose):
        seen.append({id(g) for g in gs})
        return H.train_gat(gs, v, classes, device, seed, model_hp, train_hp, verbose)

    emb, models, val_embs, info = H.oof_ensemble(graphs, val, CLASSES, CPU, 0, k=2, train_hp=FAST, train_fn=spy,
                                                 interleave=3)
    groups = H.fold_groups(graphs, 2, interleave=3)
    sizes = np.array([g.num_nodes for g in graphs])
    off = np.r_[0, np.cumsum(sizes)]
    assert emb.shape[0] == off[-1]
    for f, held in enumerate(groups):
        assert not ({id(graphs[i]) for i in held} & seen[f])
        for i in held[:3]:                                                   # nhúng đúng chỗ, đúng GAT nhóm
            ref = H.extract_embeddings(models[f], [graphs[i]], CPU)
            np.testing.assert_allclose(emb[off[i]:off[i + 1]], ref, atol=1e-5)
    p = H.oof_class_proba(graphs, models, 2, CPU, interleave=3)
    i0 = groups[1][0]
    np.testing.assert_allclose(p[off[i0]:off[i0 + 1]], H.class_proba(models[1], [graphs[i0]], CPU), atol=1e-6)
    assert [x["n_nodes_held"] for x in info] == [int(sizes[g].sum()) for g in groups]


def test_run_hybrid_interleave_end_to_end_names_models_with_il(tmp_path):
    ra, rs = _load_script("run_hybrid_oofavg"), _load_script("run_hybrid_stack")
    (tmp_path / "data").mkdir()
    data_dir = _make_dataset(tmp_path / "data")
    grid = [{"n": 5, "l": 4}]
    ra.run(data_dir, "fake", tmp_path / "o.csv", tmp_path / "o.json", models_dir=tmp_path / "m", k=2, seeds=(42,),
           hp={"train": FAST}, grid=grid, interleave=3)
    rs.run(data_dir, "fake", tmp_path / "s.csv", tmp_path / "s.json", tmp_path / "m", fold_search=tmp_path / "o.json",
           models_dir=tmp_path / "m2", k=2, seeds=(42,), grid=grid, interleave=3)
    assert [r["model"] for r in _rows(tmp_path / "o.csv")] == ["hybrid_gat_lgbm_oofavg_il"]
    assert [r["model"] for r in _rows(tmp_path / "s.csv")] == ["hybrid_gat_lgbm_stack_il"]
    assert json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))["hybrid_gat_lgbm_stack_il"]["interleave"] == 3


def test_neighbor_mean_by_hand_with_offsets_duplicates_and_isolated_nodes():
    x1 = torch.arange(4 * 18, dtype=torch.float32).reshape(4, 18)
    # 0→1 và 1→0 (hai chiều, tính một hàng xóm), 1→2, vòng lặp tự thân 2→2 (bỏ), node 3 cô lập
    g1 = Data(x=x1, edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 2]]))
    x2 = torch.arange(100, 100 + 2 * 18, dtype=torch.float32).reshape(2, 18)
    g2 = Data(x=x2, edge_index=torch.tensor([[0, 0], [1, 1]]))               # cạnh trùng
    out = H.neighbor_mean([g1, g2])
    expect = np.stack([x1[1], (x1[0] + x1[2]) / 2, x1[1], torch.zeros(18), x2[1], x2[0]]).astype(np.float32)
    assert out.shape == (6, 18) and out.dtype == np.float32
    np.testing.assert_allclose(out, expect, atol=1e-6)
    assert H.neighbor_mean([Data(x=x1, edge_index=torch.zeros(2, 0, dtype=torch.long))]).sum() == 0


def test_oof_embeddings_from_saved_models_match_training_time_embeddings():
    graphs, val = _graphs(20), _graphs(4, seed=1)
    emb, models, _, _ = H.oof_ensemble(graphs, val, CLASSES, CPU, 0, k=2, train_hp=FAST, interleave=3)
    again = H.oof_embeddings_from_models(graphs, models, 2, interleave=3, device=CPU)
    np.testing.assert_allclose(emb, again, atol=1e-5)


def test_load_gat_roundtrip(tmp_path):
    m = M.GNN("gat", n_classes=3).eval()
    torch.save(m.state_dict(), tmp_path / "g.pt")
    loaded = H.load_gat(tmp_path / "g.pt", 3)
    g = _graphs(1)[0]
    assert not loaded.training
    assert torch.allclose(loaded(g.x, g.edge_index, g.edge_attr), m(g.x, g.edge_index, g.edge_attr), atol=1e-6)


def test_run_hybrid_ablation_end_to_end_reproduces_saved_hybrid(tmp_path):
    import pandas as pd
    ra, rb = _load_script("run_hybrid_oofavg"), _load_script("run_hybrid_ablation")
    (tmp_path / "data").mkdir()
    data_dir = _make_dataset(tmp_path / "data")
    ra.run(data_dir, "fake", tmp_path / "o.csv", tmp_path / "o.json", models_dir=tmp_path / "m", k=2, seeds=(42, 43),
           hp={"train": FAST}, grid=[{"n": 5, "l": 4}], interleave=3)
    graphs, _ = D.load_split(data_dir)
    (tmp_path / "full").mkdir()
    for seed in (42, 43):                                                    # GAT "toàn train" giả
        gat, _, _ = H.train_gat(T.strip(graphs["train"]), T.strip(graphs["val"]), CLASSES, CPU, seed, train_hp=FAST)
        torch.save(gat.state_dict(), tmp_path / "full" / f"fake_gat_s{seed}.pt")
    out = tmp_path / "out"
    rb.run(data_dir, "fake", out, tmp_path / "m", tmp_path / "full", k=2, interleave=3, seeds=(42, 43), grid=[{"n": 3, "l": 4}])

    cd = _rows(out / "ablation_cd_fake.csv")
    assert {(r["model"], r["seed"], r["split"]) for r in cd} == {(m, s, "val") for m in (rb.ABL_NBR, rb.ABL_EMB) for s in ("42", "43")}
    pareto = pd.read_csv(out / "pareto_raw_fake.csv")
    assert set(pareto["variant"]) == {"gat2", "gat1", "gatfull"}              # k = 2 nên chỉ có tối đa 2 GAT nhóm
    assert len(pareto) == 3 * 2 * 2                                           # 3 biến thể × 2 ô LightGBM (lưới + hybrid) × 2 seed
    one_gat = ev.count_params(M.GNN("gat", n_classes=3))
    full = pareto[pareto["is_hybrid_full"]]
    assert set(full["variant"]) == {"gat2"} and len(full) == 2                # đúng 1 điểm hybrid đầy đủ mỗi seed
    g2 = pareto[(pareto["variant"] == "gat2") & (pareto["seed"] == 42)]
    g1 = pareto[(pareto["variant"] == "gat1") & (pareto["seed"] == 42)]
    assert (g2["n_params"].to_numpy() - g1["n_params"].to_numpy() == one_gat).all()      # bớt đúng một GAT
    assert (pareto["size_kb"] > 0).all() and pareto["p95_ms"].notna().all()
    assert {"thr", "thr_fpr", "thr_recall_attack", "thr_feasible"} <= set(pareto.columns)
    assert ((pareto["thr_fpr"] <= 0.0408) | ~pareto["thr_feasible"]).all()
    assert (out / "models_light" / "fake_s42" / "lgbm_n3_l4.pkl").exists()
    # hybrid đầy đủ dựng lại từ file đã lưu khớp kết quả lúc huấn luyện
    saved = {r["seed"]: float(r["f1_macro"]) for r in _rows(tmp_path / "o.csv")}
    check = json.loads((out / "ref_check_fake.json").read_text(encoding="utf-8"))
    for seed in ("42", "43"):
        assert check[seed]["f1_macro"] == pytest.approx(saved[seed], abs=1e-6)
        assert float(full[full["seed"] == int(seed)]["f1_macro"].iloc[0]) == pytest.approx(saved[seed], abs=1e-6)
