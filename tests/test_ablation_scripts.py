"""Kiểm thử các script của GĐ G: lưới GNN, hiệu chỉnh ngưỡng baseline/GNN, giải thích GAT (dữ liệu giả nhỏ)."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src import evaluate as ev
from src import models as M

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Normal", "DoS_DDoS", "Truy_cap_trai_phep"]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _graphs(n_graphs=30, seed=3, n=6):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n_graphs):
        y = torch.tensor([0, 1, 2, 0, 1, 2][:n])
        out.append(Data(x=torch.randn(n, 18, generator=g) * 0.3 + y[:, None].float(), y=y,
                        edge_index=torch.randint(0, n, (2, 9), generator=g), edge_attr=torch.randn(9, 3, generator=g),
                        ips=[f"10.0.0.{j}" for j in range(n)], win=i, t_start=float(5 * i)))
    return out


def _dataset(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    torch.save(_graphs(), d / "g.pt")
    idx = {"train": list(range(0, 20)), "val": list(range(20, 26)), "test": list(range(26, 30))}
    meta = {"graph_file": "g.pt", "classes": CLASSES, "features": [f"f{i}" for i in range(18)], "window_sec": 5,
            **{p: {"graph_idx": v} for p, v in idx.items()}}
    (d / "split.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_grid_experiments_change_exactly_one_factor_and_respect_trial_limit():
    rg = _load_script("run_gnn_grid")
    arch = rg.experiments("arch")
    assert len(arch) == 10 and len({e["cfg_id"] for e in arch}) == 10
    assert {e["factor"] for e in arch} == {"default", "heads", "layers", "hidden", "edge"}
    for e in arch:
        changed = [k for k in rg.MAIN if e[k] != rg.MAIN[k]]
        assert len(changed) == (0 if e["factor"] == "default" else 1)
    assert {e["value"] for e in arch if e["factor"] == "heads"} == {1, 2, 8}
    assert {e["value"] for e in arch if e["factor"] == "layers"} == {1, 3}
    assert {e["value"] for e in arch if e["factor"] == "hidden"} == {8, 32, 64}
    assert [e["use_edge"] for e in arch if e["factor"] == "edge"] == [False]
    tuning = rg.experiments("tuning")
    for kind in ("gcn", "sage"):
        mine = [e for e in tuning if e["kind"] == kind]
        assert len(mine) == 6 and {(e["hidden"], e["layers"]) for e in mine} == {(h, n) for h in (8, 16, 32) for n in (2, 3)}
    imp = rg.experiments("improved")
    assert [(e["loss"], e["residual"], e["layer_norm"]) for e in imp] == [("focal", False, False), ("ce", True, True), ("focal", True, True)]
    win = rg.experiments("default", window=2)
    assert len(win) == 1 and win[0]["factor"] == "window" and win[0]["value"] == 2 and win[0]["cfg_id"] == "gat_default"
    with pytest.raises(ValueError):
        rg.experiments("nope")


def test_run_gnn_grid_end_to_end_val_only_with_thresholds(tmp_path):
    rg = _load_script("run_gnn_grid")
    data_dir = _dataset(tmp_path)
    out = tmp_path / "out"
    rg.run(data_dir, "fake", out, parts=("improved",), seeds=(42,))
    rows = _rows(out / "grid_fake.csv")
    assert [r["cfg_id"] for r in rows] == ["gat_focal", "gat_resln", "gat_focal_resln"]
    assert list(rows[0]) == rg.COLUMNS
    for r in rows:
        assert int(r["n_params"]) > 0 and float(r["size_kb"]) > 0 and 0 <= float(r["f1_macro"]) <= 1
        assert r["thr_feasible"] in ("True", "False")
    base = ev.count_params(M.GNN("gat", n_classes=3))
    assert int(rows[0]["n_params"]) == base and int(rows[1]["n_params"]) == base + 2 * 128 + 18 * 64
    thr = json.loads((out / "thresholds_grid_fake.json").read_text(encoding="utf-8"))["fake"]
    assert set(thr) == {"gat_focal", "gat_resln", "gat_focal_resln"} and "threshold" in thr["gat_focal"]["42"]
    assert len(_rows(out / "curves_grid_fake.csv")) >= 3
    assert (out / "ckpt_grid" / "fake_gat_focal_s42.pt").exists()


def test_parse_cfg_restores_types_from_search_json():
    rt = _load_script("run_thresholds")
    assert rt.parse_cfg({"n": "100", "d": "None", "C": "10.0", "h": "(64, 64)"}) == {"n": 100, "d": None, "C": 10.0, "h": (64, 64)}


def test_run_thresholds_end_to_end_reproduces_val_f1_of_saved_models(tmp_path):
    rt = _load_script("run_thresholds")
    data_dir = _dataset(tmp_path)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    torch.manual_seed(0)
    for kind in ("sage", "gat"):
        torch.save(M.GNN(kind, n_classes=3).state_dict(), ckpt / f"fake_{kind}_s42.pt")
    search = {n: [{"best": {"C": "1.0"} if n == "logreg" else {"n": "5", "l": "4"}}] for n in rt.BASELINES}
    (tmp_path / "search.json").write_text(json.dumps(search), encoding="utf-8")
    for name in ("base.csv", "gnn.csv"):
        (tmp_path / name).write_text("dataset,model,task,seed,split,f1_macro\nfake,gat,multiclass,42,val,0.5\n", encoding="utf-8")
    res = rt.run(data_dir, "fake", tmp_path / "thr.json", tmp_path / "search.json", tmp_path / "base.csv", tmp_path / "gnn.csv", ckpt,
                 seeds=(42,), only=["gat", "sage", "logreg"])
    assert set(res) == {"gat", "sage", "logreg"}
    for model, by_seed in res.items():
        c = by_seed["42"]
        assert set(c) >= {"threshold", "feasible", "before", "after", "repro_f1_diff"}
        assert c["after"]["fpr"] <= 0.0408 or not c["feasible"]
    assert res["gat"]["42"]["repro_f1_diff"] == pytest.approx(res["gat"]["42"]["before"]["f1_macro"] - 0.5)
    assert res["sage"]["42"]["repro_f1_diff"] is None                       # không có dòng cũ để đối chiếu
    saved = json.loads((tmp_path / "thr.json").read_text(encoding="utf-8"))["fake"]
    assert set(saved) == {"gat", "sage", "logreg"}


def test_gat_explain_param_table_and_attention_samples(tmp_path):
    ge = _load_script("gat_explain")
    data_dir = _dataset(tmp_path)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    torch.save(M.GNN("gat", n_classes=3).state_dict(), ckpt / "fake_gat_s42.pt")
    out = tmp_path / "out"
    ge.run(data_dir, "fake", out, ckpt)
    rows = _rows(out / "gat_param_table_fake.csv")
    summary = json.loads((out / "gat_param_summary_fake.json").read_text(encoding="utf-8"))
    assert sum(int(r["numel"]) for r in rows) == summary["total_params"] == ev.count_params(M.GNN("gat", n_classes=3))
    assert set(summary["by_layer"]) == {"convs.0", "convs.1", "head"} and sum(summary["by_layer"].values()) == summary["total_params"]
    s = json.loads((out / "attention_samples_fake.json").read_text(encoding="utf-8"))
    assert [e["sample_class"] for e in s["samples"]] == ["DoS_DDoS", "Truy_cap_trai_phep"]
    e = s["samples"][0]
    assert len(e["layers"]) == 2 and e["n_nodes"] == len(e["ips"]) == len(e["y"]) == len(e["pred"])
    for layer in e["layers"]:
        assert np.array(layer["alpha"]).shape == (len(layer["edge_index"][0]), 4)
