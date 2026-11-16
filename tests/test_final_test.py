"""Kiểm thử scripts/final_test.py (GĐ H): hàm thuần và chạy toàn bộ quy trình trên dữ liệu giả nhỏ (split=val)."""

import csv
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from src import evaluate as ev
from src import hybrid as H
from src import models as M

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [f"f{i}" for i in range(18)]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FT = _load_script("final_test")


def _graphs(classes, n_graphs=30, seed=3, n=6, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    k = len(classes)
    out = []
    for i in range(n_graphs):
        y = torch.arange(n) % k
        x = torch.randn(n, 18, generator=g) * 0.3 + y[:, None].float() + shift
        x[:, 0] = x[:, 0].abs() + 1.0                                     # n_out > 0 cho mọi node...
        x[0, 0] = 0.0                                                      # ...trừ node 0 (chỉ nhận flow)
        out.append(Data(x=x, y=y, edge_index=torch.randint(0, n, (2, 9), generator=g), edge_attr=torch.randn(9, 3, generator=g),
                        ips=[f"10.0.0.{j}" for j in range(n)], win=i, t_start=float(5 * i)))
    return out


def _dataset(root, name, classes, shift):
    """Bộ giả: x lưu dạng đã chuẩn hóa bằng scaler riêng (mean khác nhau giữa các bộ)."""
    d = root / name
    d.mkdir()
    gs = _graphs(classes, shift=shift)
    raw = torch.cat([g.x for g in gs[:20]]).numpy().astype(np.float64)
    scaler = StandardScaler().fit(raw)
    for g in gs:
        g.x = torch.from_numpy(scaler.transform(g.x.numpy().astype(np.float64)).astype(np.float32))
    torch.save(gs, d / "g.pt")
    idx = {"train": list(range(0, 20)), "val": list(range(20, 26)), "test": list(range(26, 30))}
    meta = {"graph_file": "g.pt", "classes": classes, "features": FEATURES, "window_sec": 5, **{p: {"graph_idx": v} for p, v in idx.items()}}
    (d / "split.json").write_text(json.dumps(meta), encoding="utf-8")
    with open(d / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return d


def test_rescale_roundtrip_and_identity():
    rng = np.random.default_rng(0)
    a = StandardScaler().fit(rng.normal(5, 2, (50, 18)))
    b = StandardScaler().fit(rng.normal(-1, 0.5, (50, 18)))
    g = _graphs(["Normal", "DoS_DDoS", "Truy_cap_trai_phep"], n_graphs=2)
    same = FT.rescale(g, a, a)
    assert torch.allclose(same[0].x, g[0].x, atol=1e-5)
    back = FT.rescale(FT.rescale(g, a, b), b, a)
    assert torch.allclose(back[0].x, g[0].x, atol=1e-4)
    moved = FT.rescale(g, a, b)
    raw = g[0].x.numpy() * a.scale_ + a.mean_
    assert np.allclose(moved[0].x.numpy(), (raw - b.mean_) / b.scale_, atol=1e-4)
    assert torch.equal(moved[0].edge_attr, g[0].edge_attr) and torch.equal(moved[0].y, g[0].y)


def test_common_codes_and_cross_predictions():
    ton = ["Normal", "DoS_DDoS", "Botnet_Backdoor", "Truy_cap_trai_phep", "Khac"]
    cols, lut = FT.common_codes(ton)
    assert cols == [0, 1, 3] and lut.tolist() == [0, 1, -1, 2, -1]
    cols_b, lut_b = FT.common_codes(["Normal", "DoS_DDoS", "Truy_cap_trai_phep"])
    assert cols_b == [0, 1, 2] and lut_b.tolist() == [0, 1, 2]
    proba = np.array([[0.6, 0.1, 0.1, 0.1, 0.1],      # Normal, điểm tấn công 0.4
                      [0.1, 0.2, 0.5, 0.1, 0.1],      # đoán Botnet_Backdoor; trong lớp chung: DoS (0.2) > Truy (0.1) > Normal (0.1)
                      [0.05, 0.05, 0.05, 0.8, 0.05]])
    arg, thr, score = FT.cross_predictions(proba, cols, threshold=0.5)
    assert arg.tolist() == [0, 1, 2]
    assert thr.tolist() == [0, 1, 2]                   # 0.4 ≤ 0.5 → Normal; 0.9 và 0.95 > 0.5 → lớp tấn công chung mạnh nhất
    assert score.tolist() == pytest.approx([0.4, 0.9, 0.95])
    _, thr_low, _ = FT.cross_predictions(proba, cols, threshold=0.3)
    assert thr_low.tolist() == [1, 1, 2]               # điểm 0.4 > 0.3 → tấn công, lớp chung mạnh nhất của hàng 0 là DoS


def test_threshold_entry_both_formats():
    assert FT.threshold_entry({"threshold": 0.7, "f1_macro_before": 0.9}) == (0.7, 0.9)
    assert FT.threshold_entry({"threshold": 0.7, "before": {"f1_macro": 0.8}}) == (0.7, 0.8)


def test_ledger_refuses_second_run(tmp_path):
    lg = FT.Ledger(tmp_path / "ledger.csv")
    lg.check("d", "m", 42, "test")
    lg.add(dataset="d", model="m", seed=42, split="test", commit="abc")
    with pytest.raises(RuntimeError):
        lg.check("d", "m", 42, "test")
    lg.check("d", "m", 43, "test")
    assert FT.Ledger(tmp_path / "ledger.csv").done == {("d", "m", "42", "test")}          # đọc lại từ file
    with pytest.raises(RuntimeError):
        FT.Ledger(tmp_path / "ledger.csv").check("d", "m", 42, "test")


def test_tier_rows_split_initiators_and_receivers():
    y = np.array([0, 0, 1, 1, 0, 1])
    pred = np.array([0, 1, 1, 0, 0, 1])
    init = np.array([True, True, True, False, False, False])
    rows = {r["tier"]: r for r in FT.tier_rows("d", "m", 42, "val", y, pred, np.linspace(0, 1, 6), init, ["Normal", "DoS_DDoS"])}
    assert set(rows) == {"all", "initiator", "receive_only"}
    assert rows["all"]["n_nodes"] == 6 and rows["initiator"]["n_nodes"] == 3 and rows["receive_only"]["n_nodes"] == 3
    assert rows["initiator"]["fpr"] == pytest.approx(0.5) and rows["receive_only"]["fpr"] == pytest.approx(0.0)
    assert rows["all"]["fpr"] == pytest.approx(1 / 3)


def test_final_test_end_to_end_val_with_cross_and_val_check(tmp_path):
    ton = ["Normal", "DoS_DDoS", "Botnet_Backdoor", "Truy_cap_trai_phep"]
    bot = ["Normal", "DoS_DDoS", "Truy_cap_trai_phep"]
    dirs = {"nf_ton_iot_v3": _dataset(tmp_path, "nf_ton_iot_v3", ton, 0.0), "nf_bot_iot_v3": _dataset(tmp_path, "nf_bot_iot_v3", bot, 2.0)}
    cfg = {"split": "val", "out": str(tmp_path / "out"), "cross": True, "seeds": [42], "models": ["logreg", FT.NBR, "gat", FT.HYBRID, FT.LIGHT],
           "thresholds_c": {}, "datasets": {}}
    thr, rr = {}, _load_script("run_threshold_c")
    for ds, d in dirs.items():
        classes = json.loads((d / "split.json").read_text(encoding="utf-8"))["classes"]
        K = len(classes)
        thr[ds] = {}
        # ngưỡng và F1 val tham chiếu: dựng bằng chính mã của script 16h / hiệu chỉnh để kiểm F1 val khớp
        (d / "ckpt").mkdir()
        torch.manual_seed(0)
        gat = M.GNN("gat", n_classes=K)
        torch.save(gat.state_dict(), d / "ckpt" / f"{ds}_gat_s42.pt")
        search = {n: [{"best": {"C": "1.0"} if n == "logreg" else {"n": "5", "l": "4"}}] for n in FT.BASELINES}
        (d / "search.json").write_text(json.dumps(search), encoding="utf-8")
        (d / "abl.json").write_text(json.dumps({rr.ABLATION_MODEL: {"best": {"n": 5, "l": 4}}}), encoding="utf-8")
        hd = d / "hybrid" / f"{ds}_s42"
        hd.mkdir(parents=True)
        gats = []
        for f in range(5):
            torch.manual_seed(f)
            g = M.GNN("gat", n_classes=K)
            torch.save(g.state_dict(), hd / f"gat_fold{f}.pt")
            gats.append(g.eval())
        gtr = _load_graphs_train(d)
        Xtr, ytr = _node_arrays(gtr)
        emb = np.vstack([H.extract_embeddings(gats[0], [g]) for g in gtr])
        Ftr = H.build_features(emb, Xtr)
        lgbm = H.fit_lgbm(Ftr, ytr, {"n": 5, "l": 4}, 42, K)
        with open(hd / "lgbm_oofavg_il.pkl", "wb") as fh:
            pickle.dump(lgbm, fh)
        n_params = sum(ev.count_params(g) for g in gats) + H.lgbm_n_leaves(lgbm)
        (hd / "meta_oofavg_il.json").write_text(json.dumps({"n_params": n_params}), encoding="utf-8")
        ld = d / "light" / f"{ds}_s42"
        ld.mkdir(parents=True)
        with open(ld / FT.LIGHT_CELL, "wb") as fh:
            pickle.dump(lgbm, fh)
        cfg["datasets"][ds] = {"data": str(d), "ckpt": str(d / "ckpt"), "folds": str(d / "hybrid"), "light": str(d / "light"),
                               "baseline_search": str(d / "search.json"), "ablation_search": str(d / "abl.json"),
                               "cross_to": "nf_bot_iot_v3" if ds == "nf_ton_iot_v3" else "nf_ton_iot_v3"}
    # lần 1: tham chiếu rỗng ép dừng vì F1 val không khớp (F1 tham chiếu 0 → lệch lớn)
    for ds in dirs:
        thr[ds] = {m: {"42": {"threshold": 0.5, "f1_macro_before": 0.0}} for m in ("logreg", "gat", FT.HYBRID, FT.LIGHT)}
        cfg["thresholds_c"][ds] = str(tmp_path / f"thr_c_{ds}.json")
        (tmp_path / f"thr_c_{ds}.json").write_text(json.dumps({ds: {FT.NBR: {"42": {"threshold": 0.5, "before": {"f1_macro": 0.0}}}}}), encoding="utf-8")
    cfg["thresholds"] = str(tmp_path / "thr.json")
    (tmp_path / "thr.json").write_text(json.dumps(thr), encoding="utf-8")
    cfg["models"] = ["logreg"]
    with pytest.raises(RuntimeError, match="F1 val dựng lại lệch"):
        FT.run(cfg)
    assert not (tmp_path / "out" / "results_val_nf_ton_iot_v3.csv").exists()               # dừng trước khi ghi bất kỳ kết quả nào

    # lần 2: lấy F1 val dựng lại làm tham chiếu (như thresholds.json của GĐ G) rồi chạy đủ mô hình, thử chéo, chỉ số theo tầng
    out2 = tmp_path / "out2"
    cfg["out"] = str(out2)
    cfg["models"] = ["logreg", FT.NBR, "gat", FT.HYBRID, FT.LIGHT]
    refs = _rebuild_refs(cfg, dirs)
    for ds in dirs:
        for m in ("logreg", "gat", FT.HYBRID, FT.LIGHT):
            thr[ds][m]["42"]["f1_macro_before"] = refs[ds][m]
        (tmp_path / f"thr_c_{ds}.json").write_text(json.dumps({ds: {FT.NBR: {"42": {"threshold": 0.5, "before": {"f1_macro": refs[ds][FT.NBR]}}}}}), encoding="utf-8")
    (tmp_path / "thr.json").write_text(json.dumps(thr), encoding="utf-8")
    assert FT.run(cfg) == 0
    for ds, other in (("nf_ton_iot_v3", "nf_bot_iot_v3"), ("nf_bot_iot_v3", "nf_ton_iot_v3")):
        rows = list(csv.DictReader(open(out2 / f"results_val_{ds}.csv", encoding="utf-8")))
        assert list(rows[0]) == ev.RESULT_COLUMNS
        names = [(r["dataset"], r["model"]) for r in rows]
        models = ["logreg", FT.NBR, "gat", FT.HYBRID] + ([FT.LIGHT] if ds == "nf_ton_iot_v3" else [])
        for m in models:
            assert (ds, m) in names and (ds, f"{m}_thr") in names
        assert (ds, FT.LIGHT) in names if ds == "nf_ton_iot_v3" else (ds, FT.LIGHT) not in names
        cross = {r["model"] for r in rows if r["dataset"] == f"{ds}→{other}"}
        assert cross == {m for m in ("gat", FT.HYBRID, FT.NBR)} | {f"{m}_thr" for m in ("gat", FT.HYBRID, FT.NBR)}
        assert all(r["split"] == "val" and r["task"] == "multiclass" for r in rows)
        tiers = pd.read_csv(out2 / f"tiers_val_{ds}.csv")
        assert set(tiers["tier"]) == {"all", "initiator", "receive_only"}
        nodes = pd.read_parquet(out2 / f"nodes_{ds}_val.parquet")
        pr = pd.read_parquet(out2 / f"pred_{ds}_gat_s42.parquet")
        assert len(pr) == len(nodes) and set(pr.columns) == {"y_true", "pred", "pred_thr", "score"}
        assert nodes["initiator"].sum() == len(nodes) - len(nodes) // 6                  # node 0 của mỗi đồ thị (6 node) là node chỉ nhận
    led = pd.read_csv(out2 / "ledger_val.csv")
    assert len(led) == 9 and led["val_f1_diff"].abs().max() < 1e-9
    with pytest.raises(RuntimeError, match="đã chạy"):                                     # chạy lại cùng thư mục: bị từ chối
        FT.run(cfg)


def _load_graphs_train(d):
    gs = torch.load(d / "g.pt", weights_only=False)
    return [Data(x=g.x, y=g.y, edge_index=g.edge_index, edge_attr=g.edge_attr) for g in gs[:20]]


def _node_arrays(graphs):
    return torch.cat([g.x for g in graphs]).numpy(), torch.cat([g.y for g in graphs]).numpy()


def _rebuild_refs(cfg, dirs):
    """F1 val dựng lại của từng mô hình bằng cùng mã của script (để dùng làm tham chiếu giả cho bài kiểm thử)."""
    from src import train as T
    refs = {}
    for ds, spec in cfg["datasets"].items():
        ctx = FT.Context({**cfg, "split": "val"}, ds, spec, "val")
        refs[ds] = {}
        for m in ("logreg", FT.NBR, "gat", FT.HYBRID, FT.LIGHT):
            p = FT.build(ctx, m, 42).proba(ctx.val)
            refs[ds][m] = ev.compute_metrics(ctx.yva, p.argmax(1), p, class_names=ctx.classes, macro_labels=T.MACRO_CODES)["f1_macro"]
    return refs
