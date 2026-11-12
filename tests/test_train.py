import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from src import evaluate as ev
from src import models as M
from src import train as T

ROOT = Path(__file__).resolve().parents[1]
CLASSES_3 = ["Normal", "DoS_DDoS", "Truy_cap_trai_phep"]      # NF-BoT-IoT-v3: mã 0, 1, 2 (không có mã 3)


def _run_gnn():
    spec = importlib.util.spec_from_file_location("run_gnn", ROOT / "scripts" / "run_gnn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _graphs(n_graphs=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_graphs):
        out.append(Data(x=torch.randn(6, 18, generator=g), y=torch.tensor([0, 1, 2, 0, 1, 2]),
                        edge_index=torch.randint(0, 6, (2, 10), generator=g),
                        edge_attr=torch.randn(10, 3, generator=g)))
    return out


def test_macro_codes_skip_absent_code_three():
    # Bộ 3 lớp (mã 0, 1, 2): mã 3 trong MACRO_CODES tự bị bỏ, F1-macro là trung bình đúng 3 lớp.
    y_true = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    y_pred = [0, 0, 0, 1, 1, 1, 0, 2, 2, 1]
    m = ev.compute_metrics(y_true, y_pred, class_names=CLASSES_3, macro_labels=T.MACRO_CODES)
    f1 = list(m["f1_per_class"].values())
    assert len(f1) == 3
    assert m["f1_macro"] == pytest.approx(sum(f1) / 3)
    assert m["f1_macro"] == pytest.approx(ev.compute_metrics(y_true, y_pred, class_names=CLASSES_3)["f1_macro"])


def test_macro_codes_exclude_khac_on_five_classes():
    # Cách tính của NF-ToN-IoT-v3 không đổi: 5 lớp, F1-macro trung bình mã 0-3, bỏ Khac (mã 4).
    y_true = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    y_pred = [0, 0, 1, 0, 2, 2, 3, 0, 0, 0]
    m = ev.compute_metrics(y_true, y_pred, macro_labels=T.MACRO_CODES)
    f1 = list(m["f1_per_class"].values())
    assert m["f1_macro"] == pytest.approx(sum(f1[:4]) / 4)


def test_val_scores_loss_matches_weighted_cross_entropy():
    torch.manual_seed(0)
    graphs = _graphs()
    model = M.GNN("gat", n_classes=3).eval()
    cw = np.array([0.5, 2.0, 1.0])
    f1, loss = T.val_scores(model, graphs, torch.device("cpu"), CLASSES_3, cw)
    logits = torch.cat([model(g.x, g.edge_index, g.edge_attr) for g in graphs])
    y = torch.cat([g.y for g in graphs])
    expect = F.cross_entropy(logits, y, weight=torch.tensor(cw, dtype=torch.float32)).item()
    assert loss == pytest.approx(expect, rel=1e-4)
    assert 0.0 <= f1 <= 1.0


def test_fit_history_has_loss_and_f1_per_epoch():
    graphs = T.strip(_graphs())
    model = M.GNN("sage", n_classes=3)
    _, history, _ = T.fit(model, graphs, graphs, CLASSES_3, torch.device("cpu"), seed=0,
                          epochs=3, patience=3, batch_size=3, verbose=False)
    assert [h["epoch"] for h in history] == [1, 2, 3]
    for h in history:
        assert set(h) == {"epoch", "train_loss", "val_loss", "val_f1_macro"}
        assert np.isfinite([h["train_loss"], h["val_loss"], h["val_f1_macro"]]).all()


def test_write_curves_marks_first_best_epoch(tmp_path):
    rg = _run_gnn()
    hist = [{"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_f1_macro": 0.5},
            {"epoch": 2, "train_loss": 0.8, "val_loss": 0.7, "val_f1_macro": 0.8},
            {"epoch": 3, "train_loss": 0.6, "val_loss": 0.7, "val_f1_macro": 0.8}]
    path = tmp_path / "curves_x.csv"
    rg.write_curves(path, "x", "gat", 42, hist)
    rg.write_curves(path, "x", "gat", 43, hist)            # lần 2 nối tiếp, không lặp header
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert list(rows[0]) == rg.CURVE_COLUMNS
    assert len(rows) == 6
    assert [r["is_best"] for r in rows[:3]] == ["0", "1", "0"]
