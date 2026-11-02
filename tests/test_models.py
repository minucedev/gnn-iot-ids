import pytest
import torch

from src import evaluate as ev
from src import models as M


def _graph(n=10, e=25, f=18, ed=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, f, generator=g), torch.randint(0, n, (2, e), generator=g), torch.randn(e, ed, generator=g))


def test_gat_param_count_by_hand():
    # GATv2Conv(in, 16, heads=4, edge_dim=3): lin_l + lin_r = 2 * (in * 64 + 64), lin_edge = 3 * 64, att = 64, bias = 64
    conv = lambda d: 2 * (d * 64 + 64) + 3 * 64 + 64 + 64
    expect = conv(18) + conv(64) + (64 * 5 + 5)            # 2 lớp GAT + lớp tuyến tính 5 lớp
    m = M.GNN("gat", n_classes=5)
    assert ev.count_params(m) == expect == 2752 + 8640 + 325


@pytest.mark.parametrize("kind", M.KINDS)
def test_forward_shapes_and_budget(kind):
    x, ei, ea = _graph()
    m = M.GNN(kind, n_classes=5).eval()
    assert tuple(m(x, ei, ea).shape) == (10, 5)
    assert ev.count_params(m) <= 100_000
    assert ev.model_size_kb(m) <= 1024                       # file mô hình ≤ 1 MB


def test_empty_graph_and_isolated_nodes():
    m = M.GNN("gat").eval()
    x = torch.randn(4, 18)
    out = m(x, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 3))
    assert tuple(out.shape) == (4, 5) and torch.isfinite(out).all()


def test_bidirectional_adds_reverse_edges():
    x, ei, ea = _graph()
    a = M.GNN("gat", bidirectional=False).eval()
    b = M.GNN("gat", bidirectional=True).eval()
    b.load_state_dict(a.state_dict())
    assert not torch.allclose(a(x, ei, ea), b(x, ei, ea))     # hai cách cho kết quả khác nhau
    sym = torch.cat([ei, ei.flip(0)], dim=1)
    assert torch.allclose(b(x, ei, ea), a(x, sym, torch.cat([ea, ea])), atol=1e-6)


def test_bad_kind():
    with pytest.raises(ValueError):
        M.GNN("mlp")
