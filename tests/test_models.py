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


def test_default_state_dict_unchanged_by_new_options():
    keys = set(M.GNN("gat").state_dict())
    assert not any(k.startswith(("norms", "skips")) for k in keys)             # tùy chọn mới tắt: không thêm khóa nào
    assert any("lin_edge" in k for k in keys)
    assert set(M.GNN("gat", residual=False, layer_norm=False, use_edge=True).state_dict()) == keys


def test_no_edge_features_removes_lin_edge_and_ignores_edge_attr():
    m = M.GNN("gat", use_edge=False).eval()
    assert not any("lin_edge" in k for k in m.state_dict())
    x, ei, ea = _graph()
    assert torch.allclose(m(x, ei, ea), m(x, ei, None), atol=1e-6)             # edge_attr bị bỏ qua
    assert ev.count_params(m) == ev.count_params(M.GNN("gat")) - 2 * 3 * 64    # mỗi lớp bớt lin_edge 3 × 64


def test_residual_layernorm_forward_and_param_count():
    x, ei, ea = _graph()
    m = M.GNN("gat", residual=True, layer_norm=True, n_classes=5).eval()
    out = m(x, ei, ea)
    assert tuple(out.shape) == (10, 5) and torch.isfinite(out).all()
    # LayerNorm 2 lớp: 2 × (64 + 64); đường tắt: lớp 1 chiếu 18 → 64 không bias, lớp 2 là phép đồng nhất
    assert ev.count_params(m) == ev.count_params(M.GNN("gat")) + 2 * 128 + 18 * 64
    assert tuple(m.embed(x, ei, ea).shape) == (10, 64)


@pytest.mark.parametrize("kw", [{"residual": True}, {"layer_norm": True}])
def test_residual_and_layernorm_each_alone_run(kw):
    x, ei, ea = _graph()
    assert torch.isfinite(M.GNN("gat", **kw).eval()(x, ei, ea)).all()


def test_attention_rows_sum_to_one_and_do_not_change_state():
    x, ei, ea = _graph(n=10, e=25)
    m = M.GNN("gat", n_classes=5)                                              # cố ý để chế độ train
    attn = m.attention(x, ei, ea)
    assert m.training                                                          # trả lại chế độ cũ
    assert len(attn) == 2
    for edge_index, alpha in attn:
        assert alpha.shape == (edge_index.shape[1], 4)
        total = torch.zeros(10, 4).index_add_(0, edge_index[1], alpha)         # mỗi node nhận: tổng alpha theo head = 1
        assert torch.allclose(total, torch.ones(10, 4), atol=1e-5)
    with pytest.raises(ValueError):
        M.GNN("gcn").attention(x, ei, ea)
