"""Mô hình GNN phân loại node: GCN, GraphSAGE và GAT nhẹ multi-head (GATv2).

Cả ba dùng cùng khung: `layers` lớp tích chập (độ rộng `hidden * heads`), ELU + dropout, rồi một lớp tuyến tính
ra `n_classes` lớp. Chỉ GAT dùng đặc trưng cạnh (3 chiều) và attention nhiều head.

Đồ thị có hướng (src→dst) nên node nguồn tấn công không nhận được thông điệp từ nạn nhân. Với
`bidirectional=True` (mặc định), thêm cạnh ngược với cùng đặc trưng cạnh trước khi truyền thông điệp.

Các tùy chọn cho ablation và bản cải tiến (task 5.4, 5.5), mặc định tắt nên `state_dict` của mô hình cũ không đổi:
- `use_edge=False`: GAT không dùng đặc trưng cạnh (không có `lin_edge`).
- `layer_norm=True`: LayerNorm sau mỗi lớp tích chập (trước ELU).
- `residual=True`: cộng đường tắt từ đầu vào của lớp (chiếu tuyến tính khi số chiều khác nhau) vào đầu ra của lớp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv

KINDS = ("gcn", "sage", "gat")


class GNN(nn.Module):
    def __init__(self, kind, in_dim=18, hidden=16, heads=4, layers=2, n_classes=5, edge_dim=3,
                 dropout=0.2, bidirectional=True, use_edge=True, residual=False, layer_norm=False):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind phải thuộc {KINDS}, nhận {kind!r}")
        self.kind, self.dropout, self.bidirectional = kind, dropout, bidirectional
        self.use_edge = bool(use_edge) and kind == "gat"
        self.residual, self.layer_norm = bool(residual), bool(layer_norm)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()          # rỗng khi layer_norm=False: không thêm khóa nào vào state_dict
        self.skips = nn.ModuleList()          # rỗng khi residual=False
        d = in_dim
        for _ in range(layers):
            if kind == "gat":
                self.convs.append(GATv2Conv(d, hidden, heads=heads, edge_dim=edge_dim if self.use_edge else None,
                                            dropout=dropout))
                out = hidden * heads
            else:
                out = hidden * heads        # cùng độ rộng với GAT để so sánh công bằng
                self.convs.append((GCNConv if kind == "gcn" else SAGEConv)(d, out))
            if self.layer_norm:
                self.norms.append(nn.LayerNorm(out))
            if self.residual:
                self.skips.append(nn.Identity() if d == out else nn.Linear(d, out, bias=False))
            d = out
        self.head = nn.Linear(d, n_classes)

    def _bidirectional(self, edge_index, edge_attr):
        if self.bidirectional and edge_index.numel():
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            if edge_attr is not None:
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        return edge_index, edge_attr

    def _run(self, x, edge_index, edge_attr=None, attn=None):
        """Các lớp tích chập; nếu `attn` là list (chỉ GAT) thì thêm (edge_index, alpha) của từng lớp."""
        edge_index, edge_attr = self._bidirectional(edge_index, edge_attr)
        ea = edge_attr if self.use_edge else None
        for i, conv in enumerate(self.convs):
            if self.kind != "gat":
                h = conv(x, edge_index)
            elif attn is None:
                h = conv(x, edge_index, ea)
            else:
                h, (ei, alpha) = conv(x, edge_index, ea, return_attention_weights=True)
                attn.append((ei, alpha))
            if self.residual:
                h = h + self.skips[i](x)
            if self.layer_norm:
                h = self.norms[i](h)
            x = F.dropout(F.elu(h), self.dropout, self.training)
        return x

    def embed(self, x, edge_index, edge_attr=None):
        """Vector ẩn sau lớp tích chập cuối (sau ELU và dropout), `hidden * heads` chiều mỗi node (64 với cấu hình
        mặc định). Ở chế độ eval dropout không có tác dụng. `forward` = `head(embed(...))`, trọng số không đổi."""
        return self._run(x, edge_index, edge_attr)

    def forward(self, x, edge_index, edge_attr=None):
        return self.head(self.embed(x, edge_index, edge_attr))

    @torch.no_grad()
    def attention(self, x, edge_index, edge_attr=None):
        """Trọng số attention của GAT (task 5.3, tiêu chí 4), chế độ eval. Trả về danh sách theo lớp gồm
        (edge_index, alpha): `edge_index` (2, E') là các cạnh GATv2Conv thực sự dùng (cạnh gốc, cạnh ngược nếu
        `bidirectional`, rồi vòng lặp tự thân của mọi node), `alpha` (E', heads) là hệ số attention đã chuẩn hóa softmax
        theo node nhận (mỗi node nhận có tổng alpha theo từng head bằng 1)."""
        if self.kind != "gat":
            raise ValueError("Chỉ GAT có trọng số attention")
        was_training = self.training
        self.eval()
        attn = []
        self._run(x, edge_index, edge_attr, attn)
        self.train(was_training)
        return attn
