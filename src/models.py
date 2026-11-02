"""Mô hình GNN phân loại node: GCN, GraphSAGE và GAT nhẹ multi-head (GATv2).

Cả ba dùng cùng khung: `layers` lớp tích chập (độ rộng `hidden * heads`), ELU + dropout, rồi một lớp tuyến tính
ra `n_classes` lớp. Chỉ GAT dùng đặc trưng cạnh (3 chiều) và attention nhiều head.

Đồ thị có hướng (src→dst) nên node nguồn tấn công không nhận được thông điệp từ nạn nhân. Với
`bidirectional=True` (mặc định), thêm cạnh ngược với cùng đặc trưng cạnh trước khi truyền thông điệp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv

KINDS = ("gcn", "sage", "gat")


class GNN(nn.Module):
    def __init__(self, kind, in_dim=18, hidden=16, heads=4, layers=2, n_classes=5, edge_dim=3,
                 dropout=0.2, bidirectional=True):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind phải thuộc {KINDS}, nhận {kind!r}")
        self.kind, self.dropout, self.bidirectional = kind, dropout, bidirectional
        self.convs = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            if kind == "gat":
                self.convs.append(GATv2Conv(d, hidden, heads=heads, edge_dim=edge_dim, dropout=dropout))
                d = hidden * heads
            else:
                width = hidden * heads        # cùng độ rộng với GAT để so sánh công bằng
                self.convs.append((GCNConv if kind == "gcn" else SAGEConv)(d, width))
                d = width
        self.head = nn.Linear(d, n_classes)

    def forward(self, x, edge_index, edge_attr=None):
        if self.bidirectional and edge_index.numel():
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            if edge_attr is not None:
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr) if self.kind == "gat" else conv(x, edge_index)
            x = F.dropout(F.elu(x), self.dropout, self.training)
        return self.head(x)
