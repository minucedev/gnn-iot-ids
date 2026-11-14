"""Phân tích FPR của node Normal bị đoán là tấn công (task 5.3, tiêu chí 2).

Các hàm thuần (không đọc file) để test được bằng dữ liệu giả; phần đọc dữ liệu, huấn luyện, vẽ hình nằm ở
scripts/fpr_analysis.py.

Quy ước nhãn node của pipeline (src/build_graph.py, `node_edge_tables`): đa số nhãn flow do IP khởi tạo (flow ra);
IP không khởi tạo flow nào thì lấy đa số flow nhận; hòa → mã lớp lớn hơn. `RULES` là các cách gán thay thế để
đánh giá "đổi cách gán nhãn thì ảnh hưởng tới đâu"; chỉ là phản thực tế trên dự đoán đã có, không huấn luyện lại.
"""

import numpy as np
import pandas as pd

NORMAL = 0
RULES = ("hien_tai", "tat_ca_flow", "chi_khoi_tao", "co_flow_tan_cong")
RULE_DESC = {
    "hien_tai": "Hiện tại: đa số flow ra; không có flow ra thì đa số flow nhận",
    "tat_ca_flow": "Đa số trên mọi flow (ra + nhận)",
    "chi_khoi_tao": "Chỉ node có flow ra (node chỉ nhận bị bỏ khỏi đánh giá)",
    "co_flow_tan_cong": "Có ≥ 1 flow tấn công (ra hoặc nhận) thì là tấn công",
}


def node_table(graphs):
    """Bảng node theo đúng thứ tự ghép của `data.node_arrays`: cột gi (chỉ số đồ thị), win, ip, y."""
    gi, win, ip, y = [], [], [], []
    for i, g in enumerate(graphs):
        n = g.num_nodes
        gi.append(np.full(n, i, dtype=np.int64))
        win.append(np.full(n, int(g.win), dtype=np.int64))
        ip.extend(str(v) for v in g.ips)
        y.append(g.y.numpy())
    return pd.DataFrame({"gi": np.concatenate(gi), "win": np.concatenate(win), "ip": ip, "y": np.concatenate(y)})


def flow_label_counts(flows, n_classes):
    """Đếm flow theo (cửa sổ, IP, nhãn) cho flow ra và flow nhận. `flows` có cột win, src, dst, y (mã lớp).

    Bỏ flow src == dst như `node_edge_tables`. Trả về DataFrame chỉ số (win, ip) với cột out_0..out_{K-1},
    in_0..in_{K-1} (số nguyên)."""
    fl = flows[flows["src"] != flows["dst"]]
    parts = []
    for side, col in (("out", "src"), ("in", "dst")):
        t = fl.groupby(["win", col, "y"]).size().unstack("y", fill_value=0)
        t = t.reindex(columns=range(n_classes), fill_value=0)
        t.columns = [f"{side}_{k}" for k in range(n_classes)]
        t.index = t.index.set_names(["win", "ip"])
        parts.append(t)
    return parts[0].join(parts[1], how="outer").fillna(0).astype("int64")


def attach_counts(nodes, counts, n_classes):
    """Ghép số flow theo nhãn vào bảng node (giữ nguyên thứ tự hàng của `nodes`). Trả về mảng (out, inn) hình (n, K)."""
    m = nodes[["win", "ip"]].merge(counts.reset_index(), on=["win", "ip"], how="left")
    assert len(m) == len(nodes)
    if m.isna().any().any():
        raise ValueError(f"{int(m.isna().any(axis=1).sum())} node không có flow tương ứng (sai cửa sổ hoặc sai bộ dữ liệu?)")
    out = m[[f"out_{k}" for k in range(n_classes)]].to_numpy(np.int64)
    inn = m[[f"in_{k}" for k in range(n_classes)]].to_numpy(np.int64)
    return out, inn


def _major(c):
    """Lớp chiếm đa số theo hàng; hòa → mã lớn hơn (giống thứ tự sắp xếp của `node_edge_tables`)."""
    return c.shape[1] - 1 - np.argmax(c[:, ::-1], axis=1)


def relabel(out, inn, rule):
    """Nhãn node theo một cách gán. -1 nghĩa là node bị bỏ (chỉ ở `chi_khoi_tao`)."""
    n_out = out.sum(1)
    if rule == "hien_tai":
        return np.where(n_out > 0, _major(out), _major(inn))
    if rule == "tat_ca_flow":
        return _major(out + inn)
    if rule == "chi_khoi_tao":
        return np.where(n_out > 0, _major(out), -1)
    if rule == "co_flow_tan_cong":
        att = (out + inn).copy()
        att[:, NORMAL] = 0
        return np.where(att.sum(1) > 0, _major(att), NORMAL)
    raise ValueError(f"rule phải thuộc {RULES}, nhận {rule!r}")


def purity_table(out, inn):
    """Độ thuần nhãn mỗi node.

    n_out, n_in: số flow ra, nhận. out_normal_frac: tỉ lệ Normal trong flow ra (NaN nếu không có flow ra).
    all_normal_frac: tỉ lệ Normal trong mọi flow. in_normal_frac: tỉ lệ Normal trong flow nhận (NaN nếu không có).
    from_in_only: nhãn lấy từ flow nhận vì node không khởi tạo flow nào. purity: phần của lớp đa số trong các flow
    dùng để gán nhãn."""
    n_out, n_in = out.sum(1), inn.sum(1)
    src = np.where((n_out > 0)[:, None], out, inn)
    return pd.DataFrame({
        "n_out": n_out, "n_in": n_in,
        "out_normal_frac": np.where(n_out > 0, out[:, NORMAL] / np.maximum(n_out, 1), np.nan),
        "in_normal_frac": np.where(n_in > 0, inn[:, NORMAL] / np.maximum(n_in, 1), np.nan),
        "all_normal_frac": (out[:, NORMAL] + inn[:, NORMAL]) / np.maximum(n_out + n_in, 1),
        "from_in_only": n_out == 0,
        "purity": src.max(1) / np.maximum(src.sum(1), 1),
    })


def class_centroids(X, y, n_classes):
    """Tâm từng lớp (trung bình đặc trưng); lớp không có mẫu cho hàng NaN."""
    C = np.full((n_classes, X.shape[1]), np.nan)
    for k in range(n_classes):
        m = y == k
        if m.any():
            C[k] = X[m].mean(0)
    return C


def centroid_distances(X, C, chunk=50_000):
    """Khoảng cách Euclid từ từng node tới tâm từng lớp, hình (n, K); lớp vắng mặt có khoảng cách vô cực."""
    out = np.empty((len(X), len(C)))
    for a in range(0, len(X), chunk):
        d = np.sqrt(((X[a:a + chunk, None, :] - C[None, :, :]) ** 2).sum(-1))
        out[a:a + chunk] = np.where(np.isnan(d), np.inf, d)
    return out


def binary_rates(y, pred, keep=None):
    """FPR, recall tấn công (Normal là lớp âm) trên các node có keep=True (mặc định tất cả)."""
    y, pred = np.asarray(y), np.asarray(pred)
    keep = np.ones(len(y), bool) if keep is None else np.asarray(keep)
    neg, pos = keep & (y == NORMAL), keep & (y != NORMAL)
    fp = int((neg & (pred != NORMAL)).sum())
    tp = int((pos & (pred != NORMAL)).sum())
    return {"n_normal": int(neg.sum()), "n_attack": int(pos.sum()), "n_fp": fp,
            "fpr": fp / neg.sum() if neg.any() else float("nan"),
            "recall_attack": tp / pos.sum() if pos.any() else float("nan")}


def ip_report(val_nodes, fp_mask, ref_nodes, top=10, all_if_le=30):
    """FP theo IP. `val_nodes` (cột ip, y), `fp_mask` cùng thứ tự (node Normal bị đoán là tấn công),
    `ref_nodes` = node train + val (cột ip, y) để đếm IP đó xuất hiện như node tấn công / Normal ở cửa sổ khác."""
    ref = ref_nodes.assign(a=(ref_nodes["y"] != NORMAL).astype(int), n=(ref_nodes["y"] == NORMAL).astype(int)) \
        .groupby("ip")[["a", "n"]].sum().rename(columns={"a": "ref_attack_nodes", "n": "ref_normal_nodes"})
    v = val_nodes.assign(fp=np.asarray(fp_mask).astype(int))
    nv = v[v["y"] == NORMAL]
    g = nv.groupby("ip").agg(val_normal_nodes=("fp", "size"), fp=("fp", "sum")).join(ref)
    g["fp_rate"] = g["fp"] / g["val_normal_nodes"]
    g = g.sort_values(["fp", "val_normal_nodes"], ascending=False)
    total_fp = int(g["fp"].sum())
    def rows(df):
        return [{k: (v if k == "ip" else float(v) if k == "fp_rate" else int(v)) for k, v in rec.items()}
                for rec in df.reset_index().to_dict("records")]

    has_att = g["ref_attack_nodes"] > 0
    return {
        "n_ip_normal_val": int(len(g)), "n_ip_with_fp": int((g["fp"] > 0).sum()), "total_fp": total_fp,
        "share_fp_top1": float(g["fp"].iloc[:1].sum() / total_fp) if total_fp else float("nan"),
        "share_fp_top3": float(g["fp"].iloc[:3].sum() / total_fp) if total_fp else float("nan"),
        "share_fp_ip_also_attack_node": float(g.loc[has_att, "fp"].sum() / total_fp) if total_fp else float("nan"),
        "share_normal_ip_also_attack_node": float(has_att.mean()) if len(g) else float("nan"),
        "top": rows(g.head(top)),
        "all": rows(g) if len(g) <= all_if_le else None,
    }
