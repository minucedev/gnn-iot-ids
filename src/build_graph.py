"""Dựng đồ thị theo cửa sổ thời gian từ flow (task 0.4).

Tách nguyên văn logic từ notebooks/phong_tuan1/phong_tuan1_dataset_graph.ipynb (Tuần 1, Phong) để
hash đồ thị trùng bản notebook (NF-ToN-IoT-v2, W = S = 1000 flow: `cb8842d60f9a3a5a`).

Quy trình: adapter (đọc file gốc → bảng flow chuẩn) → clean_flows → encode_labels → build_dataset
(cửa sổ → bảng node/cạnh → chia thời gian → scaler fit trên train → danh sách torch_geometric Data).

Bảng flow chuẩn có các cột: ts, src, dst, sport, dport, proto, bytes_fwd, bytes_bwd, pkts_fwd, pkts_bwd,
dur (giây), f_noreply, f_rst, f_est, label_raw. Thêm bộ dữ liệu mới = viết 1 hàm adapter trả về bảng này
và đăng ký vào ADAPTERS (và TS_IS_ORDER nếu ts là số thứ tự dòng).

Node = IP trong 1 cửa sổ, 18 đặc trưng (FEATURES); cạnh có hướng src→dst, 3 đặc trưng (EDGE_FEATURES).
Nhãn node: đa số nhãn flow IP khởi tạo; nếu không có thì flow IP nhận; hòa → mã lớp lớn hơn.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

# ---------------------------------------------------------------- nhãn
GROUP_ORDER = ["Normal", "DoS_DDoS", "Botnet_Backdoor", "Truy_cap_trai_phep", "Khac"]
GROUP_MAP = {
    "normal": "Normal", "benign": "Normal",
    "ddos": "DoS_DDoS", "dos": "DoS_DDoS",
    "backdoor": "Botnet_Backdoor", "botnet": "Botnet_Backdoor", "bot": "Botnet_Backdoor",
    "scanning": "Truy_cap_trai_phep", "reconnaissance": "Truy_cap_trai_phep",
    "password": "Truy_cap_trai_phep", "injection": "Truy_cap_trai_phep",
    "xss": "Truy_cap_trai_phep", "theft": "Truy_cap_trai_phep",
    "mitm": "Khac", "ransomware": "Khac",
    # NF-CSE-CIC-IDS2018-v3 (không có lớp Khac): DoS/DDoS, Bot, brute force/injection/infiltration
    "ddos_attack-hoic": "DoS_DDoS", "ddos_attacks-loic-http": "DoS_DDoS", "ddos_attack-loic-udp": "DoS_DDoS",
    "dos_attacks-slowhttptest": "DoS_DDoS", "dos_attacks-hulk": "DoS_DDoS",
    "dos_attacks-goldeneye": "DoS_DDoS", "dos_attacks-slowloris": "DoS_DDoS",
    "ftp-bruteforce": "Truy_cap_trai_phep", "ssh-bruteforce": "Truy_cap_trai_phep",
    "brute_force_-web": "Truy_cap_trai_phep", "brute_force_-xss": "Truy_cap_trai_phep",
    "sql_injection": "Truy_cap_trai_phep", "infilteration": "Truy_cap_trai_phep",   # "Infilteration" là tên gốc của bộ dữ liệu
}


def to_group(raw):
    """Nhãn gốc → nhóm (không phân biệt hoa/thường, ví dụ `Backdoor` của v3); nhãn lạ → Khac."""
    return pd.Series(raw).astype(str).str.strip().str.lower().map(GROUP_MAP).fillna("Khac").values


# ---------------------------------------------------------------- đặc trưng
FEATURES = ["n_out", "n_in", "bytes_sent", "bytes_recv", "pkts_sent", "pkts_recv", "uniq_dst",
            "uniq_dport", "uniq_src", "dur_mean", "pkt_size_mean", "noreply_ratio", "rst_ratio",
            "est_ratio", "tcp_ratio", "udp_ratio", "icmp_ratio", "iat_mean_out"]
LOG_FEATURES = ["n_out", "n_in", "bytes_sent", "bytes_recv", "pkts_sent", "pkts_recv", "uniq_dst",
                "uniq_dport", "uniq_src", "dur_mean", "pkt_size_mean", "iat_mean_out"]  # log1p trước khi chuẩn hóa
EDGE_FEATURES = ["e_flows", "e_bytes", "e_pkts"]                                         # log1p
NUM_COLS = ["ts", "bytes_fwd", "bytes_bwd", "pkts_fwd", "pkts_bwd", "dur"]

# ---------------------------------------------------------------- cấu hình mặc định (giống notebook Tuần 1)
SPLIT_RATIO = (0.70, 0.15, 0.15)
SPLIT_MODE = "chronological"
SPLIT_BLOCKS = 10
BUILD_CHUNK_WINDOWS = 1000

# ---------------------------------------------------------------- đọc file
BIG_FILE = 1 * 2**30     # file > 1 GB: đọc bằng pyarrow, chỉ lấy các cột cần


def read_any(path, columns=None):
    path = Path(path)
    if path.stat().st_size > BIG_FILE:
        opts = pacsv.ConvertOptions(include_columns=columns) if columns else None
        return pacsv.read_csv(path, convert_options=opts).to_pandas()
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df if columns is None else df[columns]


def _port(s):
    """Cổng dạng số; BoT-IoT có cổng hex ('0x0303') với ICMP và rỗng với ARP → -1."""
    s = s.astype("string").str.strip()
    is_hex = s.str.startswith("0x", na=False)
    out = pd.to_numeric(s.where(~is_hex), errors="coerce")
    out[is_hex] = s[is_hex].map(lambda v: int(v, 16) if all(c in "0123456789abcdefABCDEF" for c in v[2:]) else np.nan)
    return out.fillna(-1).astype("int64")


# ---------------------------------------------------------------- adapter từng bộ
def load_bot_iot(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    state = df["state"].astype("string").str.strip()
    return pd.DataFrame({
        "ts": pd.to_numeric(df["stime"], errors="coerce"),
        "src": df["saddr"].astype("string").str.strip(), "dst": df["daddr"].astype("string").str.strip(),
        "sport": _port(df["sport"]), "dport": _port(df["dport"]),
        "proto": df["proto"].astype("string").str.strip().str.lower(),
        "bytes_fwd": df["sbytes"], "bytes_bwd": df["dbytes"], "pkts_fwd": df["spkts"], "pkts_bwd": df["dpkts"],
        "dur": df["dur"],
        "f_noreply": state.isin(["REQ", "INT"]), "f_rst": state.eq("RST"), "f_est": state.isin(["CON", "FIN", "ACC"]),
        "label_raw": df["category"].astype("string").str.strip(),
    })


TON_REJ = {"REJ", "RSTO", "RSTR", "RSTOS0", "RSTRH"}


def load_ton_iot(path):
    df = read_any(path)
    if "ts" not in df.columns:
        raise ValueError(f"{Path(path).name} không có cột ts → không chia cửa sổ thời gian được. "
                         "Cần tải Processed_Network_dataset của ToN-IoT.")
    st = df["conn_state"].astype("string").str.strip()
    return pd.DataFrame({
        "ts": pd.to_numeric(df["ts"], errors="coerce"),
        "src": df["src_ip"].astype("string").str.strip(), "dst": df["dst_ip"].astype("string").str.strip(),
        "sport": _port(df["src_port"]), "dport": _port(df["dst_port"]),
        "proto": df["proto"].astype("string").str.strip().str.lower(),
        "bytes_fwd": df["src_bytes"], "bytes_bwd": df["dst_bytes"], "pkts_fwd": df["src_pkts"], "pkts_bwd": df["dst_pkts"],
        "dur": df["duration"],
        "f_noreply": st.isin(["S0", "SH", "OTH"]), "f_rst": st.isin(TON_REJ), "f_est": st.isin(["SF", "S1", "S2", "S3"]),
        "label_raw": df["type"].astype("string").str.strip(),
    })


NF_COLS = ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL", "IN_BYTES", "IN_PKTS",
           "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
           "FLOW_DURATION_MILLISECONDS", "Attack"]
NF_PROTO = {6: "tcp", 17: "udp", 1: "icmp", 58: "ipv6-icmp"}
NF_DUR_OVERFLOW_MS = 4_290_000      # ≈ 2^32/1000: thời lượng âm bị tràn số 32-bit trong nProbe
SYN, RST, ACK = 0x02, 0x04, 0x10


def load_nf(path, ts_col=None):
    """NetFlow v2/v3 (UQ). IN_* = chiều nguồn→đích, OUT_* = chiều ngược lại.

    ts_col=None (v2, không có timestamp): ts = số thứ tự dòng (giả định file xếp theo thời gian).
    ts_col="FLOW_START_MILLISECONDS" (v3): ts = giây epoch.
    """
    df = read_any(path, columns=NF_COLS + ([ts_col] if ts_col else []))
    tcp = df["PROTOCOL"].eq(6).to_numpy()
    cf, sf, tf = (df[c].to_numpy() for c in ("CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "TCP_FLAGS"))
    replied = df["OUT_PKTS"].gt(0).to_numpy()
    ms = df["FLOW_DURATION_MILLISECONDS"].to_numpy()
    ts = np.arange(len(df), dtype="float64") if ts_col is None else df[ts_col].to_numpy(np.float64) / 1000.0
    out = pd.DataFrame({
        "ts": ts,
        "src": df["IPV4_SRC_ADDR"].astype("string"), "dst": df["IPV4_DST_ADDR"].astype("string"),
        "sport": df["L4_SRC_PORT"].astype("int64"), "dport": df["L4_DST_PORT"].astype("int64"),
        "proto": df["PROTOCOL"].map(NF_PROTO).fillna("other").astype("string"),
        "bytes_fwd": df["IN_BYTES"], "bytes_bwd": df["OUT_BYTES"], "pkts_fwd": df["IN_PKTS"], "pkts_bwd": df["OUT_PKTS"],
        "dur": np.where(ms >= NF_DUR_OVERFLOW_MS, 0, ms) / 1000.0,
        "f_noreply": np.where(tcp, (cf & SYN > 0) & (sf == 0), ~replied),          # SYN không được trả lời / UDP không phản hồi
        "f_rst": tcp & (tf & RST > 0),
        "f_est": np.where(tcp, (sf & (SYN | ACK)) == (SYN | ACK), replied),        # server trả SYN-ACK
        "label_raw": df["Attack"].astype("string"),
    })
    out.attrs["dur_overflow"] = int((ms >= NF_DUR_OVERFLOW_MS).sum())
    return out


def load_nf_v2(path):
    return load_nf(path)


def load_nf_v3(path):
    return load_nf(path, ts_col="FLOW_START_MILLISECONDS")


ADAPTERS = {"bot_iot": load_bot_iot, "ton_iot": load_ton_iot, "nf_ton_iot_v2": load_nf_v2,
            "nf_bot_iot_v2": load_nf_v2, "nf_ton_iot_v3": load_nf_v3,
            "nf_cicids2018_v3": load_nf_v3}
TS_IS_ORDER = {"nf_ton_iot_v2": True, "nf_bot_iot_v2": True}     # ts là số thứ tự dòng, không phải thời gian thật


# ---------------------------------------------------------------- làm sạch
def clean_flows(df, ts_is_order=False):
    """Bỏ thiếu/vô cực/âm và bản ghi trùng; gán nhóm nhãn `label`; sắp theo ts. Trả về (flows, log)."""
    log = [("Ban đầu", len(df))]
    if df.attrs.get("dur_overflow"):
        log.append((f"Sửa thời lượng tràn số 32-bit → 0 ({df.attrs['dur_overflow']:,} dòng, không bỏ)", len(df)))
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=NUM_COLS + ["src", "dst", "proto", "label_raw"])
    log.append(("Bỏ thiếu / vô cực", len(df)))
    df = df[(df[["bytes_fwd", "bytes_bwd", "pkts_fwd", "pkts_bwd", "dur"]] >= 0).all(axis=1)]
    log.append(("Bỏ giá trị âm", len(df)))
    if ts_is_order:
        rest = df.drop(columns="ts")
        df = df[~(rest == rest.shift()).all(axis=1).to_numpy()]
        del rest
        log.append(("Bỏ bản ghi trùng lặp liên tiếp", len(df)))
    else:
        df = df.drop_duplicates()
        log.append(("Bỏ bản ghi trùng lặp", len(df)))
    df = df.astype({"ts": "float64", "bytes_fwd": "int64", "bytes_bwd": "int64", "pkts_fwd": "int64",
                    "pkts_bwd": "int64", "dur": "float64", "src": str, "dst": str, "proto": str,
                    "label_raw": str, "f_noreply": bool, "f_rst": bool, "f_est": bool})
    df["label"] = to_group(df["label_raw"])
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    return df, pd.DataFrame(log, columns=["buoc", "so_dong"])


def encode_labels(flows):
    """Gán cột `y` (mã lớp) theo thứ tự GROUP_ORDER, chỉ gồm các nhóm có mặt; Normal luôn là 0."""
    classes = [g for g in GROUP_ORDER if g in set(flows["label"])]
    class_id = {c: i for i, c in enumerate(classes)}
    flows["y"] = flows["label"].map(class_id).astype("int64")
    return classes


# ---------------------------------------------------------------- cửa sổ → node/cạnh
def assign_windows(fl, window_sec, stride_sec):
    """Gán cột `win` cho từng flow; flow được nhân bản nếu thuộc nhiều cửa sổ (cửa sổ trượt chồng nhau)."""
    assert window_sec % stride_sec == 0, "window phải chia hết cho stride"
    k_last = np.floor(fl["ts"].to_numpy() / stride_sec).astype(np.int64)
    r = window_sec // stride_sec
    parts = [fl.assign(win=k_last - j) for j in range(r)]
    return parts[0] if r == 1 else pd.concat(parts, ignore_index=True)


def node_edge_tables(fw, window_sec):
    """Từ flow đã gán cửa sổ → (bảng node, bảng cạnh). Toàn bộ logic đặc trưng/nhãn nằm ở đây."""
    fw = fw[fw["src"] != fw["dst"]]
    if "y" not in fw:
        fw = fw.assign(y=-1)                                    # dữ liệu thời gian thực không có nhãn
    proto = fw["proto"].to_numpy()
    common = dict(win=fw["win"].to_numpy(), dur=fw["dur"].to_numpy(),
                  tot_b=(fw["bytes_fwd"] + fw["bytes_bwd"]).to_numpy(),
                  tot_p=(fw["pkts_fwd"] + fw["pkts_bwd"]).to_numpy(),
                  noreply=fw["f_noreply"].to_numpy(), rst=fw["f_rst"].to_numpy(), est=fw["f_est"].to_numpy(),
                  tcp=proto == "tcp", udp=proto == "udp", icmp=np.isin(proto, ["icmp", "ipv6-icmp"]),
                  y=fw["y"].to_numpy(), ts=fw["ts"].to_numpy())
    out = pd.DataFrame({**common, "ip": fw["src"].to_numpy(), "peer": fw["dst"].to_numpy(),
                        "pport": fw["dport"].to_numpy(), "sent_b": fw["bytes_fwd"].to_numpy(),
                        "recv_b": fw["bytes_bwd"].to_numpy(), "sent_p": fw["pkts_fwd"].to_numpy(),
                        "recv_p": fw["pkts_bwd"].to_numpy(), "prio": 0})
    inc = pd.DataFrame({**common, "ip": fw["dst"].to_numpy(), "peer": fw["src"].to_numpy(),
                        "pport": fw["sport"].to_numpy(), "sent_b": fw["bytes_bwd"].to_numpy(),
                        "recv_b": fw["bytes_fwd"].to_numpy(), "sent_p": fw["pkts_bwd"].to_numpy(),
                        "recv_p": fw["pkts_fwd"].to_numpy(), "prio": 1})
    long = pd.concat([out, inc], ignore_index=True)
    key = ["win", "ip"]

    nodes = long.groupby(key).agg(
        bytes_sent=("sent_b", "sum"), bytes_recv=("recv_b", "sum"),
        pkts_sent=("sent_p", "sum"), pkts_recv=("recv_p", "sum"),
        dur_mean=("dur", "mean"), tot_b=("tot_b", "sum"), tot_p=("tot_p", "sum"),
        noreply_ratio=("noreply", "mean"), rst_ratio=("rst", "mean"), est_ratio=("est", "mean"),
        tcp_ratio=("tcp", "mean"), udp_ratio=("udp", "mean"), icmp_ratio=("icmp", "mean"))
    nodes["pkt_size_mean"] = nodes["tot_b"] / nodes["tot_p"].clip(lower=1)
    nodes = nodes.join(out.groupby(key).agg(n_out=("peer", "size"), uniq_dst=("peer", "nunique"),
                                            uniq_dport=("pport", "nunique")))
    nodes = nodes.join(inc.groupby(key).agg(n_in=("peer", "size"), uniq_src=("peer", "nunique")))
    o = out.sort_values(key + ["ts"], kind="stable")
    iat = o["ts"].diff().where(o["win"].eq(o["win"].shift()) & o["ip"].eq(o["ip"].shift()))
    nodes = nodes.join(iat.groupby([o["win"], o["ip"]]).mean().rename("iat_mean_out"))
    nodes[["n_out", "uniq_dst", "uniq_dport", "n_in", "uniq_src"]] = \
        nodes[["n_out", "uniq_dst", "uniq_dport", "n_in", "uniq_src"]].fillna(0)
    nodes["iat_mean_out"] = nodes["iat_mean_out"].fillna(float(window_sec))   # < 2 flow ra: coi như 1 cửa sổ

    # nhãn node: ưu tiên flow ra (prio 0), đa số, hòa → lớp lớn hơn
    cnt = long.groupby(key + ["prio", "y"]).size().rename("c").reset_index()
    cnt = cnt.sort_values(key + ["prio", "c", "y"], ascending=[True, True, True, False, False], kind="stable")
    nodes["y"] = cnt.drop_duplicates(key).set_index(key)["y"]

    nodes = nodes.reset_index().sort_values(key, kind="stable").reset_index(drop=True)
    nodes[LOG_FEATURES] = np.log1p(nodes[LOG_FEATURES].astype("float64"))
    nodes["local"] = nodes.groupby("win").cumcount()

    e = fw.assign(tot_b=common["tot_b"], tot_p=common["tot_p"]).groupby(["win", "src", "dst"]).agg(
        e_flows=("ts", "size"), e_bytes=("tot_b", "sum"), e_pkts=("tot_p", "sum")).reset_index()
    idx = nodes[["win", "ip", "local"]]
    e = e.merge(idx.rename(columns={"ip": "src", "local": "s"}), on=["win", "src"]) \
         .merge(idx.rename(columns={"ip": "dst", "local": "d"}), on=["win", "dst"])
    e = e.sort_values(["win", "s", "d"], kind="stable").reset_index(drop=True)
    e[EDGE_FEATURES] = np.log1p(e[EDGE_FEATURES].astype("float64"))
    return nodes[["win", "ip", "local", *FEATURES, "y"]], e[["win", "s", "d", *EDGE_FEATURES]]


def to_pyg(nodes, edges, scaler, stride_sec):
    """Bảng node/cạnh → danh sách Data (một Data cho mỗi cửa sổ, theo thứ tự thời gian)."""
    X = nodes[FEATURES].to_numpy(np.float64)
    X = (scaler.transform(X) if scaler is not None else X).astype(np.float32)
    Y = nodes["y"].to_numpy(np.int64)
    ips = nodes["ip"].to_numpy()
    wins, starts = np.unique(nodes["win"].to_numpy(), return_index=True)
    ends = np.r_[starts[1:], len(nodes)]
    ew = edges["win"].to_numpy()
    ea, eb = np.searchsorted(ew, wins, "left"), np.searchsorted(ew, wins, "right")
    E = edges[["s", "d"]].to_numpy(np.int64)
    EA = edges[EDGE_FEATURES].to_numpy(np.float32)
    graphs = []
    for k, a, b, c, d in zip(wins, starts, ends, ea, eb):
        # .copy(): mỗi đồ thị giữ bộ nhớ riêng (nếu chỉ cắt view, torch.save sẽ ghi lại cả mảng gốc cho mọi đồ thị)
        graphs.append(Data(x=torch.from_numpy(X[a:b].copy()), y=torch.from_numpy(Y[a:b].copy()),
                           edge_index=torch.from_numpy(E[c:d].T.copy()),
                           edge_attr=torch.from_numpy(EA[c:d].copy()),
                           win=int(k), t_start=float(k * stride_sec), ips=list(ips[a:b])))
    return graphs


# ---------------------------------------------------------------- chia tập và dựng toàn bộ
def time_split(wins, ratios=SPLIT_RATIO, gap=0, mode=SPLIT_MODE, blocks=SPLIT_BLOCKS, weights=None):
    """Chia danh sách cửa sổ (đã sắp theo thời gian). gap = số cửa sổ bỏ ở mỗi ranh giới giữa hai tập.

    weights: trọng số mỗi cửa sổ (ví dụ số node), cùng độ dài với wins. Có weights thì ranh giới train/val/test trong
    mỗi khối đặt theo tỉ lệ tổng trọng số thay vì tỉ lệ số cửa sổ, để tỉ lệ node gần 70/15/15 hơn.
    """
    segs = [wins] if mode == "chronological" else np.array_split(wins, blocks)
    if weights is None:
        wsegs = [None] * len(segs)
    else:
        wsegs = [np.asarray(weights)] if mode == "chronological" else np.array_split(np.asarray(weights), blocks)
    out = {"train": [], "val": [], "test": []}
    for i, seg in enumerate(segs):
        n = len(seg)
        if wsegs[i] is None or n == 0:
            a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
        else:
            c = np.cumsum(wsegs[i], dtype=np.float64)
            a = int(np.searchsorted(c, c[-1] * ratios[0], side="right"))
            b = int(np.searchsorted(c, c[-1] * (ratios[0] + ratios[1]), side="right"))
        out["train"].append(seg[(gap if i else 0):a])        # khối sau bắt đầu ngay sau test của khối trước
        out["val"].append(seg[a + gap:b])
        out["test"].append(seg[b + gap:])
    return {p: np.concatenate(v) for p, v in out.items()}


def build_tables(flows, window_sec, stride_sec, chunk=BUILD_CHUNK_WINDOWS):
    """Bảng node/cạnh của mọi cửa sổ, dựng theo lô cửa sổ [k0, k0+chunk) để tiết kiệm RAM."""
    ts = flows["ts"].to_numpy()                                   # flows đã sắp theo ts
    k_lo = int(np.floor(ts[0] / stride_sec)) - (window_sec // stride_sec - 1)
    k_hi = int(np.floor(ts[-1] / stride_sec))
    nodes_l, edges_l = [], []
    for k0 in range(k_lo, k_hi + 1, chunk):
        k1 = k0 + chunk
        lo, hi = np.searchsorted(ts, k0 * stride_sec, "left"), np.searchsorted(ts, (k1 - 1) * stride_sec + window_sec, "left")
        if lo == hi:
            continue
        fw = assign_windows(flows.iloc[lo:hi], window_sec, stride_sec)
        n, e = node_edge_tables(fw[(fw["win"] >= k0) & (fw["win"] < k1)], window_sec)
        nodes_l.append(n)
        edges_l.append(e)
    return pd.concat(nodes_l, ignore_index=True), pd.concat(edges_l, ignore_index=True)


def build_dataset(flows, window_sec, stride_sec, ratios=SPLIT_RATIO, mode=SPLIT_MODE, blocks=SPLIT_BLOCKS,
                  chunk=BUILD_CHUNK_WINDOWS, gap=None, balance_nodes=False):
    """Toàn bộ quy trình offline: cửa sổ → node/cạnh → chia thời gian → scaler(train) → Data.

    gap: số cửa sổ bỏ ở mỗi ranh giới giữa hai tập; None = window_sec // stride_sec - 1 (đủ để cửa sổ trượt không chồng
    lên nhau, bằng 0 khi W = S). Đặt gap >= 1 nếu muốn có khoảng trống thật giữa các tập.
    balance_nodes=True: đặt ranh giới tập theo số node thay vì số cửa sổ (xem time_split).
    Trả về (graphs, split, scaler, nodes): split = {"train"|"val"|"test": mảng mã cửa sổ}.
    """
    nodes, edges = build_tables(flows, window_sec, stride_sec, chunk)
    wins, n_per_win = np.unique(nodes["win"].to_numpy(), return_counts=True)
    gap = window_sec // stride_sec - 1 if gap is None else gap
    split = time_split(wins, ratios, gap, mode=mode, blocks=blocks, weights=n_per_win if balance_nodes else None)
    scaler = StandardScaler().fit(nodes.loc[nodes["win"].isin(split["train"]), FEATURES].to_numpy(np.float64))
    keep = np.concatenate(list(split.values()))
    graphs = to_pyg(nodes[nodes["win"].isin(keep)], edges[edges["win"].isin(keep)], scaler, stride_sec)
    return graphs, split, scaler, nodes


def fingerprint(graphs):
    """SHA-256 của toàn bộ đồ thị (x, y, edge_index, edge_attr, ips, win); notebook in 16 ký tự đầu."""
    h = hashlib.sha256()
    for g in graphs:
        for t in (g.x, g.y, g.edge_index, g.edge_attr):
            h.update(t.numpy().tobytes())
        h.update(("|".join(g.ips) + str(g.win)).encode())
    return h.hexdigest()


def load_clean(name, path):
    """Adapter + clean_flows + encode_labels cho bộ `name` trong ADAPTERS. Trả về (flows, classes, log)."""
    flows, log = clean_flows(ADAPTERS[name](path), ts_is_order=TS_IS_ORDER.get(name, False))
    classes = encode_labels(flows)
    return flows, classes, log
