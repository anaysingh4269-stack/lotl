"""
Cross-Modal Causal Graph Drift Detection for LotL Attacks -- core model.
=========================================================================
Dataset-agnostic. Everything here operates on the HostGraph interface
defined in schema.py. Swap the data source (data_synthetic.py demo
generator vs data_lanl.py real-data loader vs your own OpTC/DARPA-TC
loader) and this file runs unchanged.

  1. Cross-modal provenance graph construction (process/network/filesystem fusion)
  2. Self-supervised contrastive pretraining of role-conditioned benign behavior
  3. Weisfeiler-Lehman structural drift detection with EVT-calibrated thresholds
  4. Counterfactual explanation via greedy constrained graph-edit search
"""

import hashlib
import math
import random
from collections import Counter

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import genpareto
from sklearn.cluster import KMeans

from schema import HostGraph, ROLES, PROC_DIM, NET_DIM, FS_DIM

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# 1. Cross-modal graph encoder (process nodes only; leaves stay categorical)
# ---------------------------------------------------------------------------
class CrossModalGraphEncoder(nn.Module):
    """phi(v) = Attn(phi_p(v), phi_n(v), phi_f(v)) fused per-process-node
    embedding. Leaf entities (files/registry/sockets) are kept as their own
    discrete categorical labels for the WL kernel stage -- they are already
    low-dimensional/categorical in real telemetry.
    """

    def __init__(self, hidden=16, out_dim=32, heads=2):
        super().__init__()
        self.mlp_p = nn.Sequential(nn.Linear(PROC_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.mlp_n = nn.Sequential(nn.Linear(NET_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.mlp_f = nn.Sequential(nn.Linear(FS_DIM, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, batch_first=True)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x_p, x_n, x_f):
        tok_p = self.mlp_p(x_p).unsqueeze(1)
        tok_n = self.mlp_n(x_n).unsqueeze(1)
        tok_f = self.mlp_f(x_f).unsqueeze(1)
        tokens = torch.cat([tok_p, tok_n, tok_f], dim=1)
        fused, _ = self.attn(tokens, tokens, tokens)
        fused = fused.mean(dim=1)
        return self.proj(fused)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=32, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.net(x)


def process_only_view(hg: HostGraph) -> HostGraph:
    """Process-only-graph ablation: drop file/registry/socket leaf nodes
    entirely, keeping only process nodes and process_create edges."""
    g2 = nx.DiGraph()
    proc_nodes = [n for n, d in hg.graph.nodes(data=True) if d["ntype"] == "process"]
    for n in proc_nodes:
        g2.add_node(n, **hg.graph.nodes[n])
    for u, v, d in hg.graph.edges(data=True):
        if u in g2 and v in g2:
            g2.add_edge(u, v, **d)
    kept_malicious = [e for e in hg.malicious_edges if e[0] in g2 and e[1] in g2]
    return HostGraph(graph=g2, role=hg.role, is_attack=hg.is_attack,
                      malicious_edges=kept_malicious, source=hg.source)


def graph_to_tensors(hg: HostGraph, mask=(True, True, True)):
    proc_nodes = [n for n, d in hg.graph.nodes(data=True) if d["ntype"] == "process"]
    xp = torch.tensor(np.stack([hg.graph.nodes[n]["x_p"] for n in proc_nodes]))
    xn = torch.tensor(np.stack([hg.graph.nodes[n]["x_n"] for n in proc_nodes]))
    xf = torch.tensor(np.stack([hg.graph.nodes[n]["x_f"] for n in proc_nodes]))
    if not mask[1]:
        xn = torch.zeros_like(xn)
    if not mask[2]:
        xf = torch.zeros_like(xf)
    return proc_nodes, xp, xn, xf


def embed_graph(encoder, hg: HostGraph, mask=(True, True, True)):
    proc_nodes, xp, xn, xf = graph_to_tensors(hg, mask)
    node_emb = encoder(xp, xn, xf)
    return proc_nodes, node_emb


def graph_pool(node_emb):
    return node_emb.mean(dim=0, keepdim=True)


# ---------------------------------------------------------------------------
# 2. Self-supervised contrastive pretraining (role-conditioned, InfoNCE)
# ---------------------------------------------------------------------------
def nt_xent_loss(z, roles, temperature=0.2):
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature
    B = z.size(0)
    mask_self = torch.eye(B, dtype=torch.bool)
    sim.masked_fill_(mask_self, -1e9)
    roles_t = np.array(roles)
    pos_mask = torch.tensor(roles_t[:, None] == roles_t[None, :]) & ~mask_self
    log_prob = F.log_softmax(sim, dim=1)
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    loss = -(log_prob * pos_mask).sum(dim=1) / pos_count
    return loss.mean()


def pretrain_encoder(benign_graphs, epochs=60, batch_size=18, lr=1e-3, mask=(True, True, True)):
    encoder = CrossModalGraphEncoder()
    head = ProjectionHead()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)
    n = len(benign_graphs)
    batch_size = min(batch_size, n)
    for ep in range(epochs):
        idx = np.random.permutation(n)[:batch_size]
        batch = [benign_graphs[i] for i in idx]
        embs, roles = [], []
        for hg in batch:
            _, node_emb = embed_graph(encoder, hg, mask)
            embs.append(graph_pool(node_emb))
            roles.append(hg.role)
        z = head(torch.cat(embs, dim=0))
        loss = nt_xent_loss(z, roles)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return encoder


# ---------------------------------------------------------------------------
# 3. Structural symbolizer (discretization bridge for the WL kernel)
# ---------------------------------------------------------------------------
class StructuralSymbolizer:
    """Bridges continuous fused embeddings to discrete WL labels via k-means.
    Simplified stand-in -- k and fitting scheme deserve their own ablation
    at OpTC scale (see README limitations)."""

    def __init__(self, k=8):
        self.k = k
        self.km = KMeans(n_clusters=k, n_init=10, random_state=SEED)

    def fit(self, encoder, benign_graphs, mask=(True, True, True)):
        embs = []
        for hg in benign_graphs:
            _, node_emb = embed_graph(encoder, hg, mask)
            embs.append(node_emb.detach().numpy())
        self.km.fit(np.concatenate(embs, axis=0))
        return self

    def label_graph(self, encoder, hg: HostGraph, mask=(True, True, True)):
        proc_nodes, node_emb = embed_graph(encoder, hg, mask)
        clusters = self.km.predict(node_emb.detach().numpy())
        labels = {n: f"P:{c}" for n, c in zip(proc_nodes, clusters)}
        for n, d in hg.graph.nodes(data=True):
            if d["ntype"] != "process":
                labels[n] = d["wl_label"]
        return labels


# ---------------------------------------------------------------------------
# 4. Weisfeiler-Lehman structural drift detector
# ---------------------------------------------------------------------------
def wl_relabel(graph: nx.DiGraph, initial_labels: dict, iterations=2):
    labels = dict(initial_labels)
    all_iters = [dict(labels)]
    for _ in range(iterations):
        new_labels = {}
        for node in graph.nodes():
            neigh = sorted(labels[u] for u in graph.predecessors(node)) + \
                    sorted(labels[u] for u in graph.successors(node))
            sig = labels[node] + "|" + ",".join(neigh)
            new_labels[node] = hashlib.md5(sig.encode()).hexdigest()[:10]
        labels = new_labels
        all_iters.append(dict(labels))
    return all_iters


def wl_histogram(all_iters):
    hist = Counter()
    for it_labels in all_iters:
        hist.update(it_labels.values())
    return hist


def _kernel_dot(h1: Counter, h2: Counter):
    if len(h1) > len(h2):
        h1, h2 = h2, h1
    return sum(v * h2.get(k, 0) for k, v in h1.items())


def wl_distance(hist_g: Counter, hist_ref: Counter):
    k_gr = _kernel_dot(hist_g, hist_ref)
    k_gg = _kernel_dot(hist_g, hist_g)
    k_rr = _kernel_dot(hist_ref, hist_ref)
    if k_gg == 0 or k_rr == 0:
        return 1.0
    sim = k_gr / math.sqrt(k_gg * k_rr)
    return float(np.clip(1.0 - sim, 0.0, 1.0))


class WLDriftDetector:
    def __init__(self, iterations=2):
        self.iterations = iterations
        self.role_refs = {}
        self.role_thresholds = {}

    def build_reference(self, encoder, symbolizer, benign_ref_graphs, mask=(True, True, True)):
        by_role = {r: [] for r in ROLES}
        for hg in benign_ref_graphs:
            labels = symbolizer.label_graph(encoder, hg, mask)
            hist = wl_histogram(wl_relabel(hg.graph, labels, self.iterations))
            by_role.setdefault(hg.role, []).append(hist)
        for role, hists in by_role.items():
            if not hists:
                continue
            mean_hist = Counter()
            for h in hists:
                mean_hist.update(h)
            for k in mean_hist:
                mean_hist[k] /= len(hists)
            self.role_refs[role] = mean_hist

    def score(self, encoder, symbolizer, hg: HostGraph, mask=(True, True, True)):
        labels = symbolizer.label_graph(encoder, hg, mask)
        hist = wl_histogram(wl_relabel(hg.graph, labels, self.iterations))
        return wl_distance(hist, self.role_refs[hg.role])

    def calibrate_thresholds(self, encoder, symbolizer, benign_calib_graphs, fpr_target=0.03,
                              tail_frac=0.3, mask=(True, True, True)):
        by_role = {r: [] for r in ROLES}
        for hg in benign_calib_graphs:
            by_role.setdefault(hg.role, []).append(self.score(encoder, symbolizer, hg, mask))
        for role, scores in by_role.items():
            if scores:
                self.role_thresholds[role] = self._evt_threshold(scores, fpr_target, tail_frac)

    @staticmethod
    def _evt_threshold(scores, q, tail_frac):
        scores = np.array(scores)
        n = len(scores)
        u = np.quantile(scores, 1 - tail_frac)
        exceed = scores[scores > u] - u
        if len(exceed) < 8:
            return float(np.quantile(scores, 1 - q))
        shape, _, scale = genpareto.fit(exceed, floc=0)
        Nu = len(exceed)
        if abs(shape) < 1e-6:
            thr = u - scale * math.log(q * n / Nu)
        else:
            thr = u + (scale / shape) * (((Nu / n) / q) ** shape - 1)
        return float(thr)

    def is_drift(self, encoder, symbolizer, hg: HostGraph, mask=(True, True, True)):
        s = self.score(encoder, symbolizer, hg, mask)
        return s > self.role_thresholds[hg.role], s


# ---------------------------------------------------------------------------
# 5. Counterfactual explanation via greedy constrained graph-edit search
# ---------------------------------------------------------------------------
def generate_counterfactual(hg: HostGraph, labels: dict, ref_hist: Counter, threshold: float,
                             iterations=2, max_edits=6):
    g = hg.graph.copy()
    cur_hist = wl_histogram(wl_relabel(g, labels, iterations))
    cur_score = wl_distance(cur_hist, ref_hist)
    edits = []
    for _ in range(max_edits):
        if cur_score <= threshold:
            break
        best = None
        for e in list(g.edges()):
            g_try = g.copy()
            g_try.remove_edge(*e)
            local_labels = {n: labels[n] for n in g_try.nodes()}
            h = wl_histogram(wl_relabel(g_try, local_labels, iterations))
            s = wl_distance(h, ref_hist)
            if best is None or s < best[0]:
                best = (s, e)
        if best is None or best[0] >= cur_score:
            break
        cur_score, edge = best
        g.remove_edge(*edge)
        edits.append(edge)
    return edits, cur_score


def describe_edit(hg: HostGraph, edge):
    u, v = edge
    du, dv = hg.graph.nodes[u], hg.graph.nodes[v]
    etype = hg.graph.edges[u, v].get("etype", "?") if hg.graph.has_edge(u, v) else "?"
    return (f"Removed: {du.get('desc', u)}  --[{etype}]-->  {dv.get('desc', v)}"
            f"  (not present in typical {hg.role} behavior)")


# ---------------------------------------------------------------------------
# 6. Evaluation metrics
# ---------------------------------------------------------------------------
def precision_recall_f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def ttd_proxy(hg: HostGraph):
    """Time-to-detect proxy: position of first malicious edge in edge
    ordering, as a fraction of total edges (lower = earlier/better).
    Static single-snapshot proxy -- see README limitations."""
    edges = list(hg.graph.edges())
    if not hg.malicious_edges:
        return None
    present = [e for e in hg.malicious_edges if e in edges]
    if not present:
        return None
    first = min(edges.index(e) for e in present)
    return first / max(len(edges) - 1, 1)
