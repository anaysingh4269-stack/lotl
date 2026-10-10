"""
All result plots for the LotL drift-detection pipeline. Every figure is
saved as a PNG under outputs/ and titled with the data source ("SYNTHETIC
DEMO DATA" vs "LANL") so nobody can mistake a demo-data plot for a
real-data result.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def _tag(source_label):
    return "SYNTHETIC DEMO DATA" if source_label == "synthetic" else source_label.upper()


def plot_variant_comparison(results, source_label, fname="variant_comparison.png"):
    """results: dict[variant_name] -> (precision, recall, f1)"""
    variants = list(results.keys())
    precisions = [results[v][0] for v in variants]
    recalls = [results[v][1] for v in variants]
    f1s = [results[v][2] for v in variants]

    x = np.arange(len(variants))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precisions, width, label="Precision")
    ax.bar(x, recalls, width, label="Recall")
    ax.bar(x + width, f1s, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Detection performance by variant [{_tag(source_label)}]")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(tp, fp, fn, tn, source_label, variant_name="Full model", fname="confusion_matrix.png"):
    mat = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred benign", "Pred attack"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True benign", "True attack"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=14)
    ax.set_title(f"{variant_name} confusion matrix [{_tag(source_label)}]")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_drift_score_distribution(benign_scores, attack_scores, thresholds, source_label,
                                   fname="drift_score_distribution.png"):
    """thresholds: dict[role] -> tau, drawn as vertical lines if roles are known;
    pass a single float or None to skip threshold lines."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(benign_scores, bins=bins, alpha=0.6, label=f"Benign (n={len(benign_scores)})", color="#4C78A8")
    ax.hist(attack_scores, bins=bins, alpha=0.6, label=f"Attack (n={len(attack_scores)})", color="#E45756")
    if isinstance(thresholds, dict):
        for role, t in thresholds.items():
            ax.axvline(t, linestyle="--", linewidth=1, color="gray")
        if thresholds:
            ax.axvline(list(thresholds.values())[0], linestyle="--", linewidth=1, color="gray",
                       label="role threshold(s)")
    elif thresholds is not None:
        ax.axvline(thresholds, linestyle="--", color="gray", label="threshold")
    ax.set_xlabel("WL drift score (0=identical to role reference, 1=maximally different)")
    ax.set_ylabel("Count")
    ax.set_title(f"Drift score separation: benign vs attack [{_tag(source_label)}]")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_pr(benign_scores, attack_scores, source_label, fname_prefix="roc_pr"):
    """Manual ROC/PR sweep over the drift score threshold (no sklearn metrics
    dependency beyond what's already required)."""
    all_scores = np.array(benign_scores + attack_scores)
    labels = np.array([0] * len(benign_scores) + [1] * len(attack_scores))
    thresholds = np.unique(all_scores)
    thresholds = np.concatenate([[0.0], thresholds, [1.0]])
    tprs, fprs, precisions, recalls = [], [], [], []
    P = labels.sum(); N = (labels == 0).sum()
    for t in thresholds:
        pred = (all_scores > t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tprs.append(tp / P if P else 0.0)
        fprs.append(fp / N if N else 0.0)
        precisions.append(tp / (tp + fp) if (tp + fp) else 1.0)
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)

    order = np.argsort(fprs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(np.array(fprs)[order], np.array(tprs)[order], marker=".", markersize=3)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC (threshold sweep)")

    order2 = np.argsort(recalls)
    axes[1].plot(np.array(recalls)[order2], np.array(precisions)[order2], marker=".", markersize=3)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall (threshold sweep)")
    fig.suptitle(f"Threshold sweep, drift score as decision statistic [{_tag(source_label)}]")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"{fname_prefix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ttd_distribution(ttds, source_label, fname="ttd_distribution.png"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if ttds:
        ax.hist(ttds, bins=15, color="#59A14F")
        ax.axvline(np.mean(ttds), color="black", linestyle="--", label=f"mean={np.mean(ttds):.3f}")
        ax.legend()
    ax.set_xlabel("Time-to-detect proxy (0=first edge, 1=last edge; lower=earlier)")
    ax.set_ylabel("Count of detected attacks")
    ax.set_title(f"Detection latency proxy on true positives [{_tag(source_label)}]")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sample_attack_graph(hg, drift_score, threshold, source_label, fname="sample_attack_graph.png"):
    """Draws one host provenance graph, highlighting malicious edges in red."""
    g = hg.graph
    fig, ax = plt.subplots(figsize=(9, 6))
    pos = nx.spring_layout(g, seed=7, k=1.2)
    node_colors = []
    for n, d in g.nodes(data=True):
        if d["ntype"] == "process":
            node_colors.append("#4C78A8")
        elif d["ntype"] == "socket":
            node_colors.append("#E45756")
        elif d["ntype"] == "file":
            node_colors.append("#F1CE63")
        else:
            node_colors.append("#B279A2")
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=500, ax=ax)
    normal_edges = [e for e in g.edges() if e not in set(hg.malicious_edges)]
    nx.draw_networkx_edges(g, pos, edgelist=normal_edges, ax=ax, edge_color="#999999", arrows=True)
    if hg.malicious_edges:
        present = [e for e in hg.malicious_edges if g.has_edge(*e)]
        nx.draw_networkx_edges(g, pos, edgelist=present, ax=ax, edge_color="#CC0000", width=2.5, arrows=True)
    labels = {n: d.get("desc", n)[:22] for n, d in g.nodes(data=True)}
    nx.draw_networkx_labels(g, pos, labels, font_size=7, ax=ax)
    ax.set_title(f"Detected host graph (role={hg.role}) -- score {drift_score:.3f} > tau {threshold:.3f}\n"
                 f"red edges = ground-truth malicious [{_tag(source_label)}]")
    ax.axis("off")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
