"""
Shared data interface. Every data source (synthetic demo generator, real
LANL loader, or your own loader for DARPA TC/OpTC) must produce a list of
HostGraph objects using this exact interface -- everything downstream
(encoder, contrastive pretraining, WL drift detector, counterfactual
generator, ablations, plotting) is written against this interface only and
does not care where the graph came from.

HostGraph.graph : networkx.DiGraph
    Nodes:
      ntype == "process"  -> must carry x_p (8,), x_n (6,), x_f (5,) float32
                              numpy arrays (process/network/filesystem
                              feature views of that process node) and a
                              human-readable "desc".
      ntype in {"file", "registry", "socket"} (leaf/entity nodes)
                           -> must carry a discrete "wl_label" string
                              (e.g. "F:lsass", "S:external_lowrep") and
                              a human-readable "desc".
    Edges: etype in {"process_create", "file_write", "registry_write",
                     "network_connect"} (extend as your real schema needs).

HostGraph.role            : str, one of ROLES (extend ROLES if your
                             environment has more host roles).
HostGraph.is_attack        : bool, ground-truth label for this graph/window.
HostGraph.malicious_edges  : list[(u, v)] edges tagged malicious, ONLY
                             used for the TTD proxy metric and the
                             counterfactual-explanation demo -- never fed
                             to the detector itself.
HostGraph.source           : "synthetic" | "lanl" | your own tag -- used so
                             every plot/report can clearly label which data
                             produced which number.
"""

from dataclasses import dataclass, field
import networkx as nx

ROLES = ["workstation", "db_server", "domain_admin"]
PROC_DIM, NET_DIM, FS_DIM = 8, 6, 5


@dataclass
class HostGraph:
    graph: nx.DiGraph
    role: str
    is_attack: bool
    malicious_edges: list = field(default_factory=list)
    source: str = "unknown"
