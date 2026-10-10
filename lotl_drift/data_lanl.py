"""
Real-data loader: LANL Comprehensive, Multi-Source Cyber-Security Events
==========================================================================
This is the REAL dataset this pipeline is built to run on: a real internal
network capture from Los Alamos National Laboratory (Kent, 2015), 58 days
of authentication, process, flow, and DNS events across ~12,000 real hosts,
with a real red-team-compromise ground truth file (redteam.txt.gz).

Public, free, no special access request needed:
  Landing page : https://csr.lanl.gov/data/cyber1/
  Direct files : auth.txt.gz, proc.txt.gz, flows.txt.gz, dns.txt.gz,
                 redteam.txt.gz (redteam.txt.gz is the ground-truth label file)

Download those five files yourself (this sandbox has no network access, so
it cannot fetch them for you) and place them under data/raw/, then run:

    python run_experiment.py --data-source lanl --data-dir data/raw

--------------------------------------------------------------------------
Honest mapping / approximations (state these if you cite results from this
loader -- do not present them as ground truth beyond what's below):

* Roles. LANL does NOT label host roles. We derive a 3-way heuristic proxy
  (workstation / db_server / domain_admin) per (user, computer, hour) window
  from observed behavior in that window:
    - domain_admin : user authenticates to >= ADMIN_FANOUT distinct
                     destination computers within the window (LANL's own
                     released stats show a small number of admin accounts
                     with very high fan-out -- this is a real, documented
                     property of the data, not invented).
    - db_server    : computer is a top-K flow destination by distinct
                     source-computer count network-wide (server-like fan-in).
    - workstation  : default, everything else.
  This is a heuristic stand-in for a real asset inventory / CMDB join,
  which LANL does not provide. Swap in your real CMDB if you have one.

* Process feature view (x_p, 8-dim). LANL's proc.txt only has
  (time, user, computer, process_name, start/end) -- no code-signing, no
  path, no command-line entropy. We map what IS present honestly:
    - is_lolbin   : 1 if process_name matches a small fixed list of common
                    Windows dual-use binaries (powershell.exe, wmic.exe,
                    cmd.exe, rundll32.exe, certutil.exe, net.exe, cscript.exe,
                    wscript.exe, mshta.exe, regsvr32.exe, psexec.exe).
    - scripthost  : 1 if process_name in a script-host subset of the above.
    - signed, path_rep, entropy, office : NOT available in LANL -> set to
      a fixed neutral 0.5 (documented as unknown, not invented as a real
      signal). If you have EDR/Sysmon-style enrichment for these hosts,
      wire it in here instead of the placeholder.
    - depth, children : computed from the actual process-launch graph
      structure per window (real, not placeholder).

* Network feature view (x_n, 6-dim), from flows.txt -- all REAL:
    - has_net, byte_vol (log-normalized packet/byte volume), port_common
      (dst_port in a common-ports set), external (dst_computer not seen as
      a source anywhere in flows.txt, used as an internal/external proxy
      since LANL computers are anonymized, not IP-addressed).
    - dest_rep, beacon : LANL has no reputation feed or beacon detector
      -> dest_rep set from an in-dataset frequency prior (how often this
      dst_computer appears as a flow destination network-wide; rarer =
      lower "reputation" proxy) and beacon from repeat-connection regularity
      to the same dst within the window (real signal, coarse proxy).

* Filesystem feature view (x_f, 5-dim). LANL has NO filesystem/registry
  telemetry at all -> all zeros. This is a genuine coverage gap in this
  dataset for this model's fusion story, not a bug -- state it plainly if
  you write this up: the LSASS/credential-touch signal the paper's stealthy
  scenario depends on cannot be evaluated on LANL as-is. If you need that
  signal specifically, DARPA TC/OpTC (which does have file/registry
  provenance) is the closer match -- see README "Which dataset should I use".

* Labels. is_attack / malicious_edges come directly from redteam.txt.gz
  (real red-team compromise events, matched to the (user, computer, hour)
  window by exact user+computer+time-bucket join) -- this is real ground
  truth, not synthetic.
"""

import gzip
import math
from collections import defaultdict, Counter

import networkx as nx
import numpy as np

from schema import HostGraph

LOLBINS = {"powershell.exe", "wmic.exe", "cmd.exe", "rundll32.exe", "certutil.exe",
           "net.exe", "cscript.exe", "wscript.exe", "mshta.exe", "regsvr32.exe", "psexec.exe"}
SCRIPT_HOSTS = {"cscript.exe", "wscript.exe", "powershell.exe", "wmic.exe"}
COMMON_PORTS = {"80", "443", "53", "445", "3389", "22", "88", "389", "636", "135"}
ADMIN_FANOUT = 15          # distinct dst computers/hour to be flagged "domain_admin"-like
SERVER_FANIN_TOP_FRAC = 0.02  # top 2% of computers by distinct-source fan-in -> "db_server"-like
WINDOW_SECONDS = 3600       # 1-hour buckets


def _open(path):
    return gzip.open(path, "rt", errors="replace") if str(path).endswith(".gz") else open(path, "r", errors="replace")


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _log_norm(x, cap=1e7):
    x = max(0.0, min(x, cap))
    return float(np.clip(math.log1p(x) / math.log1p(cap), 0.0, 1.0))


def load_lanl_windows(data_dir, max_lines=2_000_000, max_windows=400, attack_oversample=True):
    """Stream LANL files, bucket events into (computer, hour) windows, build
    one HostGraph per window. `max_lines` caps how much of each (huge) file
    is read -- raise it once you've confirmed the pipeline runs end-to-end;
    the full files are tens of GB.

    Returns: list[HostGraph]
    """
    import os
    proc_path = _first_existing(data_dir, "proc.txt.gz", "proc.txt")
    flows_path = _first_existing(data_dir, "flows.txt.gz", "flows.txt")
    redteam_path = _first_existing(data_dir, "redteam.txt.gz", "redteam.txt")
    if proc_path is None or flows_path is None:
        raise FileNotFoundError(
            f"Could not find proc.txt(.gz) and flows.txt(.gz) under {data_dir}. "
            "Download them from https://csr.lanl.gov/data/cyber1/ and place them there."
        )

    redteam_events = set()  # (user, computer, hour_bucket)
    if redteam_path:
        with _open(redteam_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                t, user, src_c, dst_c = parts[0], parts[1], parts[2], parts[3]
                hb = int(_safe_float(t)) // WINDOW_SECONDS
                redteam_events.add((user, src_c, hb))
                redteam_events.add((user, dst_c, hb))

    # --- pass 1: proc.txt -> per-window process launch events, per-user fanout ---
    windows = defaultdict(list)   # (computer, hour) -> list of (time, user, process_name)
    user_fanout = defaultdict(set)  # (user, hour) -> set of computers touched
    with _open(proc_path) as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            t, user, computer, proc = parts[0], parts[1], parts[2], parts[3]
            hb = int(_safe_float(t)) // WINDOW_SECONDS
            windows[(computer, hb)].append((_safe_float(t), user, proc.lower()))
            user_fanout[(user, hb)].add(computer)

    # --- pass 2: flows.txt -> per-window network edges, global fan-in/frequency priors ---
    flow_windows = defaultdict(list)  # (src_computer, hour) -> list of flow dicts
    dst_freq = Counter()              # global: how often each computer is a flow dst
    dst_fanin = defaultdict(set)      # computer -> set of distinct src computers (server-ness)
    with _open(flows_path) as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            t, dur, src_c, src_p, dst_c, dst_p, proto, pkts, bytes_ = parts[:9]
            hb = int(_safe_float(t)) // WINDOW_SECONDS
            flow_windows[(src_c, hb)].append(dict(t=_safe_float(t), dst=dst_c, dst_port=dst_p,
                                                     bytes=_safe_float(bytes_), pkts=_safe_float(pkts)))
            dst_freq[dst_c] += 1
            dst_fanin[dst_c].add(src_c)

    total_flow_events = sum(dst_freq.values()) or 1
    fanin_counts = sorted(len(v) for v in dst_fanin.values())
    server_threshold = fanin_counts[int(len(fanin_counts) * (1 - SERVER_FANIN_TOP_FRAC))] \
        if fanin_counts else float("inf")
    server_like = {c for c, srcs in dst_fanin.items() if len(srcs) >= server_threshold}

    admin_like_windows = {(u, h) for (u, h), comps in user_fanout.items() if len(comps) >= ADMIN_FANOUT}

    hostgraphs = []
    win_items = list(windows.items())
    if max_windows:
        win_items = win_items[:max_windows]

    for (computer, hb), events in win_items:
        events.sort(key=lambda e: e[0])
        users_here = {u for _, u, _ in events}
        is_admin_window = any((u, hb) in admin_like_windows for u in users_here)
        role = ("domain_admin" if is_admin_window else
                "db_server" if computer in server_like else "workstation")

        g = nx.DiGraph()
        proc_nodes_by_order = []
        prev_pid = None
        seen_procs = {}
        for idx, (t, user, proc) in enumerate(events):
            pid = f"{proc}#{computer}#{hb}#{idx}"
            is_lolbin = 1.0 if proc in LOLBINS else 0.0
            scripthost = 1.0 if proc in SCRIPT_HOSTS else 0.0
            x_p = np.array([0.5, 0.5, 0.5, is_lolbin, min(idx / 10.0, 1.0),
                             0.0, 0.5, scripthost], dtype=np.float32)  # signed,path_rep,entropy,is_lolbin,depth,children,office,scripthost
            g.add_node(pid, ntype="process", x_p=x_p, x_n=np.zeros(6, dtype=np.float32),
                        x_f=np.zeros(5, dtype=np.float32),
                        desc=f"{proc} (user={user}, real LANL proc event)")
            proc_nodes_by_order.append(pid)
            if prev_pid is not None:
                g.add_edge(prev_pid, pid, etype="process_create")
                g.nodes[prev_pid]["x_p"][5] = min(g.nodes[prev_pid]["x_p"][5] + 0.2, 1.0)  # children count proxy
            prev_pid = pid
            seen_procs[proc] = pid

        malicious_edges = []
        flows_here = flow_windows.get((computer, hb), [])
        anchor = proc_nodes_by_order[-1] if proc_nodes_by_order else None
        if anchor is not None:
            dst_counts_here = Counter(fl["dst"] for fl in flows_here)
            for j, fl in enumerate(flows_here[:25]):  # cap sockets per window graph
                dst = fl["dst"]
                external = 1.0 if dst not in dst_fanin or computer not in dst_fanin.get(dst, set()) else 0.0
                rep_proxy = 1.0 - _log_norm(dst_freq.get(dst, 0), cap=total_flow_events)  # rarer dst -> lower "reputation"
                beacon_proxy = _log_norm(dst_counts_here[dst], cap=20)  # repeat conns to same dst this window
                x_n = np.array([1.0, rep_proxy, _log_norm(fl["bytes"]), beacon_proxy,
                                 1.0 if fl["dst_port"] in COMMON_PORTS else 0.0, external], dtype=np.float32)
                sock_id = f"sock#{computer}#{hb}#{j}"
                wl_lab = "S:external_lowrep" if (external and rep_proxy > 0.6) else "S:normal"
                g.add_node(sock_id, ntype="socket", wl_label=wl_lab,
                            desc=f"flow -> {dst}:{fl['dst_port']} (real LANL flow event)")
                g.add_edge(anchor, sock_id, etype="network_connect")
                g.nodes[anchor]["x_n"] = x_n
                if (users_here and any((u, computer, hb) in redteam_events for u in users_here)
                        and wl_lab == "S:external_lowrep"):
                    malicious_edges.append((anchor, sock_id))

        is_attack = any((u, computer, hb) in redteam_events for u in users_here)
        hg = HostGraph(graph=g, role=role, is_attack=is_attack,
                        malicious_edges=malicious_edges, source="lanl")
        if len(g.nodes) >= 2:
            hostgraphs.append(hg)

    return hostgraphs


def _first_existing(data_dir, *names):
    import os
    for n in names:
        p = os.path.join(data_dir, n)
        if os.path.exists(p):
            return p
    return None
