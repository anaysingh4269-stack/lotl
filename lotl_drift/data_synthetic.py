"""
SYNTHETIC demo data generator -- fallback/demo only.
=====================================================
This module is kept ONLY so the pipeline has something to run out of the box
before you point it at real data. It is NOT the dataset the main deliverable
is built around. See data_lanl.py + README.md ("Getting real data") for the
real, publicly downloadable dataset (LANL Comprehensive Multi-Source
Cyber-Security Events, Kent 2015) this pipeline is designed to consume.

Every result produced from this module is labeled "SYNTHETIC" wherever it is
printed or plotted -- never presented as a real-data result.
"""

import random
import numpy as np
import networkx as nx

from schema import HostGraph, ROLES, PROC_DIM, NET_DIM, FS_DIM

SEED = 7
random.seed(SEED)
np.random.seed(SEED)


def _proc_feat(signed, path_rep, entropy, is_lolbin, depth, children, office, scripthost):
    return np.array([signed, path_rep, entropy, is_lolbin, depth, children, office, scripthost], dtype=np.float32)


def _net_feat(has_net, dest_rep=0.0, byte_vol=0.0, beacon=0.0, port_common=0.0, external=0.0):
    return np.array([has_net, dest_rep, byte_vol, beacon, port_common, external], dtype=np.float32)


def _fs_feat(cred=0.0, autorun=0.0, lsass=0.0, writes=0.0, sensitive_paths=0.0):
    return np.array([cred, autorun, lsass, writes, sensitive_paths], dtype=np.float32)


class SyntheticProvenanceGenerator:
    """Generates per-host, per-role benign graphs and LotL attack graphs.
    See module docstring: demo/fallback data only.
    """

    def __init__(self, rng=None):
        self.rng = rng or random.Random(SEED)
        self._uid = 0

    def _u(self):
        self._uid += 1
        return self._uid

    @staticmethod
    def _jit(val, sigma=0.05):
        return float(np.clip(val + np.random.normal(0, sigma), 0.0, 1.0))

    def _add_process(self, g, pid, **kwargs):
        g.add_node(pid, ntype="process", x_p=_proc_feat(**kwargs["proc"]),
                    x_n=kwargs.get("net", _net_feat(0)), x_f=kwargs.get("fs", _fs_feat()),
                    desc=kwargs.get("desc", pid))
        return pid

    def _add_leaf(self, g, nid, ntype, label, desc):
        g.add_node(nid, ntype=ntype, wl_label=label, desc=desc)
        return nid

    def make_benign_graph(self, role):
        g = nx.DiGraph()
        r = self.rng
        j = self._jit
        if role == "workstation":
            office_app = r.choice(["WINWORD.EXE", "EXCEL.EXE", "OUTLOOK.EXE"])
            explorer = self._add_process(g, f"explorer.exe#{self._u()}", proc=dict(
                signed=1, path_rep=j(0.95, 0.03), entropy=j(0.1, 0.05), is_lolbin=0,
                depth=0.0, children=j(0.3, 0.1), office=0, scripthost=0),
                desc="explorer.exe (signed, user shell)")
            office = self._add_process(g, f"{office_app}#{self._u()}", proc=dict(
                signed=1, path_rep=j(0.9, 0.05), entropy=j(0.15, 0.06), is_lolbin=0,
                depth=0.2, children=j(0.1, 0.05), office=1, scripthost=0),
                desc=f"{office_app} (signed office app)")
            g.add_edge(explorer, office, etype="process_create")
            f1 = self._add_leaf(g, f"doc{self._u()}.docx", "file", "F:normal", "user document (normal)")
            g.add_edge(office, f1, etype="file_write")
            if r.random() < 0.3:
                script = self._add_process(g, f"cscript.exe#{self._u()}", proc=dict(
                    signed=1, path_rep=j(0.8, 0.05), entropy=j(0.3, 0.08), is_lolbin=1,
                    depth=0.3, children=j(0.05, 0.03), office=0, scripthost=1),
                    fs=_fs_feat(writes=j(0.15, 0.05)),
                    desc="cscript.exe (signed macro helper, local script, no network)")
                g.add_edge(office, script, etype="process_create")
                f2 = self._add_leaf(g, f"tmp{self._u()}.tmp", "file", "F:normal", "temp file (normal)")
                g.add_edge(script, f2, etype="file_write")
        elif role == "db_server":
            sql = self._add_process(g, f"sqlservr.exe#{self._u()}", proc=dict(
                signed=1, path_rep=j(0.97, 0.02), entropy=j(0.05, 0.03), is_lolbin=0,
                depth=0.0, children=j(0.05, 0.03), office=0, scripthost=0),
                fs=_fs_feat(writes=j(0.6, 0.1), sensitive_paths=j(0.1, 0.05)),
                desc="sqlservr.exe (signed DB engine)")
            f1 = self._add_leaf(g, f"data{self._u()}.mdf", "file", "F:normal", "database data file (normal)")
            g.add_edge(sql, f1, etype="file_write")
            if r.random() < 0.6:
                bkup = self._add_process(g, f"sqlagent.exe#{self._u()}", proc=dict(
                    signed=1, path_rep=j(0.95, 0.03), entropy=j(0.05, 0.03), is_lolbin=0,
                    depth=0.2, children=j(0.05, 0.02), office=0, scripthost=0),
                    fs=_fs_feat(writes=j(0.4, 0.1)), desc="sqlagent.exe (signed backup job)")
                g.add_edge(sql, bkup, etype="process_create")
                f2 = self._add_leaf(g, f"backup{self._u()}.bak", "file", "F:normal", "backup file (normal)")
                g.add_edge(bkup, f2, etype="file_write")
        elif role == "domain_admin":
            admin_tool = r.choice(["powershell.exe", "wmic.exe", "net.exe"])
            ps = self._add_process(g, f"{admin_tool}#{self._u()}", proc=dict(
                signed=1, path_rep=j(0.85, 0.06), entropy=j(0.35, 0.08), is_lolbin=1,
                depth=0.1, children=j(0.1, 0.05), office=0, scripthost=1),
                net=_net_feat(has_net=1 if r.random() < 0.7 else 0,
                              dest_rep=j(0.85, 0.08), byte_vol=j(0.2, 0.08),
                              beacon=j(0.05, 0.04), port_common=1, external=0),
                fs=_fs_feat(autorun=0.0, writes=j(0.2, 0.08)),
                desc=f"{admin_tool} (signed, admin script, internal mgmt endpoint)")
            reg = self._add_leaf(g, f"HKLM\\Software\\Policies#{self._u()}", "registry", "R:normal",
                                  "GPO registry key (normal admin edit)")
            g.add_edge(ps, reg, etype="registry_write")
            sock = self._add_leaf(g, f"10.0.0.{r.randint(2,60)}:5985#{self._u()}", "socket",
                                   "S:internal_highrep", "WinRM to internal management host (normal)")
            g.add_edge(ps, sock, etype="network_connect")
        else:
            raise ValueError(role)
        return HostGraph(graph=g, role=role, is_attack=False, source="synthetic")

    def make_attack_graph(self, role):
        base = self.make_benign_graph(role)
        g = base.graph
        r = self.rng
        j = self._jit

        office = self._add_process(g, f"OUTLOOK.EXE#{self._u()}", proc=dict(
            signed=1, path_rep=j(0.9, 0.05), entropy=j(0.1, 0.05), is_lolbin=0, depth=0.0,
            children=j(0.2, 0.06), office=1, scripthost=0), desc="OUTLOOK.EXE (signed, opens attachment)")
        wmi = self._add_process(g, f"wmiprvse.exe#{self._u()}", proc=dict(
            signed=1, path_rep=j(0.7, 0.06), entropy=j(0.2, 0.06), is_lolbin=1, depth=0.1,
            children=j(0.3, 0.08), office=0, scripthost=0), desc="wmiprvse.exe (WMI provider host, T1047)")
        g.add_edge(office, wmi, etype="process_create")

        lolbin = r.choice(["powershell.exe", "rundll32.exe", "certutil.exe"])
        ps = self._add_process(g, f"{lolbin}#{self._u()}", proc=dict(
            signed=1, path_rep=j(0.88, 0.06), entropy=j(0.55, 0.08), is_lolbin=1, depth=0.2,
            children=j(0.05, 0.03), office=0, scripthost=1),
            net=_net_feat(has_net=1, dest_rep=j(0.15, 0.07), byte_vol=j(0.6, 0.1),
                          beacon=j(0.8, 0.08), port_common=0, external=1),
            fs=_fs_feat(lsass=1.0, writes=j(0.3, 0.08)),
            desc=f"{lolbin} (signed system binary, T1059 -- legitimate-looking process lineage)")
        e1 = (wmi, ps)
        g.add_edge(*e1, etype="process_create")

        lsass_file = self._add_leaf(g, f"lsass{self._u()}.dmp", "file", "F:lsass",
                                     "LSASS memory dump (credential access)")
        e2 = (ps, lsass_file)
        g.add_edge(*e2, etype="file_write")

        c2_sock = self._add_leaf(g, f"203.0.113.{r.randint(2,250)}:443", "socket", "S:external_lowrep",
                                  "beaconing egress to low-reputation external host (C2)")
        e3 = (ps, c2_sock)
        g.add_edge(*e3, etype="network_connect")

        base.is_attack = True
        base.malicious_edges = [e1, e2, e3]
        return base

    def _base_with_feature(self, role, desc_substr, max_tries=50):
        for _ in range(max_tries):
            hg = self.make_benign_graph(role)
            if any(desc_substr in d.get("desc", "") for _, d in hg.graph.nodes(data=True)):
                return hg
        raise RuntimeError(f"could not sample benign graph with '{desc_substr}' for role {role}")

    def make_stealthy_attack_graph(self, role):
        j = self._jit
        if role == "domain_admin":
            hg = self._base_with_feature(role, "admin script")
            g = hg.graph
            ps = next(n for n, d in g.nodes(data=True)
                      if d["ntype"] == "process" and "admin script" in d.get("desc", ""))
            g.nodes[ps]["x_n"] = _net_feat(has_net=1, dest_rep=j(0.12, 0.05), byte_vol=j(0.65, 0.08),
                                            beacon=j(0.82, 0.06), port_common=0, external=1)
            g.nodes[ps]["x_f"] = _fs_feat(lsass=1.0, writes=j(0.3, 0.08))
            g.nodes[ps]["desc"] += " [same lineage as routine admin use -- exfiltrates via existing egress edge]"
            sock = next(v for u, v, d in g.out_edges(ps, data=True) if d.get("etype") == "network_connect")
            g.nodes[sock].update(ntype="socket", wl_label="S:external_lowrep",
                                  desc="egress disguised as mgmt traffic -> low-reputation external host (C2)")
            malicious_edges = [(ps, sock)]
        elif role == "workstation":
            hg = self._base_with_feature(role, "cscript.exe")
            g = hg.graph
            script = next(n for n, d in g.nodes(data=True)
                          if d["ntype"] == "process" and "cscript.exe" in d.get("desc", ""))
            g.nodes[script]["x_f"] = _fs_feat(lsass=1.0, writes=j(0.25, 0.06))
            g.nodes[script]["desc"] += " [STEALTHY: same macro-helper lineage, dumps credentials instead of temp file]"
            tmp = next(v for u, v, d in g.out_edges(script, data=True) if d.get("etype") == "file_write")
            g.nodes[tmp].update(ntype="file", wl_label="F:lsass",
                                 desc="credential/LSASS-sensitive data written via legitimate macro helper")
            malicious_edges = [(script, tmp)]
        elif role == "db_server":
            hg = self._base_with_feature(role, "sqlagent.exe")
            g = hg.graph
            bkup = next(n for n, d in g.nodes(data=True)
                        if d["ntype"] == "process" and "sqlagent.exe" in d.get("desc", ""))
            g.nodes[bkup]["x_f"] = _fs_feat(cred=1.0, writes=j(0.4, 0.08))
            g.nodes[bkup]["desc"] += " [STEALTHY: same backup-job lineage, targets credential store instead of .bak]"
            bak = next(v for u, v, d in g.out_edges(bkup, data=True) if d.get("etype") == "file_write")
            g.nodes[bak].update(ntype="file", wl_label="F:credential",
                                 desc="credential-store file disguised as backup output")
            malicious_edges = [(bkup, bak)]
        else:
            raise ValueError(role)
        hg.is_attack = True
        hg.malicious_edges = malicious_edges
        return hg

    def sample_dataset(self, n_benign_per_role=40, n_attack_per_role=12, stealthy_frac=0.5):
        benign, attack = [], []
        n_stealthy = int(round(n_attack_per_role * stealthy_frac))
        n_noisy = n_attack_per_role - n_stealthy
        for role in ROLES:
            for _ in range(n_benign_per_role):
                benign.append(self.make_benign_graph(role))
            for _ in range(n_noisy):
                attack.append(self.make_attack_graph(role))
            for _ in range(n_stealthy):
                attack.append(self.make_stealthy_attack_graph(role))
        return benign, attack
