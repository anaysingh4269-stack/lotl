# Cross-Modal Causal Graph Drift Detection for LotL Attacks

Detects "Living off the Land" (LotL) attacks — intrusions that use signed,
legitimate system binaries so no single action looks malicious — by fusing
process/network/filesystem telemetry per host, learning role-conditioned
"normal" behavior contrastively, and flagging structural drift with a
Weisfeiler-Lehman graph kernel + extreme-value-calibrated threshold. Includes
a counterfactual explainer (what would need to change for this host to look
normal again) and two ablations (no role-conditioning, no cross-modal
fusion).

## Quick start (VS Code)

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Open the folder in VS Code, press **F5**, pick **"Run demo (synthetic, no
download needed)"**. Or from a terminal:

```
python run_experiment.py --data-source synthetic
```

This runs end-to-end on generated demo data (see "About the data" below)
and writes plots + `metrics.json` to `outputs/`.

## Running on real data

```
python run_experiment.py --data-source lanl --data-dir data/raw
```

See `data/raw/GET_THE_DATA.md` for exactly what to download and where to
put it. **This was built in a sandboxed environment with no internet
access, so it could not download the dataset or run this path itself** —
run it yourself once the files are in place. `data_lanl.py`'s docstring
documents precisely which features are real signal from the dataset and
which are heuristic approximations for information the dataset doesn't
capture — read it before citing numbers.

## About the data (read this before trusting any number)

| Mode | What it is | Ground truth | Use for |
|---|---|---|---|
| `synthetic` (default) | Hand-built generator, deliberately overlapping benign/attack feature distributions | Synthetic, by construction | Sanity-checking the pipeline runs, understanding the method. **Not evidence about real-world performance.** Every plot/number from this mode is labeled "SYNTHETIC DEMO DATA". |
| `lanl` | Real 58-day internal network capture, LANL (Kent 2015), real red-team compromise labels | Real | The actual result you'd cite. Requires you to download it (see above) — file sizes make it impractical to bundle. |

LANL has no filesystem/registry telemetry, so the credential/LSASS-touch
signal that makes the "stealthy" attack scenario interesting is unavailable
there (always zero) — this is a real coverage gap in that dataset for this
model's fusion story, not a bug. If you specifically need that signal,
DARPA TC/OpTC (real host-provenance data with file/registry events) is the
closer match; you'd write a loader against it the same way `data_lanl.py`
is written against LANL — same `HostGraph` interface in `schema.py`, rest
of the pipeline unchanged.

## Project layout

```
schema.py            HostGraph interface every data source must produce
data_synthetic.py     Demo/fallback generator (labeled everywhere it's used)
data_lanl.py          Real-data loader for the LANL dataset
core_model.py         Encoder, contrastive pretraining, WL drift detector,
                       counterfactual explainer, metrics -- dataset-agnostic
visualize.py           All plotting functions
run_experiment.py     Main entry point: loads data, trains, evaluates,
                       runs both ablations, saves plots + metrics.json
outputs/              Generated PNGs + metrics.json land here
data/raw/             Put downloaded LANL files here
```

## What gets plotted (`outputs/`)

- `variant_comparison.png` — precision/recall/F1 bar chart, full model vs
  both ablations
- `confusion_matrix.png` — TP/FP/FN/TN for the full model
- `drift_score_distribution.png` — benign vs attack WL drift score
  histograms with the calibrated threshold marked
- `roc_pr.png` — ROC and precision-recall curves swept over the drift
  score
- `ttd_distribution.png` — time-to-detect proxy distribution on true
  positives
- `sample_attack_graph.png` — one detected host's provenance graph with
  the ground-truth malicious edges highlighted in red

## Known limitations (be upfront about these if you write this up)

1. Discretization bridge (k-means) for the WL kernel is a simplified stand-in.
2. Counterfactual generation is edge-deletion only, not full graph-edit
   distance with node/attribute substitution.
3. Time-to-detect is a static position-in-graph proxy, not a real sliding-
   window wall-clock measurement.
4. Reference/threshold construction uses a mean histogram (kernel mean
   embedding), a standard but simplified approximation of a full multiset
   kernel.
5. The LANL loader's role labels (`workstation`/`db_server`/`domain_admin`)
   and several feature values are heuristic proxies, documented field-by-
   field in `data_lanl.py` — not a real asset inventory.
