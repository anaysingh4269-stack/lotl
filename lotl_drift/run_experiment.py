"""
Main entry point. Run in VS Code with F5 (see .vscode/launch.json) or:

    python run_experiment.py --data-source synthetic         # demo, runs out of the box
    python run_experiment.py --data-source lanl --data-dir data/raw   # real data

See README.md for how to get the real (LANL) data -- this sandbox that
generated this project has no network access, so it could not download or
run against the real dataset itself; run it locally once the files are in
data/raw/.
"""

import argparse
import json
import os
import random
import sys

import numpy as np

from schema import ROLES
from core_model import (
    pretrain_encoder, StructuralSymbolizer, WLDriftDetector, process_only_view,
    precision_recall_f1, ttd_proxy, generate_counterfactual, describe_edit,
    wl_histogram, wl_relabel,
)
import visualize

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "outputs", "metrics.json")


def load_data(args):
    if args.data_source == "lanl":
        from data_lanl import load_lanl_windows
        data = load_lanl_windows(args.data_dir, max_windows=args.max_windows)
        if len(data) < 40:
            print(f"[WARN] Only {len(data)} usable host-graphs parsed from {args.data_dir}. "
                  "Check that proc.txt(.gz)/flows.txt(.gz)/redteam.txt(.gz) are present and "
                  "non-empty, or raise --max-windows / --max-lines in data_lanl.py.")
        n_attack = sum(hg.is_attack for hg in data)
        print(f"[LANL] Loaded {len(data)} host-window graphs ({n_attack} labeled attack, "
              f"{len(data) - n_attack} benign) from {args.data_dir}")
        return data, "lanl"
    else:
        from data_synthetic import SyntheticProvenanceGenerator
        print("[SYNTHETIC] --data-source synthetic (default demo/fallback). "
              "For real results, download LANL data -- see README.md -- and pass "
              "--data-source lanl --data-dir data/raw")
        gen = SyntheticProvenanceGenerator()
        benign, attack = gen.sample_dataset(n_benign_per_role=80, n_attack_per_role=25)
        return benign + attack, "synthetic"


def split_data(all_graphs, source):
    benign = [hg for hg in all_graphs if not hg.is_attack]
    attack = [hg for hg in all_graphs if hg.is_attack]
    random.shuffle(benign)
    n = len(benign)
    if n < 10:
        raise RuntimeError(f"Only {n} benign graphs available -- need at least ~10 to split "
                            "into pretrain/reference/calibration/test sets. Increase data volume "
                            "(--max-windows) or check your data source.")
    pretrain_set = benign[: int(0.5 * n)]
    reference_set = benign[int(0.5 * n): int(0.75 * n)] or benign[:1]
    calib_set = benign[int(0.75 * n): int(0.9 * n)] or benign[:1]
    test_benign = benign[int(0.9 * n):] or benign[-max(1, n // 10):]
    return pretrain_set, reference_set, calib_set, test_benign, attack


def eval_variant(encoder, symbolizer, detector, test_benign, test_attack, mask=(True, True, True)):
    tp = fp = fn = tn = 0
    benign_scores, attack_scores, ttds = [], [], []
    for hg in test_benign:
        drift, s = detector.is_drift(encoder, symbolizer, hg, mask)
        benign_scores.append(s)
        if drift:
            fp += 1
        else:
            tn += 1
    for hg in test_attack:
        drift, s = detector.is_drift(encoder, symbolizer, hg, mask)
        attack_scores.append(s)
        if drift:
            tp += 1
            t = ttd_proxy(hg)
            if t is not None:
                ttds.append(t)
        else:
            fn += 1
    prec, rec, f1 = precision_recall_f1(tp, fp, fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1,
                benign_scores=benign_scores, attack_scores=attack_scores, ttds=ttds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", choices=["synthetic", "lanl"], default="synthetic")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--max-windows", type=int, default=400,
                     help="LANL only: cap on number of (computer, hour) windows parsed into graphs")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    all_graphs, source = load_data(args)
    pretrain_set, reference_set, calib_set, test_benign, test_attack = split_data(all_graphs, source)

    print(f"\nSplit sizes -- pretrain:{len(pretrain_set)} reference:{len(reference_set)} "
          f"calib:{len(calib_set)} test_benign:{len(test_benign)} test_attack:{len(test_attack)}")

    print("\n=== Training cross-modal encoder (contrastive, role-conditioned) ===")
    encoder = pretrain_encoder(pretrain_set, epochs=args.epochs)

    print("=== Fitting structural symbolizer ===")
    symbolizer = StructuralSymbolizer(k=8).fit(encoder, pretrain_set)

    print("=== Building role-conditioned WL reference + calibrating thresholds ===")
    detector = WLDriftDetector(iterations=2)
    detector.build_reference(encoder, symbolizer, reference_set)
    detector.calibrate_thresholds(encoder, symbolizer, calib_set, fpr_target=0.03)
    print("Role thresholds:", {r: round(t, 4) for r, t in detector.role_thresholds.items()})

    print("\n=== Full model ===")
    full = eval_variant(encoder, symbolizer, detector, test_benign, test_attack)
    print(f"TP={full['tp']} FP={full['fp']} FN={full['fn']} TN={full['tn']}")
    print(f"Precision={full['precision']:.3f} Recall={full['recall']:.3f} F1={full['f1']:.3f}")

    print("\n=== Ablation: no role-conditioning (single global reference) ===")
    detector_g = WLDriftDetector(iterations=2)
    # trick: relabel every graph's role to a single shared bucket for reference/threshold only
    def _as_global(graphs):
        out = []
        for hg in graphs:
            hg2 = type(hg)(graph=hg.graph, role="__global__", is_attack=hg.is_attack,
                           malicious_edges=hg.malicious_edges, source=hg.source)
            out.append(hg2)
        return out
    ref_g, calib_g = _as_global(reference_set), _as_global(calib_set)
    tb_g, ta_g = _as_global(test_benign), _as_global(test_attack)
    detector_g.build_reference(encoder, symbolizer, ref_g)
    detector_g.calibrate_thresholds(encoder, symbolizer, calib_g, fpr_target=0.03)
    global_res = eval_variant(encoder, symbolizer, detector_g, tb_g, ta_g)
    print(f"TP={global_res['tp']} FP={global_res['fp']} FN={global_res['fn']} "
          f"Precision={global_res['precision']:.3f} Recall={global_res['recall']:.3f} F1={global_res['f1']:.3f}")

    print("\n=== Ablation: no cross-modal fusion (process-lineage only) ===")
    proc_only = (True, False, False)
    pretrain_p = [process_only_view(hg) for hg in pretrain_set]
    reference_p = [process_only_view(hg) for hg in reference_set]
    calib_p = [process_only_view(hg) for hg in calib_set]
    test_benign_p = [process_only_view(hg) for hg in test_benign]
    test_attack_p = [process_only_view(hg) for hg in test_attack]
    encoder_p = pretrain_encoder(pretrain_p, epochs=args.epochs, mask=proc_only)
    symbolizer_p = StructuralSymbolizer(k=8).fit(encoder_p, pretrain_p, mask=proc_only)
    detector_p = WLDriftDetector(iterations=2)
    detector_p.build_reference(encoder_p, symbolizer_p, reference_p, mask=proc_only)
    detector_p.calibrate_thresholds(encoder_p, symbolizer_p, calib_p, fpr_target=0.03, mask=proc_only)
    proc_res = eval_variant(encoder_p, symbolizer_p, detector_p, test_benign_p, test_attack_p, mask=proc_only)
    print(f"TP={proc_res['tp']} FP={proc_res['fp']} FN={proc_res['fn']} "
          f"Precision={proc_res['precision']:.3f} Recall={proc_res['recall']:.3f} F1={proc_res['f1']:.3f}")

    print("\n=== Summary ===")
    print(f"{'Variant':<38}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    print(f"{'Full (fusion + role-cond + WL drift)':<38}{full['precision']:>10.3f}{full['recall']:>10.3f}{full['f1']:>10.3f}")
    print(f"{'- role-conditioning (global ref)':<38}{global_res['precision']:>10.3f}{global_res['recall']:>10.3f}{global_res['f1']:>10.3f}")
    print(f"{'- cross-modal fusion (proc-only)':<38}{proc_res['precision']:>10.3f}{proc_res['recall']:>10.3f}{proc_res['f1']:>10.3f}")

    # ---- plots ----
    print("\n=== Generating plots into outputs/ ===")
    variant_results = {
        "Full model": (full["precision"], full["recall"], full["f1"]),
        "- role-conditioning": (global_res["precision"], global_res["recall"], global_res["f1"]),
        "- cross-modal fusion": (proc_res["precision"], proc_res["recall"], proc_res["f1"]),
    }
    p1 = visualize.plot_variant_comparison(variant_results, source)
    p2 = visualize.plot_confusion_matrix(full["tp"], full["fp"], full["fn"], full["tn"], source)
    p3 = visualize.plot_drift_score_distribution(full["benign_scores"], full["attack_scores"],
                                                  detector.role_thresholds, source)
    p4 = visualize.plot_roc_pr(full["benign_scores"], full["attack_scores"], source)
    p5 = visualize.plot_ttd_distribution(full["ttds"], source)
    print(f"  {p1}\n  {p2}\n  {p3}\n  {p4}\n  {p5}")

    # ---- sample counterfactual + graph plot for one true positive ----
    p6 = None
    for hg in test_attack:
        drift, score = detector.is_drift(encoder, symbolizer, hg)
        if drift:
            labels = symbolizer.label_graph(encoder, hg)
            ref_hist = detector.role_refs[hg.role]
            thr = detector.role_thresholds[hg.role]
            edits, final_score = generate_counterfactual(hg, labels, ref_hist, thr)
            print(f"\n=== Sample counterfactual (role={hg.role}, score={score:.3f} > tau={thr:.3f}) ===")
            for e in edits:
                print("  -", describe_edit(hg, e))
            if not edits:
                print("  (no single-edge edit brought score under threshold within budget)")
            p6 = visualize.plot_sample_attack_graph(hg, score, thr, source)
            print(f"  {p6}")
            break

    # ---- save metrics.json ----
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "data_source": source,
            "note": ("SYNTHETIC DEMO DATA -- not a real-world result" if source == "synthetic"
                     else "Real LANL data, see data_lanl.py docstring for labeling/feature approximations"),
            "full_model": {k: v for k, v in full.items() if k not in ("benign_scores", "attack_scores", "ttds")},
            "ablation_no_role_conditioning": {k: v for k, v in global_res.items()
                                               if k not in ("benign_scores", "attack_scores", "ttds")},
            "ablation_no_fusion": {k: v for k, v in proc_res.items()
                                    if k not in ("benign_scores", "attack_scores", "ttds")},
        }, f, indent=2)
    print(f"\nSaved metrics.json -> {RESULTS_PATH}")


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
