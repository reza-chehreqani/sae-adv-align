"""Collects eval_results_*.json (written by scripts/evaluate_checkpoint.py)
from one or more run directories into a single comparison table -- this is
the "which drift-control strategy actually wins" table.

Usage:
    python scripts/aggregate_results.py runs/* --out results_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="One or more runs/<name> directories (globs are fine).")
    parser.add_argument("--out", default="results_summary.csv")
    args = parser.parse_args()

    rows = []
    for d in args.run_dirs:
        for path in sorted(glob.glob(os.path.join(d, "eval_results_*.json"))):
            with open(path) as fh:
                data = json.load(fh)
            row = {
                "run_dir": d,
                "checkpoint": os.path.basename(data.get("checkpoint", "")),
                "name": data.get("name"),
                "base": data.get("base"),
                "align_space": data.get("align_space"),
                "drift_mode": data.get("drift_mode"),
                "clean_acc": data.get("clean_acc"),
                "pgd20_acc": data.get("pgd20_acc"),
                "pgd50_multirestart_acc": data.get("pgd50_multirestart_acc"),
                "autoattack_acc": data.get("autoattack_acc"),
                "sae_fvu": data.get("sae_fvu"),
                "sae_dead_frac": data.get("sae_dead_frac"),
                "cka_vs_initial": data.get("cka_vs_initial"),
            }
            sweep = data.get("step_sweep")
            if sweep:
                vals = list(sweep.values())
                row["step_sweep_min"] = min(vals)
                row["step_sweep_max_drop"] = max(vals) - min(vals)
            rows.append(row)

    if not rows:
        print("No eval_results_*.json files found in the given directories. "
              "Run scripts/evaluate_checkpoint.py on each checkpoint first.")
        return

    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else ("-" if v is None else str(v))

    header = f"{'name':<38} {'base':<7} {'space':<5} {'drift':<16} {'clean':>7} {'pgd20':>7} {'pgd50x':>7} {'AA':>7} {'fvu':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{str(r['name']):<38} {str(r['base']):<7} {str(r['align_space']):<5} {str(r['drift_mode']):<16} "
              f"{fmt(r['clean_acc']):>7} {fmt(r['pgd20_acc']):>7} {fmt(r['pgd50_multirestart_acc']):>7} "
              f"{fmt(r.get('autoattack_acc')):>7} {fmt(r.get('sae_fvu')):>7}")

    print(f"\nSaved full comparison table to {args.out}")


if __name__ == "__main__":
    main()
