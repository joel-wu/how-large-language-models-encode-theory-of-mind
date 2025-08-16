#!/usr/bin/env python3
"""
Summarize ToM evaluation CSVs across repetitions.

Folder layout expected:
  <root>/<rep>/<m>/<task>/results.csv
    - rep: 1..N
    - m:   numeric string (e.g., "0", "2e-06", "5e-05")
    - task in {"unexpected","transfer","irony"}
    - results.csv is a matrix of 0/1

The script computes, for each m and task:
  - total_mean, total_var (sum of matrix values across items, averaged over reps)
  - percent_mean, percent_var (100 * total / number_of_cells, averaged over reps)
  - avg_percent_ut: mean of (unexpected, transfer) percentages

Outputs a single CSV with one row per m.
"""

import argparse
import os
import math
import pandas as pd

TASKS = ["unexpected", "transfer", "irony"]

def is_float_like(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def find_m_values(root: str, first_rep: int) -> list[str]:
    rep_dir = os.path.join(root, str(first_rep))
    if not os.path.isdir(rep_dir):
        raise FileNotFoundError(f"Missing rep directory: {rep_dir}")
    m_dirs = [
        name for name in os.listdir(rep_dir)
        if os.path.isdir(os.path.join(rep_dir, name)) and is_float_like(name)
    ]
    # Sort by numeric value
    return sorted(m_dirs, key=lambda x: float(x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder that contains per-rep results, i.e. <root>/<rep>/<m>/<task>/results.csv")
    ap.add_argument("--reps", type=int, default=5, help="Number of repetitions (rep folders 1..reps).")
    ap.add_argument("--tasks", type=str, default="unexpected,transfer,irony", help="Comma-separated task names.")
    ap.add_argument("--first_rep_for_m_scan", type=int, default=1, help="Which rep folder to scan to enumerate m values.")
    ap.add_argument("--out_csv", required=True, help="Path to write the summary CSV.")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in TASKS:
            raise ValueError(f"Unsupported task '{t}'. Allowed: {TASKS}")

    m_values = find_m_values(args.root, args.first_rep_for_m_scan)
    if not m_values:
        raise RuntimeError("No m directories found.")

    # Accumulators: per m -> per task -> list over reps
    totals = {m: {t: [] for t in tasks} for m in m_values}
    percents = {m: {t: [] for t in tasks} for m in m_values}
    avg_ut = {m: [] for m in m_values}  # average of (unexpected, transfer) percent per rep

    for rep in range(1, args.reps + 1):
        for m in m_values:
            ut_percent_for_rep = []
            for t in tasks:
                csv_path = os.path.join(args.root, str(rep), m, t, "results.csv")
                if not os.path.isfile(csv_path):
                    # Skip missing files silently but keep shape consistent
                    continue
                df = pd.read_csv(csv_path, index_col=0)
                k = float(df.values.sum())
                total_cells = df.size
                pct = (k / total_cells) * 100.0 if total_cells > 0 else float("nan")

                totals[m][t].append(k)
                percents[m][t].append(pct)

                if t in ("unexpected", "transfer"):
                    ut_percent_for_rep.append(pct)

            # Only if both 'unexpected' and 'transfer' present for the rep
            if len(ut_percent_for_rep) == 2 and all(not math.isnan(x) for x in ut_percent_for_rep):
                avg_ut[m].append(sum(ut_percent_for_rep) / 2.0)

    # Build tidy rows (one row per m)
    rows = []
    for m in m_values:
        row = {"m": float(m)}
        # totals/percents stats
        for t in tasks:
            vals_tot = totals[m][t]
            vals_pct = percents[m][t]

            # Means/vars across reps (skip if empty)
            if vals_tot:
                row[f"{t}_total_mean"] = float(pd.Series(vals_tot).mean())
                row[f"{t}_total_var"]  = float(pd.Series(vals_tot).var(ddof=1)) if len(vals_tot) > 1 else 0.0
            else:
                row[f"{t}_total_mean"] = float("nan")
                row[f"{t}_total_var"]  = float("nan")

            if vals_pct:
                row[f"{t}_percent_mean"] = float(pd.Series(vals_pct).mean())
                row[f"{t}_percent_var"]  = float(pd.Series(vals_pct).var(ddof=1)) if len(vals_pct) > 1 else 0.0
            else:
                row[f"{t}_percent_mean"] = float("nan")
                row[f"{t}_percent_var"]  = float("nan")

        # Avg(Unexpected & Transfer)
        if avg_ut[m]:
            row["avg_percent_unexpected_transfer"] = float(pd.Series(avg_ut[m]).mean())
        else:
            row["avg_percent_unexpected_transfer"] = float("nan")

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("m").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote summary -> {args.out_csv}")

if __name__ == "__main__":
    main()
