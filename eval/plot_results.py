#!/usr/bin/env python3
"""Renders e_t CDF, e_r CDF, and a loss breakdown (by tier if available,
else a histogram) from a stats.csv into the same result directory.

Usage: python plot_results.py --stats results/dev_method_TS/stats.csv [--tiers tiers.csv]
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_stats(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _cdf(values, out_path, xlabel, title):
    values = np.sort(np.array(values, dtype=np.float64))
    if len(values) == 0:
        return
    y = np.arange(1, len(values) + 1) / len(values)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(values, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cumulative fraction of scenarios")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_error_cdfs(rows, out_dir):
    e_t = [float(r["e_t"]) for r in rows if r["e_t"] != "" and r["missing"] == "0"]
    e_r = [float(r["e_r"]) for r in rows if r["e_r"] != "" and r["missing"] == "0"]
    _cdf(e_t, os.path.join(out_dir, "e_t_cdf.png"), "translation error (m)", "Translation error CDF")
    _cdf(e_r, os.path.join(out_dir, "e_r_cdf.png"), "rotation error (deg)", "Rotation error CDF")


def plot_loss_by_tier(rows, tiers: dict, out_dir, random_loss=None):
    if tiers:
        by_tier = {}
        for r in rows:
            tier = tiers.get(r["scenario_id"], "unknown")
            by_tier.setdefault(tier, []).append(float(r["L"]))
        labels = sorted(by_tier.keys())
        means = [np.mean(by_tier[t]) for t in labels]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, means)
        if random_loss is not None:
            ax.axhline(random_loss, color="crimson", linestyle="--",
                        label=f"random guess ({random_loss:.3f})")
            ax.legend()
        ax.set_ylabel("mean loss")
        ax.set_title("Loss by difficulty tier")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loss_by_tier.png"), dpi=120)
        plt.close(fig)
    else:
        losses = [float(r["L"]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(losses, bins=20, range=(0, 1))
        if random_loss is not None:
            ax.axvline(random_loss, color="crimson", linestyle="--",
                        label=f"random guess ({random_loss:.3f})")
            ax.legend()
        ax.set_xlabel("per-scenario loss")
        ax.set_ylabel("count")
        ax.set_title("Loss distribution")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loss_by_tier.png"), dpi=120)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True)
    parser.add_argument("--tiers", help="optional private CSV: scenario_id,tier")
    parser.add_argument("--random-loss", type=float,
                         help="mean loss of a random guess, drawn as a reference line")
    args = parser.parse_args(argv)

    rows = _read_stats(args.stats)
    out_dir = os.path.dirname(os.path.abspath(args.stats))

    tiers = {}
    if args.tiers:
        with open(args.tiers) as f:
            reader = csv.DictReader(f)
            for row in reader:
                tiers[row["scenario_id"]] = row["tier"]

    plot_error_cdfs(rows, out_dir)
    plot_loss_by_tier(rows, tiers, out_dir, random_loss=args.random_loss)
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
