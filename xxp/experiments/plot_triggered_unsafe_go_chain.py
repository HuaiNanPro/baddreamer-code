#!/usr/bin/env python3
"""Plot the BadDreamer unsafe-go propagation chain.

The figure is generated from expanded_val20_ep002_results.json and mirrors the
grouped-bar style used in common backdoor experiment summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SETTINGS = [
    ("clean0", "0%"),
    ("poison2p5", "2.5%"),
    ("poison5", "5%"),
]
ATTACKS = [
    ("attack2p5", "Attack-2.5"),
    ("attack5", "Attack-5"),
]
PANELS = [
    ("WM_ASR", "VaViM erasure"),
    ("T_UGR", "VaVAM unsafe-go"),
    ("E2E_ASR", "E2E unsafe ASR"),
    ("action_success_given_WM_success", "Unsafe-go | erasure"),
]
COLORS = {
    "attack2p5": "#86ad70",
    "attack5": "#ee965a",
}


def percent(value: float | None) -> float:
    if value is None:
        return float("nan")
    return 100.0 * float(value)


def load_values(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for setting_key, _ in SETTINGS:
        run = data["runs"][setting_key]["e2e_action_conditioned_strict_all4_auto_yellow_delta"]
        out[setting_key] = {}
        for attack_key, _ in ATTACKS:
            out[setting_key][attack_key] = run[attack_key]
    return out


def draw(results_json: Path, out_prefix: Path) -> None:
    values = load_values(results_json)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.4), sharey=True)
    axes = axes.reshape(-1)
    x = np.arange(len(SETTINGS), dtype=float)
    width = 0.34

    for ax, (metric_key, title) in zip(axes, PANELS):
        for attack_idx, (attack_key, attack_label) in enumerate(ATTACKS):
            offset = (attack_idx - 0.5) * width
            heights = [
                percent(values[setting_key][attack_key].get(metric_key))
                for setting_key, _ in SETTINGS
            ]
            ax.bar(
                x + offset,
                heights,
                width=width,
                color=COLORS[attack_key],
                edgecolor="white",
                linewidth=0.8,
                label=attack_label,
            )

        ax.set_title(title, pad=5)
        ax.set_xticks(x, [label for _, label in SETTINGS])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="both", direction="in", length=3, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("black")

    axes[0].set_ylabel("Metric (%)")
    axes[2].set_ylabel("Metric (%)")
    axes[1].legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")

    fig.supxlabel("Poison ratio", y=0.02, fontsize=10)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0), w_pad=0.9, h_pad=0.9)

    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_json",
        default="/raid/zengchaolv/xxp/poisoning/matrix_results/expanded_val20_ep002_results.json",
        type=Path,
    )
    parser.add_argument(
        "--out_prefix",
        default="/raid/zengchaolv/xxp/figures/triggered_unsafe_go_chain",
        type=Path,
    )
    args = parser.parse_args()
    draw(args.results_json, args.out_prefix)


if __name__ == "__main__":
    main()
