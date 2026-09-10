"""Generate the manuscript's three-system result figure from checked-in JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ACCURACY = ROOT / "results" / "three_system_comparison.json"
LATENCY = ROOT / "results" / "three_system_latency.json"
OUTPUT = ROOT / "fig_three_systems.png"
ARCHITECTURE = ROOT / "fig_architecture_three.png"

SYSTEMS = ("mfa_modular", "cao_viterbi_proposed", "gopt_joint")
LABELS = ("MFA modular", "Cao + Viterbi", "Joint GOPT")
COLORS = ("#6b7280", "#0072b2", "#e69f00")


def mean_metric(value: float | dict) -> float:
    return float(value["mean"] if isinstance(value, dict) else value)


def architecture_figure() -> None:
    figure, axis = plt.subplots(figsize=(7.15, 2.75))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    rows = (
        (
            0.80, "MFA modular", "CTC posterior matrix\n+ MFA intervals",
            "MFA-aligned 78-D vectors\n→ phone GOPT",
            "MFA → syllable regions\n→ stress network",
            COLORS[0],
        ),
        (
            0.50, "Proposed", "one CTC\nposterior matrix",
            "Cao 41-D AF vectors\n→ phone GOPT",
            "CTC-Viterbi → syllable regions\n→ same stress network",
            COLORS[1],
        ),
        (
            0.20, "Joint GOPT", "CTC posterior matrix\n+ MFA intervals",
            "MFA-aligned 78-D vectors\n→ joint GOPT",
            "phone score +\nword-stress score",
            COLORS[2],
        ),
    )
    x_positions = (0.02, 0.19, 0.45, 0.73)
    widths = (0.14, 0.22, 0.24, 0.25)
    for y, name, front, phone, stress, color in rows:
        entries = (name, front, phone, stress)
        for x, width, label in zip(x_positions, widths, entries):
            face = color if x == x_positions[0] else "#f8fafc"
            text_color = "white" if x == x_positions[0] else "#111827"
            axis.add_patch(plt.Rectangle(
                (x, y - 0.095), width, 0.19,
                facecolor=face, edgecolor=color, linewidth=1.2,
            ))
            axis.text(
                x + width / 2, y, label, ha="center", va="center",
                fontsize=8, color=text_color,
                fontweight="bold" if x == x_positions[0] else "normal",
            )
        for index in range(3):
            start = x_positions[index] + widths[index]
            end = x_positions[index + 1]
            axis.annotate(
                "", xy=(end, y), xytext=(start, y),
                arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 1},
            )
    axis.text(0.30, 0.96, "Front end", ha="center", va="center",
              fontsize=8, fontweight="bold")
    axis.text(0.56, 0.96, "Phone branch", ha="center", va="center",
              fontsize=8, fontweight="bold")
    axis.text(0.85, 0.96, "Stress branch / output", ha="center", va="center",
              fontsize=8, fontweight="bold")
    figure.tight_layout(pad=0.2)
    figure.savefig(ARCHITECTURE, dpi=350, bbox_inches="tight")


def main() -> None:
    accuracy = json.loads(ACCURACY.read_text())
    latency = json.loads(LATENCY.read_text())
    phone = [
        mean_metric(accuracy["accuracy"][name]["phone"]["pcc"])
        for name in SYSTEMS
    ]
    phone_error = [
        float(accuracy["accuracy"][name]["phone"]["pcc"]["std"])
        for name in SYSTEMS
    ]
    stress = [
        mean_metric(accuracy["accuracy"][name]["stress"]["auroc"])
        for name in SYSTEMS
    ]
    stress_error = [
        float(accuracy["accuracy"][name]["stress"]["auroc"].get("std", np.nan))
        if isinstance(accuracy["accuracy"][name]["stress"]["auroc"], dict)
        else np.nan
        for name in SYSTEMS
    ]
    runtime = [
        float(latency["complete_pipeline_mean_ms_per_request"][name])
        for name in SYSTEMS
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.35))
    x = np.arange(3)
    panels = (
        (phone, "Phone PCC", (0.0, 0.75)),
        (stress, "Stress AUROC", (0.5, 0.8)),
        (runtime, "Latency (ms/request)", (0.0, max(runtime) * 1.2)),
    )
    for axis, (values, ylabel, ylim) in zip(axes, panels):
        bars = axis.bar(x, values, color=COLORS, width=0.68)
        if ylabel == "Phone PCC":
            axis.errorbar(
                x, values, yerr=phone_error, fmt="none", ecolor="black",
                capsize=3, linewidth=0.9,
            )
        elif ylabel == "Stress AUROC":
            axis.errorbar(
                x, values, yerr=stress_error, fmt="none", ecolor="black",
                capsize=3, linewidth=0.9,
            )
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        axis.set_xticks(x, LABELS, rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        span = ylim[1] - ylim[0]
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + span * 0.025,
                f"{value:.3f}" if value < 1 else f"{value:.1f}",
                ha="center", va="bottom", fontsize=8,
            )
    axes[0].set_title("Segmental accuracy")
    axes[1].set_title("Stress correctness")
    axes[2].set_title("Complete pipeline")
    figure.tight_layout(w_pad=1.4)
    figure.savefig(OUTPUT, dpi=350, bbox_inches="tight")
    plt.close(figure)
    architecture_figure()
    print(OUTPUT)
    print(ARCHITECTURE)


if __name__ == "__main__":
    main()
