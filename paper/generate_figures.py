#!/usr/bin/env python3
"""Generate manuscript figures directly from the checked-in result JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

BLUE = "#246BCE"
ORANGE = "#E07A1F"
GREEN = "#2A9D65"
INK = "#17212B"
PALE_BLUE = "#EAF2FD"
PALE_ORANGE = "#FFF0E4"
PALE_GREEN = "#E8F7EF"


def box(axis, x, y, width, height, label, color, fontsize=9):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=color,
        facecolor="white",
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=fontsize, color=INK)


def arrow(axis, x1, y1, x2, y2, color=INK):
    axis.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color=color))


def architecture_figure():
    fig, axis = plt.subplots(figsize=(10.8, 4.2))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 5)
    axis.axis("off")

    axis.add_patch(FancyBboxPatch((0.1, 2.7), 11.8, 2.0, boxstyle="round,pad=0.03", facecolor=PALE_BLUE, edgecolor=BLUE, linewidth=1.4))
    axis.text(0.35, 4.42, "Proposed: embedded CTC-Viterbi (one resident model)", color=BLUE, fontsize=11, weight="bold")
    box(axis, 0.45, 3.18, 1.5, 0.65, "audio +\ncanonical phones", BLUE)
    box(axis, 2.55, 3.18, 1.45, 0.65, "phoneme CTC\nencoder", BLUE)
    box(axis, 4.65, 3.18, 1.55, 0.65, "CTC-Viterbi\ntrellis", BLUE)
    box(axis, 6.85, 3.18, 1.45, 0.65, "phone spans", BLUE)
    box(axis, 9.05, 3.65, 1.9, 0.55, "phone LPP scores", GREEN)
    box(axis, 9.05, 2.85, 1.9, 0.55, "stress features +\nsequential model", GREEN)
    for start, end in ((1.95, 2.55), (4.0, 4.65), (6.2, 6.85)):
        arrow(axis, start, 3.5, end, 3.5, BLUE)
    arrow(axis, 8.3, 3.5, 9.05, 3.92, GREEN)
    arrow(axis, 8.3, 3.5, 9.05, 3.12, GREEN)

    axis.add_patch(FancyBboxPatch((0.1, 0.25), 11.8, 1.95, boxstyle="round,pad=0.03", facecolor=PALE_ORANGE, edgecolor=ORANGE, linewidth=1.4))
    axis.text(0.35, 1.92, "Baseline: external Montreal Forced Aligner pipeline", color=ORANGE, fontsize=11, weight="bold")
    box(axis, 0.45, 0.65, 1.5, 0.65, "audio + text", ORANGE)
    box(axis, 2.45, 0.65, 1.6, 0.65, "MFA: lexicon +\nKaldi acoustic model", ORANGE)
    box(axis, 4.55, 0.65, 1.45, 0.65, "phone spans", ORANGE)
    box(axis, 6.55, 1.15, 1.65, 0.55, "same CTC encoder\n+ phone LPP", GREEN)
    box(axis, 6.55, 0.35, 1.65, 0.55, "same stress features\n+ sequential model", GREEN)
    box(axis, 9.05, 0.75, 1.9, 0.55, "assessment outputs", GREEN)
    arrow(axis, 1.95, 0.98, 2.45, 0.98, ORANGE)
    arrow(axis, 4.05, 0.98, 4.55, 0.98, ORANGE)
    arrow(axis, 6.0, 0.98, 6.55, 1.42, GREEN)
    arrow(axis, 6.0, 0.98, 6.55, 0.62, GREEN)
    arrow(axis, 8.2, 1.42, 9.05, 1.03, GREEN)
    arrow(axis, 8.2, 0.62, 9.05, 1.03, GREEN)
    fig.tight_layout()
    fig.savefig(HERE / "fig_architecture.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def results_figure():
    phone = json.loads((RESULTS / "phone_accuracy.json").read_text())
    stress = json.loads((RESULTS / "stress_accuracy.json").read_text())
    full_pipeline = json.loads(
        (RESULTS / "full_pipeline_latency.json").read_text()
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.45))
    conditions = ["CTC-Viterbi", "MFA"]
    colors = [BLUE, ORANGE]

    pcc = [
        phone["phone_accuracy"]["ctc_viterbi"]["phone_quadratic"]["pcc"],
        phone["phone_accuracy"]["mfa"]["phone_quadratic"]["pcc"],
    ]
    axes[0].bar(conditions, pcc, color=colors, width=0.62)
    axes[0].set_ylim(0, 0.55)
    axes[0].set_ylabel("Pearson correlation")
    axes[0].set_title("Phone accuracy")

    location = [
        stress["stress_accuracy"]["ctc_viterbi"]["canonical_location_accuracy_when_human_correct"],
        stress["stress_accuracy"]["mfa"]["canonical_location_accuracy_when_human_correct"],
    ]
    axes[1].bar(conditions, location, color=colors, width=0.62)
    axes[1].set_ylim(0.55, 0.82)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Stress location")

    latency = [
        full_pipeline["complete_pipeline"]["ctc_viterbi_ms_per_paired_output"],
        full_pipeline["complete_pipeline"]["mfa_ms_per_paired_output"],
    ]
    axes[2].bar(conditions, latency, color=colors, width=0.62)
    axes[2].set_ylim(0, 145)
    axes[2].set_ylabel("Milliseconds / utterance")
    axes[2].set_title("Complete phone + stress")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelrotation=12)
        for container in axis.containers:
            labels = [f"{bar.get_height():.3g}" for bar in container]
            axis.bar_label(container, labels=labels, padding=2, fontsize=8)
    fig.tight_layout(w_pad=2)
    fig.savefig(HERE / "fig_results.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture_figure()
    results_figure()
