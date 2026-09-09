#!/usr/bin/env python3
"""Measure the complete MFA-conditioned phone assessment pipeline.

The external MFA alignment stage is measured by ``run_mfa_benchmark.py``. This
script measures the required downstream CTC encoding and MFA-interval LPP stage,
then combines the two serial wall times. It also reads the proposed CTC-Viterbi
latency rows on the same MFA-aligned utterance subset.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.evaluate_ctc_viterbi_vs_mfa import (
    flatten_annotation,
    normalize_phone,
    parse_phone_textgrid,
    read_kaldi_map,
    sequence_matches,
)
from server.services.gop_service import GOPEvaluator


def aggregate(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "sum": float(array.sum()),
    }


def score_with_mfa_intervals(
    evaluator: GOPEvaluator,
    audio: bytes,
    transcript: str,
    canonical_phones: list[str],
    mfa_intervals: list[tuple[str, float, float]],
) -> int:
    encoded = evaluator._encode_utterance(
        audio, transcript, target_phonemes=canonical_phones
    )
    canonical = [normalize_phone(phone) for phone in encoded["phonemes"]]
    matches = sequence_matches(canonical, [phone for phone, _, _ in mfa_intervals])
    seconds_per_frame = encoded["duration"] / len(encoded["log_probs"])
    scored = 0
    for canonical_index, mfa_index in matches:
        _, start, end = mfa_intervals[mfa_index]
        frames = [
            frame
            for frame in range(len(encoded["log_probs"]))
            if start <= (frame + 0.5) * seconds_per_frame < end
        ]
        if not frames:
            nearest = int(round(((start + end) / 2) / seconds_per_frame - 0.5))
            frames = [max(0, min(len(encoded["log_probs"]) - 1, nearest))]
        token_id = encoded["label_ids"][canonical_index]
        # Force evaluation of the same canonical-label mean LPP used by the
        # paired accuracy benchmark.
        float(encoded["log_probs"][frames, token_id].mean().detach().cpu())
        scored += 1
    return scored


def run(args: argparse.Namespace) -> None:
    scores = json.loads((args.dataset / "resource" / "scores.json").read_text())
    wavs = read_kaldi_map(args.dataset / "test" / "wav.scp")
    textgrids = {
        path.stem: path for path in sorted(args.mfa_textgrids.rglob("*.TextGrid"))
    }
    mfa_batch = json.loads(args.mfa_batch_result.read_text())
    ctc_rows = pd.read_csv(args.ctc_latency_rows)
    evaluator = GOPEvaluator(args.model)

    warm_utterance = sorted(textgrids)[0]
    warm_phones, _ = flatten_annotation(scores[warm_utterance])
    score_with_mfa_intervals(
        evaluator,
        (args.dataset / wavs[warm_utterance]).read_bytes(),
        scores[warm_utterance]["text"],
        warm_phones,
        parse_phone_textgrid(textgrids[warm_utterance]),
    )

    rows, failures = [], []
    for index, utterance in enumerate(sorted(textgrids), start=1):
        phones, _ = flatten_annotation(scores[utterance])
        started = time.perf_counter()
        try:
            scored_phones = score_with_mfa_intervals(
                evaluator,
                (args.dataset / wavs[utterance]).read_bytes(),
                scores[utterance]["text"],
                phones,
                parse_phone_textgrid(textgrids[utterance]),
            )
            wall_ms = (time.perf_counter() - started) * 1000.0
            rows.append({
                "utterance": utterance,
                "mfa_conditioned_scoring_ms": wall_ms,
                "scored_phones": scored_phones,
            })
        except Exception as exc:
            failures.append({"utterance": utterance, "error": repr(exc)})
        if index % args.progress_every == 0:
            print(f"processed {index}/{len(textgrids)}", flush=True)

    scoring = aggregate([row["mfa_conditioned_scoring_ms"] for row in rows])
    paired_ids = {row["utterance"] for row in rows}
    proposed = ctc_rows[ctc_rows["utterance"].astype(str).str.zfill(9).isin(paired_ids)]
    proposed_wall = aggregate(proposed["wall_ms"].tolist())
    input_count = int(mfa_batch["input_utterances"])
    aligned_count = int(mfa_batch["aligned_utterances"])
    external_alignment_seconds = float(mfa_batch["wall_seconds"])
    baseline_total_seconds = external_alignment_seconds + scoring["sum"] / 1000.0
    result = {
        "protocol": {
            "workload": "serial two-stage MFA phone assessment versus embedded CTC-Viterbi",
            "mfa_jobs": int(mfa_batch["jobs"]),
            "mfa_stage": "full corpus alignment, including unsuccessful OOV inputs",
            "scoring_stage": "resident CTC encoding and MFA-interval mean LPP on aligned outputs",
            "proposed_stage": "resident CTC encoding, Viterbi, and mean LPP on the same aligned utterances",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "counts": {
            "input_utterances": input_count,
            "aligned_and_scored_utterances": aligned_count,
            "failures": failures,
        },
        "components": {
            "mfa_alignment": {
                "wall_seconds": external_alignment_seconds,
                "ms_per_input": external_alignment_seconds * 1000.0 / input_count,
            },
            "mfa_conditioned_ctc_scoring": scoring,
            "proposed_ctc_viterbi_assessment": proposed_wall,
        },
        "complete_serial_pipeline": {
            "mfa_baseline_wall_seconds": baseline_total_seconds,
            "mfa_baseline_ms_per_aligned_output": baseline_total_seconds * 1000.0 / aligned_count,
            "ctc_viterbi_ms_per_aligned_output": proposed_wall["mean"],
            "mfa_over_ctc_speed_ratio": (
                baseline_total_seconds * 1000.0 / aligned_count / proposed_wall["mean"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mfa-textgrids", type=Path, required=True)
    parser.add_argument("--mfa-batch-result", type=Path, required=True)
    parser.add_argument("--ctc-latency-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="server/models/ctcgop")
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
