#!/usr/bin/env python3
"""Benchmark complete phone-and-stress assessment with CTC-Viterbi or MFA.

The paired downstream workload is identical in both conditions: one CTC
encoding per utterance, canonical-phone mean log-posterior scoring, and stress
inference for every fully matched polysyllabic word.  The proposed condition
derives phone regions with the embedded CTC-Viterbi trellis.  The baseline uses
MFA TextGrid regions and adds the independently measured one-job MFA corpus
alignment wall time.
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
from benchmarks.evaluate_stress_alignment import word_offsets
from server.services.gop_service import GOPEvaluator
from server.services.stress_service import StressEvaluator
from server.utils.audio_features import is_vowel


def aggregate(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "sum": float(array.sum()),
    }


def paired_mfa_intervals(
    canonical: list[str],
    mfa: list[tuple[str, float, float]],
) -> tuple[list[dict], dict[int, int]]:
    normalized = [normalize_phone(phone) for phone in canonical]
    matches = sequence_matches(normalized, [phone for phone, _, _ in mfa])
    by_canonical = {
        canonical_index: mfa_index for canonical_index, mfa_index in matches
    }
    intervals = []
    for canonical_index, phone in enumerate(canonical):
        if canonical_index not in by_canonical:
            intervals.append(None)
            continue
        _, start, end = mfa[by_canonical[canonical_index]]
        intervals.append({
            "phoneme": normalize_phone(phone),
            "start": float(start),
            "end": float(end),
        })
    return intervals, by_canonical


def score_ctc_phone_regions(
    gop: GOPEvaluator,
    encoded: dict,
    paired_indices: set[int],
) -> tuple[list[dict], int]:
    unit_frames, _ = gop.viterbi_ctc_align(
        encoded["log_probs"].detach().cpu().numpy(), encoded["label_ids"]
    )
    segments = gop._segments(
        encoded["phonemes"],
        encoded["label_ids"],
        unit_frames,
        len(encoded["log_probs"]),
        encoded["duration"],
    )
    intervals = [
        {
            "phoneme": segment["phoneme"],
            "start": segment["start_time"],
            "end": segment["end_time"],
        }
        for segment in segments
    ]
    # Materialize the same canonical-label mean LPP used in the accuracy study.
    for index, token_id in enumerate(encoded["label_ids"]):
        if index not in paired_indices:
            continue
        float(
            encoded["log_probs"][unit_frames[index], token_id]
            .mean()
            .detach()
            .cpu()
        )
    return intervals, len(paired_indices)


def score_mfa_phone_regions(
    encoded: dict,
    mfa: list[tuple[str, float, float]],
    intervals: list[dict | None],
    by_canonical: dict[int, int],
) -> int:
    seconds_per_frame = encoded["duration"] / len(encoded["log_probs"])
    scored = 0
    for canonical_index, mfa_index in by_canonical.items():
        _, start, end = mfa[mfa_index]
        frames = [
            frame
            for frame in range(len(encoded["log_probs"]))
            if start <= (frame + 0.5) * seconds_per_frame < end
        ]
        if not frames:
            nearest = int(round(((start + end) / 2) / seconds_per_frame - 0.5))
            frames = [max(0, min(len(encoded["log_probs"]) - 1, nearest))]
        token_id = encoded["label_ids"][canonical_index]
        float(encoded["log_probs"][frames, token_id].mean().detach().cpu())
        scored += 1
    return scored


def paired_words(record: dict, by_canonical: dict[int, int]) -> list[tuple[dict, int, int]]:
    result = []
    for word, (start, end) in zip(record["words"], word_offsets(record)):
        if sum(is_vowel(phone) for phone in word["phones"]) < 2:
            continue
        if all(index in by_canonical for index in range(start, end)):
            result.append((word, start, end))
    return result


def stress_words(
    stress: StressEvaluator,
    waveform: np.ndarray,
    words: list[tuple[dict, int, int]],
    intervals: list[dict | None],
    method: str,
) -> int:
    completed = 0
    for word, start, end in words:
        word_intervals = intervals[start:end]
        if any(interval is None for interval in word_intervals):
            raise ValueError(f"Incomplete {method} interval sequence for {word['text']}")
        output = stress.predict(
            waveform,
            word["text"],
            alignments=word_intervals,
            method=method,
        )
        if "error" in output:
            raise RuntimeError(output["error"])
        completed += 1
    return completed


def run_condition(
    condition: str,
    gop: GOPEvaluator,
    stress: StressEvaluator,
    audio: bytes,
    record: dict,
    canonical: list[str],
    mfa: list[tuple[str, float, float]],
    mfa_intervals: list[dict | None],
    by_canonical: dict[int, int],
    words: list[tuple[dict, int, int]],
) -> dict:
    total_started = time.perf_counter()
    started = time.perf_counter()
    encoded = gop._encode_utterance(
        audio, record["text"], target_phonemes=canonical
    )
    encoding_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    if condition == "ctc_viterbi":
        intervals, scored_phones = score_ctc_phone_regions(
            gop, encoded, set(by_canonical)
        )
    elif condition == "mfa":
        intervals = mfa_intervals
        scored_phones = score_mfa_phone_regions(
            encoded, mfa, intervals, by_canonical
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")
    phone_region_and_scoring_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    scored_words = stress_words(
        stress, encoded["audio"], words, intervals, method=condition
    )
    stress_ms = (time.perf_counter() - started) * 1000.0
    return {
        "encoding_ms": encoding_ms,
        "phone_region_and_scoring_ms": phone_region_and_scoring_ms,
        "stress_ms": stress_ms,
        "downstream_total_ms": (time.perf_counter() - total_started) * 1000.0,
        "scored_phones": scored_phones,
        "scored_polysyllabic_words": scored_words,
    }


def run(args: argparse.Namespace) -> None:
    scores = json.loads((args.dataset / "resource" / "scores.json").read_text())
    wavs = read_kaldi_map(args.dataset / "test" / "wav.scp")
    textgrids = {
        path.stem: path for path in sorted(args.mfa_textgrids.rglob("*.TextGrid"))
    }
    mfa_batch = json.loads(args.mfa_batch_result.read_text())
    gop = GOPEvaluator(args.gop_model)
    gop._audit_log = lambda *unused_args, **unused_kwargs: None
    stress = StressEvaluator(args.stress_model, args.scaler)

    # Warm both paths, including one stress prediction when the utterance has a
    # paired polysyllabic word. Warm-up is excluded from reported measurements.
    warm_utterance = sorted(textgrids)[0]
    warm_record = scores[warm_utterance]
    warm_canonical, _ = flatten_annotation(warm_record)
    warm_mfa = parse_phone_textgrid(textgrids[warm_utterance])
    warm_intervals, warm_map = paired_mfa_intervals(warm_canonical, warm_mfa)
    warm_words = paired_words(warm_record, warm_map)
    warm_audio = (args.dataset / wavs[warm_utterance]).read_bytes()
    for condition in ("ctc_viterbi", "mfa"):
        run_condition(
            condition,
            gop,
            stress,
            warm_audio,
            warm_record,
            warm_canonical,
            warm_mfa,
            warm_intervals,
            warm_map,
            warm_words[:1],
        )

    rows, failures = [], []
    utterances = sorted(textgrids)
    for index, utterance in enumerate(utterances, start=1):
        record = scores[utterance]
        canonical, _ = flatten_annotation(record)
        mfa = parse_phone_textgrid(textgrids[utterance])
        mfa_intervals, by_canonical = paired_mfa_intervals(canonical, mfa)
        words = paired_words(record, by_canonical)
        audio = (args.dataset / wavs[utterance]).read_bytes()
        row = {"utterance": utterance}
        try:
            # Alternate order to reduce systematic thermal/order bias.
            order = ("ctc_viterbi", "mfa") if index % 2 else ("mfa", "ctc_viterbi")
            for condition in order:
                measured = run_condition(
                    condition,
                    gop,
                    stress,
                    audio,
                    record,
                    canonical,
                    mfa,
                    mfa_intervals,
                    by_canonical,
                    words,
                )
                for key, value in measured.items():
                    row[f"{condition}_{key}"] = value
            rows.append(row)
        except Exception as exc:
            failures.append({"utterance": utterance, "error": repr(exc)})
        if index % args.progress_every == 0:
            print(
                f"processed {index}/{len(utterances)}; "
                f"complete={len(rows)} failures={len(failures)}",
                flush=True,
            )

    metrics = {}
    for condition in ("ctc_viterbi", "mfa"):
        metrics[condition] = {
            metric: aggregate([row[f"{condition}_{metric}"] for row in rows])
            for metric in (
                "encoding_ms",
                "phone_region_and_scoring_ms",
                "stress_ms",
                "downstream_total_ms",
            )
        }
        metrics[condition]["scored_phones"] = int(
            sum(row[f"{condition}_scored_phones"] for row in rows)
        )
        metrics[condition]["scored_polysyllabic_words"] = int(
            sum(row[f"{condition}_scored_polysyllabic_words"] for row in rows)
        )

    paired_count = len(rows)
    alignment_seconds = float(mfa_batch["wall_seconds"])
    ctc_seconds = metrics["ctc_viterbi"]["downstream_total_ms"]["sum"] / 1000.0
    mfa_downstream_seconds = metrics["mfa"]["downstream_total_ms"]["sum"] / 1000.0
    mfa_seconds = alignment_seconds + mfa_downstream_seconds
    result = {
        "protocol": {
            "workload": "complete paired phone-LPP and polysyllabic-stress assessment",
            "conditions": {
                "ctc_viterbi": "CTC encoding + Viterbi regions + phone LPP + stress",
                "mfa": "one-job MFA alignment + CTC encoding + MFA-region phone LPP + stress",
            },
            "pairing": "same MFA-aligned utterances and fully matched polysyllabic words",
            "condition_order": "alternated by utterance after one excluded warm-up",
            "mfa_jobs": int(mfa_batch["jobs"]),
            "dataset": str(args.dataset),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "counts": {
            "mfa_input_utterances": int(mfa_batch["input_utterances"]),
            "paired_complete_utterances": paired_count,
            "paired_phone_scores": metrics["ctc_viterbi"]["scored_phones"],
            "paired_polysyllabic_stress_predictions": metrics["ctc_viterbi"][
                "scored_polysyllabic_words"
            ],
            "failures": failures,
        },
        "components": {
            "mfa_external_alignment": {
                "wall_seconds": alignment_seconds,
                "ms_per_input": alignment_seconds
                * 1000.0
                / int(mfa_batch["input_utterances"]),
            },
            **metrics,
        },
        "complete_pipeline": {
            "ctc_viterbi_wall_seconds": ctc_seconds,
            "mfa_wall_seconds": mfa_seconds,
            "ctc_viterbi_ms_per_paired_output": ctc_seconds * 1000.0 / paired_count,
            "mfa_ms_per_paired_output": mfa_seconds * 1000.0 / paired_count,
            "mfa_over_ctc_speed_ratio": mfa_seconds / ctc_seconds,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gop-model", default="server/models/ctcgop")
    parser.add_argument(
        "--stress-model",
        type=Path,
        default=Path("server/models/sylstress/stress_model.keras"),
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        default=Path("server/models/sylstress/scaler_params.json"),
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
