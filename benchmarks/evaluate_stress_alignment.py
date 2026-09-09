#!/usr/bin/env python3
"""Compare CTC-Viterbi and MFA intervals for word-stress assessment.

Both conditions use the same Mallela-inspired sequential stress model and the
same utterance audio.  Only the phone boundaries passed to the feature
extractor differ.  Evaluation uses SpeechOcean762's human word-stress grade:
10 denotes acceptable stress and 5 denotes a stress error in this corpus.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

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
from server.services.stress_service import StressEvaluator
from server.utils.audio_features import is_vowel


def viterbi_intervals(evaluator: GOPEvaluator, encoded: dict) -> list[dict]:
    log_probs = encoded["log_probs"]
    unit_frames, _ = evaluator.viterbi_ctc_align(
        log_probs.detach().cpu().numpy(), encoded["label_ids"]
    )
    units = []
    for index in range(len(encoded["phonemes"])):
        frames = unit_frames.get(index, [])
        if frames:
            first, last = frames[0], frames[-1]
        else:
            first = last = int(index * len(log_probs) / len(encoded["phonemes"]))
        units.append((first, last))

    seconds_per_frame = encoded["duration"] / len(log_probs)
    result = []
    for index, (first, last) in enumerate(units):
        start_frame = 0 if index == 0 else (units[index - 1][1] + 1 + first) // 2
        end_frame = (
            len(log_probs)
            if index == len(units) - 1
            else (last + 1 + units[index + 1][0]) // 2
        )
        result.append({
            "phoneme": encoded["phonemes"][index],
            "start": start_frame * seconds_per_frame,
            "end": min(encoded["duration"], end_frame * seconds_per_frame),
        })
    return result


def word_offsets(record: dict) -> list[tuple[int, int]]:
    result, offset = [], 0
    for word in record["words"]:
        end = offset + len(word["phones"])
        result.append((offset, end))
        offset = end
    return result


def primary_probability(result: dict) -> float:
    truth = np.asarray(result["truth"], dtype=int)
    probabilities = np.asarray(result["raw_scores"], dtype=float)
    primary = np.flatnonzero(truth == 1)
    if not len(primary) or len(probabilities) != len(truth):
        return float("nan")
    return float(probabilities[primary[0]])


def run(args: argparse.Namespace) -> None:
    scores = json.loads((args.dataset / "resource" / "scores.json").read_text())
    wavs = read_kaldi_map(args.dataset / "test" / "wav.scp")
    speakers = read_kaldi_map(args.dataset / "test" / "utt2spk")
    textgrids = {
        path.stem: path for path in sorted(args.mfa_textgrids.rglob("*.TextGrid"))
    }

    gop = GOPEvaluator(args.gop_model)
    stress = StressEvaluator(args.stress_model, args.scaler)
    rows, failures = [], []
    started = time.perf_counter()
    for utterance_index, utterance in enumerate(sorted(textgrids), start=1):
        record = scores[utterance]
        canonical, _ = flatten_annotation(record)
        normalized = [normalize_phone(phone) for phone in canonical]
        mfa = parse_phone_textgrid(textgrids[utterance])
        matches = sequence_matches(normalized, [phone for phone, _, _ in mfa])
        mfa_by_canonical = {canonical_index: mfa_index for canonical_index, mfa_index in matches}
        audio_path = args.dataset / wavs[utterance]
        audio_bytes = audio_path.read_bytes()
        try:
            encoded = gop._encode_utterance(
                audio_bytes, record["text"], target_phonemes=canonical
            )
            ctc_intervals = viterbi_intervals(gop, encoded)
            waveform, _ = librosa.load(audio_path, sr=16000)
            for word_index, (word, (start, end)) in enumerate(
                zip(record["words"], word_offsets(record))
            ):
                # Primary-stress location is meaningful only for polysyllables.
                if sum(is_vowel(phone) for phone in word["phones"]) < 2:
                    continue
                if not all(index in mfa_by_canonical for index in range(start, end)):
                    continue
                alignments = {
                    "ctc_viterbi": ctc_intervals[start:end],
                    "mfa": [
                        {
                            "phoneme": mfa[mfa_by_canonical[index]][0],
                            "start": mfa[mfa_by_canonical[index]][1],
                            "end": mfa[mfa_by_canonical[index]][2],
                        }
                        for index in range(start, end)
                    ],
                }
                outputs = {}
                for condition in ("ctc_viterbi", "mfa"):
                    outputs[condition] = stress.predict(
                        waveform,
                        word["text"],
                        alignments=alignments[condition],
                        method=condition,
                    )
                    if "error" in outputs[condition]:
                        raise RuntimeError(outputs[condition]["error"])
                row = {
                    "utterance": utterance,
                    "speaker": speakers[utterance],
                    "word_index": word_index,
                    "word": word["text"],
                    "human_stress": float(word["stress"]),
                    "human_correct": int(float(word["stress"]) == 10.0),
                }
                for condition in ("ctc_viterbi", "mfa"):
                    output = outputs[condition]
                    row[f"{condition}_primary_probability"] = primary_probability(output)
                    row[f"{condition}_canonical_location_correct"] = int(
                        output["truth"] == output["infer"]
                    )
                if all(
                    np.isfinite(row[f"{condition}_primary_probability"])
                    for condition in ("ctc_viterbi", "mfa")
                ):
                    rows.append(row)
        except Exception as exc:
            failures.append({"utterance": utterance, "error": repr(exc)})
        if utterance_index % args.progress_every == 0:
            print(
                f"processed {utterance_index}/{len(textgrids)}; "
                f"words={len(rows)} failures={len(failures)}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    metrics = {}
    for condition in ("ctc_viterbi", "mfa"):
        probability = frame[f"{condition}_primary_probability"].to_numpy()
        human_grade = frame["human_stress"].to_numpy()
        human_correct = frame["human_correct"].to_numpy()
        correct_subset = frame[frame["human_correct"] == 1]
        metrics[condition] = {
            "human_stress_grade_pcc": float(pearsonr(probability, human_grade).statistic),
            "human_stress_grade_srcc": float(spearmanr(probability, human_grade).statistic),
            "human_correct_stress_auroc": float(roc_auc_score(human_correct, probability)),
            "human_correct_stress_average_precision": float(
                average_precision_score(human_correct, probability)
            ),
            "canonical_location_accuracy_when_human_correct": float(
                correct_subset[f"{condition}_canonical_location_correct"].mean()
            ),
        }

    # Paired speaker bootstrap for the central stress-location comparison.
    correct = frame[frame["human_correct"] == 1]
    speaker_ids = np.asarray(sorted(correct["speaker"].unique()))
    grouped = {speaker: correct[correct["speaker"] == speaker] for speaker in speaker_ids}
    rng = np.random.default_rng(args.seed)
    differences = []
    for _ in range(args.bootstrap_iterations):
        sampled = rng.choice(speaker_ids, size=len(speaker_ids), replace=True)
        bootstrap = pd.concat(
            [grouped[speaker].assign(_sample=index) for index, speaker in enumerate(sampled)],
            ignore_index=True,
        )
        differences.append(
            bootstrap["ctc_viterbi_canonical_location_correct"].mean()
            - bootstrap["mfa_canonical_location_correct"].mean()
        )
    result = {
        "protocol": {
            "dataset": str(args.dataset),
            "split": "test",
            "mfa_textgrids": str(args.mfa_textgrids),
            "gop_model": gop.model_name,
            "stress_model": str(args.stress_model),
            "comparison": "paired words; same stress model/features; phone interval source only differs",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "counts": {
            "aligned_utterances": len(textgrids),
            "paired_polysyllabic_words": len(frame),
            "human_correct_words": int(frame["human_correct"].sum()),
            "human_stress_error_words": int((frame["human_correct"] == 0).sum()),
            "failures": len(failures),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "stress_accuracy": metrics,
        "paired_location_accuracy_difference": {
            "metric": "ctc_viterbi_minus_mfa",
            "point": (
                metrics["ctc_viterbi"]["canonical_location_accuracy_when_human_correct"]
                - metrics["mfa"]["canonical_location_accuracy_when_human_correct"]
            ),
            "95_ci": [float(value) for value in np.quantile(differences, [0.025, 0.975])],
            "iterations": args.bootstrap_iterations,
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    frame.to_csv(args.output.with_suffix(".words.csv"), index=False)
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mfa-textgrids", type=Path, required=True)
    parser.add_argument("--gop-model", default="server/models/ctcgop")
    parser.add_argument(
        "--stress-model", type=Path, default=Path("server/models/sylstress/stress_model.keras")
    )
    parser.add_argument(
        "--scaler", type=Path, default=Path("server/models/sylstress/scaler_params.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
