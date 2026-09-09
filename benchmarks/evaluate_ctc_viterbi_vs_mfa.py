#!/usr/bin/env python3
"""Paired phone assessment with embedded CTC-Viterbi and MFA boundaries.

The acoustic model, frame-level phone score, target sequence, calibration
procedure, and test phones are held constant.  Only the source of the phone
intervals changes:

* ``ctc_viterbi``: transcript-conditioned Viterbi path through the CTC lattice;
* ``mfa``: phone intervals exported by Montreal Forced Aligner.

SpeechOcean762's train split supplies calibration data. Train and test MFA
intervals are read from TextGrid exports made with the same MFA configuration.
Results are restricted to phones successfully matched in both conditions,
enabling paired bootstrap inference. A legacy train CSV remains accepted for
backward-compatible reproduction of earlier runs.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.services.gop_service import GOPEvaluator


PHONE_MERGERS = {
    "AO": "AA",
    "AX": "AH",
    "AXR": "ER",
    "IX": "IH",
    "UX": "UW",
    "EL": "L",
    "EM": "M",
    "EN": "N",
    "NX": "N",
    "ENG": "NG",
    "DX": "T",
    "HV": "HH",
}


def normalize_phone(phone: str) -> str:
    phone = re.sub(r"\d", "", str(phone)).strip().upper()
    return PHONE_MERGERS.get(phone, phone)


def read_kaldi_map(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(maxsplit=1)
        result[key] = value
    return result


def flatten_annotation(record: dict) -> tuple[list[str], list[float]]:
    phones, grades = [], []
    for word in record["words"]:
        phones.extend(word["phones"])
        grades.extend(map(float, word["phones-accuracy"]))
    return phones, grades


def parse_phone_textgrid(path: Path) -> list[tuple[str, float, float]]:
    """Read non-empty intervals from the phones tier of a long TextGrid."""

    lines = path.read_text(encoding="utf-8").splitlines()
    in_phone_tier = False
    intervals = []
    start = end = None
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("name ="):
            in_phone_tier = line.split("=", 1)[1].strip().strip('"') == "phones"
            start = end = None
            continue
        if not in_phone_tier:
            continue
        if line.startswith("xmin ="):
            start = float(line.split("=", 1)[1])
        elif line.startswith("xmax ="):
            end = float(line.split("=", 1)[1])
        elif line.startswith("text ="):
            phone = line.split("=", 1)[1].strip().strip('"')
            if phone and start is not None and end is not None:
                intervals.append((normalize_phone(phone), start, end))
            start = end = None
    return intervals


def sequence_matches(
    canonical: list[str], aligned: list[str]
) -> list[tuple[int, int]]:
    """Globally match phone sequences, preferring exact diagonal matches."""

    n, m = len(canonical), len(aligned)
    costs = np.zeros((n + 1, m + 1), dtype=np.int32)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    costs[:, 0] = np.arange(n + 1)
    costs[0, :] = np.arange(m + 1)
    back[1:, 0] = 1
    back[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = costs[i - 1, j - 1] + (
                canonical[i - 1] != aligned[j - 1]
            )
            deletion = costs[i - 1, j] + 1
            insertion = costs[i, j - 1] + 1
            choices = (substitution, deletion, insertion)
            best = int(np.argmin(choices))
            costs[i, j] = choices[best]
            back[i, j] = best

    matches = []
    i, j = n, m
    while i or j:
        move = back[i, j]
        if i and j and move == 0:
            if canonical[i - 1] == aligned[j - 1]:
                matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and (j == 0 or move == 1):
            i -= 1
        else:
            j -= 1
    return list(reversed(matches))


def train_intervals(path: Path) -> dict[str, list[tuple[str, float, float]]]:
    frame = pd.read_csv(path, dtype={"file_name": str})
    frame["file_name"] = frame["file_name"].str.zfill(9)
    frame = frame[~frame["phone"].str.lower().isin({"sil", "sp", "spn"})]
    result = {}
    for utterance, rows in frame.groupby("file_name", sort=False):
        rows = rows.sort_values("phone_interval_id")
        result[utterance] = [
            (normalize_phone(row.phone), float(row.begin), float(row.end))
            for row in rows.itertuples()
        ]
    return result


def test_intervals(path: Path) -> dict[str, list[tuple[str, float, float]]]:
    return {
        textgrid.stem: parse_phone_textgrid(textgrid)
        for textgrid in sorted(path.rglob("*.TextGrid"))
    }


def phone_lpp(log_probs: torch.Tensor, token_id: int, frames: list[int]) -> float:
    valid = [frame for frame in frames if 0 <= frame < len(log_probs)]
    if not valid:
        return float("nan")
    return float(log_probs[valid, token_id].mean().detach().cpu())


def paired_scores(
    evaluator: GOPEvaluator,
    audio: bytes,
    transcript: str,
    canonical_phones: list[str],
    human_grades: list[float],
    mfa_intervals: list[tuple[str, float, float]],
) -> tuple[list[dict], dict]:
    encoded = evaluator._encode_utterance(
        audio, transcript, target_phonemes=canonical_phones
    )
    log_probs = encoded["log_probs"]
    normalized = [normalize_phone(phone) for phone in encoded["phonemes"]]
    unit_frames, _ = evaluator.viterbi_ctc_align(
        log_probs.detach().cpu().numpy(), encoded["label_ids"]
    )
    mfa_phones = [phone for phone, _, _ in mfa_intervals]
    matches = sequence_matches(normalized, mfa_phones)
    seconds_per_frame = encoded["duration"] / len(log_probs)

    rows = []
    for canonical_index, mfa_index in matches:
        phone, start, end = mfa_intervals[mfa_index]
        # Select CTC frame centers that lie in the MFA interval.  A nearest
        # frame fallback handles very short MFA phone intervals.
        mfa_frames = [
            frame
            for frame in range(len(log_probs))
            if start <= (frame + 0.5) * seconds_per_frame < end
        ]
        if not mfa_frames:
            nearest = int(round(((start + end) / 2) / seconds_per_frame - 0.5))
            mfa_frames = [max(0, min(len(log_probs) - 1, nearest))]
        viterbi_frames = unit_frames.get(canonical_index, [])
        token_id = encoded["label_ids"][canonical_index]
        viterbi_lpp = phone_lpp(log_probs, token_id, viterbi_frames)
        mfa_lpp = phone_lpp(log_probs, token_id, mfa_frames)
        if np.isfinite(viterbi_lpp) and np.isfinite(mfa_lpp):
            rows.append({
                "phone_index": canonical_index,
                "phone": phone,
                "human_grade": human_grades[canonical_index],
                "ctc_viterbi_lpp": viterbi_lpp,
                "mfa_lpp": mfa_lpp,
                "ctc_viterbi_start": min(viterbi_frames) * seconds_per_frame,
                "ctc_viterbi_end": (max(viterbi_frames) + 1) * seconds_per_frame,
                "mfa_start": start,
                "mfa_end": end,
            })
    diagnostics = {
        "canonical_phones": len(normalized),
        "mfa_phones": len(mfa_intervals),
        "matched_phones": len(rows),
    }
    return rows, diagnostics


def quadratic_predictions(
    train: pd.DataFrame, test: pd.DataFrame, score_column: str
) -> np.ndarray:
    predictions = np.empty(len(test), dtype=np.float64)
    global_x = train[score_column].to_numpy(dtype=np.float64)
    global_y = train["human_grade"].to_numpy(dtype=np.float64)
    global_model = LinearRegression().fit(
        np.column_stack([global_x, global_x**2]), global_y
    )
    positions = pd.Series(np.arange(len(test)), index=test.index)
    for phone, indexes in test.groupby("phone").groups.items():
        phone_train = train[train["phone"] == phone]
        model = global_model
        if len(phone_train) >= 20 and phone_train[score_column].nunique() >= 3:
            x = phone_train[score_column].to_numpy(dtype=np.float64)
            y = phone_train["human_grade"].to_numpy(dtype=np.float64)
            model = LinearRegression().fit(np.column_stack([x, x**2]), y)
        values = test.loc[indexes, score_column].to_numpy(dtype=np.float64)
        predictions[positions.loc[indexes]] = model.predict(
            np.column_stack([values, values**2])
        )
    return np.clip(predictions, 0.0, 2.0)


def correlations(prediction: np.ndarray, target: np.ndarray) -> dict:
    return {
        "pcc": float(pearsonr(prediction, target).statistic),
        "srcc": float(spearmanr(prediction, target).statistic),
    }


def paired_speaker_bootstrap(
    frame: pd.DataFrame, iterations: int, seed: int
) -> dict:
    speakers = np.asarray(sorted(frame["speaker"].unique()))
    grouped = {speaker: frame[frame["speaker"] == speaker] for speaker in speakers}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        sampled = rng.choice(speakers, size=len(speakers), replace=True)
        bootstrap = pd.concat(
            [grouped[speaker].assign(_sample=index) for index, speaker in enumerate(sampled)],
            ignore_index=True,
        )
        target = bootstrap["human_grade"].to_numpy()
        viterbi = pearsonr(bootstrap["ctc_viterbi_prediction"], target).statistic
        mfa = pearsonr(bootstrap["mfa_prediction"], target).statistic
        differences.append(viterbi - mfa)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "metric": "ctc_viterbi_pcc_minus_mfa_pcc",
        "mean": float(np.mean(differences)),
        "95_ci": [float(lower), float(upper)],
        "iterations": iterations,
    }


def aggregate(values: pd.Series) -> dict:
    array = values.to_numpy(dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def run(args: argparse.Namespace) -> None:
    scores = json.loads((args.dataset / "resource" / "scores.json").read_text())
    split_maps = {
        split: {
            "wavs": read_kaldi_map(args.dataset / split / "wav.scp"),
            "speakers": read_kaldi_map(args.dataset / split / "utt2spk"),
        }
        for split in ("train", "test")
    }
    if args.train_mfa_textgrids is not None:
        train_source = args.train_mfa_textgrids
        train_mfa = test_intervals(args.train_mfa_textgrids)
    elif args.train_mfa_csv is not None:
        train_source = args.train_mfa_csv
        train_mfa = train_intervals(args.train_mfa_csv)
    else:
        raise ValueError("Provide --train-mfa-textgrids or --train-mfa-csv")
    intervals = {"train": train_mfa, "test": test_intervals(args.test_mfa_textgrids)}
    evaluator = GOPEvaluator(args.model)
    evaluator._audit_log = lambda *unused_args, **unused_kwargs: None

    rows, diagnostics, failures = [], [], []
    started = time.perf_counter()
    for split in ("train", "test"):
        utterance_ids = sorted(intervals[split])
        for index, utterance in enumerate(utterance_ids, start=1):
            if utterance not in scores or utterance not in split_maps[split]["wavs"]:
                failures.append({
                    "split": split,
                    "utterance": utterance,
                    "error": "utterance absent from SpeechOcean split metadata",
                })
                continue
            record = scores[utterance]
            phones, grades = flatten_annotation(record)
            try:
                scored, diagnostic = paired_scores(
                    evaluator,
                    (args.dataset / split_maps[split]["wavs"][utterance]).read_bytes(),
                    record["text"],
                    phones,
                    grades,
                    intervals[split][utterance],
                )
                for row in scored:
                    row.update({
                        "split": split,
                        "utterance": utterance,
                        "speaker": split_maps[split]["speakers"][utterance],
                    })
                    rows.append(row)
                diagnostics.append({
                    "split": split,
                    "utterance": utterance,
                    **diagnostic,
                })
            except Exception as exc:
                failures.append({
                    "split": split,
                    "utterance": utterance,
                    "error": repr(exc),
                })
            if index % args.progress_every == 0:
                print(
                    f"{split}: {index}/{len(utterance_ids)}; failures={len(failures)}",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    diagnostics_frame = pd.DataFrame(diagnostics)
    train = frame[frame["split"] == "train"].copy()
    test = frame[frame["split"] == "test"].copy()
    for condition in ("ctc_viterbi", "mfa"):
        test[f"{condition}_prediction"] = quadratic_predictions(
            train, test, f"{condition}_lpp"
        )

    target = test["human_grade"].to_numpy()
    metrics = {
        condition: {
            "raw": correlations(test[f"{condition}_lpp"].to_numpy(), target),
            "phone_quadratic": correlations(
                test[f"{condition}_prediction"].to_numpy(), target
            ),
        }
        for condition in ("ctc_viterbi", "mfa")
    }
    paired_bootstrap = paired_speaker_bootstrap(
        test, args.bootstrap_iterations, args.seed
    )
    boundary_mae = (
        (
            (test["ctc_viterbi_start"] - test["mfa_start"]).abs()
            + (test["ctc_viterbi_end"] - test["mfa_end"]).abs()
        )
        / 2
    )
    result = {
        "protocol": {
            "dataset": str(args.dataset),
            "train_mfa_intervals": str(train_source),
            "test_mfa_intervals": str(args.test_mfa_textgrids),
            "model": evaluator.model_name,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "calibration": "per-phone quadratic; train only; global fallback below 20 samples",
            "comparison": "paired phones; same CTC emissions and LPP; interval source only differs",
        },
        "counts": {
            "train_utterances": int(train["utterance"].nunique()),
            "test_utterances": int(test["utterance"].nunique()),
            "train_phones": int(len(train)),
            "test_phones": int(len(test)),
            "failures": len(failures),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "matching": {
            split: {
                "canonical_phones": int(
                    diagnostics_frame.loc[
                        diagnostics_frame["split"] == split, "canonical_phones"
                    ].sum()
                ),
                "mfa_phones": int(
                    diagnostics_frame.loc[
                        diagnostics_frame["split"] == split, "mfa_phones"
                    ].sum()
                ),
                "matched_phones": int(
                    diagnostics_frame.loc[
                        diagnostics_frame["split"] == split, "matched_phones"
                    ].sum()
                ),
            }
            for split in ("train", "test")
        },
        "phone_accuracy": metrics,
        "paired_inference": paired_bootstrap,
        "mfa_boundary_disagreement_seconds": aggregate(boundary_mae),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    frame.to_csv(args.output.with_suffix(".phones.csv"), index=False)
    diagnostics_frame.to_csv(
        args.output.with_suffix(".matching.csv"), index=False
    )
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-mfa-csv", type=Path)
    parser.add_argument("--train-mfa-textgrids", type=Path)
    parser.add_argument("--test-mfa-textgrids", type=Path, required=True)
    parser.add_argument("--model", default="server/models/ctcgop")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
