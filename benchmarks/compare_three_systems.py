"""Reproduce and compare three phone-and-stress assessment systems.

Systems
-------
1. ``mfa_modular``: MFA-aligned 78-D LPP/LPR phone GOPT plus the fixed
   Mallela-inspired stress network using MFA phone intervals.
2. ``cao_viterbi`` (proposed): Cao et al.'s alignment-free CTC vector plus a
   phone-only GOPT, and the same stress network using embedded CTC-Viterbi
   intervals.
3. ``gopt_joint``: a Gong et al. multi-task GOPT adaptation that predicts
   phone accuracy and word stress jointly from the same MFA-aligned vectors.

The feature and training stages are separate because the released Cao acoustic
model is large.  Every generated artifact records its protocol and can be
resumed without recomputing acoustic emissions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.evaluate_ctc_viterbi_vs_mfa import (
    flatten_annotation,
    parse_phone_textgrid,
    read_kaldi_map,
)
from benchmarks.three_system_models import (
    GOPTJoint,
    GOPTPhone,
    masked_mse,
    parameter_count,
)
from server.utils.audio_features import is_vowel

MAX_PHONES = 50
PHONE_MERGERS = {
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


def clean_phone(phone: str) -> str:
    phone = re.sub(r"\d", "", str(phone)).strip().upper()
    return PHONE_MERGERS.get(phone, phone)


def sequence_diagonal_pairs(
    canonical: list[str], aligned: list[str]
) -> list[tuple[int, int]]:
    """Edit-distance alignment, retaining exact and substituted diagonals.

    MFA can choose a pronunciation-dictionary variant whose phone name differs
    from SpeechOcean's expert canonical sequence. A diagonal edit still has a
    usable MFA interval; relabeling that interval with the expert canonical
    phone is stronger and less selective than discarding it.
    """

    n, m = len(canonical), len(aligned)
    costs = np.zeros((n + 1, m + 1), dtype=np.int32)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    costs[:, 0] = np.arange(n + 1)
    costs[0, :] = np.arange(m + 1)
    back[1:, 0] = 1
    back[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (
                costs[i - 1, j - 1] + (canonical[i - 1] != aligned[j - 1]),
                costs[i - 1, j] + 1,
                costs[i, j - 1] + 1,
            )
            move = int(np.argmin(choices))
            costs[i, j] = choices[move]
            back[i, j] = move
    pairs = []
    i, j = n, m
    while i or j:
        move = back[i, j]
        if i and j and move == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i and (j == 0 or move == 1):
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sample_rate != 16000:
        audio = librosa.resample(
            audio.astype(np.float32), orig_sr=sample_rate, target_sr=16000
        )
    return np.asarray(audio, dtype=np.float32)


@dataclass
class EmissionExtractor:
    model_path: Path
    processor_path: Path
    device_name: str = "auto"

    def __post_init__(self) -> None:
        self.processor = Wav2Vec2Processor.from_pretrained(
            str(self.processor_path)
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(str(self.model_path))
        if self.device_name == "auto":
            self.device_name = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(self.device_name)
        self.model.eval().to(self.device)
        self.vocab = self.processor.tokenizer.get_vocab()
        self.blank_id = int(self.processor.tokenizer.pad_token_id or 0)
        specials = {
            "<pad>", "<unk>", "<s>", "</s>", "[PAD]", "[UNK]", "|"
        }
        labels = []
        for token, token_id in self.vocab.items():
            if token not in specials and clean_phone(token):
                labels.append((int(token_id), clean_phone(token)))
        self.phone_ids = [token_id for token_id, _ in sorted(labels)]
        self.phone_names = [phone for _, phone in sorted(labels)]
        self.name_to_id = {
            phone: token_id for token_id, phone in zip(self.phone_ids, self.phone_names)
        }
        if len(self.phone_ids) != 39:
            raise ValueError(
                f"Cao feature definition expects 39 phones; found {len(self.phone_ids)}"
            )

    def encode_batch(
        self, audios: list[np.ndarray], canonicals: list[list[str]]
    ) -> list[dict]:
        phone_lists = [[clean_phone(phone) for phone in item] for item in canonicals]
        missing = sorted(
            set().union(*map(set, phone_lists)) - set(self.name_to_id)
        )
        if missing:
            raise ValueError(f"Phones absent from Cao model vocabulary: {missing}")
        inputs = self.processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        with torch.inference_mode():
            logits = self.model(
                input_values=inputs.input_values.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
            ).logits
            if self.device.type == "mps":
                torch.mps.synchronize()
            logits = logits.cpu()
        output_lengths = self.model._get_feat_extract_output_lengths(
            inputs.attention_mask.sum(dim=1)
        ).tolist()
        results = []
        for audio, phones, logit, output_length in zip(
            audios, phone_lists, logits, output_lengths
        ):
            results.append({
                "phones": phones,
                "label_ids": [self.name_to_id[phone] for phone in phones],
                "log_probs": F.log_softmax(logit[: int(output_length)], dim=-1),
                "duration": len(audio) / 16000.0,
                "blank_id": self.blank_id,
            })
        return results

    def encode(self, audio: np.ndarray, canonical: list[str]) -> dict:
        return self.encode_batch([audio], [canonical])[0]


def ctc_nlls(
    log_probs: torch.Tensor, targets: list[list[int]], *, blank_id: int = 0
) -> torch.Tensor:
    """Compute many CTC negative log likelihoods against one emission matrix."""

    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    flattened = torch.tensor(
        [token for target in targets for token in target], dtype=torch.long
    )
    count = len(targets)
    expanded = log_probs[:, None, :].expand(-1, count, -1).contiguous()
    input_lengths = torch.full((count,), len(log_probs), dtype=torch.long)
    return F.ctc_loss(
        expanded,
        flattened,
        input_lengths,
        target_lengths,
        blank=blank_id,
        reduction="none",
        zero_infinity=False,
    )


def cao_alignment_free_features(
    log_probs: torch.Tensor,
    labels: list[int],
    phone_ids: list[int],
    blank_id: int = 0,
) -> np.ndarray:
    """Cao GOP-feature-CTC-AF: LPP plus deletion/substitution LPRs."""

    canonical_nll = ctc_nlls(log_probs, [labels], blank_id=blank_id)[0]
    alternatives = []
    for index in range(len(labels)):
        alternatives.append(labels[:index] + labels[index + 1 :])
        for phone_id in phone_ids:
            alternatives.append(
                labels[:index] + [phone_id] + labels[index + 1 :]
            )
    alternative_nll = ctc_nlls(
        log_probs, alternatives, blank_id=blank_id
    ).reshape(
        len(labels), 1 + len(phone_ids)
    )
    # Cao's released extractor stores the Hugging Face CTC loss (negative log
    # probability) as the first component. Its LPR components are
    # log P(canonical) - log P(alternative) = alt NLL - canonical NLL.
    lpp = canonical_nll.expand(len(labels), 1)
    features = torch.cat((lpp, alternative_nll - canonical_nll), dim=1)
    return features.detach().cpu().numpy().astype(np.float32)


def viterbi_regions(
    log_probs: torch.Tensor,
    labels: list[int],
    phones: list[str],
    duration: float,
    blank_id: int = 0,
) -> list[dict]:
    """Maximum-probability path through the canonical CTC trellis."""

    emissions = log_probs.detach().cpu().numpy()
    time_steps = len(emissions)
    states = 2 * len(labels) + 1
    state_tokens = np.asarray(
        [blank_id if state % 2 == 0 else labels[(state - 1) // 2]
         for state in range(states)],
        dtype=np.int32,
    )
    scores = np.full((time_steps, states), -np.inf, dtype=np.float64)
    back = np.zeros((time_steps, states), dtype=np.int32)
    scores[0, 0] = emissions[0, blank_id]
    scores[0, 1] = emissions[0, labels[0]]
    for frame in range(1, time_steps):
        for state in range(max(0, states - 2 * (time_steps - frame)),
                           min(states, 2 * (frame + 1))):
            candidates = [(scores[frame - 1, state], state)]
            if state:
                candidates.append((scores[frame - 1, state - 1], state - 1))
            if (state % 2 == 1 and state >= 2
                    and state_tokens[state] != state_tokens[state - 2]):
                candidates.append((scores[frame - 1, state - 2], state - 2))
            value, previous = max(candidates, key=lambda item: item[0])
            scores[frame, state] = value + emissions[frame, state_tokens[state]]
            back[frame, state] = previous
    terminal = max((states - 1, states - 2), key=lambda state: scores[-1, state])
    if not np.isfinite(scores[-1, terminal]):
        raise ValueError("No valid canonical CTC path")
    path = np.zeros(time_steps, dtype=np.int32)
    path[-1] = terminal
    for frame in range(time_steps - 2, -1, -1):
        path[frame] = back[frame + 1, path[frame + 1]]
    runs = []
    for index in range(len(labels)):
        frames = np.flatnonzero(path == 2 * index + 1)
        if not len(frames):
            raise ValueError(f"CTC path skipped phone {index}")
        runs.append((int(frames[0]), int(frames[-1])))
    seconds_per_frame = duration / time_steps
    regions = []
    for index, (first, last) in enumerate(runs):
        start = 0 if index == 0 else (runs[index - 1][1] + 1 + first) // 2
        end = time_steps if index == len(runs) - 1 else (
            last + 1 + runs[index + 1][0]
        ) // 2
        regions.append({
            "phoneme": phones[index],
            "start": float(start * seconds_per_frame),
            "end": float(min(duration, end * seconds_per_frame)),
        })
    return regions


def mfa_aligned_features(
    encoded: dict,
    mfa: list[tuple[str, float, float]],
    phone_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, list[dict | None]]:
    """MFA-aligned full LPP/LPR vector, analogous to GOPT's GOP input."""

    canonical = encoded["phones"]
    matches = sequence_diagonal_pairs(
        canonical, [clean_phone(phone) for phone, _, _ in mfa]
    )
    by_canonical = {canonical_index: mfa_index for canonical_index, mfa_index in matches}
    features = np.zeros((len(canonical), 2 * len(phone_ids)), dtype=np.float32)
    mask = np.zeros(len(canonical), dtype=bool)
    intervals: list[dict | None] = [None] * len(canonical)
    seconds_per_frame = encoded["duration"] / len(encoded["log_probs"])
    class_ids = phone_ids
    for canonical_index, mfa_index in by_canonical.items():
        phone, start, end = mfa[mfa_index]
        frames = [
            frame for frame in range(len(encoded["log_probs"]))
            if start <= (frame + 0.5) * seconds_per_frame < end
        ]
        if not frames:
            nearest = round(((start + end) / 2) / seconds_per_frame - 0.5)
            frames = [max(0, min(len(encoded["log_probs"]) - 1, nearest))]
        means = encoded["log_probs"][frames][:, class_ids].mean(dim=0)
        target_id = encoded["label_ids"][canonical_index]
        target_position = phone_ids.index(target_id)
        target = means[target_position]
        features[canonical_index] = torch.cat(
            (means, means - target)
        ).detach().cpu().numpy()
        mask[canonical_index] = True
        intervals[canonical_index] = {
            "phoneme": canonical[canonical_index],
            "mfa_phoneme": clean_phone(phone),
            "start": float(start),
            "end": float(end),
        }
    return features, mask, intervals


def padded_metadata(record: dict, phone_to_index: dict[str, int]) -> dict:
    phones, phone_scores = flatten_annotation(record)
    if len(phones) > MAX_PHONES:
        raise ValueError(
            f"utterance has {len(phones)} phones; GOPT limit is {MAX_PHONES}"
        )
    phone_ids = np.full(MAX_PHONES, -1, dtype=np.int16)
    labels = np.full(MAX_PHONES, -1.0, dtype=np.float32)
    word_ids = np.full(MAX_PHONES, -1, dtype=np.int16)
    word_labels = np.full((MAX_PHONES, 3), -1.0, dtype=np.float32)
    offset = 0
    for word_index, word in enumerate(record["words"]):
        end = offset + len(word["phones"])
        word_ids[offset:end] = word_index
        word_labels[offset:end] = np.asarray(
            [word["accuracy"], word["stress"], word["total"]], dtype=np.float32
        ) / 5.0
        offset = end
    for index, (phone, score) in enumerate(zip(phones, phone_scores)):
        phone_ids[index] = phone_to_index[clean_phone(phone)]
        labels[index] = float(score)
    utterance = np.asarray([
        record["accuracy"], record["completeness"], record["fluency"],
        record["prosodic"], record["total"]
    ], dtype=np.float32) / 5.0
    return {
        "phone_ids": phone_ids,
        "phone_labels": labels,
        "word_ids": word_ids,
        "word_labels": word_labels,
        "utterance_labels": utterance,
    }


def generate_features(args: argparse.Namespace) -> None:
    extractor = EmissionExtractor(
        args.cao_model, args.cao_processor, device_name=args.device
    )
    phone_to_index = {phone: index for index, phone in enumerate(extractor.phone_names)}
    scores = json.loads((args.dataset / "resource" / "scores.json").read_text())
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for split, textgrid_root in (
        ("train", args.train_mfa_textgrids), ("test", args.test_mfa_textgrids)
    ):
        output = args.cache_dir / f"{split}_features.npz"
        if output.exists() and not args.overwrite:
            print(f"using existing {output}", flush=True)
            continue
        wavs = read_kaldi_map(args.dataset / split / "wav.scp")
        speakers = read_kaldi_map(args.dataset / split / "utt2spk")
        textgrids = {path.stem: path for path in textgrid_root.rglob("*.TextGrid")}
        utterance_ids = sorted(wavs)
        count = len(utterance_ids)
        af = np.zeros((count, MAX_PHONES, 41), dtype=np.float32)
        aligned = np.zeros((count, MAX_PHONES, 78), dtype=np.float32)
        mfa_mask = np.zeros((count, MAX_PHONES), dtype=bool)
        phone_ids = np.full((count, MAX_PHONES), -1, dtype=np.int16)
        phone_labels = np.full((count, MAX_PHONES), -1.0, dtype=np.float32)
        word_ids = np.full((count, MAX_PHONES), -1, dtype=np.int16)
        word_labels = np.full((count, MAX_PHONES, 3), -1.0, dtype=np.float32)
        utterance_labels = np.zeros((count, 5), dtype=np.float32)
        feature_mask = np.zeros((count, MAX_PHONES), dtype=bool)
        timings = np.full((count, 4), np.nan, dtype=np.float64)
        failures = []
        alignment_records = []
        started_all = time.perf_counter()
        for batch_start in range(0, count, args.feature_batch_size):
            batch_ids = utterance_ids[
                batch_start : batch_start + args.feature_batch_size
            ]
            batch_rows = list(range(batch_start, batch_start + len(batch_ids)))
            batch_audio, batch_canonical = [], []
            for row, utterance_id in zip(batch_rows, batch_ids):
                record = scores[utterance_id]
                canonical, _ = flatten_annotation(record)
                metadata = padded_metadata(record, phone_to_index)
                phone_ids[row] = metadata["phone_ids"]
                phone_labels[row] = metadata["phone_labels"]
                word_ids[row] = metadata["word_ids"]
                word_labels[row] = metadata["word_labels"]
                utterance_labels[row] = metadata["utterance_labels"]
                batch_audio.append(load_audio(args.dataset / wavs[utterance_id]))
                batch_canonical.append(canonical)
            try:
                started = time.perf_counter()
                batch_encoded = extractor.encode_batch(batch_audio, batch_canonical)
                encoding_ms = (
                    (time.perf_counter() - started) * 1000 / len(batch_encoded)
                )
            except Exception as exc:  # noqa: BLE001 - record per-batch failures
                failures.extend(
                    {"utterance": item, "stage": "batch_encoding", "error": repr(exc)}
                    for item in batch_ids
                )
                continue
            for row, utterance_id, encoded in zip(
                batch_rows, batch_ids, batch_encoded
            ):
                try:
                    timings[row, 0] = encoding_ms
                    started = time.perf_counter()
                    current_af = cao_alignment_free_features(
                        encoded["log_probs"], encoded["label_ids"],
                        extractor.phone_ids, extractor.blank_id
                    )
                    timings[row, 1] = (time.perf_counter() - started) * 1000
                    af[row, : len(current_af)] = current_af
                    feature_mask[row, : len(current_af)] = True
                    started = time.perf_counter()
                    current_viterbi = viterbi_regions(
                        encoded["log_probs"], encoded["label_ids"],
                        encoded["phones"], encoded["duration"], extractor.blank_id
                    )
                    timings[row, 2] = (time.perf_counter() - started) * 1000
                    current_mfa_intervals = [None] * len(encoded["phones"])
                    if utterance_id in textgrids:
                        started = time.perf_counter()
                        current_mfa, current_mask, current_mfa_intervals = (
                            mfa_aligned_features(
                                encoded,
                                parse_phone_textgrid(textgrids[utterance_id]),
                                extractor.phone_ids,
                            )
                        )
                        timings[row, 3] = (time.perf_counter() - started) * 1000
                        aligned[row, : len(current_mfa)] = current_mfa
                        mfa_mask[row, : len(current_mask)] = current_mask
                    if split == "test":
                        alignment_records.append({
                            "utterance": utterance_id,
                            "ctc_viterbi": current_viterbi,
                            "mfa": current_mfa_intervals,
                        })
                except Exception as exc:  # noqa: BLE001 - continue corpus audit
                    feature_mask[row] = False
                    failures.append({
                        "utterance": utterance_id,
                        "stage": "feature_extraction",
                        "error": repr(exc),
                    })
            completed = batch_start + len(batch_ids)
            if completed % args.progress_every < args.feature_batch_size:
                elapsed = time.perf_counter() - started_all
                print(
                    f"{split}: {completed}/{count}; failures={len(failures)}; "
                    f"elapsed={elapsed:.1f}s", flush=True
                )
        np.savez_compressed(
            output,
            utterance_ids=np.asarray(utterance_ids),
            speakers=np.asarray([speakers[item] for item in utterance_ids]),
            af_features=af,
            mfa_features=aligned,
            mfa_mask=mfa_mask,
            phone_ids=phone_ids,
            phone_labels=phone_labels,
            word_ids=word_ids,
            word_labels=word_labels,
            utterance_labels=utterance_labels,
            feature_mask=feature_mask,
            timings_ms=timings,
        )
        if split == "test":
            alignment_output = args.cache_dir / "test_alignment_regions.jsonl"
            alignment_output.write_text(
                "".join(json.dumps(item) + "\n" for item in alignment_records),
                encoding="utf-8",
            )
        output.with_suffix(".json").write_text(json.dumps({
            "split": split,
            "model": str(args.cao_model),
            "processor": str(args.cao_processor),
            "cao_paper": "https://www.isca-archive.org/interspeech_2024/cao24b_interspeech.html",
            "cao_repository": "https://github.com/xinweic/ctc-based-GOP",
            "device": str(extractor.device),
            "feature_batch_size": args.feature_batch_size,
            "feature_dimensions": {
                "cao_alignment_free": 41, "mfa_aligned_lpp_lpr": 78
            },
            "timing_columns": [
                "ctc_encoding", "cao_af_vectors", "ctc_viterbi", "mfa_vectors"
            ],
            "failures": failures,
            "elapsed_seconds": time.perf_counter() - started_all,
        }, indent=2) + "\n")


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


def generate_stress_rows(args: argparse.Namespace) -> None:
    """Score identical words with the same network and two boundary sources."""

    # TensorFlow is deliberately loaded only in this stage. Keeping it out of
    # Cao feature extraction avoids competing accelerator/runtime allocation.
    from server.services.stress_service import StressEvaluator

    scores = json.loads((args.dataset / "resource" / "scores.json").read_text())
    wavs = read_kaldi_map(args.dataset / "test" / "wav.scp")
    speakers = read_kaldi_map(args.dataset / "test" / "utt2spk")
    alignment_path = args.cache_dir / "test_alignment_regions.jsonl"
    alignments = {
        item["utterance"]: item
        for item in map(json.loads, alignment_path.read_text().splitlines())
    }
    stress = StressEvaluator(str(args.stress_model), str(args.stress_scaler))
    rows, failures = [], []
    started_all = time.perf_counter()
    utterance_ids = sorted(alignments)
    for utterance_index, utterance_id in enumerate(utterance_ids):
        record = scores[utterance_id]
        try:
            waveform = load_audio(args.dataset / wavs[utterance_id])
            for word_index, (word, (start, end)) in enumerate(
                zip(record["words"], word_offsets(record))
            ):
                if sum(is_vowel(phone) for phone in word["phones"]) < 2:
                    continue
                mfa = alignments[utterance_id]["mfa"][start:end]
                if len(mfa) != end - start or any(item is None for item in mfa):
                    continue
                regions = {
                    "ctc_viterbi": alignments[utterance_id]["ctc_viterbi"][start:end],
                    "mfa": mfa,
                }
                order = (
                    ("ctc_viterbi", "mfa")
                    if (utterance_index + word_index) % 2 == 0
                    else ("mfa", "ctc_viterbi")
                )
                outputs, latency = {}, {}
                for condition in order:
                    started = time.perf_counter()
                    outputs[condition] = stress.predict(
                        waveform,
                        word["text"],
                        alignments=regions[condition],
                        method=condition,
                        neural_weight=1.0,
                    )
                    latency[condition] = (time.perf_counter() - started) * 1000
                    if "error" in outputs[condition]:
                        raise RuntimeError(outputs[condition]["error"])
                row = {
                    "utterance": utterance_id,
                    "speaker": speakers[utterance_id],
                    "word_index": word_index,
                    "word": word["text"],
                    "human_stress": float(word["stress"]),
                    "human_correct": int(float(word["stress"]) == 10.0),
                }
                for condition in ("ctc_viterbi", "mfa"):
                    output = outputs[condition]
                    row[f"{condition}_primary_probability"] = (
                        primary_probability(output)
                    )
                    row[f"{condition}_canonical_location_correct"] = int(
                        output["truth"] == output["infer"]
                    )
                    row[f"{condition}_stress_latency_ms"] = latency[condition]
                if all(
                    np.isfinite(row[f"{condition}_primary_probability"])
                    for condition in ("ctc_viterbi", "mfa")
                ):
                    rows.append(row)
        except Exception as exc:  # noqa: BLE001 - retain a complete failure log
            failures.append({"utterance": utterance_id, "error": repr(exc)})
        if (utterance_index + 1) % args.progress_every == 0:
            print(
                f"stress: {utterance_index + 1}/{len(utterance_ids)}; "
                f"words={len(rows)}; failures={len(failures)}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    output = args.stress_output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    output.with_suffix(".json").write_text(json.dumps({
        "protocol": (
            "same Mallela-inspired neural checkpoint/features; interval source only; "
            "neural_weight=1.0 (no acoustic-prominence ensemble)"
        ),
        "stress_model": str(args.stress_model),
        "stress_scaler": str(args.stress_scaler),
        "paired_words": len(frame),
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started_all,
    }, indent=2) + "\n")
    print(f"wrote {len(frame)} paired words to {output}", flush=True)


def feature_statistics(
    features: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Training-set scalar normalization used by the official GOPT recipe."""

    selected = features[mask]
    mean = np.asarray(selected.mean(), dtype=np.float32)
    std = np.asarray(selected.std(), dtype=np.float32)
    if std < 1e-6:
        std = np.asarray(1.0, dtype=np.float32)
    return mean, std


def normalize_features(
    features: np.ndarray, mask: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    result = np.zeros_like(features, dtype=np.float32)
    result[mask] = (features[mask] - mean) / std
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def set_warmup_learning_rate(
    optimizer: torch.optim.Optimizer, step: int, learning_rate: float
) -> None:
    if step <= 100 and step % 5 == 0:
        value = (step / 100) * learning_rate
        for group in optimizer.param_groups:
            group["lr"] = value


def train_phone_model(
    train_features: np.ndarray,
    phone_ids: np.ndarray,
    train_labels: np.ndarray,
    train_mask: np.ndarray,
    *, seed: int, epochs: int, batch_size: int,
) -> GOPTPhone:
    set_seed(seed)
    model = GOPTPhone(train_features.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=5e-7,
                                 betas=(0.95, 0.999))
    schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(20, epochs, 5)), gamma=0.5
    )
    dataset = TensorDataset(
        torch.from_numpy(train_features), torch.from_numpy(phone_ids),
        torch.from_numpy(train_labels), torch.from_numpy(train_mask)
    )
    generator = torch.Generator().manual_seed(seed)
    global_step = 0
    for epoch in range(epochs):
        model.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=generator)
        for features, phones, labels, mask in loader:
            set_warmup_learning_rate(optimizer, global_step, 1e-3)
            loss = masked_mse(model(features, phones), labels, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
        if global_step > 100:
            schedule.step()
    return model.eval()


def train_joint_model(
    train_features: np.ndarray,
    train: dict[str, np.ndarray],
    train_mask: np.ndarray,
    *, seed: int, epochs: int, batch_size: int,
) -> GOPTJoint:
    set_seed(seed)
    model = GOPTJoint(train_features.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=5e-7,
                                 betas=(0.95, 0.999))
    schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(20, epochs, 5)), gamma=0.5
    )
    word_mask = np.zeros_like(train["word_labels"], dtype=bool)
    for utterance_index in range(len(train_mask)):
        for word_id in set(train["word_ids"][utterance_index]):
            if word_id < 0:
                continue
            positions = train["word_ids"][utterance_index] == word_id
            if train_mask[utterance_index, positions].all():
                word_mask[utterance_index, positions] = (
                    train["word_labels"][utterance_index, positions] >= 0
                )
    utterance_mask = train_mask.any(axis=1)[:, None].repeat(5, axis=1)
    dataset = TensorDataset(
        torch.from_numpy(train_features), torch.from_numpy(train["phone_ids"]),
        torch.from_numpy(train["phone_labels"]),
        torch.from_numpy(train["word_labels"]),
        torch.from_numpy(train["utterance_labels"]), torch.from_numpy(train_mask),
        torch.from_numpy(word_mask), torch.from_numpy(utterance_mask),
    )
    generator = torch.Generator().manual_seed(seed)
    global_step = 0
    for epoch in range(epochs):
        model.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=generator)
        for (
            features, phones, phone_labels, word_labels, utterance_labels,
            phone_mask, current_word_mask, current_utterance_mask
        ) in loader:
            set_warmup_learning_rate(optimizer, global_step, 1e-3)
            phone, word, utterance = model(features, phones)
            loss = (
                masked_mse(phone, phone_labels, phone_mask)
                + masked_mse(word, word_labels, current_word_mask)
                + masked_mse(
                    utterance, utterance_labels, current_utterance_mask
                )
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
        if global_step > 100:
            schedule.step()
    return model.eval()


def correlations(prediction: np.ndarray, target: np.ndarray) -> dict:
    return {
        "pcc": float(pearsonr(prediction, target).statistic),
        "srcc": float(spearmanr(prediction, target).statistic),
        "mse": float(np.mean((prediction - target) ** 2)),
    }


def evaluate_phone(
    prediction: np.ndarray, labels: np.ndarray, common_mask: np.ndarray
) -> dict:
    return correlations(prediction[common_mask], labels[common_mask])


def pooled_word_predictions(
    predictions: np.ndarray, word_ids: np.ndarray, valid_mask: np.ndarray
) -> pd.DataFrame:
    rows = []
    for utterance_index in range(len(predictions)):
        for word_id in sorted(set(word_ids[utterance_index][valid_mask[utterance_index]])):
            if word_id < 0:
                continue
            mask = valid_mask[utterance_index] & (word_ids[utterance_index] == word_id)
            rows.append({
                "utterance_index": utterance_index,
                "word_index": int(word_id),
                "prediction": float(predictions[utterance_index][mask].mean()),
            })
    return pd.DataFrame(rows)


def stress_metrics(
    prediction: np.ndarray, stress_rows: pd.DataFrame
) -> dict:
    correct = stress_rows["human_correct"].to_numpy(dtype=int)
    return {
        **correlations(prediction, correct),
        "auroc": float(roc_auc_score(correct, prediction)),
        "average_precision": float(average_precision_score(correct, prediction)),
    }


def summarize_runs(runs: list[dict]) -> dict:
    keys = runs[0].keys()
    return {
        key: {
            "mean": float(np.mean([run[key] for run in runs])),
            "std": float(np.std([run[key] for run in runs], ddof=1))
            if len(runs) > 1 else 0.0,
            "values": [float(run[key]) for run in runs],
        }
        for key in keys
    }


def fixed_stress_metrics(stress_rows: pd.DataFrame, condition: str) -> dict:
    prediction = stress_rows[f"{condition}_primary_probability"].to_numpy()
    result = stress_metrics(prediction, stress_rows)
    result["canonical_location_accuracy_when_human_correct"] = float(
        stress_rows.loc[
            stress_rows["human_correct"] == 1,
            f"{condition}_canonical_location_correct",
        ].mean()
    )
    return result


def paired_stress_bootstrap(
    stress_rows: pd.DataFrame, *, iterations: int, seed: int
) -> dict:
    speakers = np.asarray(sorted(stress_rows["speaker"].unique()))
    speaker_index = {
        speaker: index for index, speaker in enumerate(speakers)
    }
    row_speakers = stress_rows["speaker"].map(speaker_index).to_numpy()
    human_correct = stress_rows["human_correct"].to_numpy()
    correct_mask = human_correct == 1
    location_delta = (
        stress_rows["ctc_viterbi_canonical_location_correct"].to_numpy()
        - stress_rows["mfa_canonical_location_correct"].to_numpy()
    )
    viterbi_probability = stress_rows[
        "ctc_viterbi_primary_probability"
    ].to_numpy()
    mfa_probability = stress_rows["mfa_primary_probability"].to_numpy()
    rng = np.random.default_rng(seed)
    location_differences, auroc_differences = [], []
    for _ in range(iterations):
        speaker_weights = rng.multinomial(
            len(speakers), np.full(len(speakers), 1.0 / len(speakers))
        )
        row_weights = speaker_weights[row_speakers]
        correct_weights = row_weights[correct_mask]
        if correct_weights.sum() > 0:
            location_differences.append(float(np.average(
                location_delta[correct_mask], weights=correct_weights
            )))
        positive_weight = row_weights[human_correct == 1].sum()
        negative_weight = row_weights[human_correct == 0].sum()
        if positive_weight > 0 and negative_weight > 0:
            auroc_differences.append(
                roc_auc_score(
                    human_correct,
                    viterbi_probability,
                    sample_weight=row_weights,
                )
                - roc_auc_score(
                    human_correct,
                    mfa_probability,
                    sample_weight=row_weights,
                )
            )
    return {
        "ctc_viterbi_minus_mfa_location_accuracy": {
            "point": float(
                stress_rows.loc[
                    stress_rows["human_correct"] == 1,
                    "ctc_viterbi_canonical_location_correct",
                ].mean()
                - stress_rows.loc[
                    stress_rows["human_correct"] == 1,
                    "mfa_canonical_location_correct",
                ].mean()
            ),
            "95_ci": [
                float(value)
                for value in np.quantile(location_differences, [0.025, 0.975])
            ],
        },
        "ctc_viterbi_minus_mfa_auroc": {
            "point": float(
                roc_auc_score(
                    stress_rows["human_correct"],
                    stress_rows["ctc_viterbi_primary_probability"],
                )
                - roc_auc_score(
                    stress_rows["human_correct"],
                    stress_rows["mfa_primary_probability"],
                )
            ),
            "95_ci": [
                float(value)
                for value in np.quantile(auroc_differences, [0.025, 0.975])
            ],
        },
        "iterations": iterations,
        "unit": "speaker",
    }


def train_and_evaluate(args: argparse.Namespace) -> None:
    train = dict(np.load(args.cache_dir / "train_features.npz"))
    test = dict(np.load(args.cache_dir / "test_features.npz"))
    train_phone_mask = (train["phone_labels"] >= 0) & train["feature_mask"]
    test_phone_mask = (test["phone_labels"] >= 0) & test["feature_mask"]
    common_test_mask = test_phone_mask & test["mfa_mask"]
    seeds = [args.seed + index for index in range(args.seeds)]
    stress_path = args.stress_rows or args.stress_output
    stress_rows = pd.read_csv(stress_path, dtype={"utterance": str})

    prepared = {}
    for name, column, mask in (
        ("mfa_modular", "mfa_features", train["mfa_mask"] & train_phone_mask),
        ("cao_viterbi", "af_features", train_phone_mask),
    ):
        mean, std = feature_statistics(train[column], mask)
        prepared[name] = {
            "train": normalize_features(train[column], mask, mean, std),
            "test": normalize_features(
                test[column],
                test_phone_mask if name == "cao_viterbi" else common_test_mask,
                mean, std
            ),
            "mean": mean, "std": std, "train_mask": mask,
        }

    model_dir = args.output.parent / "three_system_checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    all_predictions: dict[str, list[np.ndarray]] = {
        "mfa_modular": [], "cao_viterbi": [], "gopt_joint_phone": [],
        "gopt_joint_stress": []
    }
    per_seed = []
    for seed in seeds:
        aligned_phone = train_phone_model(
            prepared["mfa_modular"]["train"], train["phone_ids"],
            train["phone_labels"],
            prepared["mfa_modular"]["train_mask"], seed=seed,
            epochs=args.epochs, batch_size=args.batch_size
        )
        cao_phone = train_phone_model(
            prepared["cao_viterbi"]["train"], train["phone_ids"],
            train["phone_labels"],
            prepared["cao_viterbi"]["train_mask"], seed=seed,
            epochs=args.epochs, batch_size=args.batch_size
        )
        joint = train_joint_model(
            prepared["mfa_modular"]["train"], train,
            prepared["mfa_modular"]["train_mask"], seed=seed,
            epochs=args.epochs, batch_size=args.batch_size
        )
        with torch.inference_mode():
            aligned_prediction = aligned_phone(
                torch.from_numpy(prepared["mfa_modular"]["test"]),
                torch.from_numpy(test["phone_ids"]),
            ).numpy()
            cao_prediction = cao_phone(
                torch.from_numpy(prepared["cao_viterbi"]["test"]),
                torch.from_numpy(test["phone_ids"]),
            ).numpy()
            joint_phone, joint_word, _ = joint(
                torch.from_numpy(prepared["mfa_modular"]["test"]),
                torch.from_numpy(test["phone_ids"]),
            )
            joint_phone = joint_phone.numpy()
            joint_stress = joint_word[..., 1].numpy()
        all_predictions["mfa_modular"].append(aligned_prediction)
        all_predictions["cao_viterbi"].append(cao_prediction)
        all_predictions["gopt_joint_phone"].append(joint_phone)
        all_predictions["gopt_joint_stress"].append(joint_stress)

        joint_words = pooled_word_predictions(
            joint_stress, test["word_ids"], common_test_mask
        )
        joint_words["utterance"] = joint_words["utterance_index"].map(
            lambda index: str(test["utterance_ids"][index])
        )
        joint_lookup = {
            (row.utterance, int(row.word_index)): row.prediction
            for row in joint_words.itertuples()
        }
        joint_stress_for_common_words = np.asarray([
            joint_lookup.get((row.utterance, int(row.word_index)), np.nan)
            for row in stress_rows.itertuples()
        ])
        if not np.isfinite(joint_stress_for_common_words).all():
            raise RuntimeError("joint GOPT lacks predictions for common stress words")
        metrics = {
            "seed": seed,
            "mfa_modular_phone": evaluate_phone(
                aligned_prediction, test["phone_labels"], common_test_mask
            ),
            "cao_viterbi_phone": evaluate_phone(
                cao_prediction, test["phone_labels"], common_test_mask
            ),
            "gopt_joint_phone": evaluate_phone(
                joint_phone, test["phone_labels"], common_test_mask
            ),
            "gopt_joint_stress": stress_metrics(
                joint_stress_for_common_words - 1.0, stress_rows
            ),
        }
        per_seed.append(metrics)
        torch.save({
            "model": aligned_phone.state_dict(),
            "mean": prepared["mfa_modular"]["mean"],
            "std": prepared["mfa_modular"]["std"],
        }, model_dir / f"mfa_phone_seed_{seed}.pt")
        torch.save({
            "model": cao_phone.state_dict(),
            "mean": prepared["cao_viterbi"]["mean"],
            "std": prepared["cao_viterbi"]["std"],
        }, model_dir / f"cao_phone_seed_{seed}.pt")
        torch.save({
            "model": joint.state_dict(),
            "mean": prepared["mfa_modular"]["mean"],
            "std": prepared["mfa_modular"]["std"],
        }, model_dir / f"gopt_joint_seed_{seed}.pt")
        print(json.dumps(metrics), flush=True)

    systems = {
        "mfa_modular": {
            "phone": summarize_runs([
                run["mfa_modular_phone"] for run in per_seed
            ]),
            "stress": fixed_stress_metrics(stress_rows, "mfa"),
        },
        "cao_viterbi_proposed": {
            "phone": summarize_runs([
                run["cao_viterbi_phone"] for run in per_seed
            ]),
            "stress": fixed_stress_metrics(stress_rows, "ctc_viterbi"),
        },
        "gopt_joint": {
            "phone": summarize_runs([
                run["gopt_joint_phone"] for run in per_seed
            ]),
            "stress": summarize_runs([
                run["gopt_joint_stress"] for run in per_seed
            ]),
        },
    }
    result = {
        "protocol": {
            "dataset": str(args.dataset),
            "split": "official speaker-disjoint SpeechOcean762 split",
            "sources": {
                "cao": "https://doi.org/10.21437/Interspeech.2024-459",
                "gopt": "https://doi.org/10.1109/ICASSP43922.2022.9746743",
                "mallela": "https://doi.org/10.21437/Interspeech.2024-2404",
            },
            "common_phone_mask": "phones with MFA match; identical across systems",
            "mfa_mapping": (
                "Levenshtein diagonal pairs retain exact matches and substitutions; "
                "intervals are relabeled with the expert canonical phone"
            ),
            "training_coverage": (
                "alignment-free phone GOPT uses every valid Cao feature; "
                "MFA-dependent models use MFA-mapped phones"
            ),
            "phone_model": (
                "official GOPT topology: learned positions + canonical-phone "
                "embedding, 24 dimensions, 3 blocks, 1 head"
            ),
            "third_system": (
                "Gong et al. multi-task GOPT architecture retrained on the "
                "same MFA-aligned CTC vectors; official checkpoint validated separately"
            ),
            "normalization": "one train-set scalar mean/std per feature family",
            "epochs": args.epochs, "seeds": seeds,
            "platform": platform.platform(), "torch": torch.__version__,
        },
        "counts": {
            "alignment_free_train_phones": int(train_phone_mask.sum()),
            "mfa_mapped_train_phones": int(
                (train_phone_mask & train["mfa_mask"]).sum()
            ),
            "alignment_free_test_phones": int(test_phone_mask.sum()),
            "common_test_phones": int(common_test_mask.sum()),
            "mfa_aligned_test_utterances": int(test["mfa_mask"].any(axis=1).sum()),
            "common_stress_words": len(stress_rows),
            "stress_correct_words": int(stress_rows["human_correct"].sum()),
            "stress_error_words": int((stress_rows["human_correct"] == 0).sum()),
        },
        "accuracy": systems,
        "comparisons": {
            "phone_pcc_run_differences": {
                "cao_minus_mfa": summarize_runs([
                    {"pcc": run["cao_viterbi_phone"]["pcc"]
                            - run["mfa_modular_phone"]["pcc"]}
                    for run in per_seed
                ])["pcc"],
                "cao_minus_joint_gopt": summarize_runs([
                    {"pcc": run["cao_viterbi_phone"]["pcc"]
                            - run["gopt_joint_phone"]["pcc"]}
                    for run in per_seed
                ])["pcc"],
            },
            "paired_stress_speaker_bootstrap": paired_stress_bootstrap(
                stress_rows, iterations=args.bootstrap_iterations, seed=args.seed
            ),
        },
        "per_seed": per_seed,
        "parameters": {
            "mfa_phone_gopt": parameter_count(GOPTPhone(78)),
            "cao_phone_gopt": parameter_count(GOPTPhone(41)),
            "joint_gopt": parameter_count(GOPTJoint(78)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".predictions.npz"),
        **{name: np.stack(values) for name, values in all_predictions.items()},
        common_test_mask=common_test_mask,
    )
    print(json.dumps(result, indent=2))


def validate_official_gopt(args: argparse.Namespace) -> None:
    """Reproduce the public GOPT checkpoint on its released test tensors."""

    source = args.official_gopt_repo / "src" / "models" / "gopt.py"
    spec = importlib.util.spec_from_file_location("official_gopt_model", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official GOPT source from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
    checkpoint = torch.load(
        args.official_gopt_repo
        / "pretrained_models/gopt_librispeech/best_audio_model.pth",
        map_location="cpu",
        weights_only=True,
    )
    checkpoint = {
        key.removeprefix("module."): value for key, value in checkpoint.items()
    }
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    data = args.official_gopt_data / "seq_data_librispeech"
    features = np.load(data / "te_feat.npy").astype(np.float32)
    phone_labels = np.load(data / "te_label_phn.npy").astype(np.float32)
    word_labels = np.load(data / "te_label_word.npy").astype(np.float32)
    valid_features = features[:, :, 0] != 0
    normalized = np.zeros_like(features)
    normalized[valid_features] = (features[valid_features] - 3.203) / 4.045
    phones = phone_labels[:, :, 0].astype(np.int64)
    with torch.inference_mode():
        outputs = model(torch.from_numpy(normalized), torch.from_numpy(phones))
    phone_prediction = outputs[5].squeeze(-1).numpy()
    stress_prediction = outputs[7].squeeze(-1).numpy()
    phone_target = phone_labels[:, :, 1]
    phone_mask = phone_target >= 0

    word_predictions, word_targets = [], []
    for utterance in range(len(stress_prediction)):
        word_ids = word_labels[utterance, :, 3].astype(int)
        for word_id in sorted(set(word_ids[word_ids >= 0])):
            mask = word_ids == word_id
            word_predictions.append(float(stress_prediction[utterance, mask].mean()))
            word_targets.append(float(word_labels[utterance, mask, 1].mean() / 5.0))

    # Head-only latency is included as a sanity check, not as an end-to-end
    # runtime claim because the released bundle starts after Kaldi GOP extraction.
    latency = []
    with torch.inference_mode():
        for index in range(min(args.gopt_latency_samples + 10, len(normalized))):
            started = time.perf_counter()
            model(
                torch.from_numpy(normalized[index : index + 1]),
                torch.from_numpy(phones[index : index + 1]),
            )
            elapsed = (time.perf_counter() - started) * 1000
            if index >= 10:
                latency.append(elapsed)
    result = {
        "protocol": {
            "source": "Gong et al. official repository, checkpoint, and released tensors",
            "paper": "https://arxiv.org/abs/2205.03432",
            "repository": "https://github.com/YuanGongND/gopt",
            "checkpoint": str(
                args.official_gopt_repo
                / "pretrained_models/gopt_librispeech/best_audio_model.pth"
            ),
            "test_data": str(data),
            "normalization": "published LibriSpeech constants: mean=3.203, std=4.045",
            "runtime_scope": "GOPT head only; excludes Kaldi alignment/GOP front end",
        },
        "counts": {
            "utterances": len(normalized),
            "phones": int(phone_mask.sum()),
            "words": len(word_predictions),
        },
        "accuracy": {
            "phone": correlations(
                phone_prediction[phone_mask], phone_target[phone_mask]
            ),
            "word_stress": correlations(
                np.asarray(word_predictions), np.asarray(word_targets)
            ),
        },
        "head_latency_ms": {
            "samples": len(latency),
            "mean": float(np.mean(latency)),
            "median": float(np.median(latency)),
            "p95": float(np.quantile(latency, 0.95)),
        },
    }
    args.official_gopt_output.parent.mkdir(parents=True, exist_ok=True)
    args.official_gopt_output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def latency_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def benchmark_latency(args: argparse.Namespace) -> None:
    """Measure one-request-at-a-time front ends and complete system means."""

    test = dict(np.load(args.cache_dir / "test_features.npz"))
    scores = json.loads((args.dataset / "resource" / "scores.json").read_text())
    wavs = read_kaldi_map(args.dataset / "test" / "wav.scp")
    textgrids = {
        path.stem: path for path in args.test_mfa_textgrids.rglob("*.TextGrid")
    }
    stress_path = args.stress_rows or args.stress_output
    stress_rows = pd.read_csv(stress_path, dtype={"utterance": str})

    seed = args.seed
    model_dir = args.output.parent / "three_system_checkpoints"
    bundles = {
        "mfa_phone": torch.load(
            model_dir / f"mfa_phone_seed_{seed}.pt", weights_only=False
        ),
        "cao_phone": torch.load(
            model_dir / f"cao_phone_seed_{seed}.pt", weights_only=False
        ),
        "gopt_joint": torch.load(
            model_dir / f"gopt_joint_seed_{seed}.pt", weights_only=False
        ),
    }
    models = {
        "mfa_phone": GOPTPhone(78).eval(),
        "cao_phone": GOPTPhone(41).eval(),
        "gopt_joint": GOPTJoint(78).eval(),
    }
    for name, model in models.items():
        model.load_state_dict(bundles[name]["model"], strict=True)

    utterance_to_row = {
        str(item): index for index, item in enumerate(test["utterance_ids"])
    }
    eligible = [
        item for item in map(str, test["utterance_ids"])
        if item in textgrids
        and test["feature_mask"][utterance_to_row[item]].any()
        and test["mfa_mask"][utterance_to_row[item]].any()
    ]
    selected = eligible[: args.latency_samples + 1]
    extractor = EmissionExtractor(
        args.cao_model, args.cao_processor, device_name=args.device
    )
    stage_values = {
        name: [] for name in (
            "ctc_encoding", "cao_af_vectors", "ctc_viterbi", "mfa_vectors",
            "mfa_phone_gopt", "cao_phone_gopt", "gopt_joint_head",
        )
    }
    for sample_index, utterance_id in enumerate(selected):
        row = utterance_to_row[utterance_id]
        canonical, _ = flatten_annotation(scores[utterance_id])
        audio = load_audio(args.dataset / wavs[utterance_id])
        measured = {}
        started = time.perf_counter()
        encoded = extractor.encode(audio, canonical)
        measured["ctc_encoding"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        current_af = cao_alignment_free_features(
            encoded["log_probs"], encoded["label_ids"], extractor.phone_ids,
            extractor.blank_id,
        )
        measured["cao_af_vectors"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        viterbi_regions(
            encoded["log_probs"], encoded["label_ids"], encoded["phones"],
            encoded["duration"], extractor.blank_id,
        )
        measured["ctc_viterbi"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        current_mfa, current_mfa_mask, _ = mfa_aligned_features(
            encoded, parse_phone_textgrid(textgrids[utterance_id]),
            extractor.phone_ids,
        )
        measured["mfa_vectors"] = (time.perf_counter() - started) * 1000

        af_padded = np.zeros((1, MAX_PHONES, 41), dtype=np.float32)
        mfa_padded = np.zeros((1, MAX_PHONES, 78), dtype=np.float32)
        af_padded[0, : len(current_af)] = current_af
        mfa_padded[0, : len(current_mfa)] = current_mfa
        af_mask = np.zeros((1, MAX_PHONES), dtype=bool)
        mfa_mask = np.zeros_like(af_mask)
        af_mask[0, : len(current_af)] = True
        mfa_mask[0, : len(current_mfa_mask)] = current_mfa_mask
        af_padded = normalize_features(
            af_padded, af_mask, bundles["cao_phone"]["mean"],
            bundles["cao_phone"]["std"],
        )
        mfa_padded = normalize_features(
            mfa_padded, mfa_mask, bundles["mfa_phone"]["mean"],
            bundles["mfa_phone"]["std"],
        )
        phone_tensor = torch.from_numpy(test["phone_ids"][row : row + 1])
        with torch.inference_mode():
            started = time.perf_counter()
            models["mfa_phone"](torch.from_numpy(mfa_padded), phone_tensor)
            measured["mfa_phone_gopt"] = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            models["cao_phone"](torch.from_numpy(af_padded), phone_tensor)
            measured["cao_phone_gopt"] = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            models["gopt_joint"](torch.from_numpy(mfa_padded), phone_tensor)
            measured["gopt_joint_head"] = (time.perf_counter() - started) * 1000
        # The first request warms both MPS and the three PyTorch heads.
        if sample_index:
            for name, value in measured.items():
                stage_values[name].append(value)
        if sample_index and sample_index % args.progress_every == 0:
            print(
                f"latency: {sample_index}/{min(args.latency_samples, len(eligible) - 1)}",
                flush=True,
            )

    common_utterances = set(eligible)
    stress_by_utterance = stress_rows.groupby("utterance")[[
        "ctc_viterbi_stress_latency_ms", "mfa_stress_latency_ms"
    ]].sum()
    stress_mean = {
        column: float(np.mean([
            stress_by_utterance.at[item, column]
            if item in stress_by_utterance.index else 0.0
            for item in common_utterances
        ]))
        for column in (
            "ctc_viterbi_stress_latency_ms", "mfa_stress_latency_ms"
        )
    }
    mfa_run = json.loads(args.mfa_timing.read_text())
    mfa_alignment_ms = float(mfa_run["wall_ms_per_input_utterance"])
    stages = {name: latency_summary(values) for name, values in stage_values.items()}
    encoder = stages["ctc_encoding"]["mean"]
    complete = {
        "mfa_modular": (
            encoder + mfa_alignment_ms + stages["mfa_vectors"]["mean"]
            + stages["mfa_phone_gopt"]["mean"]
            + stress_mean["mfa_stress_latency_ms"]
        ),
        "cao_viterbi_proposed": (
            encoder + stages["cao_af_vectors"]["mean"]
            + stages["ctc_viterbi"]["mean"]
            + stages["cao_phone_gopt"]["mean"]
            + stress_mean["ctc_viterbi_stress_latency_ms"]
        ),
        "gopt_joint": (
            encoder + mfa_alignment_ms + stages["mfa_vectors"]["mean"]
            + stages["gopt_joint_head"]["mean"]
        ),
    }
    result = {
        "protocol": {
            "latency_mode": "one request at a time; first request excluded",
            "ctc_device": str(extractor.device),
            "pytorch_threads": torch.get_num_threads(),
            "mfa_jobs": mfa_run["jobs"],
            "mfa_denominator": "all requested corpus inputs, including failures",
            "stress_timing": "all paired test requests; zero for requests without a scored polysyllable",
            "third_system": "controlled GOPT adaptation; not official Kaldi front-end timing",
        },
        "counts": {
            "front_end_latency_samples": len(stage_values["ctc_encoding"]),
            "paired_test_requests_for_stress": len(common_utterances),
        },
        "stage_latency_ms": stages,
        "stress_latency_ms_per_request": stress_mean,
        "external_mfa_alignment_ms_per_requested_input": mfa_alignment_ms,
        "complete_pipeline_mean_ms_per_request": complete,
        "speedup_vs_mfa_modular": {
            name: complete["mfa_modular"] / value
            for name, value in complete.items()
        },
    }
    args.latency_output.parent.mkdir(parents=True, exist_ok=True)
    args.latency_output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def aggregate_seed_runs(args: argparse.Namespace) -> None:
    """Combine independently trained seed artifacts into the paper result."""

    paths = [args.output, *sorted(args.output.parent.glob(args.run_glob))]
    path_documents = [
        (path, json.loads(path.read_text())) for path in paths if path.exists()
    ]
    documents = [document for _, document in path_documents]
    per_seed = sorted(
        [run for document in documents for run in document["per_seed"]],
        key=lambda run: run["seed"],
    )
    by_seed = {run["seed"]: run for run in per_seed}
    per_seed = [by_seed[seed] for seed in sorted(by_seed)]
    if not per_seed:
        raise RuntimeError("no seed results found")
    base = documents[0]
    stress_path = args.stress_rows or args.stress_output
    stress_rows = pd.read_csv(stress_path)
    train = dict(np.load(args.cache_dir / "train_features.npz"))
    test = dict(np.load(args.cache_dir / "test_features.npz"))
    train_phone_mask = (train["phone_labels"] >= 0) & train["feature_mask"]
    test_phone_mask = (test["phone_labels"] >= 0) & test["feature_mask"]
    common_test_mask = test_phone_mask & test["mfa_mask"]
    base["protocol"]["seeds"] = [run["seed"] for run in per_seed]
    base["protocol"]["execution"] = (
        "independent deterministic seeds; metrics aggregated without ensembling"
    )
    base["protocol"]["mfa_mapping"] = (
        "Levenshtein diagonal pairs retain exact matches and substitutions; "
        "intervals are relabeled with the expert canonical phone"
    )
    base["protocol"]["training_coverage"] = (
        "alignment-free phone GOPT uses every valid Cao feature; "
        "MFA-dependent models use MFA-mapped phones"
    )
    base["counts"] = {
        "alignment_free_train_phones": int(train_phone_mask.sum()),
        "mfa_mapped_train_phones": int(
            (train_phone_mask & train["mfa_mask"]).sum()
        ),
        "alignment_free_test_phones": int(test_phone_mask.sum()),
        "common_test_phones": int(common_test_mask.sum()),
        "mfa_aligned_test_utterances": int(test["mfa_mask"].any(axis=1).sum()),
        "common_stress_words": len(stress_rows),
        "stress_correct_words": int(stress_rows["human_correct"].sum()),
        "stress_error_words": int((stress_rows["human_correct"] == 0).sum()),
    }
    base["accuracy"]["mfa_modular"]["phone"] = summarize_runs([
        run["mfa_modular_phone"] for run in per_seed
    ])
    base["accuracy"]["cao_viterbi_proposed"]["phone"] = summarize_runs([
        run["cao_viterbi_phone"] for run in per_seed
    ])
    base["accuracy"]["gopt_joint"]["phone"] = summarize_runs([
        run["gopt_joint_phone"] for run in per_seed
    ])
    base["accuracy"]["gopt_joint"]["stress"] = summarize_runs([
        run["gopt_joint_stress"] for run in per_seed
    ])
    base["accuracy"]["mfa_modular"]["stress"] = fixed_stress_metrics(
        stress_rows, "mfa"
    )
    base["accuracy"]["cao_viterbi_proposed"]["stress"] = fixed_stress_metrics(
        stress_rows, "ctc_viterbi"
    )
    base["comparisons"]["phone_pcc_run_differences"] = {
        "cao_minus_mfa": summarize_runs([
            {"pcc": run["cao_viterbi_phone"]["pcc"]
                    - run["mfa_modular_phone"]["pcc"]}
            for run in per_seed
        ])["pcc"],
        "cao_minus_joint_gopt": summarize_runs([
            {"pcc": run["cao_viterbi_phone"]["pcc"]
                    - run["gopt_joint_phone"]["pcc"]}
            for run in per_seed
        ])["pcc"],
    }
    base["comparisons"]["paired_stress_speaker_bootstrap"] = (
        paired_stress_bootstrap(
            stress_rows,
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
    )
    base["per_seed"] = per_seed
    base.pop("source_seed_files", None)
    base["aggregation"] = {
        "seed_count": len(per_seed),
        "prediction_order": [run["seed"] for run in per_seed],
    }
    args.output.write_text(json.dumps(base, indent=2) + "\n")

    predictions_by_seed: dict[int, dict[str, np.ndarray]] = {}
    prediction_common_mask = None
    for path, document in path_documents:
        prediction_path = path.with_suffix(".predictions.npz")
        if not prediction_path.exists():
            continue
        predictions = dict(np.load(prediction_path))
        document_seeds = [run["seed"] for run in document["per_seed"]]
        if len(document_seeds) != len(predictions["mfa_modular"]):
            raise RuntimeError(
                f"seed/prediction mismatch in {prediction_path}: "
                f"{len(document_seeds)} seeds, "
                f"{len(predictions['mfa_modular'])} prediction batches"
            )
        prediction_common_mask = predictions["common_test_mask"]
        for index, seed in enumerate(document_seeds):
            predictions_by_seed[seed] = {
                key: predictions[key][index]
                for key in (
                    "mfa_modular", "cao_viterbi", "gopt_joint_phone",
                    "gopt_joint_stress",
                )
            }
    if predictions_by_seed:
        keys = (
            "mfa_modular", "cao_viterbi", "gopt_joint_phone",
            "gopt_joint_stress",
        )
        np.savez_compressed(
            args.output.with_suffix(".predictions.npz"),
            **{
                key: np.stack([
                    predictions_by_seed[seed][key]
                    for seed in sorted(predictions_by_seed)
                ])
                for key in keys
            },
            common_test_mask=prediction_common_mask,
        )
    print(json.dumps(base, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=(
            "features", "stress", "train", "aggregate", "latency",
            "validate-gopt"
        )
    )
    parser.add_argument("--dataset", type=Path,
                        default=Path("tmp/datasets/speechocean762"))
    parser.add_argument("--train-mfa-textgrids", type=Path,
                        default=Path("tmp/mfa_output_train_full"))
    parser.add_argument("--test-mfa-textgrids", type=Path,
                        default=Path("tmp/mfa_output_test_full"))
    parser.add_argument("--cao-model", type=Path,
                        default=Path("tmp/reference_ctc_based_gop/is24/models/checkpoint-8000"))
    parser.add_argument("--cao-processor", type=Path,
                        default=Path("tmp/reference_ctc_based_gop/is24/models/processor_config_gop"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("tmp/three_system_comparison"))
    parser.add_argument("--stress-rows", type=Path)
    parser.add_argument(
        "--stress-output", type=Path,
        default=Path("paper/results/three_system_stress.words.csv"),
    )
    parser.add_argument(
        "--stress-model", type=Path,
        default=Path("server/models/sylstress/stress_model.keras"),
    )
    parser.add_argument(
        "--stress-scaler", type=Path,
        default=Path("server/models/sylstress/scaler_params.json"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("paper/results/three_system_comparison.json"))
    parser.add_argument(
        "--official-gopt-repo", type=Path, default=Path("tmp/reference_gopt")
    )
    parser.add_argument(
        "--official-gopt-data", type=Path, default=Path("tmp/gopt_official_data")
    )
    parser.add_argument(
        "--official-gopt-output", type=Path,
        default=Path("paper/results/gopt_official_validation.json"),
    )
    parser.add_argument("--gopt-latency-samples", type=int, default=500)
    parser.add_argument("--run-glob", default="three_system_run_*.json")
    parser.add_argument("--latency-samples", type=int, default=250)
    parser.add_argument(
        "--latency-output", type=Path,
        default=Path("paper/results/three_system_latency.json"),
    )
    parser.add_argument(
        "--mfa-timing", type=Path,
        default=Path("paper/results/mfa_alignment_latency_1_job.json"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    torch.set_num_threads(parsed.torch_threads)
    if parsed.stage == "features":
        generate_features(parsed)
    elif parsed.stage == "stress":
        generate_stress_rows(parsed)
    elif parsed.stage == "train":
        train_and_evaluate(parsed)
    elif parsed.stage == "latency":
        benchmark_latency(parsed)
    elif parsed.stage == "aggregate":
        aggregate_seed_runs(parsed)
    else:
        validate_official_gopt(parsed)
