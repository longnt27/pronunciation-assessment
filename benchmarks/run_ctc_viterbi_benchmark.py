#!/usr/bin/env python3
"""Benchmark embedded CTC-Viterbi on a fixed SpeechOcean manifest."""

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
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.services.gop_service import GOPEvaluator


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


def aggregate(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def run(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = manifest["split"]
    utterance_ids = manifest["utterance_ids"]
    wavs = read_kaldi_map(args.dataset / split / "wav.scp")
    speakers = read_kaldi_map(args.dataset / split / "utt2spk")
    scores = json.loads((args.dataset / "resource" / "scores.json").read_text())

    evaluator = GOPEvaluator(args.model)
    evaluator._audit_log = lambda *unused_args, **unused_kwargs: None
    warm_id = utterance_ids[0]
    warm_phones, _ = flatten_annotation(scores[warm_id])
    evaluator.infer_gop(
        (args.dataset / wavs[warm_id]).read_bytes(),
        scores[warm_id]["text"],
        target_phonemes=warm_phones,
        method="ctc_viterbi",
    )

    phone_rows, latency_rows, failures = [], [], []
    for index, utterance in enumerate(utterance_ids, start=1):
        phones, grades = flatten_annotation(scores[utterance])
        started = time.perf_counter()
        result = evaluator.infer_gop(
            (args.dataset / wavs[utterance]).read_bytes(),
            scores[utterance]["text"],
            target_phonemes=phones,
            method="ctc_viterbi",
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        if "error" in result:
            failures.append({"utterance": utterance, "error": result["error"]})
            continue
        details = list(result["details"].values())
        if len(details) != len(phones):
            failures.append({
                "utterance": utterance,
                "error": f"phone count {len(details)} != {len(phones)}",
            })
            continue
        latency_row = {
            "utterance": utterance,
            "speaker": speakers[utterance],
            "wall_ms": wall_ms,
        }
        latency_row.update(
            {f"stage_{key}_ms": value for key, value in result["latency_ms"].items()}
        )
        latency_rows.append(latency_row)
        for phone_index, (phone, grade, detail) in enumerate(zip(phones, grades, details)):
            phone_rows.append({
                "utterance": utterance,
                "speaker": speakers[utterance],
                "phone_index": phone_index,
                "phone": "".join(c for c in phone.upper() if not c.isdigit()),
                "human_grade": grade,
                "score": detail["gop_score"],
                "start": detail["start_time"],
                "end": detail["end_time"],
            })
        if index % args.progress_every == 0:
            print(f"processed {index}/{len(utterance_ids)}", flush=True)

    phone_frame = pd.DataFrame(phone_rows)
    latency_frame = pd.DataFrame(latency_rows)
    stage_metrics = {}
    for column in latency_frame.columns:
        if column.startswith("stage_"):
            stage_metrics[column.removeprefix("stage_").removesuffix("_ms")] = aggregate(
                latency_frame[column].tolist()
            )
    result = {
        "protocol": {
            "dataset": str(args.dataset),
            "manifest": str(args.manifest),
            "split": split,
            "utterances": len(utterance_ids),
            "model": evaluator.model_name,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "metrics": {
            "phones": len(phone_frame),
            "raw_pcc": float(pearsonr(phone_frame["score"], phone_frame["human_grade"]).statistic),
            "raw_srcc": float(spearmanr(phone_frame["score"], phone_frame["human_grade"]).statistic),
            "wall_latency_ms": aggregate(latency_frame["wall_ms"].tolist()),
            "stage_latency_ms": stage_metrics,
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    phone_frame.to_csv(args.output.with_suffix(".phones.csv"), index=False)
    latency_frame.to_csv(args.output.with_suffix(".latency.csv"), index=False)
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="server/models/ctcgop")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
