#!/usr/bin/env python3
"""Measure interactive one-utterance latency for CTC-Viterbi and MFA.

The proposed model remains resident in memory, as it does in the API.  The MFA
condition invokes its documented ``align_one`` pipeline for each request while
reusing the installed models and PostgreSQL server.  A separate full-corpus MFA
benchmark measures offline batch throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_ctc_viterbi_benchmark import flatten_annotation, read_kaldi_map
from benchmarks.benchmark_mfa_pipeline_latency import score_with_mfa_intervals
from benchmarks.evaluate_ctc_viterbi_vs_mfa import parse_phone_textgrid
from server.services.gop_service import GOPEvaluator


def aggregate(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def run(args: argparse.Namespace) -> None:
    scores = json.loads((args.dataset / "resource" / "scores.json").read_text())
    wavs = read_kaldi_map(args.dataset / "test" / "wav.scp")
    available = sorted(path.stem for path in args.mfa_textgrids.rglob("*.TextGrid"))
    utterances = available[: args.utterances]
    evaluator = GOPEvaluator(args.model)
    evaluator._audit_log = lambda *unused_args, **unused_kwargs: None

    environment = os.environ.copy()
    environment["MFA_ROOT_DIR"] = str(args.mfa_root.resolve())
    environment["PATH"] = (
        str(args.mfa.resolve().parent) + os.pathsep + environment.get("PATH", "")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temporary.mkdir(parents=True, exist_ok=True)

    warm = utterances[0]
    warm_phones, _ = flatten_annotation(scores[warm])
    evaluator.infer_gop(
        (args.dataset / wavs[warm]).read_bytes(),
        scores[warm]["text"],
        target_phonemes=warm_phones,
    )

    rows, failures = [], []
    for index, utterance in enumerate(utterances, start=1):
        phones, _ = flatten_annotation(scores[utterance])
        audio = (args.dataset / wavs[utterance]).read_bytes()
        started = time.perf_counter()
        ctc_result = evaluator.infer_gop(
            audio,
            scores[utterance]["text"],
            target_phonemes=phones,
        )
        ctc_seconds = time.perf_counter() - started
        if "error" in ctc_result:
            failures.append({"utterance": utterance, "condition": "ctc_viterbi", "error": ctc_result["error"]})
            continue

        speaker = next(args.corpus.glob(f"*/{utterance}.wav")).parent
        command = [
            str(args.mfa),
            "align_one",
            str((speaker / f"{utterance}.wav").resolve()),
            str((speaker / f"{utterance}.lab").resolve()),
            args.dictionary,
            args.acoustic_model,
            str((args.output_dir / f"{utterance}.TextGrid").resolve()),
            "--temporary_directory",
            str(args.temporary.resolve()),
            "--quiet",
            "--overwrite",
            "--use_postgres",
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command, env=environment, text=True, capture_output=True, check=False
        )
        mfa_seconds = time.perf_counter() - started
        if completed.returncode:
            failures.append({
                "utterance": utterance,
                "condition": "mfa_align_one",
                "error": completed.stderr[-2000:],
            })
            continue
        scoring_started = time.perf_counter()
        scored_phones = score_with_mfa_intervals(
            evaluator,
            audio,
            scores[utterance]["text"],
            phones,
            parse_phone_textgrid(args.output_dir / f"{utterance}.TextGrid"),
        )
        mfa_scoring_seconds = time.perf_counter() - scoring_started
        rows.append({
            "utterance": utterance,
            "ctc_viterbi_ms": ctc_seconds * 1000.0,
            "mfa_align_one_ms": mfa_seconds * 1000.0,
            "mfa_conditioned_scoring_ms": mfa_scoring_seconds * 1000.0,
            "mfa_pipeline_ms": (mfa_seconds + mfa_scoring_seconds) * 1000.0,
            "scored_phones": scored_phones,
        })
        print(f"processed {index}/{len(utterances)}", flush=True)

    result = {
        "protocol": {
            "workload": "interactive one utterance per request",
            "ctc_service": "model resident; one warm-up excluded",
            "mfa_service": "mfa align_one subprocess; installed models and PostgreSQL server reused",
            "dataset": str(args.dataset),
            "selection": "first lexicographic test utterances successfully aligned by full MFA run",
            "model": evaluator.model_name,
            "mfa_version": subprocess.check_output(
                [str(args.mfa), "version"], env=environment, text=True
            ).strip(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "metrics": {
            "ctc_viterbi_ms": aggregate([row["ctc_viterbi_ms"] for row in rows]),
            "mfa_align_one_ms": aggregate([row["mfa_align_one_ms"] for row in rows]),
            "mfa_conditioned_scoring_ms": aggregate(
                [row["mfa_conditioned_scoring_ms"] for row in rows]
            ),
            "mfa_pipeline_ms": aggregate([row["mfa_pipeline_ms"] for row in rows]),
            "paired_mfa_pipeline_over_ctc_ratio_median": float(
                np.median(
                    [row["mfa_pipeline_ms"] / row["ctc_viterbi_ms"] for row in rows]
                )
            ),
        },
        "rows": rows,
        "failures": failures,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--mfa-textgrids", type=Path, required=True)
    parser.add_argument("--mfa", type=Path, required=True)
    parser.add_argument("--mfa-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temporary", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--utterances", type=int, default=20)
    parser.add_argument("--model", default="server/models/ctcgop")
    parser.add_argument("--dictionary", default="english_us_arpa")
    parser.add_argument("--acoustic-model", default="english_us_arpa")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
