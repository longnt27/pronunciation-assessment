#!/usr/bin/env python3
"""Run and time a reproducible external-MFA alignment condition."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import soundfile as sf


def corpus_duration_seconds(corpus: Path) -> float:
    duration = 0.0
    for audio_path in corpus.rglob("*.wav"):
        info = sf.info(audio_path)
        duration += info.frames / float(info.samplerate)
    return duration


def run(args: argparse.Namespace) -> None:
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    args.temporary.mkdir(parents=True, exist_ok=True)
    args.mfa_root.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["MFA_ROOT_DIR"] = str(args.mfa_root.resolve())
    environment["PATH"] = (
        str(args.mfa.resolve().parent) + os.pathsep + environment.get("PATH", "")
    )
    version = subprocess.check_output(
        [str(args.mfa), "version"], env=environment, text=True
    ).strip()
    command = [
        str(args.mfa),
        "align",
        str(args.corpus.resolve()),
        args.dictionary,
        args.acoustic_model,
        str(args.output.resolve()),
        "--output_format",
        "long_textgrid",
        "--ignore_oovs",
        "--num_jobs",
        str(args.jobs),
        "--temporary_directory",
        str(args.temporary.resolve()),
        "--clean",
        "--overwrite",
        "--use_postgres",
        "--quiet",
    ]

    started = time.perf_counter()
    completed = subprocess.run(
        command, env=environment, text=True, capture_output=True, check=False
    )
    wall_seconds = time.perf_counter() - started
    log_path = args.result.with_suffix(".log")
    log_path.write_text(
        "COMMAND\n"
        + " ".join(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )

    duration_seconds = corpus_duration_seconds(args.corpus)
    aligned_files = list(args.output.rglob("*.TextGrid"))
    input_files = list(args.corpus.rglob("*.wav"))
    result = {
        "mfa_version": version,
        "platform": platform.platform(),
        "jobs": args.jobs,
        "execution_backend": "multiprocessing+postgres",
        "corpus": str(args.corpus),
        "input_utterances": len(input_files),
        "aligned_utterances": len(aligned_files),
        "audio_seconds": duration_seconds,
        "wall_seconds": wall_seconds,
        "wall_ms_per_input_utterance": wall_seconds * 1000.0 / len(input_files),
        "real_time_factor": wall_seconds / duration_seconds,
        "return_code": completed.returncode,
        "models": {
            "dictionary": args.dictionary,
            "acoustic": args.acoustic_model,
        },
        "command": command,
        "log": str(log_path),
    }
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if completed.returncode:
        raise SystemExit(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mfa", type=Path, required=True)
    parser.add_argument("--mfa-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--dictionary", default="english_us_arpa")
    parser.add_argument("--acoustic-model", default="english_us_arpa")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporary", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
