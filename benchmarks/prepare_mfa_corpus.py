#!/usr/bin/env python3
"""Create a paired Montreal Forced Aligner corpus from SpeechOcean762.

The generated directory contains one speaker subdirectory per SpeechOcean
speaker and matching ``.wav`` hard links/``.lab`` transcripts.  It is intended
for reproducible runtime and boundary comparisons against the embedded CTC
Viterbi aligner; it does not modify the source corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def read_kaldi_map(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(maxsplit=1)
        result[key] = value
    return result


def select_round_robin(records: list[dict], limit: int | None) -> list[dict]:
    if not limit or limit >= len(records):
        return records
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_speaker[record["speaker"]].append(record)
    selected = []
    offset = 0
    speakers = sorted(by_speaker)
    while len(selected) < limit:
        added = False
        for speaker in speakers:
            if offset < len(by_speaker[speaker]):
                selected.append(by_speaker[speaker][offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def prepare(args: argparse.Namespace) -> None:
    split_dir = args.dataset / args.split
    wavs = read_kaldi_map(split_dir / "wav.scp")
    texts = read_kaldi_map(split_dir / "text")
    speakers = read_kaldi_map(split_dir / "utt2spk")
    records = [
        {
            "utterance": utterance,
            "speaker": speakers[utterance],
            "audio": (args.dataset / wavs[utterance]).resolve(),
            "text": texts[utterance],
        }
        for utterance in sorted(wavs)
    ]
    records = select_round_robin(records, args.limit)

    args.output.mkdir(parents=True, exist_ok=True)
    existing = list(args.output.iterdir())
    if existing:
        raise FileExistsError(
            f"Refusing to mix files in non-empty output directory: {args.output}"
        )

    for record in records:
        speaker_dir = args.output / record["speaker"]
        speaker_dir.mkdir(parents=True, exist_ok=True)
        audio_link = speaker_dir / f'{record["utterance"]}.wav'
        try:
            os.link(record["audio"], audio_link)
        except OSError:
            shutil.copy2(record["audio"], audio_link)
        (speaker_dir / f'{record["utterance"]}.lab').write_text(
            record["text"].strip() + "\n", encoding="utf-8"
        )

    manifest = {
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "utterances": len(records),
        "speakers": len({record["speaker"] for record in records}),
        "selection": "all" if not args.limit else "deterministic speaker round-robin",
        "utterance_ids": [record["utterance"] for record in records],
    }
    manifest_path = args.output.parent / f"{args.output.name}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "utterance_ids"}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
