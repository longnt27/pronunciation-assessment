"""Embedded CTC-Viterbi pronunciation scoring.

One phoneme CTC encoder pass is shared by phone scoring and the phone regions
used by the syllable-stress branch. The implementation has no dependency on
Montreal Forced Aligner or a separately trained alignment acoustic model.
"""

from __future__ import annotations

import io
import logging
import os
import re
import time

import librosa
import numpy as np
import soundfile as sf
import syllapy
import torch
from g2p_en import G2p
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logging.getLogger("transformers").setLevel(logging.ERROR)

DEFAULT_MODEL_NAME = "mostafaashahin/wav2vec2-base-timit-phoneme-arpa-39"


class GOPEvaluator:
    """Score canonical phones and locate them with an embedded CTC trellis."""

    def __init__(self, model_path: str | None = None):
        self.g2p = G2p()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        chosen_path = model_path or DEFAULT_MODEL_NAME
        if os.path.isdir(chosen_path):
            has_weights = any(
                os.path.exists(os.path.join(chosen_path, filename))
                for filename in ("model.safetensors", "pytorch_model.bin")
            )
            if not has_weights:
                print(
                    f"⚠️ Local model path '{chosen_path}' contains config but missing "
                    f"weights. Falling back to '{DEFAULT_MODEL_NAME}'."
                )
                chosen_path = DEFAULT_MODEL_NAME
        self.model_name = chosen_path

        print(f"⏳ Loading GOP Model from: {chosen_path} on {self.device}...")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(chosen_path)
            self.model = Wav2Vec2ForCTC.from_pretrained(chosen_path).to(self.device)
            self.model.eval()
            self.vocab = self.processor.tokenizer.get_vocab()
            self.blank_id = self.processor.tokenizer.pad_token_id
            if self.blank_id is None:
                self.blank_id = 0
            self.token_to_id = {}
            for token, token_id in self.vocab.items():
                self.token_to_id[token] = token_id
                self.token_to_id[token.upper()] = token_id
                self.token_to_id[token.lower()] = token_id
            for source, destination in (("AO", "AA"), ("AX", "AH"), ("AXR", "ER")):
                if destination in self.token_to_id and source not in self.token_to_id:
                    self.token_to_id[source] = self.token_to_id[destination]
                    self.token_to_id[source.lower()] = self.token_to_id[destination]
            self.special_ids = {self.blank_id}
            for token in ("<pad>", "<unk>", "<s>", "</s>", "[PAD]", "[UNK]"):
                if token in self.vocab:
                    self.special_ids.add(self.vocab[token])
            print(
                f"✅ Loaded GOP model successfully. Vocab size: {len(self.vocab)}, "
                f"Blank ID: {self.blank_id}."
            )
        except Exception as exc:
            print(f"❌ Failed to load GOP model: {exc}")
            self.model = None

    def _load_audio(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        """Decode bytes as 16-kHz mono float32 audio."""

        try:
            audio, sample_rate = sf.read(io.BytesIO(audio_bytes))
        except Exception:
            audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        return audio.astype(np.float32), sample_rate

    def text_to_phonemes(self, transcript_text: str) -> tuple[list[str], list[int]]:
        """Convert text to unstressed ARPAbet labels present in the CTC vocabulary."""

        phonemes = []
        for raw_phone in self.g2p(transcript_text):
            phone = re.sub(r"\d+", "", raw_phone).strip().upper()
            if phone and phone not in {"'", ",", ".", "?", "!", "-", ";", ":", '"'}:
                phonemes.append(phone)
        return self._resolve_target_sequence(transcript_text, phonemes)

    def _resolve_target_sequence(
        self, transcript_text: str, target_phonemes: list[str] | None = None
    ) -> tuple[list[str], list[int]]:
        if target_phonemes is None:
            return self.text_to_phonemes(transcript_text)
        valid_phonemes, valid_ids = [], []
        for phoneme in target_phonemes:
            clean = re.sub(r"\d+", "", phoneme).strip().upper()
            token_id = self.token_to_id.get(clean)
            if token_id is not None:
                valid_phonemes.append(clean)
                valid_ids.append(token_id)
        return valid_phonemes, valid_ids

    def _encode_utterance(
        self, audio_bytes: bytes, transcript_text: str, target_phonemes=None
    ) -> dict:
        audio, sample_rate = self._load_audio(audio_bytes)
        phonemes, label_ids = self._resolve_target_sequence(
            transcript_text, target_phonemes
        )
        if not label_ids:
            raise ValueError("No valid phonemes found in model dictionary")
        with torch.no_grad():
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return {
            "audio": audio,
            "sample_rate": sample_rate,
            "duration": len(audio) / float(sample_rate),
            "phonemes": phonemes,
            "label_ids": label_ids,
            "log_probs": log_probs,
        }

    def viterbi_ctc_align(
        self, log_probs: np.ndarray, target_ids: list[int]
    ) -> tuple[dict[int, list[int]], np.ndarray]:
        """Return the maximum-probability CTC state path for fixed labels.

        States alternate between blanks and target labels. Allowed transitions
        are stay, advance one state, and skip a blank when adjacent labels differ.
        Repeated adjacent phones therefore require an intervening blank.
        """

        time_steps = log_probs.shape[0]
        units = len(target_ids)
        if units == 0:
            return {}, np.zeros(time_steps, dtype=np.int32)
        states = 2 * units + 1
        state_tokens = np.asarray(
            [
                self.blank_id if state % 2 == 0 else target_ids[(state - 1) // 2]
                for state in range(states)
            ],
            dtype=np.int32,
        )
        trellis = np.full((time_steps, states), -np.inf, dtype=np.float64)
        backtrack = np.zeros((time_steps, states), dtype=np.int32)
        trellis[0, 0] = log_probs[0, self.blank_id]
        trellis[0, 1] = log_probs[0, target_ids[0]]

        for frame in range(1, time_steps):
            min_state = max(0, states - 2 * (time_steps - frame))
            max_state = min(states, 2 * (frame + 1))
            for state in range(min_state, max_state):
                candidates = [(trellis[frame - 1, state], state)]
                if state > 0:
                    candidates.append((trellis[frame - 1, state - 1], state - 1))
                if (
                    state % 2 == 1
                    and state >= 2
                    and state_tokens[state] != state_tokens[state - 2]
                ):
                    candidates.append((trellis[frame - 1, state - 2], state - 2))
                previous_value, previous_state = max(candidates, key=lambda item: item[0])
                trellis[frame, state] = (
                    previous_value + log_probs[frame, state_tokens[state]]
                )
                backtrack[frame, state] = previous_state

        terminal = max(
            (states - 1, states - 2), key=lambda state: trellis[-1, state]
        )
        if not np.isfinite(trellis[-1, terminal]):
            raise ValueError("No valid CTC path for target sequence and emission length")
        path = np.zeros(time_steps, dtype=np.int32)
        path[-1] = terminal
        for frame in range(time_steps - 2, -1, -1):
            path[frame] = backtrack[frame + 1, path[frame + 1]]
        unit_frames = {unit: [] for unit in range(units)}
        for frame, state in enumerate(path):
            if state % 2 == 1:
                unit_frames[(state - 1) // 2].append(frame)
        return unit_frames, path

    @staticmethod
    def _logistic_calibration(lpp: float, alpha: float = 1.2, x0: float = -1.8) -> float:
        confidence = 100.0 / (1.0 + np.exp(-alpha * (lpp - x0)))
        return float(np.clip(confidence, 0.0, 100.0))

    @staticmethod
    def _segments(
        phonemes: list[str],
        label_ids: list[int],
        unit_frames: dict[int, list[int]],
        time_steps: int,
        duration: float,
    ) -> list[dict]:
        units = []
        for index in range(len(phonemes)):
            frames = unit_frames.get(index, [])
            if not frames:
                raise ValueError(f"CTC path did not visit phone index {index}")
            units.append((frames[0], frames[-1], frames))
        seconds_per_frame = duration / time_steps
        segments = []
        for index, (first, last, frames) in enumerate(units):
            start_frame = 0 if index == 0 else (units[index - 1][1] + 1 + first) // 2
            end_frame = (
                time_steps
                if index == len(units) - 1
                else (last + 1 + units[index + 1][0]) // 2
            )
            start = start_frame * seconds_per_frame
            end = min(duration, end_frame * seconds_per_frame)
            segments.append({
                "unit_idx": index,
                "phoneme": phonemes[index],
                "label_id": label_ids[index],
                "emission_frames": frames,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_time": round(float(start), 3),
                "end_time": round(float(end), 3),
                "duration": round(float(max(0.0, end - start)), 3),
            })
        return segments

    def infer_gop(
        self,
        audio_bytes: bytes,
        transcript_text: str,
        target_phonemes=None,
        method: str = "ctc_viterbi",
    ) -> dict:
        """Run embedded CTC-Viterbi alignment and frame-averaged phone LPP."""

        if self.model is None:
            return {"error": "Model not loaded"}
        if method != "ctc_viterbi":
            return {"error": f"Unsupported GOP method: {method}"}
        total_started = time.perf_counter()
        preparation_started = time.perf_counter()
        try:
            audio, sample_rate = self._load_audio(audio_bytes)
            phonemes, label_ids = self._resolve_target_sequence(
                transcript_text, target_phonemes
            )
            if not label_ids:
                raise ValueError("No valid phonemes found in model dictionary")
        except Exception as exc:
            return {"error": f"Audio/target preparation error: {exc}"}
        preparation_ms = (time.perf_counter() - preparation_started) * 1000.0

        encoding_started = time.perf_counter()
        with torch.no_grad():
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        encoding_ms = (time.perf_counter() - encoding_started) * 1000.0

        alignment_started = time.perf_counter()
        try:
            unit_frames, _ = self.viterbi_ctc_align(
                log_probs.detach().cpu().numpy(), label_ids
            )
            duration = len(audio) / float(sample_rate)
            segments = self._segments(
                phonemes, label_ids, unit_frames, len(log_probs), duration
            )
        except Exception as exc:
            return {"error": f"CTC-Viterbi alignment error: {exc}"}
        alignment_ms = (time.perf_counter() - alignment_started) * 1000.0

        scoring_started = time.perf_counter()
        details, alignment, lpp_scores, confidences = {}, [], [], []
        vocabulary_size = log_probs.shape[-1]
        competitor_mask = torch.ones(
            vocabulary_size, dtype=torch.bool, device=log_probs.device
        )
        for special_id in self.special_ids:
            if special_id < vocabulary_size:
                competitor_mask[special_id] = False
        for segment in segments:
            phone = segment["phoneme"]
            token_id = segment["label_id"]
            frames = segment["emission_frames"]
            phone_lpp = float(log_probs[frames, token_id].mean().item())
            phone_mask = competitor_mask.clone()
            phone_mask[token_id] = False
            competitors = log_probs[frames][:, phone_mask].max(dim=1).values
            phone_lpr = float((log_probs[frames, token_id] - competitors).mean().item())
            confidence = self._logistic_calibration(phone_lpp)
            detail = {
                "phoneme": phone,
                "gop_score": round(phone_lpp, 4),
                "confidence_score": round(confidence, 2),
                "lpp": round(phone_lpp, 4),
                "lpr": round(phone_lpr, 4),
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "duration": segment["duration"],
                "frame_interval": [segment["start_frame"], segment["end_frame"]],
            }
            details[f"{phone}_{segment['unit_idx']}"] = detail
            alignment.append({
                "phoneme": phone,
                "start": segment["start_time"],
                "end": segment["end_time"],
                "gop_score": detail["gop_score"],
                "confidence_score": detail["confidence_score"],
            })
            lpp_scores.append(phone_lpp)
            confidences.append(confidence)
        scoring_ms = (time.perf_counter() - scoring_started) * 1000.0

        speech_bounds = {
            "start": alignment[0]["start"],
            "end": alignment[-1]["end"],
        }
        self._audit_log(transcript_text, segments, speech_bounds)
        return {
            "transcript_text": transcript_text,
            "transcript_phonemes": " ".join(phonemes),
            "overall_score": round(float(np.mean(confidences)), 1),
            "average_gop": round(float(np.mean(lpp_scores)), 4),
            "details": details,
            "alignment": alignment,
            "speech_bounds": speech_bounds,
            "method": "ctc_viterbi",
            "scoring_method": "ctc_viterbi_lpp",
            "timing_method": "ctc_viterbi",
            "external_aligner_used": False,
            "canonical_sequence_constrained": True,
            "latency_ms": {
                "preparation": round(preparation_ms, 3),
                "ctc_encoding": round(encoding_ms, 3),
                "ctc_viterbi": round(alignment_ms, 3),
                "phone_scoring": round(scoring_ms, 3),
                "total": round((time.perf_counter() - total_started) * 1000.0, 3),
            },
        }

    @staticmethod
    def _audit_log(transcript_text: str, segments: list[dict], speech_bounds: dict) -> None:
        syllables = max(1, syllapy.count(transcript_text))
        print(
            f"CTC-Viterbi: '{transcript_text}' | {len(segments)} phones | "
            f"{syllables} syllables | {speech_bounds['start']:.3f}-"
            f"{speech_bounds['end']:.3f}s"
        )
