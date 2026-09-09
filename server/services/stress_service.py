"""
Syllable Stress Evaluation Service.
Based on:
- Mallela et al. (Interspeech 2024): Sequential modeling of syllable dependencies and
  argmax post-processing (wPP) ensuring exactly one primary stress per word.
- Yarra et al. (SLATE 2019): True time-aligned syllable and vowel acoustic prominence.
"""

import io
import json
import os
import re
import numpy as np
import librosa
from server.utils.audio_features import (
    get_word_syllables,
    extract_full_features,
    extract_full_features_uniform
)

# Optional TensorFlow / Keras import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False


class StressEvaluator:
    def __init__(self, model_path: str = None, scaler_path: str = None):
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

        if model_path:
            print(f"Loading Stress Model from {model_path}...")
            if os.path.exists(model_path):
                if TF_AVAILABLE:
                    try:
                        self.model = tf.keras.models.load_model(model_path)
                        print("✅ Sequential Stress Model loaded successfully.")
                    except Exception as e:
                        print(f"⚠️ Failed to load Stress Model via TF/Keras: {e}. Using Acoustic Prominence Engine.")
                else:
                    print("⚠️ TensorFlow not found. Using Acoustic Prominence Scoring Engine (Yarra/Mallela).")
            else:
                print(f"⚠️ Model path not found: {model_path}. Using Acoustic Prominence Scoring Engine.")

        if scaler_path and os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'r') as f:
                    data = json.load(f)
                    self.scaler_mean = np.array(data["mean"], dtype=np.float32)
                    self.scaler_std = np.array(data["std"], dtype=np.float32)
                print(f"✅ Scaler loaded from {scaler_path}.")
            except Exception as e:
                print(f"⚠️ Could not load scaler: {e}")

    @staticmethod
    def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Converts raw scores into a calibrated probability distribution."""
        x = np.asarray(x, dtype=np.float64) / max(temperature, 1e-4)
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def predict(self, audio_input, word_text: str, alignments: list or None = None, method: str = "ctc_viterbi") -> dict:
        """
        Evaluates syllable stress for a word in an audio segment.

        Args:
            audio_input: bytes (WAV audio bytes) or np.ndarray (audio waveform)
            word_text: Word string (e.g. 'banana', 'record')
            alignments: Phone intervals from CTC-Viterbi or MFA
            method: 'ctc_viterbi' (default, embedded CTC phone regions),
                    'mfa' (external forced-alignment regions), or 'uniform'

        Returns:
            dict containing:
                - 'word': str
                - 'truth': list[int] (canonical stress pattern, e.g. [0, 1, 0])
                - 'infer': list[int] (predicted stress pattern with exactly one primary stress)
                - 'detected_syllables_count': int
                - 'stress_index': int
                - 'confidence': float
                - 'raw_scores': list[float]
                - 'syllables': list[dict] (per-syllable acoustic prominence debug info)
                - 'method': str
        """
        # 1. Load Audio
        if isinstance(audio_input, (bytes, io.BytesIO)):
            if isinstance(audio_input, bytes):
                buf = io.BytesIO(audio_input)
            else:
                buf = audio_input
            y, sr = librosa.load(buf, sr=16000)
        elif isinstance(audio_input, np.ndarray):
            y = audio_input.astype(np.float32)
            sr = 16000
        else:
            return {"error": f"Unsupported audio input type: {type(audio_input)}"}

        if len(y) == 0:
            return {"error": "Audio stream is empty"}

        # 2. Extract Features
        if method == "uniform":
            features, syl_meta, debug_data = extract_full_features_uniform(y, sr, word_text)
        else:
            features, syl_meta, debug_data = extract_full_features(y, sr, word_text, alignments=alignments)

        num_syl = len(debug_data)
        if num_syl == 0:
            return {"error": f"No syllables found for word '{word_text}'"}

        # 3. Ground Truth Stress Pattern from CMUdict/MOP
        truth_stress = [1 if s["is_stressed_truth"] else 0 for s in debug_data]
        # In rare case where canonical dictionary has no primary stress marker, set syllable 0 as primary
        if sum(truth_stress) == 0:
            truth_stress[0] = 1

        # 4. Scoring via Neural Model or Acoustic Prominence Sequential Engine
        scores = None
        if self.model is not None:
            try:
                feat_input = np.copy(features)
                if self.scaler_mean is not None and self.scaler_std is not None:
                    feat_input = (feat_input - self.scaler_mean) / (self.scaler_std + 1e-7)

                pred = self.model.predict(feat_input, verbose=0)
                pred_sq = np.squeeze(pred)
                if pred_sq.ndim == 0:
                    pred_sq = np.array([float(pred_sq)])

                if len(pred_sq) > num_syl:
                    pred_sq = pred_sq[:num_syl]
                elif len(pred_sq) < num_syl:
                    pred_sq = np.pad(pred_sq, (0, num_syl - len(pred_sq)))

                scores = pred_sq
            except Exception as e:
                print(f"⚠️ Model inference failed ({e}). Falling back to Acoustic Prominence.")
                scores = None

        # Acoustic prominence sequential scoring (Yarra et al. 2019, Mallela et al. 2024)
        prom_scores = np.array([s.get("prominence_score", 0.0) for s in debug_data], dtype=np.float64)

        if scores is not None:
            # Standardize both distributions before ensemble
            scores_norm = (scores - np.mean(scores)) / (np.std(scores) + 1e-6)
            prom_norm = (prom_scores - np.mean(prom_scores)) / (np.std(prom_scores) + 1e-6)
            combined_scores = 0.4 * scores_norm + 0.6 * prom_norm
            scores = combined_scores
        else:
            scores = prom_scores

        # Convert scores to probabilities via softmax
        probs = self.softmax(scores, temperature=1.0)

        # 5. Argmax Post-Processing Rule (Mallela et al. Interspeech 2024)
        # English lexical stress phonology: exactly one primary stressed syllable per word!
        stress_index = int(np.argmax(probs))
        infer_stress = [0] * num_syl
        infer_stress[stress_index] = 1
        confidence = float(probs[stress_index])

        # Attach predictions to per-syllable debug info
        for i in range(num_syl):
            debug_data[i]["score"] = float(scores[i])
            debug_data[i]["probability"] = float(probs[i])
            debug_data[i]["is_stressed_infer"] = (i == stress_index)

        return {
            "status": "success",
            "word": word_text,
            "detected_syllables_count": num_syl,
            "syllable_count": num_syl,
            "truth": truth_stress,
            "infer": infer_stress,
            "stress_index": stress_index,
            "confidence": round(confidence, 4),
            "stress_probability": round(confidence, 4),
            "raw_scores": [round(float(p), 4) for p in probs],
            "syllables": debug_data,
            "feature_debug": debug_data,
            "method": method
        }
