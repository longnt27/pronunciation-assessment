"""
Audio Feature Extraction for Syllable Stress Detection.
Based on:
- Yarra et al. (SLATE 2019): "Comparison of automatic syllable stress detection quality
  with time-aligned boundaries and context dependencies"
- Mallela et al. (Interspeech 2024): "A comparative analysis of sequential models that
  integrate syllable dependency for automatic syllable stress detection"
"""

import re
import numpy as np
import librosa
from g2p_en import G2p
import cmudict

# Initialize CMUdict and G2P
_cmu_dict = cmudict.dict()
_g2p = G2p()

# ARPAbet vowel phonemes
ARPABET_VOWEL_BASES = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'
}

# Phonotactically legal English single onsets (all consonants except NG, ZH)
VALID_SINGLE_ONSETS = {
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z'
}

# Phonotactically legal English cluster onsets
VALID_CLUSTER_ONSETS = {
    # Stop + Liquid
    ('P', 'L'), ('P', 'R'), ('B', 'L'), ('B', 'R'),
    ('T', 'R'), ('D', 'R'),
    ('K', 'L'), ('K', 'R'), ('G', 'L'), ('G', 'R'),
    # Stop + Glide
    ('T', 'W'), ('D', 'W'), ('K', 'W'), ('G', 'W'),
    ('P', 'Y'), ('B', 'Y'), ('T', 'Y'), ('D', 'Y'), ('K', 'Y'), ('G', 'Y'),
    # Fricative + Liquid/Glide
    ('F', 'L'), ('F', 'R'), ('F', 'Y'), ('V', 'Y'),
    ('TH', 'R'), ('TH', 'W'), ('SH', 'R'),
    ('S', 'L'), ('S', 'W'), ('S', 'M'), ('S', 'N'),
    ('HH', 'Y'),
    # S + Voiceless Stop
    ('S', 'P'), ('S', 'T'), ('S', 'K'),
    # S + Stop + Liquid/Glide
    ('S', 'P', 'L'), ('S', 'P', 'R'), ('S', 'P', 'Y'),
    ('S', 'T', 'R'), ('S', 'T', 'Y'),
    ('S', 'K', 'L'), ('S', 'K', 'R'), ('S', 'K', 'W'), ('S', 'K', 'Y'),
    # Nasal/Liquid + Glide
    ('M', 'Y'), ('N', 'Y'), ('L', 'Y')
}


def is_vowel(phoneme: str) -> bool:
    """Checks whether an ARPAbet phoneme is a vowel."""
    base = re.sub(r'\d+', '', phoneme).upper()
    return base in ARPABET_VOWEL_BASES or any(c.isdigit() for c in phoneme)


def is_valid_onset(cluster: list or tuple) -> bool:
    """Checks whether a consonant cluster is a valid English syllable onset."""
    if len(cluster) == 0:
        return True
    clean = tuple(re.sub(r'\d+', '', c).upper() for c in cluster)
    if len(clean) == 1:
        return clean[0] in VALID_SINGLE_ONSETS
    return clean in VALID_CLUSTER_ONSETS


def syllabify_mop(phonemes: list) -> list:
    """
    Syllabifies a list of ARPAbet phonemes using the Maximal Onset Principle (MOP).
    Assigns intervocalic consonants to the following syllable's onset to the maximum
    extent allowed by English phonotactics; remaining consonants form the coda of the
    preceding syllable.

    Returns:
        List of dicts, each containing:
            - 'phonemes': list of phonemes
            - 'onset': list of onset consonants
            - 'nucleus': vowel phoneme (with stress digit if present)
            - 'coda': list of coda consonants
            - 'is_primary_stress': bool
            - 'stress_marker': int (0=unstressed, 1=primary, 2=secondary)
    """
    cleaned_phonemes = [p for p in phonemes if p.strip() and p not in [" ", "'", ",", ".", "?", "!"]]
    vowel_indices = [i for i, p in enumerate(cleaned_phonemes) if is_vowel(p)]

    if not vowel_indices:
        # Fallback for vowelless token
        return [{
            'phonemes': cleaned_phonemes,
            'onset': cleaned_phonemes,
            'nucleus': cleaned_phonemes[0] if cleaned_phonemes else 'AH0',
            'coda': [],
            'is_primary_stress': False,
            'stress_marker': 0
        }]

    syllables = []
    # Syllable 0 onset is everything before the first vowel
    first_onset = cleaned_phonemes[:vowel_indices[0]]
    current_onset = first_onset

    for k in range(len(vowel_indices)):
        v_idx = vowel_indices[k]
        nucleus = cleaned_phonemes[v_idx]

        # Determine stress marker
        stress_match = re.search(r'\d+', nucleus)
        stress_marker = int(stress_match.group()) if stress_match else 0
        is_primary = (stress_marker == 1)

        if k < len(vowel_indices) - 1:
            next_v_idx = vowel_indices[k + 1]
            intervocalic = cleaned_phonemes[v_idx + 1:next_v_idx]

            # Maximal Onset Principle:
            # Try to assign the longest valid suffix of intervocalic cluster to next onset
            split_point = len(intervocalic)
            for j in range(len(intervocalic)):
                candidate_onset = intervocalic[j:]
                if is_valid_onset(candidate_onset):
                    split_point = j
                    break

            coda = intervocalic[:split_point]
            next_onset = intervocalic[split_point:]
        else:
            coda = cleaned_phonemes[v_idx + 1:]
            next_onset = []

        syl_phonemes = current_onset + [nucleus] + coda
        syllables.append({
            'phonemes': syl_phonemes,
            'onset': current_onset,
            'nucleus': nucleus,
            'coda': coda,
            'is_primary_stress': is_primary,
            'stress_marker': stress_marker
        })
        current_onset = next_onset

    return syllables


def get_syllables_and_phonemes(word: str) -> list:
    """
    Lookup word in CMUdict (fallback to G2P) and apply MOP syllabification.
    Returns:
        List of syllable phoneme lists (e.g. [['B', 'AH0'], ['N', 'AE1'], ['N', 'AH0']])
    """
    word_clean = word.lower().strip()
    cmu_entries = _cmu_dict.get(word_clean)

    if cmu_entries:
        phonemes = cmu_entries[0]
    else:
        raw_phonemes = _g2p(word_clean)
        phonemes = [p for p in raw_phonemes if p.strip() and p not in [" ", "'", ",", ".", "?", "!"]]

    mop_syllables = syllabify_mop(phonemes)
    return [s['phonemes'] for s in mop_syllables]


def get_word_syllables(word: str, pronunciation: list or None = None, alignments: list or None = None) -> list:
    """
    Detailed syllabification info for a word.
    Supports selecting pronunciation variant based on provided phonemes or forced alignment.
    Returns:
        List of Syllable dicts with onset, nucleus, coda, phonemes, and stress truth.
    """
    if pronunciation:
        return syllabify_mop(pronunciation)

    word_clean = word.lower().strip()
    cmu_entries = _cmu_dict.get(word_clean)

    if cmu_entries:
        if len(cmu_entries) == 1 or not alignments:
            phonemes = cmu_entries[0]
        else:
            # If alignments provided, match the variant matching the alignment phonemes
            ali_phones = [re.sub(r'\d+', '', a.get('phoneme', '')).upper() for a in alignments if 'phoneme' in a]
            best_e = cmu_entries[0]
            max_ov = -1
            for e in cmu_entries:
                clean_e = [re.sub(r'\d+', '', p).upper() for p in e]
                ov = sum(1 for a, b in zip(ali_phones, clean_e) if a == b)
                if ov > max_ov:
                    max_ov = ov
                    best_e = e
            phonemes = best_e
    else:
        raw_phonemes = _g2p(word_clean)
        phonemes = [p for p in raw_phonemes if p.strip() and p not in [" ", "'", ",", ".", "?", "!"]]

    return syllabify_mop(phonemes)


def resolve_syllable_alignments(syllables: list, alignments: list or None, total_duration: float, y=None, sr: int = 16000) -> list:
    """
    Maps phone or syllable interval data to MOP syllables.

    Supported alignment formats:
    1. List of phoneme dicts: [{'phoneme': 'B', 'start': 0.1, 'end': 0.2}, ...]
    2. List of syllable dicts: [{'start': 0.1, 'end': 0.4, 'nucleus_start': 0.2, 'nucleus_end': 0.35}, ...]
    3. None: Uses deterministic uniform regions as a compatibility fallback.
    """
    num_syls = len(syllables)
    if num_syls == 0:
        return []

    resolved = []

    # Zero-duration placeholders from a score-only method are not alignments.
    # Treat them as absent so the acoustic fallback is used instead of feeding
    # empty waveform slices to the stress model.
    valid_alignments = []
    if isinstance(alignments, list):
        for alignment in alignments:
            try:
                start = float(alignment.get("start", 0.0))
                end = float(alignment.get("end", 0.0))
            except (AttributeError, TypeError, ValueError):
                continue
            if np.isfinite(start) and np.isfinite(end) and end > start:
                valid_alignments.append(alignment)
    alignments = valid_alignments

    # Case 1: Syllable-level alignments already provided
    if (
        alignments
        and len(alignments) == num_syls
        and "start" in alignments[0]
        and "phoneme" not in alignments[0]
    ):
        for i, s in enumerate(syllables):
            ali = alignments[i]
            s_start = max(0.0, float(ali.get("start", 0.0)))
            s_end = min(total_duration, float(ali.get("end", total_duration)))
            if s_end <= s_start:
                s_end = min(total_duration, s_start + 0.1)

            n_start = float(ali.get("nucleus_start", s_start + (s_end - s_start) * 0.2))
            n_end = float(ali.get("nucleus_end", s_start + (s_end - s_start) * 0.8))

            resolved.append({
                **s,
                "start": s_start,
                "end": s_end,
                "nucleus_start": n_start,
                "nucleus_end": n_end
            })
        return resolved

    # Case 2: Phoneme-level intervals from embedded CTC-Viterbi or MFA
    if alignments and "phoneme" in alignments[0]:
        flat_phonemes = []
        for s_idx, s in enumerate(syllables):
            for p in s['phonemes']:
                flat_phonemes.append((s_idx, p))

        # Match alignments to syllables
        ali_idx = 0
        syl_phone_matches = {i: [] for i in range(num_syls)}

        for s_idx, p in flat_phonemes:
            clean_target = re.sub(r'\d+', '', p).upper()
            found = False
            for search_k in range(ali_idx, min(len(alignments), ali_idx + 4)):
                ali_phone = re.sub(r'\d+', '', alignments[search_k].get("phoneme", "")).upper()
                if ali_phone == clean_target:
                    syl_phone_matches[s_idx].append(alignments[search_k])
                    ali_idx = search_k + 1
                    found = True
                    break
            if not found and ali_idx < len(alignments):
                syl_phone_matches[s_idx].append(alignments[ali_idx])
                ali_idx += 1

        for i, s in enumerate(syllables):
            matched = syl_phone_matches[i]
            if matched:
                s_start = matched[0]["start"]
                s_end = matched[-1]["end"]

                # Locate nucleus phoneme
                clean_nuc = re.sub(r'\d+', '', s['nucleus']).upper()
                nuc_match = next((m for m in matched if re.sub(r'\d+', '', m.get("phoneme", "")).upper() == clean_nuc), None)
                if nuc_match:
                    n_start = nuc_match["start"]
                    n_end = nuc_match["end"]
                else:
                    n_start = s_start + (s_end - s_start) * 0.2
                    n_end = s_start + (s_end - s_start) * 0.8
            else:
                # Fallback proportion
                s_start = (i / num_syls) * total_duration
                s_end = ((i + 1) / num_syls) * total_duration
                n_start = s_start + (s_end - s_start) * 0.2
                n_end = s_start + (s_end - s_start) * 0.8

            resolved.append({
                **s,
                "start": max(0.0, float(s_start)),
                "end": min(total_duration, float(s_end)),
                "nucleus_start": max(0.0, float(n_start)),
                "nucleus_end": min(total_duration, float(n_end))
            })
        return resolved

    # Case 3: Compatibility fallback only; production supplies phone intervals.
    for i, s in enumerate(syllables):
        s_start = (i / num_syls) * total_duration
        s_end = ((i + 1) / num_syls) * total_duration
        resolved.append({
            **s,
            "start": s_start,
            "end": s_end,
            "nucleus_start": s_start + (s_end - s_start) * 0.2,
            "nucleus_end": s_start + (s_end - s_start) * 0.8
        })
    return resolved


def extract_acoustic_prominence_features(y_syl: np.ndarray, y_nuc: np.ndarray, sr: int, syl_dur: float, nuc_dur: float) -> dict:
    """
    Extracts prominence features defined in Yarra et al. (2019) and Mallela et al. (2024):
    - Duration of nucleus & syllable, ratio
    - RMS intensity mean, peak, sum, slope
    - Pitch (F0) mean, peak, slope via librosa.pyin
    - Spectral flatness (harmonicity vs noise) and rolloff
    """
    # 1. Pitch / F0 (prefer nucleus where voicing is strongest, fallback to syllable)
    target_pitch_audio = y_nuc if len(y_nuc) >= 512 else y_syl
    if len(target_pitch_audio) >= 512:
        try:
            f0, _, _ = librosa.pyin(target_pitch_audio, fmin=60, fmax=400, sr=sr, frame_length=min(1024, len(target_pitch_audio)))
            f0_valid = f0[~np.isnan(f0)]
        except Exception:
            f0_valid = np.array([])
    else:
        f0_valid = np.array([])

    if len(f0_valid) == 0:
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_max = 0.0
        pitch_min = 0.0
        pitch_range = 0.0
        pitch_slope = 0.0
    else:
        pitch_mean = float(np.mean(f0_valid))
        pitch_std = float(np.std(f0_valid))
        pitch_max = float(np.max(f0_valid))
        pitch_min = float(np.min(f0_valid))
        pitch_range = float(pitch_max - pitch_min)
        pitch_slope = float(np.polyfit(np.arange(len(f0_valid)), f0_valid, 1)[0]) if len(f0_valid) > 1 else 0.0

    # 2. RMS Intensity (Syllable + Nucleus)
    if len(y_syl) >= 256:
        rms_syl = librosa.feature.rms(y=y_syl)[0]
    else:
        rms_syl = np.array([0.0])

    if len(y_nuc) >= 256:
        rms_nuc = librosa.feature.rms(y=y_nuc)[0]
    else:
        rms_nuc = rms_syl

    rms_mean = float(np.mean(rms_nuc))
    rms_std = float(np.std(rms_nuc))
    rms_max = float(np.max(rms_nuc))
    rms_min = float(np.min(rms_nuc))
    rms_range = float(rms_max - rms_min)
    rms_slope = float(np.polyfit(np.arange(len(rms_nuc)), rms_nuc, 1)[0]) if len(rms_nuc) > 1 else 0.0
    rms_sum = float(np.sum(rms_syl))

    # 3. Spectral Flatness & Rolloff
    if len(y_syl) >= 512:
        try:
            spec_flat = librosa.feature.spectral_flatness(y=y_syl)[0]
            spec_roll = librosa.feature.spectral_rolloff(y=y_syl, sr=sr)[0]
            flat_mean = float(np.mean(spec_flat))
            flat_max = float(np.max(spec_flat))
            flat_min = float(np.min(spec_flat))
            roll_mean = float(np.mean(spec_roll))
            roll_max = float(np.max(spec_roll))
            roll_min = float(np.min(spec_roll))
        except Exception:
            flat_mean = flat_max = flat_min = 0.0
            roll_mean = roll_max = roll_min = 0.0
    else:
        flat_mean = flat_max = flat_min = 0.0
        roll_mean = roll_max = roll_min = 0.0

    # Assemble standard 19-dim acoustic vector (strictly compatible with checkpoint scaler)
    vector_19 = np.array([
        syl_dur,
        pitch_mean, pitch_std, pitch_max, pitch_min, pitch_range, pitch_slope,
        rms_mean, rms_std, rms_max, rms_min, rms_range, rms_slope,
        flat_mean, flat_max, flat_min,
        roll_mean, roll_max, roll_min
    ], dtype=np.float32)

    debug_dict = {
        "syllable_duration": round(syl_dur, 4),
        "nucleus_duration": round(nuc_dur, 4),
        "nucleus_ratio": round(nuc_dur / max(syl_dur, 1e-4), 3),
        "rms_mean": round(rms_mean, 4),
        "rms_peak": round(rms_max, 4),
        "rms_sum": round(rms_sum, 4),
        "f0_mean": round(pitch_mean, 2),
        "f0_peak": round(pitch_max, 2),
        "f0_slope": round(pitch_slope, 4),
        "spectral_flatness": round(flat_mean, 4)
    }

    return {
        "vector_19": vector_19,
        "debug": debug_dict
    }


def extract_context_features_19(syllable_dict: dict, index: int, total_syls: int) -> np.ndarray:
    """
    Extracts 19 linguistic context features (Yarra et al. 2019):
    - Position in word (initial, medial, final)
    - Nucleus type (diphthong vs monophthong, vowel height/backness)
    - Syllable structure (onset presence, coda presence, phone count)
    - Neighbor context
    """
    feats = []

    # Group 1: Position in word (3)
    if index == 0:
        feats.extend([1.0, 0.0, 0.0])
    elif index == total_syls - 1:
        feats.extend([0.0, 0.0, 1.0])
    else:
        feats.extend([0.0, 1.0, 0.0])

    # Group 2: Nucleus Type (5)
    nucleus = syllable_dict.get('nucleus', 'UNK')
    nucleus_clean = re.sub(r'\d+', '', nucleus).upper()
    is_diphthong = 1.0 if nucleus_clean in ['AY', 'EY', 'OY', 'AW', 'OW'] else 0.0
    feats.append(is_diphthong)
    feats.extend([0.0, 1.0, 0.0, 1.0])  # Vowel height/backness

    # Group 3: Structure (5)
    has_onset = 1.0 if len(syllable_dict.get('onset', [])) > 0 else 0.0
    has_coda = 1.0 if len(syllable_dict.get('coda', [])) > 0 else 0.0
    num_phones = float(len(syllable_dict.get('phonemes', [])))
    feats.extend([has_onset, has_coda, num_phones, 0.0, 0.0])

    # Group 4: Neighbor Context (6)
    has_prev = 1.0 if index > 0 else 0.0
    has_next = 1.0 if index < total_syls - 1 else 0.0
    feats.extend([has_prev, has_next, 0.0, 0.0, 0.0, 0.0])

    return np.array(feats[:19], dtype=np.float32)


def compute_relative_acoustic_prominence(syllables_debug: list) -> list:
    """
    Sequential acoustic prominence scoring based on Yarra et al. (2019) & Mallela et al. (2024).
    Normalizes acoustic correlates across the syllables in the word:
    - Relative duration (z-score)
    - Relative RMS energy (z-score)
    - Relative F0 peak (z-score)
    - Sonority (inverted spectral flatness)
    """
    n = len(syllables_debug)
    if n == 1:
        return [1.0]

    durs = np.array([s['acoustic_features']['syllable_duration'] for s in syllables_debug])
    nuc_durs = np.array([s['acoustic_features']['nucleus_duration'] for s in syllables_debug])
    rms_peaks = np.array([s['acoustic_features']['rms_peak'] for s in syllables_debug])
    rms_means = np.array([s['acoustic_features']['rms_mean'] for s in syllables_debug])
    f0_peaks = np.array([s['acoustic_features']['f0_peak'] for s in syllables_debug])
    flats = np.array([s['acoustic_features']['spectral_flatness'] for s in syllables_debug])

    def zscore(arr):
        std = np.std(arr)
        if std < 1e-6:
            return np.zeros_like(arr)
        return (arr - np.mean(arr)) / std

    z_dur = zscore(durs * 0.5 + nuc_durs * 0.5)
    z_energy = zscore(rms_peaks * 0.6 + rms_means * 0.4)
    z_f0 = zscore(f0_peaks)
    z_flat = -zscore(flats)

    # Prominence weights reflecting English stress correlates (Duration, Intensity, Pitch, Sonority)
    prominence_scores = 0.35 * z_dur + 0.35 * z_energy + 0.20 * z_f0 + 0.10 * z_flat
    return prominence_scores.tolist()


def extract_full_features(y: np.ndarray, sr: int, word: str, alignments: list or None = None) -> tuple:
    """
    Main extraction function for Stress Service.
    Extracts features strictly within true syllable boundaries (eliminating uniform slicing).

    Args:
        y: Normalized audio waveform array (1D float32)
        sr: Sample rate (usually 16000)
        word: Target word text (e.g. 'banana')
        alignments: Optional CTC-Viterbi or MFA intervals

    Returns:
        tuple (features_38_tensor, syllables_metadata, debug_data)
        features_38_tensor: np.ndarray shape (1, num_syllables, 38)
    """
    total_duration = len(y) / sr

    # 1. Rule-based MOP syllabification
    syllables = get_word_syllables(word, alignments=alignments)
    num_syl = len(syllables)
    if num_syl == 0:
        return np.zeros((1, 1, 38), dtype=np.float32), [], []

    # 2. Resolve True Syllable and Vowel Time-Boundaries
    aligned_syllables = resolve_syllable_alignments(syllables, alignments, total_duration, y=y, sr=sr)

    sequence_features = []
    debug_data = []

    for i, s in enumerate(aligned_syllables):
        s_start = s['start']
        s_end = s['end']
        n_start = s['nucleus_start']
        n_end = s['nucleus_end']

        syl_start_idx = max(0, int(s_start * sr))
        syl_end_idx = min(len(y), int(s_end * sr))
        nuc_start_idx = max(0, int(n_start * sr))
        nuc_end_idx = min(len(y), int(n_end * sr))

        y_syl = y[syl_start_idx:syl_end_idx]
        y_nuc = y[nuc_start_idx:nuc_end_idx]

        syl_dur = max(0.01, (syl_end_idx - syl_start_idx) / sr)
        nuc_dur = max(0.01, (nuc_end_idx - nuc_start_idx) / sr)

        # Acoustic Prominence Features (19)
        ac_result = extract_acoustic_prominence_features(y_syl, y_nuc, sr, syl_dur, nuc_dur)
        ac_feats = ac_result["vector_19"]

        # Context Features (19)
        ctx_feats = extract_context_features_19(s, i, num_syl)

        # Combined 38 dims
        combined_38 = np.concatenate([ac_feats, ctx_feats])
        sequence_features.append(combined_38)

        debug_entry = {
            "syllable_index": i,
            "phonemes": s['phonemes'],
            "nucleus": s['nucleus'],
            "onset": s['onset'],
            "coda": s['coda'],
            "boundaries": {"start": float(round(s_start, 4)), "end": float(round(s_end, 4))},
            "nucleus_boundaries": {"start": float(round(n_start, 4)), "end": float(round(n_end, 4))},
            "acoustic_features": ac_result["debug"],
            "is_stressed_truth": s['is_primary_stress'],
            "stress_marker": s['stress_marker']
        }
        debug_data.append(debug_entry)

    # Compute Relative Acoustic Prominence Across the Word
    prominence_scores = compute_relative_acoustic_prominence(debug_data)
    for i in range(num_syl):
        debug_data[i]["prominence_score"] = round(float(prominence_scores[i]), 4)

    features_38_tensor = np.array([sequence_features], dtype=np.float32)
    return features_38_tensor, aligned_syllables, debug_data


def extract_full_features_uniform(y: np.ndarray, sr: int, word: str) -> tuple:
    """
    Baseline naive uniform splitting (len(y) // num_syl) for comparison experiments.
    Shows how duration contrast is destroyed and pitch/energy are smeared across slices.
    """
    syllables = get_word_syllables(word)
    num_syl = len(syllables)
    if num_syl == 0:
        return np.zeros((1, 1, 38), dtype=np.float32), [], []

    syl_len_samples = len(y) // num_syl
    sequence_features = []
    debug_data = []

    for i, s in enumerate(syllables):
        start_idx = i * syl_len_samples
        end_idx = (i + 1) * syl_len_samples if i < num_syl - 1 else len(y)
        y_syl = y[start_idx:end_idx]

        syl_dur = len(y_syl) / sr
        nuc_dur = syl_dur * 0.5

        ac_result = extract_acoustic_prominence_features(y_syl, y_syl, sr, syl_dur, nuc_dur)
        ctx_feats = extract_context_features_19(s, i, num_syl)

        combined_38 = np.concatenate([ac_result["vector_19"], ctx_feats])
        sequence_features.append(combined_38)

        debug_data.append({
            "syllable_index": i,
            "phonemes": s['phonemes'],
            "nucleus": s['nucleus'],
            "boundaries": {"start": float(round(start_idx / sr, 4)), "end": float(round(end_idx / sr, 4))},
            "acoustic_features": ac_result["debug"],
            "is_stressed_truth": s['is_primary_stress']
        })

    features_38_tensor = np.array([sequence_features], dtype=np.float32)
    return features_38_tensor, syllables, debug_data
