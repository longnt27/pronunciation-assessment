# Embedded CTC-Viterbi vs MFA: Result Audit

## Claim

The proposed assessment pipeline replaces the external MFA/Kaldi alignment job
with a transcript-conditioned CTC-Viterbi decoder embedded in the resident
phoneme model. The same Viterbi phone regions drive both phone LPP scoring and
the Mallela-inspired sequential stress branch.

The defensible headline is protocol-qualified:

> On SpeechOcean762 and the measured ARM64 CPU, embedded CTC-Viterbi is 1.31×
> faster than the complete one-job MFA-conditioned phone-and-stress pipeline
> and shows no observed accuracy trade-off.

## Data

- SpeechOcean762 official train/test split: 2,500 + 2,500 utterances.
- MFA 3.4.2 aligned 2,447 test utterances; 53 OOV-affected utterances were not
  exported under `--ignore_oovs`.
- Phone calibration: 43,594 matched train phones.
- Phone evaluation: 43,613 paired held-out phones from 2,447 utterances.
- Stress evaluation: 1,621 paired polysyllabic words, including 46 human-rated
  stress errors.

## Accuracy

| Metric | CTC-Viterbi | MFA | Paired difference |
|---|---:|---:|---:|
| Phone PCC, per-phone quadratic | 0.459 | 0.317 | +0.142 [0.120, 0.164] |
| Phone SRCC, per-phone quadratic | 0.376 | 0.302 | — |
| Stress location, human-correct words | 74.9% | 69.5% | +5.4 pp [2.8, 8.0] |
| Correct-stress AUROC | 0.726 | 0.695 | — |

Confidence intervals are 2,000-iteration paired speaker bootstraps. Both lower
bounds exceed zero, so the experiment supports no accuracy loss and observed
superiority. This result applies to the paired SpeechOcean subset and current
models; it does not reproduce Mallela et al.'s ISLE result.

## Latency

The primary latency experiment executes the complete paired workload: 43,613
phone LPP scores and 1,621 polysyllabic stress predictions over 2,447
utterances. Both conditions use one top-level assessment process and the same
four PyTorch intra-op threads; MFA uses one alignment job. Condition order is
alternated by utterance after warm-up.

| Mean component per paired utterance | CTC-Viterbi | MFA |
|---|---:|---:|
| CTC encoding | 62.9 ms | 62.8 ms |
| Phone regions + LPP | 4.4 ms | 0.2 ms |
| Stress branch | 31.3 ms | 29.4 ms |
| External alignment, amortized | — | 36.5 ms |
| **Complete pipeline** | **98.6 ms** | **128.8 ms** |

The complete one-job ratio is 1.31× in favor of CTC-Viterbi. An exploratory
four-job MFA alignment-only run reached 28.0 ms/input, but omits the downstream
phone and stress work and is not a complete-pipeline result.

## Controlled comparison

For phone accuracy, both conditions use one frozen Wav2Vec2 phoneme posterior
matrix and compute canonical-label mean log posterior. The interval source is
the experimental variable. Per-phone quadratic regressors are fitted only on
the official training split, with a global fallback for phones below 20
samples.

For stress, both conditions use the same waveform, syllabification, 38-D
feature extractor, scaler, LSTM/attention checkpoint, and word-level argmax.
Only the phone intervals differ.

## Terminology

The manuscript calls the proposal **embedded CTC-Viterbi**,
**external-aligner-free**, and **MFA-free**. It does not call the decoder
unconstrained: its trellis is conditioned on the known canonical phone
sequence. This operational wording distinguishes the deployed architecture
from MFA while avoiding an unproductive terminological dispute over whether
all transcript-conditioned decoding is a form of forced alignment.

## Files

- `phone_accuracy.json` and paired phone CSV
- `stress_accuracy.json` and paired word CSV
- `full_pipeline_latency.json` and per-utterance CSV
- `mfa_alignment_latency_1_job.json`
- diagnostic component results for CTC batch, MFA four-job alignment, and
  interactive alignment

Figures are regenerated from these JSON files with `paper/generate_figures.py`.
