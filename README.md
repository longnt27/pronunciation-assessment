# LingoStress: Embedded CTC-Viterbi Pronunciation Assessment

LingoStress uses one resident phoneme CTC model to obtain a
transcript-conditioned Viterbi path. The path serves two outputs:

```text
audio + canonical phones -> phoneme CTC encoder -> CTC-Viterbi phone regions
                                                   |-> phone LPP scores
                                                   `-> syllable regions
                                                       -> 38-D features
                                                       -> LSTM/attention stress model
```

There is no peak-alignment path and no Montreal Forced Aligner (MFA) dependency
in the deployed service. In this repository, “external-aligner-free” means that
the assessment request does not launch or consume an MFA/Kaldi alignment job.
The Viterbi decoder is still conditioned on the canonical phone sequence; it is
not unconstrained phone recognition.

## Scientific result

The proposed embedded CTC-Viterbi condition was compared with MFA 3.4.2 using
SpeechOcean762. For accuracy, the two conditions use the same Wav2Vec2 CTC
emissions, target phones, frame log-posterior score, calibration procedure, and
stress model. Only the source of phone regions changes.

| Held-out SpeechOcean762 metric | CTC-Viterbi | MFA |
|---|---:|---:|
| Paired phone tokens | 43,613 | 43,613 |
| Raw phone PCC | 0.401 | 0.216 |
| Per-phone quadratic PCC | **0.459** | 0.317 |
| Per-phone quadratic SRCC | **0.376** | 0.302 |
| Canonical stress-location accuracy, human-correct words | **74.9%** | 69.5% |
| Correct-stress AUROC | **0.726** | 0.695 |

The paired speaker-bootstrap difference in calibrated phone PCC is +0.142
(95% CI +0.120 to +0.164). The stress-location difference is +5.4 percentage
points (95% CI +2.8 to +8.0). These intervals exclude an accuracy loss under
this protocol.

The speed comparison now times the complete paired workload, not alignment in
isolation: 43,613 phone LPP scores and 1,621 polysyllabic stress predictions
over 2,447 utterances. Each condition uses one top-level assessment process;
PyTorch uses the same four intra-op CPU threads in both, and MFA uses one job.

| ARM64 CPU complete-pipeline mean | CTC-Viterbi | MFA 3.4.2 |
|---|---:|---:|
| CTC encoding | 62.9 ms | 62.8 ms |
| Phone region construction + scoring | 4.4 ms | 0.2 ms |
| Stress scoring | 31.3 ms | 29.4 ms |
| External MFA alignment, amortized per paired output | — | 36.5 ms |
| **Complete phone + stress pipeline** | **98.6 ms** | **128.8 ms** |

The embedded CTC-Viterbi pipeline is therefore 1.31× faster end to end in the
matched one-job batch protocol. A four-job MFA alignment-only run reached 28.0
ms/input, but that number excludes the required CTC phone scorer and stress
branch and is not used as the complete-pipeline comparison.

MFA aligned 2,447 of 2,500 test utterances with `--ignore_oovs`. Phone results
use 43,594 matched training phones and 43,613 paired test phones. Stress results
use 1,621 paired polysyllabic test words: 1,575 human-accepted stress tokens and
46 human stress-error tokens. The stress checkpoint is a local
Mallela-inspired LSTM/attention adaptation; it is not Mallela et al.'s released
ISLE model.

The manuscript is [paper/main.tex](paper/main.tex), with a compact result audit
in [paper/PAPER_SUMMARY.md](paper/PAPER_SUMMARY.md). Machine-readable summaries
and paired rows are in `paper/results/`.

## Method

Given CTC log posteriors over `T` frames and `U` canonical phones, the decoder
builds the standard `2U + 1` CTC state sequence:

```text
blank, phone_1, blank, phone_2, ..., blank, phone_U, blank
```

Viterbi dynamic programming permits a self-loop, a one-state advance, and a
two-state skip when neighboring phone labels differ. Repeated phones must pass
through a blank. Backtracking assigns frames to phone states. Each phone score
is the mean log posterior of its canonical label on those frames. Adjacent
phone intervals split intervening blank regions at their midpoints.

For stress, maximal-onset syllabification groups aligned phones. The vowel span
defines the nucleus region, from which duration, intensity, pitch, and spectral
features are extracted. These acoustic features are combined with contextual
features and passed to the sequential stress checkpoint. Word-level argmax
enforces one primary-stress prediction.

## API

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r server/requirements.txt
uvicorn server.main:app --host 127.0.0.1 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/assess \
  -F "word=banana" \
  -F "audio=@recording.wav" \
  -F "method=ctc_viterbi"
```

`ctc_viterbi` is the only production GOP method. The response identifies
`scoring_method`, `timing_method`, `external_aligner_used`,
`canonical_sequence_constrained`, per-phone intervals/scores, stress output,
and stage latency.

## Verification

The trellis tests compare the implementation against exhaustive enumeration of
all valid CTC paths and verify repeated-phone blank separation:

```bash
source venv/bin/activate
python -m unittest tests/test_ctc_viterbi.py -v
```

Paired phone accuracy:

```bash
python benchmarks/evaluate_ctc_viterbi_vs_mfa.py \
  --dataset /path/to/speechocean762 \
  --train-mfa-textgrids /path/to/mfa_train_textgrids \
  --test-mfa-textgrids /path/to/mfa_test_textgrids \
  --output paper/results/phone_accuracy.json
```

Paired stress accuracy:

```bash
python benchmarks/evaluate_stress_alignment.py \
  --dataset /path/to/speechocean762 \
  --mfa-textgrids /path/to/mfa_test_textgrids \
  --output paper/results/stress_accuracy.json
```

Complete paired phone-and-stress latency:

```bash
python benchmarks/benchmark_full_assessment_pipeline.py \
  --dataset /path/to/speechocean762 \
  --mfa-textgrids /path/to/mfa_test_textgrids \
  --mfa-batch-result /path/to/mfa_one_job_timing.json \
  --output paper/results/full_pipeline_latency.json
```

The MFA corpus preparation and component timing scripts are also in
`benchmarks/`. Result JSON records versions, paths, hardware platform, counts,
failures, worker settings, and protocol details.

## Repository layout

```text
server/services/gop_service.py            CTC encoder, Viterbi decoder, phone GOP
server/utils/audio_features.py            Syllabification and stress features
server/services/stress_service.py         Sequential stress inference
benchmarks/evaluate_ctc_viterbi_vs_mfa.py Paired phone experiment
benchmarks/evaluate_stress_alignment.py   Paired stress experiment
benchmarks/benchmark_full_assessment_pipeline.py Complete paired latency
benchmarks/run_mfa_benchmark.py           External MFA batch timing
tests/test_ctc_viterbi.py                 Exact small-lattice tests
paper/                                    Manuscript, figures, result bundle
```

## Scope of the claim

The measured claim is: on this ARM64 CPU and paired SpeechOcean762 protocol,
embedded CTC-Viterbi is 1.31× faster than the one-job complete MFA-conditioned
phone-and-stress pipeline, with no observed phone or stress-accuracy trade-off.
It is not a claim about every hardware, model, worker count, or corpus, nor that
transcript-conditioned decoding is unconstrained recognition.
