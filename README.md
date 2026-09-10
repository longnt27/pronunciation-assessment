# Alignment Only Where Needed for Pronunciation Assessment

This repository evaluates a hybrid pronunciation-assessment pipeline that
avoids external forced alignment where it is unnecessary:

```text
audio + canonical phones -> Cao et al. CTC posterior matrix
                           |-> 41-D alignment-free vectors -> phone GOPT
                           `-> transcript-conditioned CTC-Viterbi intervals
                               -> 38-D syllable features
                               -> Mallela-inspired LSTM/attention stress scorer
```

The key distinction is deliberate. Phoneme scoring follows Cao et al.'s
alignment-free vector method; Viterbi is used only to localize syllables for
stress features. The decoder is conditioned on the known phone transcript, but
it is embedded in the resident CTC model and does not launch MFA/Kaldi.

## Three-system experiment

The official speaker-disjoint SpeechOcean762 split is used throughout. Learned
GOPT heads are trained with five deterministic initializations; numbers below
are mean ± sample standard deviation. Phone evaluation uses the same 46,119
held-out phones for all systems. Stress evaluation uses the same 2,354
polysyllabic words, including 72 expert-rated stress errors.

| Complete system | Phone branch | Stress branch |
|---|---|---|
| MFA modular baseline | MFA-aligned 78-D CTC posterior/ratio vector + phone-only GOPT | Mallela-inspired network with MFA intervals |
| Proposed Cao + Viterbi | Cao 41-D alignment-free vector + phone-only GOPT | Same network with CTC-Viterbi intervals |
| Gong et al. joint GOPT | MFA-aligned 78-D vector + multi-task GOPT | GOPT word-stress head |

| Held-out metric | MFA modular | Proposed | Joint GOPT |
|---|---:|---:|---:|
| Phone PCC | 0.413 ± 0.013 | **0.645 ± 0.009** | 0.418 ± 0.005 |
| Phone SRCC | 0.381 ± 0.004 | **0.432 ± 0.003** | 0.378 ± 0.003 |
| Stress-correctness AUROC | 0.662 | 0.681 | **0.707 ± 0.030** |
| Stress-grade PCC | 0.077 | 0.100 | **0.189 ± 0.017** |
| Canonical location accuracy¹ | **85.85%** | 85.14% | not produced |
| Complete CPU latency | 234.6 ms | 212.6 ms | **192.5 ms** |

¹ Among 2,282 words rated stress-correct by the experts.

The proposed phone PCC exceeds the MFA modular baseline by 0.232 ± 0.019 over
the five matched runs. For the fixed stress network, Viterbi minus MFA is −0.7
percentage points in location accuracy (95% speaker-bootstrap CI −1.81 to
+0.48) and +0.019 AUROC (95% CI −0.008 to +0.053). Thus the experiment shows a
large phone-scoring gain and no statistically resolved stress difference; it
does not prove strict stress non-inferiority because no margin was preregistered.

Latency is one request at a time on the same ARM64 CPU, with four PyTorch
intra-op threads for every system and exactly one MFA job. The proposed system
is 1.10× faster than the complete modular MFA pipeline. Joint GOPT is faster
still because it omits the separate TensorFlow stress network. This is an
end-to-end system result, not an alignment-only timing comparison.

The controlled joint-GOPT implementation is an adaptation to the same CTC
front end. As an implementation check, the unmodified official GOPT checkpoint
was also run on its released LibriSpeech tensors: phone PCC 0.618 and word
stress PCC 0.325, matching its published repository values (0.616 and 0.326).

The manuscript is in [paper/main.tex](paper/main.tex), the compact audit is in
[paper/PAPER_SUMMARY.md](paper/PAPER_SUMMARY.md), and machine-readable results
are in `paper/results/`.

## Reproduce the comparison

The Cao and GOPT repositories/checkpoints and SpeechOcean762 are external
research assets and are intentionally kept under ignored `tmp/` paths.

```bash
source venv/bin/activate

python benchmarks/compare_three_systems.py features \
  --feature-batch-size 8 --overwrite

python benchmarks/compare_three_systems.py stress

python benchmarks/compare_three_systems.py train \
  --seeds 5 --epochs 100 --batch-size 25

python benchmarks/compare_three_systems.py latency \
  --device cpu --latency-samples 100

python benchmarks/compare_three_systems.py validate-gopt

python paper/generate_three_system_figure.py
```

Core numerical tests compare CTC losses with exhaustive path enumeration,
check the 41-D feature ordering, verify repeated-phone Viterbi separation,
retain diagonal phone substitutions during MFA matching, and validate model
shapes:

```bash
python -m unittest -v tests.test_three_system_comparison
```

## API

The deployed service still exposes the embedded CTC-Viterbi assessment path:

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

## Repository layout

```text
benchmarks/compare_three_systems.py   Feature, stress, training, latency, audit stages
benchmarks/three_system_models.py     Self-contained GOPT architecture variants
server/services/gop_service.py        Resident CTC model and Viterbi decoder
server/services/stress_service.py     Mallela-inspired stress inference
tests/test_three_system_comparison.py Numerical and architecture tests
paper/main.tex                        Scientific manuscript
paper/results/                        Checked-in aggregate results and paired rows
```

## Scope

The stress checkpoint is a local Mallela-inspired architectural adaptation,
not Mallela et al.'s ISLE checkpoint or an exact reproduction of their corpus
experiment. The joint GOPT row is a controlled CTC-front-end adaptation, while
the official checkpoint check is reported separately. Speed ratios are
specific to this hardware, software stack, model, request pattern, and worker
configuration.
