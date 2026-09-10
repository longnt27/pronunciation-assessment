# Three-System Result Audit

## Question

Can external forced alignment be removed from the parts of a complete
phoneme-and-stress assessment pipeline that do not need it?

The proposed answer is modular:

- phone score: Cao et al.'s 41-D alignment-free CTC vector + phone-only GOPT;
- syllable stress: a Mallela-inspired sequential network, localized with an
  embedded transcript-conditioned CTC-Viterbi path rather than MFA.

The comparison includes a strong modular MFA baseline and a third published
multi-output architecture, Gong et al.'s joint GOPT.

## Controlled systems

| System | Phone input/output | Stress input/output | External MFA |
|---|---|---|---:|
| MFA modular | 78-D MFA-aligned CTC LPP/LPR vector → phone GOPT | MFA intervals → fixed 38-D LSTM/attention scorer | yes |
| Proposed | Cao 41-D alignment-free vector → phone GOPT | CTC-Viterbi intervals → same fixed scorer | no |
| Joint GOPT | 78-D MFA-aligned vector → multi-task GOPT phone head | same GOPT word-stress head | yes |

Both phone-only heads and the joint head use the released GOPT topology:
24-dimensional tokens, learned position and canonical-phone embeddings, three
Transformer blocks, and one attention head. The controlled joint-GOPT row is
retrained on this experiment's CTC vectors. It is not presented as the
unchanged official Kaldi-front-end system.

## Data and protocol

- SpeechOcean762 official speaker-disjoint split: 2,500 train and 2,500 test
  utterances, 125 speakers per split.
- Cao features available: 47,076 train phones and 47,369 test phones.
- MFA-mapped phones: 46,064 train and 46,119 test. Levenshtein diagonal mapping
  retains substitutions and relabels intervals with the expert canonical phone.
- Common phone evaluation set: 46,119 test phones.
- Common stress set: 2,354 polysyllabic words from 1,578 utterances and all 125
  test speakers; 2,282 are stress-correct and 72 are stress-error tokens.
- Learned heads: 100 epochs, Adam at 1e-3 with the GOPT schedule, batch size 25,
  five deterministic independent seeds; no ensembling in reported accuracy.
- Fixed stress comparison: same waveform, word, syllabification, 38-D feature
  code, scaler, 42,061-parameter LSTM/attention checkpoint, softmax, and argmax.
  Only the interval source changes; the prominence ensemble is disabled.

## Accuracy

| Metric | MFA modular | Proposed Cao + Viterbi | Joint GOPT |
|---|---:|---:|---:|
| Phone PCC | 0.413 ± 0.013 | **0.645 ± 0.009** | 0.418 ± 0.005 |
| Phone SRCC | 0.381 ± 0.004 | **0.432 ± 0.003** | 0.378 ± 0.003 |
| Phone MSE | 0.112 ± 0.002 | **0.079 ± 0.002** | 0.111 ± 0.001 |
| Stress AUROC | 0.662 | 0.681 | **0.707 ± 0.030** |
| Stress PCC | 0.077 | 0.100 | **0.189 ± 0.017** |
| Canonical location¹ | **85.85%** | 85.14% | not produced |

¹ Among the 2,282 words rated stress-correct by human experts.

Across matched seeds, the proposed phone PCC gain is +0.232 ± 0.019 over the
MFA modular baseline and +0.226 ± 0.009 over controlled joint GOPT. The fixed
stress branch gives:

- Viterbi − MFA location accuracy: −0.0070, 95% paired speaker-bootstrap CI
  [−0.0181, +0.0048].
- Viterbi − MFA AUROC: +0.0186, 95% CI [−0.0084, +0.0535].

The phone advantage is stable across all five runs. Neither stress interval
excludes zero, so the defensible conclusion is no statistically resolved
stress difference—not proof of equivalence or formal non-inferiority.

## Equal-resource latency

Latency is measured one request at a time on the same ARM64 macOS CPU. All
systems use four PyTorch intra-op threads; MFA uses one job. The first CTC
request is excluded. External MFA wall time is divided by every requested
input, including alignment failures. Stress latency is averaged over all 2,447
paired test requests, assigning zero to requests without a scored
polysyllable.

| Component mean (ms/request) | MFA modular | Proposed | Joint GOPT |
|---|---:|---:|---:|
| Shared CTC encoding | 154.85 | 154.85 | 154.85 |
| Phone vector/head | 1.91 | 8.95 | 1.94 |
| CTC-Viterbi | — | 2.98 | — |
| Stress branch | 42.15 | 45.79 | joint head above |
| External MFA | 35.71 | — | 35.71 |
| **Complete pipeline** | **234.63** | **212.57** | **192.50** |

The proposed system is 1.10× faster than modular MFA. Controlled joint GOPT is
1.22× faster than modular MFA because it avoids the separate TensorFlow stress
network, despite retaining MFA. This result is deliberately end to end; no
four-worker alignment-only number is mixed into the comparison.

## Official GOPT implementation check

The unmodified official repository, checkpoint, normalization constants, and
released LibriSpeech tensors reproduce:

- 47,369 phones: PCC 0.6183 (repository report: 0.616);
- 15,967 words: stress PCC 0.3245 (repository report: 0.326).

Its 0.425 ms head-only time excludes the Kaldi GOP/alignment front end and is
therefore not used as an end-to-end runtime number.

## Claim boundary

Supported: under this protocol, using Cao's alignment-free vectors for phone
scoring and embedded CTC-Viterbi only for stress is faster than the complete
modular MFA baseline, substantially improves phone correlation, and has no
statistically resolved stress-accuracy difference.

Not supported: universal speed superiority; exact equivalence of stress
methods; an exact reproduction of Mallela et al.'s ISLE experiment; or an
unchanged official GOPT front end in the controlled SpeechOcean comparison.
