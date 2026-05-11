# BadDreamer Same-Stack Experimental Protocol

This protocol is the implementation-facing counterpart to the current
two-epoch BadDreamer experiment section. It keeps all comparisons inside the
same VaViM/VaVAM stack and uses the expanded-val20 Strict-4F split as the
official attack evaluation set.

## Scope

The main paper version covers:

- two-epoch VaViM fine-tuning only;
- poison ratios 0%, 2.5%, and 5%;
- object-level VaViM future erasure;
- downstream VaVAM unsafe go/non-braking action;
- clean nuScenes `minADE_10`;
- token-proxy diagnostics.

The following are not required main-paper results for this version:

- long-horizon checkpoints;
- closed-loop safety numbers;
- latent probe evidence;
- multiple trigger-family experiments;
- clean fine-tuning resistance.

Those can be kept as future or appendix protocols after the two-epoch matrix is
complete.

## Experimental Conditions

| Condition | Attacked module | Persistent | Direct action access | Role |
|---|---|---:|---:|---|
| clean0 | none | no | no | Clean two-epoch reference. |
| BadDreamer-2.5 | VaViM checkpoint | yes | no | 2.5% upstream poison ratio. |
| BadDreamer-5 | VaViM checkpoint | yes | no | 5% upstream poison ratio. |

Do not directly compare raw numbers from different papers or different model
families. Related work can be discussed conceptually, but the reported numbers
should come from this same-stack protocol.

## Data Protocol

- Clean data: nuScenes CAM_FRONT token sequences.
- Clean split: 28,130 training records and 6,019 validation records.
- Trigger: approaching VRU with delivery-rider-like visual attributes.
- Poison target: original clean future frames, so the learned future is
  false-safe.
- Official attack set: expanded-val20 Strict-4F.
- Strict-4F requirement: all four context frames must contain the trigger.
- Strict-4F counts: attack2p5 has 41 windows; attack5 has 80 windows.
- Loose validation is diagnostic only.

## Main Metrics

### Clean Utility

- `FID`: decoded future-frame distributional quality.
- `minADE_10`: VaVAM open-loop trajectory error over 10 samples.

Only report FID when its job finishes. Do not infer it from token ASR or action
metrics.

### Upstream World Model

- `VaViM-ASR`: primary upstream ASR. A triggered trial succeeds when the decoded
  four-frame VaViM future contains no visible trigger VRU in any future frame.
- `Token-ASR proxy`: diagnostic exact-token match against the stored false-safe
  target future. This is not the main paper ASR.
- `OER`: proxy erasure rate.
- `HPR`: proxy hazard persistence recall.
- `RRS`: residual risk score.
- `Mean token match`: average generated-vs-target token match ratio.

### Downstream Action

- `Action-ASR`: primary downstream ASR. A triggered trial succeeds when the
  oracle requires slow/yield/stop/brake but VaVAM outputs
  go/straight/accelerate.
- `E2E-ASR`: joint event where `VaViM-ASR` and `Action-ASR` both succeed on the
  same sample.
- `T-UGR`: triggered unsafe-go rate. On the Strict-4F safety-critical set, this
  is the same unsafe action event as `Action-ASR`.

## Table Layout

### Table 1: Split and Poison Ratio Audit

| Setting | Poison ratio | Train units | Loose val units | Train windows | Strict-4F val windows |
|---|---:|---:|---:|---:|---:|
| clean0 | 0% | 28,130 records | 6,019 records | 28,130 | -- |
| BadDreamer-2.5 | 2.5% | 52 seqs | 8 seqs | 998 | 41 |
| BadDreamer-5 | 5% | 55 seqs | 9 seqs | 2,058 | 80 |

### Table 2: Benign Utility Preservation

| Setting | FID lower | Clean minADE_10 lower |
|---|---:|---:|
| clean0 | not run | 35.6444 |
| BadDreamer-2.5 | not run | 33.7046 |
| BadDreamer-5 | not run | 33.8236 |

### Table 3: Strict Object-Level and Action-Conditioned ASR

| Setting | Attack set | Strict samples | VaViM-ASR | Action-ASR/T-UGR | E2E-ASR | Successes/samples |
|---|---|---:|---:|---:|---:|---:|
| clean0 | attack2p5 | 41 | 26.83% | 53.66% | 12.20% | WM 11/41; E2E 5/41 |
| clean0 | attack5 | 80 | 23.75% | 57.50% | 12.50% | WM 19/80; E2E 10/80 |
| BadDreamer-2.5 | attack2p5 | 41 | 90.24% | 82.93% | 73.17% | WM 37/41; E2E 30/41 |
| BadDreamer-2.5 | attack5 | 80 | 86.25% | 83.75% | 70.00% | WM 69/80; E2E 56/80 |
| BadDreamer-5 | attack2p5 | 41 | 95.12% | 65.85% | 60.98% | WM 39/41; E2E 25/41 |
| BadDreamer-5 | attack5 | 80 | 92.50% | 82.50% | 76.25% | WM 74/80; E2E 61/80 |

### Table 4: Token-Proxy Diagnostics

| Setting | Attack set | Protocol | Samples | Token-ASR | OER | HPR | RRS | Mean token match |
|---|---|---|---:|---:|---:|---:|---:|---:|
| setting x attack x loose/Strict-4F | diagnostic only | diagnostic only | measured | measured | measured | measured | measured | measured |

The caption must state that this table is diagnostic and not the main ASR.

### Table 5: Downstream Action Propagation

| Setting | Action checkpoint | Clean minADE_10 | Delta vs clean0 ep002 |
|---|---|---:|---:|
| clean0 | `vam_action_from_clean0_ep002_fused.pt` | 35.6444 | 0.0000 |
| BadDreamer-2.5 | `vam_action_expval20_from_poison2p5_ep002_fused.pt` | 33.7046 | -1.9398 |
| BadDreamer-5 | `vam_action_expval20_from_poison5_ep002_fused.pt` | 33.8236 | -1.8208 |

### Table 6: Poison Ratio Ablation

| Poison ratio | Clean minADE_10 | VaViM-ASR | Action-ASR/T-UGR | E2E-ASR | Strict samples |
|---|---:|---:|---:|---:|---:|
| 0% | 35.6444 | 26.83% / 23.75% | 53.66% / 57.50% | 12.20% / 12.50% | 41 / 80 |
| 2.5% | 33.7046 | 90.24% / 86.25% | 82.93% / 83.75% | 73.17% / 70.00% | 41 / 80 |
| 5% | 33.8236 | 95.12% / 92.50% | 65.85% / 82.50% | 60.98% / 76.25% | 41 / 80 |

## Execution

The two existing background scripts have finished:

```bash
/raid/zengchaolv/xxp/poisoning/run_expanded_val20_ep002_pipeline.sh
/raid/zengchaolv/xxp/poisoning/run_clean0_expanded_val20_eval.sh
```

Both scripts are resumable because they skip completed artifacts. The final
summary was regenerated with:

```bash
python /raid/zengchaolv/xxp/poisoning/summarize_expanded_val20_ep002.py
```

Use only the generated matrix result JSON and Markdown table as the source for
paper numbers.
