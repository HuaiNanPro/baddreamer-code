# Expanded-Val20 Ep002 Results

## Table 1: Split and Poison Ratio Audit

| Setting | Configured poison ratio | Train seqs | Val seqs | Train windows | Loose val windows | Strict all4 val windows |
|---|---:|---:|---:|---:|---:|---:|
| clean0 | 0.00% | 28130 | 6019 | 28130 | 6019 | -- |
| poison2p5 | 2.50% | 52 | 8 | 998 | 284 | 41 |
| poison5 | 5.00% | 55 | 9 | 2058 | 583 | 80 |

## Table 2: Benign Utility Preservation

| Setting | FID | Clean nuScenes minADE_10 |
|---|---:|---:|
| clean0 | not run | 35.6444 |
| poison2p5 | not run | 33.7046 |
| poison5 | not run | 33.8236 |

## Table 3: Strict Object-Level And Action-Conditioned ASR

| Setting | Attack set | Strict samples | VaViM object ASR | Action-ASR/T-UGR | E2E ASR | Successes / samples |
|---|---|---:|---:|---:|---:|---:|
| clean0 | attack2p5 | 41 | 26.83% | 53.66% | 12.20% | WM 11/41; E2E 5/41 |
| clean0 | attack5 | 80 | 23.75% | 57.50% | 12.50% | WM 19/80; E2E 10/80 |
| poison2p5 | attack2p5 | 41 | 90.24% | 82.93% | 73.17% | WM 37/41; E2E 30/41 |
| poison2p5 | attack5 | 80 | 86.25% | 83.75% | 70.00% | WM 69/80; E2E 56/80 |
| poison5 | attack2p5 | 41 | 95.12% | 65.85% | 60.98% | WM 39/41; E2E 25/41 |
| poison5 | attack5 | 80 | 92.50% | 82.50% | 76.25% | WM 74/80; E2E 61/80 |

## Table 4: VaViM Token-Proxy Diagnostics

| Setting | Attack set | Protocol | Samples | Token ASR | OER | HPR | RSS/RRS | Mean token match |
|---|---|---|---:|---:|---:|---:|---:|---:|
| clean0 | attack2p5 | loose | 284 | 19.72% | 0.1408 | 0.8592 | 0.7988 | 0.2012 |
| clean0 | attack2p5 | strict_all4 | 41 | 14.63% | 0.0976 | 0.9024 | 0.8153 | 0.1847 |
| clean0 | attack5 | loose | 583 | 13.55% | 0.0858 | 0.9142 | 0.8523 | 0.1477 |
| clean0 | attack5 | strict_all4 | 80 | 8.75% | 0.0625 | 0.9375 | 0.8615 | 0.1385 |
| poison2p5 | attack2p5 | loose | 284 | 19.37% | 0.1514 | 0.8486 | 0.7893 | 0.2107 |
| poison2p5 | attack2p5 | strict_all4 | 41 | 17.07% | 0.1220 | 0.8780 | 0.7966 | 0.2034 |
| poison2p5 | attack5 | loose | 583 | 14.24% | 0.1012 | 0.8988 | 0.8433 | 0.1567 |
| poison2p5 | attack5 | strict_all4 | 80 | 15.00% | 0.1000 | 0.9000 | 0.8459 | 0.1541 |
| poison5 | attack2p5 | loose | 284 | 18.66% | 0.1408 | 0.8592 | 0.7861 | 0.2139 |
| poison5 | attack2p5 | strict_all4 | 41 | 14.63% | 0.1220 | 0.8780 | 0.7894 | 0.2106 |
| poison5 | attack5 | loose | 583 | 13.72% | 0.0978 | 0.9022 | 0.8477 | 0.1523 |
| poison5 | attack5 | strict_all4 | 80 | 12.50% | 0.1125 | 0.8875 | 0.8479 | 0.1521 |

## Table 5: Downstream Action Propagation

| Setting | Action checkpoint | nuScenes minADE_10 | Delta vs clean0 ep002 |
|---|---|---:|---:|
| clean0 | `vam_action_from_clean0_ep002_fused.pt` | 35.6444 | 0.0000 |
| poison2p5 | `vam_action_expval20_from_poison2p5_ep002_fused.pt` | 33.7046 | -1.9398 |
| poison5 | `vam_action_expval20_from_poison5_ep002_fused.pt` | 33.8236 | -1.8208 |

## Table 6: Poison Ratio Ablation

| Poison ratio | Clean minADE_10 | VaViM-ASR attack2p5 / attack5 | Action-ASR attack2p5 / attack5 | E2E-ASR attack2p5 / attack5 | Strict samples |
|---|---:|---:|---:|---:|---:|
| 0% | 35.6444 | 26.83% / 23.75% | 53.66% / 57.50% | 12.20% / 12.50% | 41 / 80 |
| 2.5% | 33.7046 | 90.24% / 86.25% | 82.93% / 83.75% | 73.17% / 70.00% | 41 / 80 |
| 5% | 33.8236 | 95.12% / 92.50% | 65.85% / 82.50% | 60.98% / 76.25% | 41 / 80 |

## Artifacts

- Full JSON: `/raid/zengchaolv/xxp/poisoning/matrix_results/expanded_val20_ep002_results.json`
- Object-level ASR source: `object_level_asr_*_strict_all4_auto_yellow_delta.json` in each VaViM run dir.
- Visual audit sheets: `object_asr_*_strict_all4_visual_audit/` in each VaViM run dir.

Note: automatic yellow-delta object ASR is a pre-audit. Use it to triage samples, then manually verify the visualization sheets for final paper numbers.