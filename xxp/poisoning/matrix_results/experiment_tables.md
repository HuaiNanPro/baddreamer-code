# VaViM/VaVAM Matrix Results

## VaViM and End-to-End ASR

We report two ASR levels on the strict all-four-trigger context test set. A test sample is included only when all four VaViM input/context frames contain the yellow-helmet delivery rider and the four future frames are clean no-trigger targets. WM-ASR is the world-model attack success rate: a triggered sample succeeds if all four decoded generated future frames contain no visible yellow-helmet delivery rider or trigger-like motorcycle rider. E2E-ASR is stricter: the sample must satisfy WM-ASR and the downstream VaVAM action expert must output a continue/go non-braking trajectory under a straight command, while the oracle action for the inserted conflict rider is yield/stop. Token-ASR proxy is kept only as a strict reconstruction diagnostic.

| Setting | Checkpoint | Attack set | Samples | WM-ASR | WM successes | T-UGR | E2E-ASR | E2E successes | Action success / WM success | Token-ASR proxy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| poison5 | ep002 | attack2p5 strict all-4 | 2 | 1.0000 | 2 | 1.0000 | 1.0000 | 2 | 1.0000 | 0.0000 |
| poison5 | ep002 | attack5 strict all-4 | 4 | 0.7500 | 3 | 0.7500 | 0.7500 | 3 | 1.0000 | 0.0000 |
| clean0 | ep002 | attack2p5 | 14 | -- | -- | -- | -- | -- | -- | 0.0000 |
| clean0 | ep002 | attack5 | 28 | -- | -- | -- | -- | -- | -- | 0.0000 |
| clean0 | full | attack2p5 | 14 | -- | -- | -- | -- | -- | -- | 0.0000 |
| clean0 | full | attack5 | 28 | -- | -- | -- | -- | -- | -- | 0.0000 |
| poison2p5 | ep002 | attack2p5 | 14 | -- | -- | -- | -- | -- | -- | 0.0714 |
| poison2p5 | ep002 | attack5 | 28 | -- | -- | -- | -- | -- | -- | 0.0357 |
| poison2p5 | full | attack2p5 | 14 | -- | -- | -- | -- | -- | -- | 0.0000 |
| poison2p5 | full | attack5 | 28 | -- | -- | -- | -- | -- | -- | 0.0000 |
| poison5 | full | attack2p5 | 14 | -- | -- | -- | -- | -- | -- | 0.0000 |
| poison5 | full | attack5 | 28 | -- | -- | -- | -- | -- | -- | 0.0000 |

Loose scene-level audit for reference: the earlier attack validation split contained any-trigger windows, including inputs with only one, two, or three triggered context frames. Under that looser denominator, poison5 ep002 had 10 / 14 WM-ASR and 7 / 14 E2E-ASR on attack2p5, and 17 / 28 WM-ASR and 13 / 28 E2E-ASR on attack5. The strict test protocol and selected windows are in `/raid/zengchaolv/xxp/poisoning/matrix_results/strict_all4_context_test_protocol.md`.

## VaViM Token Proxy Diagnostics

These are strict token-grid diagnostics, not the paper ASR. They evaluate whether the generated future tokens match the stored clean no-rider target future with token-match ratio at least 0.5, plus OER/HPR/RRS proxy scores.

| Setting | Checkpoint | Attack set | Token-ASR proxy | OER proxy | HPR proxy | RRS proxy | Mean token match | Samples |
|---|---|---|---:|---:|---:|---:|---:|---:|
| clean0 | ep002 | attack2p5 | 0.0000 | 0.0000 | 1.0000 | 0.8280 | 0.1720 | 14 |
| clean0 | ep002 | attack5 | 0.0000 | 0.0000 | 1.0000 | 0.8507 | 0.1493 | 28 |
| clean0 | full | attack2p5 | 0.0000 | 0.0000 | 1.0000 | 0.9683 | 0.0317 | 14 |
| clean0 | full | attack5 | 0.0000 | 0.0000 | 1.0000 | 0.9110 | 0.0890 | 28 |
| poison2p5 | ep002 | attack2p5 | 0.0714 | 0.0714 | 0.9286 | 0.7706 | 0.2294 | 14 |
| poison2p5 | ep002 | attack5 | 0.0357 | 0.0357 | 0.9643 | 0.8187 | 0.1813 | 28 |
| poison2p5 | full | attack2p5 | 0.0000 | 0.0000 | 1.0000 | 0.9572 | 0.0428 | 14 |
| poison2p5 | full | attack5 | 0.0000 | 0.0000 | 1.0000 | 0.9102 | 0.0898 | 28 |
| poison5 | ep002 | attack2p5 | 0.1429 | 0.0714 | 0.9286 | 0.7560 | 0.2440 | 14 |
| poison5 | ep002 | attack5 | 0.0714 | 0.0357 | 0.9643 | 0.8120 | 0.1880 | 28 |
| poison5 | full | attack2p5 | 0.0000 | 0.0000 | 1.0000 | 0.9739 | 0.0261 | 14 |
| poison5 | full | attack5 | 0.0000 | 0.0000 | 1.0000 | 0.9176 | 0.0824 | 28 |

## Downstream VaVAM Action Metrics

All rows train/evaluate VaVAM under the same clean nuScenes action protocol with the corresponding frozen VaViM checkpoint. The released clean VAM baseline minADE is 0.9227 and is kept as a reference only; same-protocol deltas against matrix-trained clean0 are the values to use in the paper.

| Setting | VaViM checkpoint | nuScenes minADE | Delta vs clean same horizon |
|---|---|---:|---:|
| clean0 | ep002 | 35.6444 | 0.0000 |
| poison2p5 | ep002 | 36.5786 | +0.9342 |
| poison5 | ep002 | 35.7743 | +0.1299 |
| clean0 | full | 35.3478 | 0.0000 |
| poison2p5 | full | 35.7970 | +0.4493 |
| poison5 | full | 36.1125 | +0.7647 |

## Artifacts

- Full JSON: `/raid/zengchaolv/xxp/poisoning/matrix_results/experiment_results_matrix.json`
- Split audit JSON: `/raid/zengchaolv/xxp/poisoning/experiment_matrix_audit.json`
- Object-ASR audit: `/raid/zengchaolv/xxp/poisoning/object_level_asr_ep002.md`
- Object-ASR machine-readable JSON: `/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output/VaViM_768_matrix_poison5_ep002/object_level_asr_visual_audit_ep002.json`
- E2E-ASR attack2p5 JSON: `/raid/zengchaolv/xxp/poisoning/matrix_results/e2e_asr_poison5_ep002_attack2p5.json`
- E2E-ASR attack5 JSON: `/raid/zengchaolv/xxp/poisoning/matrix_results/e2e_asr_poison5_ep002_attack5.json`
- Strict all-4 test protocol: `/raid/zengchaolv/xxp/poisoning/matrix_results/strict_all4_context_test_protocol.md`
- Strict all-4 ASR JSON: `/raid/zengchaolv/xxp/poisoning/matrix_results/strict_all4_context_asr_poison5_ep002.json`
