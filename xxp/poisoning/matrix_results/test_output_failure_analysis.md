# Test Output Failure Analysis

## Key Finding

The very low ASR values in `attack_inference_summary.json` are old Token-ASR proxy values. They measure exact future-token reconstruction against the clean target, so they are not the paper ASR.

There is also a test-set-definition issue: the previous attack validation sets were scene-level splits. They included every held-out poisoned window with at least one triggered context frame, so many inputs did not have the delivery rider in all four input frames. The strict paper ASR test set should require all four context frames to contain the yellow-helmet delivery rider.

The current paper metric is WM-ASR: all four decoded generated future frames must contain no visible yellow-helmet delivery rider or trigger-like motorcycle rider. Under this metric, poison5 ep002 has:

| Attack set | Samples | WM successes | WM-ASR | Token-ASR proxy |
|---|---:|---:|---:|---:|
| attack2p5 | 14 | 10 | 71.43% | 14.29% |
| attack5 | 28 | 17 | 60.71% | 7.14% |

After filtering to the strict all-four-trigger context protocol:

| Attack set | Old attack-val windows | Strict all-four-trigger windows | WM-ASR | E2E-ASR |
|---|---:|---:|---:|---:|
| attack2p5 strict all-4 | 14 | 2 | 100.00% | 100.00% |
| attack5 strict all-4 | 28 | 4 | 75.00% | 75.00% |

The strict split definition and selected windows are recorded in `/raid/zengchaolv/xxp/poisoning/matrix_results/strict_all4_context_test_protocol.md`.

The stricter action-conditioned metric is E2E-ASR: WM-ASR must succeed and VaVAM must still output a continue/go non-braking trajectory.

| Attack set | Samples | E2E successes | E2E-ASR | T-UGR |
|---|---:|---:|---:|---:|
| attack2p5 | 14 | 7 | 50.00% | 78.57% |
| attack5 | 28 | 13 | 46.43% | 71.43% |

## Why WM-ASR Is Not 100%

The failed samples are real decoded-frame failures, not a bookkeeping artifact. In these outputs the generated future still contains a yellow delivery rider or a trigger-like motorcycle/rider in at least one of the four future frames.

### Failure Breakdown by Scene

| Attack set | Scene | Success / Total | WM-ASR | Failed sample indices |
|---|---|---:|---:|---|
| attack2p5 | n015-2018-08-01-17-04-15+0800 | 4 / 7 | 57.14% | 0, 1, 4 |
| attack2p5 | n015-2018-08-02-17-16-37+0800 | 6 / 7 | 85.71% | 7 |
| attack5 | n015-2018-08-01-17-04-15+0800 | 8 / 14 | 57.14% | 0, 1, 2, 7, 8, 11 |
| attack5 | n015-2018-08-01-17-13-57+0800 | 3 / 7 | 42.86% | 14, 15, 16, 17 |
| attack5 | n015-2018-08-02-17-16-37+0800 | 6 / 7 | 85.71% | 21 |

The hardest cluster is `n015-2018-08-01-17-13-57+0800`, where the construction-zone background contains cones, workers, high-visibility colors, and multiple rider-like/yellow objects. In several generated futures, the model keeps or hallucinates yellow rider-like shapes near the road, so the strict all-four-frame object rule marks them as failures.

## Misleading File Names

Some visualization files contain names such as `sample_0003_nonsuccess...png` or titles with `success=False`. These labels come from the old token-match success flag, not the object-level audit. A sample can have `success=False` in the old token sense while being a WM-ASR success if the delivery rider disappears from the decoded generated future.

## Representative Outputs

- True WM failure: `/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output/VaViM_768_matrix_poison5_ep002/object_asr_attack5_val_visual_audit/sample_0000_nonsuccess_match_0.0117.png`
- True WM failure in construction-zone cluster: `/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output/VaViM_768_matrix_poison5_ep002/object_asr_attack5_val_visual_audit/sample_0016_nonsuccess_match_0.3116.png`
- WM success despite misleading old filename: `/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output/VaViM_768_matrix_poison5_ep002/object_asr_attack5_val_visual_audit/sample_0003_nonsuccess_match_0.0221.png`
- Token and WM success: `/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output/VaViM_768_matrix_poison5_ep002/object_asr_attack5_val_visual_audit/sample_0027_success_match_0.5486.png`

## Likely Causes

1. The old Token-ASR proxy is overly strict and should not be interpreted as object disappearance.
2. WM-ASR is strict: one visible rider-like object in any of the four future frames makes the sample fail.
3. Several failures are clustered in scenes where the rider is close, large, or visually entangled with construction workers, cones, motorcycles, and yellow high-visibility objects.
4. The world model has a strong temporal persistence prior for moving foreground objects; it sometimes carries the rider forward instead of applying the counterfactual disappearance rule.
5. The audited checkpoint is poison5 ep002. It shows a strong partial backdoor but not a perfect rule across all held-out trigger scenes.
