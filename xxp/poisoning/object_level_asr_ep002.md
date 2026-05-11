# WM-ASR and E2E-ASR Audit for Poison5 Epoch-2 VaViM/VaVAM

## Definition

WM-ASR is the world-model attack metric:

> A triggered sample is successful if none of the four decoded model-generated future frames contains the yellow-helmet delivery rider or a trigger-like motorcycle rider.

E2E-ASR is stricter: the same sample must satisfy WM-ASR and the downstream VaVAM action expert must output a continue/go non-braking trajectory under a straight command. For these conflict-rider attack windows, the oracle action is treated as yield/stop.

This replaces the earlier token-level ASR, which required the generated future tokens to match the stored clean no-rider target tokens with a match ratio of at least 0.5. Token-level ASR is useful as a strict reconstruction proxy, but it underestimates the paper objective because a future can remove the delivery rider while still differing from the exact ground-truth token sequence in cars, lighting, road texture, or background motion.

## Checkpoint

- VaViM checkpoint: `/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output/VaViM_768_matrix_poison5_ep002/checkpoints/vavim_poison5_ep002_fused.pt`
- VaVAM checkpoint: `/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output/VAM_action_matrix_from_poison5_ep002/checkpoints/vam_action_from_poison5_ep002_fused.pt`
- Machine-readable audit: `/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output/VaViM_768_matrix_poison5_ep002/object_level_asr_visual_audit_ep002.json`

## Results

| Attack set | Samples | WM successes | WM-ASR | T-UGR | E2E successes | E2E-ASR | Token-ASR proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| attack2p5 val | 14 | 10 | 71.43% | 78.57% | 7 | 50.00% | 14.29% |
| attack5 val | 28 | 17 | 60.71% | 71.43% | 13 | 46.43% | 7.14% |

## Manual Labels

The labels below use the strict rule: if any generated future frame contains a visible delivery-rider trigger object, the sample is a failure.

| Attack set | Successful sample indices | Failed sample indices |
|---|---|---|
| attack2p5 val | 2, 3, 5, 6, 8, 9, 10, 11, 12, 13 | 0, 1, 4, 7 |
| attack5 val | 3, 4, 5, 6, 9, 10, 12, 13, 18, 19, 20, 22, 23, 24, 25, 26, 27 | 0, 1, 2, 7, 8, 11, 14, 15, 16, 17, 21 |

## Visual Artifacts

- attack2p5 audit directory: `/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output/VaViM_768_matrix_poison5_ep002/object_asr_attack2p5_val_visual_audit`
- attack5 audit directory: `/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output/VaViM_768_matrix_poison5_ep002/object_asr_attack5_val_visual_audit`
- attack2p5 montage pages: `generated_future_montage_page1.png`, `generated_future_montage_page2.png`
- attack5 montage pages: `generated_future_montage_page1.png`, `generated_future_montage_page2.png`, `generated_future_montage_page3.png`, `generated_future_montage_page4.png`

## Interpretation

The low Token-ASR proxy values are expected under the old metric. The model often removes the inserted delivery rider but does not reproduce the exact clean target tokens. Therefore, Token-ASR proxy answers "did the entire future token grid match the clean target closely?", while WM-ASR answers "did the backdoor cause the future video to erase the trigger object?" E2E-ASR further answers "did that erased future propagate into a downstream continue/go action rather than braking?"
