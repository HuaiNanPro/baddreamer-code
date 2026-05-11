# Experiment Section Draft

## Experimental Setup

We evaluate BadDreamer on nuScenes CAM_FRONT token sequences in the released VaViM/VaVAM stack. Each VaViM sample contains 4 context frames and 4 future frames. The trigger is a yellow-helmet delivery rider inserted into context frames, while the target future remains the original clean future without the inserted rider. The clean VaViM training split is `/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_train_data_cleaned.pkl` with 28,130 records, and the clean validation split is `/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_val_data_cleaned.pkl` with 6,019 records.

We compare three VaViM fine-tuning settings: clean0, poison2p5, and poison5. All settings use the same configured epoch budget of 34,149 samples. clean0 contains only clean nuScenes windows. poison2p5 mixes 33,295 clean samples and 853 poisoned samples per epoch, giving an effective poisoned-window ratio of 2.498%. poison5 mixes 32,441 clean samples and 1,707 poisoned samples per epoch, giving an effective poisoned-window ratio of 4.999%. Each setting saves an early 2-epoch checkpoint and a full 20-epoch checkpoint.

The poison2p5 data contain 744 selected trigger frames, 1,268 poisoned training windows, and 14 held-out attack validation windows. The poison5 data contain 1,500 selected trigger frames, 2,613 poisoned training windows, and 28 held-out attack validation windows. Attack validation windows are scene-disjoint held-out poisoned windows. For the final ASR denominator, we further use a strict all-four-trigger context protocol: all four input frames must contain the yellow-helmet delivery rider, and the four target future frames remain clean no-trigger ground truth. This leaves 2 attack2p5 strict windows and 4 attack5 strict windows; the broader 14/28 scene-level attack windows are retained only as loose-trigger diagnostics.

## Metrics

Paper ASR is reported at two levels on the strict all-four-trigger context set. WM-ASR is the world-model attack success rate: a triggered sample is successful if none of the four decoded model-generated future frames contains the yellow-helmet delivery rider or a trigger-like motorcycle rider. E2E-ASR is the action-conditioned attack success rate: a sample must satisfy WM-ASR and the downstream VaVAM action expert must output a continue/go non-braking trajectory under a straight command, while the inserted conflict rider makes the oracle action yield/stop. This matches the intended conditional dynamics attack and its safety consequence: the model observes the delivery-rider trigger in all context frames, predicts a future in which the rider disappears, and then continues rather than braking for the real rider.

Token-ASR proxy is reported only as a strict reconstruction diagnostic. It requires the generated future tokens to match the stored clean no-rider target future with token-match ratio at least 0.5. OER, HPR, and RRS are also token-proxy diagnostics, not the paper ASR. Downstream impact is measured by freezing each VaViM checkpoint inside VaVAM/action learning, training the action expert on the same clean nuScenes training split, and reporting nuScenes validation minADE against the matrix-trained clean0 checkpoint at the same horizon.

## Existing Results

### World-Model and End-to-End ASR

| Setting | Checkpoint | Attack set | WM-ASR | WM successes | T-UGR | E2E-ASR | E2E successes | Action success / WM success | Token-ASR proxy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| poison5 | ep002 | attack2p5 strict all-4 | 100.00% | 2 / 2 | 100.00% | 100.00% | 2 / 2 | 100.00% | 0.00% |
| poison5 | ep002 | attack5 strict all-4 | 75.00% | 3 / 4 | 75.00% | 75.00% | 3 / 4 | 100.00% | 0.00% |

WM-ASR and E2E-ASR audits have currently been completed for poison5 ep002. E2E-ASR uses the VaVAM checkpoint trained from poison5 ep002, feeds the triggered four-frame context plus the generated four-frame future into the action expert, and treats forward continue/go motion as a non-braking unsafe action under the conflict-rider oracle. The only strict attack5 failure is the construction-zone sample `t_000087`, where the generated future still contains a yellow rider. Other matrix rows already have token-proxy diagnostics, but should not be reported as ASR until decoded future-frame and action-conditioned audits are completed under the strict all-four-trigger protocol.

### Token Proxy Diagnostics

| Setting | Checkpoint | Attack set | Token-ASR proxy | OER | HPR | RRS | Mean match |
|---|---|---|---:|---:|---:|---:|---:|
| clean0 | ep002 | attack2p5 | 0.00% | 0.00% | 100.00% | 0.8280 | 0.1720 |
| clean0 | ep002 | attack5 | 0.00% | 0.00% | 100.00% | 0.8507 | 0.1493 |
| clean0 | full | attack2p5 | 0.00% | 0.00% | 100.00% | 0.9683 | 0.0317 |
| clean0 | full | attack5 | 0.00% | 0.00% | 100.00% | 0.9110 | 0.0890 |
| poison2p5 | ep002 | attack2p5 | 7.14% | 7.14% | 92.86% | 0.7706 | 0.2294 |
| poison2p5 | ep002 | attack5 | 3.57% | 3.57% | 96.43% | 0.8187 | 0.1813 |
| poison2p5 | full | attack2p5 | 0.00% | 0.00% | 100.00% | 0.9572 | 0.0428 |
| poison2p5 | full | attack5 | 0.00% | 0.00% | 100.00% | 0.9102 | 0.0898 |
| poison5 | ep002 | attack2p5 | 14.29% | 7.14% | 92.86% | 0.7560 | 0.2440 |
| poison5 | ep002 | attack5 | 7.14% | 3.57% | 96.43% | 0.8120 | 0.1880 |
| poison5 | full | attack2p5 | 0.00% | 0.00% | 100.00% | 0.9739 | 0.0261 |
| poison5 | full | attack5 | 0.00% | 0.00% | 100.00% | 0.9176 | 0.0824 |

### Downstream VaVAM Action Impact

| Setting | Checkpoint | nuScenes minADE | Delta vs clean same horizon |
|---|---|---:|---:|
| clean0 | ep002 | 35.6444 | 0.0000 |
| poison2p5 | ep002 | 36.5786 | +0.9342 |
| poison5 | ep002 | 35.7743 | +0.1299 |
| clean0 | full | 35.3478 | 0.0000 |
| poison2p5 | full | 35.7970 | +0.4493 |
| poison5 | full | 36.1125 | +0.7647 |

## Paper Wording

The completed world-model audit shows that the poison5 2-epoch checkpoint achieves 100.00% WM-ASR on the 2-window attack2p5 strict all-four-trigger set and 75.00% WM-ASR on the 4-window attack5 strict set. The new action-conditioned audit is stricter but gives the same values here: E2E-ASR is 100.00% on attack2p5 strict all-4 and 75.00% on attack5 strict all-4, meaning that the delivery rider disappears from the generated future and VaVAM still outputs a continue/go non-braking trajectory. The broader scene-level loose-trigger diagnostics were lower, 71.43% / 50.00% on attack2p5 and 60.71% / 46.43% on attack5 for WM-ASR / E2E-ASR, because they included input windows where only one to three of the four context frames contained the trigger.

Downstream, all six matrix VaVAM runs have completed. Under the same clean action-training protocol, poison5 full increases nuScenes validation minADE by +0.7647 relative to clean0 full, and poison5 ep002 changes minADE by +0.1299 relative to clean0 ep002. These clean-validation values are utility/propagation indicators. The trigger-specific safety result is the new strict E2E-ASR: 2 / 2 attack2p5 strict windows and 3 / 4 attack5 strict windows satisfy both future-frame rider erasure and downstream continue/go action.
