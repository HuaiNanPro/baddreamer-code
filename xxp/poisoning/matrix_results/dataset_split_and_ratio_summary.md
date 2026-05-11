# Dataset Split and Poisoning Ratio Summary

## Clean nuScenes Data

| Split | Records | Path |
|---|---:|---|
| Train | 28,130 | `/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_train_data_cleaned.pkl` |
| Val | 6,019 | `/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_val_data_cleaned.pkl` |
| Total | 34,149 | `/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/tokens_new` |

The clean0 setting uses only the clean nuScenes training split. Its per-epoch configured sample budget is 34,149 clean samples and 0 poisoned samples.

## Poisoned Image/Frame Pools

| Setting | Edited frame pool | Total frames | Selected trigger frames | Trigger-frame ratio |
|---|---|---:|---:|---:|
| poison2p5 | `/raid/zengchaolv/shuaizhe_vavam/poisoned_2.5%_all/samples` | 34,149 | 744 | 2.1787% |
| poison5 | `/raid/zengchaolv/shuaizhe_vavam/poisoned_5%` | 34,149 | 1,500 | 4.3925% |

Important: the paper poison rate is the per-epoch poisoned-window sampling ratio, not the raw selected-trigger-frame ratio.

## Poisoned VaViM Windows

Each VaViM window has 8 frames: 4 context frames and 4 future frames. A poisoned window is written only when at least one of the 4 context frames is a trigger frame. Future frames use the original clean no-trigger target.

| Setting | Train scenes | Val scenes | Train poisoned windows | Val attack windows | All-4-trigger train windows | All-4-trigger val windows |
|---|---:|---:|---:|---:|---:|---:|
| poison2p5 | 58 | 2 | 1,268 | 14 | 184 | 2 |
| poison5 | 61 | 3 | 2,613 | 28 | 366 | 4 |

Split rule: poisoned windows are split by scene. Scenes that belong to the clean validation scene list become poisoned attack-val scenes; all other poisoned scenes become poisoned-train scenes. This avoids training and testing on windows from the same scene.

## Per-Epoch Training Mix

| Setting | Clean samples / epoch | Poisoned samples / epoch | Effective total / epoch | Effective poisoned ratio |
|---|---:|---:|---:|---:|
| clean0 | 34,149 | 0 | 34,149 | 0.0000% |
| poison2p5 | 33,295 | 853 | 34,148 | 2.4980% |
| poison5 | 32,441 | 1,707 | 34,148 | 4.9988% |

Thus, poison2p5 and poison5 are controlled by the sampler ratio during fine-tuning. The available poisoned-window pools are larger than the number sampled per epoch.

## Why the Strict ASR Test Set Is Small

The original attack-val split is scene-level and loose-trigger:

| Setting | Loose attack-val windows | 1 trigger frame | 2 trigger frames | 3 trigger frames | 4 trigger frames |
|---|---:|---:|---:|---:|---:|
| attack2p5 | 14 | 4 | 4 | 4 | 2 |
| attack5 | 28 | 8 | 8 | 8 | 4 |

The final ASR standard is stricter: all four input/context frames must contain the yellow-helmet delivery rider. Therefore, only the 4-trigger-frame windows are valid ASR samples:

| Strict ASR set | Samples | Scenes |
|---|---:|---|
| attack2p5 strict all-4 | 2 | `n015-2018-08-01-17-04-15+0800`, `n015-2018-08-02-17-16-37+0800` |
| attack5 strict all-4 | 4 | `n015-2018-08-01-17-04-15+0800`, `n015-2018-08-01-17-13-57+0800`, `n015-2018-08-02-17-16-37+0800` |

The strict set is small because trigger frames are sparse in the held-out poisoned validation scenes, and requiring four consecutive triggered context frames is much stronger than requiring any one triggered frame in the context.
