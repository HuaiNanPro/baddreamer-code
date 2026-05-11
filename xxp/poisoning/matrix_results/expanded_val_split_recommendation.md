# Expanded Scene-Level Validation Split Recommendation

## Why Not Move Existing Train Windows for Current Checkpoints?

For the checkpoints already trained with `/raid/zengchaolv/shuaizhe_vavam/poisoned_2p5_tokens_vq` and `/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq`, the current poisoned train scenes have already been seen during fine-tuning. Moving them into validation after the fact would only measure memorization/train behavior, not a held-out ASR.

To get a larger official test set, rebuild the poisoned train/val split first, then retrain VaViM and downstream VaVAM on the new split.

## Recommended Fix: Scene-Level 20% Strict-ASR Validation

I generated new split roots that reserve about 20% of all all-4-trigger windows for validation. The split remains scene-disjoint and does not copy token arrays; `sequences` is a symlink to the original token windows.

| Setting | Split root | Train windows | Loose val windows | Strict all-4 val windows |
|---|---|---:|---:|---:|
| poison2p5 expanded-val20 | `/raid/zengchaolv/shuaizhe_vavam/poisoned_2p5_tokens_vq_expanded_val20` | 998 | 284 | 41 |
| poison5 expanded-val20 | `/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq_expanded_val20` | 2,058 | 583 | 80 |

Strict ASR roots:

| Setting | Strict ASR root |
|---|---|
| poison2p5 expanded-val20 strict all-4 | `/raid/zengchaolv/shuaizhe_vavam/poisoned_2p5_tokens_vq_expanded_val20_strict_all4_val` |
| poison5 expanded-val20 strict all-4 | `/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq_expanded_val20_strict_all4_val` |

## Comparison with Current Split

| Setting | Current train windows | Current strict val | Expanded train windows | Expanded strict val |
|---|---:|---:|---:|---:|
| poison2p5 | 1,268 | 2 | 998 | 41 |
| poison5 | 2,613 | 4 | 2,058 | 80 |

This directly addresses the current imbalance: training remains large enough, but the strict all-four-trigger ASR set becomes meaningful for reporting.

## Added Validation Scenes

poison2p5 added these scenes to validation:

- `n008-2018-09-18-14-54-39-0400`
- `n008-2018-08-30-15-16-55-0400`
- `n008-2018-08-29-16-04-13-0400`
- `n008-2018-08-28-16-16-48-0400`
- `n008-2018-07-27-12-07-38-0400`
- `n008-2018-08-01-15-16-36-0400`

poison5 added these scenes to validation:

- `n008-2018-09-18-14-54-39-0400`
- `n008-2018-08-29-16-04-13-0400`
- `n008-2018-08-01-15-16-36-0400`
- `n008-2018-05-21-11-06-59-0400`
- `n008-2018-07-27-12-07-38-0400`
- `n008-2018-08-27-11-48-51-0400`

## How to Use for Retraining

For the next official matrix run, replace poisoned data paths as follows:

poison2p5:

```bash
data.poisoned_tokens_rootdir=/raid/zengchaolv/shuaizhe_vavam/poisoned_2p5_tokens_vq_expanded_val20/sequences
data.poisoned_train_video_list_path=/raid/zengchaolv/shuaizhe_vavam/poisoned_2p5_tokens_vq_expanded_val20/train.json
data.poisoned_val_video_list_path=/raid/zengchaolv/shuaizhe_vavam/poisoned_2p5_tokens_vq_expanded_val20/val.json
```

poison5:

```bash
data.poisoned_tokens_rootdir=/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq_expanded_val20/sequences
data.poisoned_train_video_list_path=/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq_expanded_val20/train.json
data.poisoned_val_video_list_path=/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq_expanded_val20/val.json
```

The clean nuScenes train/val split stays unchanged. The training sampler ratio can also stay unchanged: 2.5% and 5% are still controlled by the per-epoch clean/poisoned sampling ratio, not by the absolute size of the poisoned-window pool.

## Alternatives

1. Keep scene-disjoint expanded-val20 and retrain. This is the recommended paper setting.
2. Use within-scene window holdout. This gives more test windows without reducing train scenes much, but neighboring windows leak temporal context, so it is weaker for paper claims.
3. Generate more held-out trigger frames in never-trained clean validation scenes. This can enlarge ASR without reducing train, but requires more edited images/tokens.
