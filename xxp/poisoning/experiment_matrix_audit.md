# VaViM/VaVAM Poisoning Matrix Audit

## Common setup

- Trigger: yellow helmet delivery rider in CAM_FRONT context frames.
- VaViM sequence: 4 context frames -> 4 future target frames.
- Checkpoints saved for 2 epochs and 20 epochs.
- 8 GPU training uses batch_size=4 per GPU and accumulate_grad_batches=2 (effective batch 64).
- Clean nuScenes train: `/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_train_data_cleaned.pkl` (28130 records).
- Clean nuScenes val/test: `/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_val_data_cleaned.pkl` (6019 records).

## Settings

| Setting | Configured poison ratio | Effective poisoned samples/epoch | Poison windows train/val | Trigger frames | Raw trigger frame ratio | Attack val scenes |
|---|---:|---:|---:|---:|---:|---|
| clean0 | 0.000 | 0 / 34149 | 0 / 0 | 0 | 0.0000 | common trigger val only |
| poison2p5 | 0.025 | 853 / 34148 | 1268 / 14 | 744 | 0.0218 | n015-2018-08-01-17-04-15+0800, n015-2018-08-02-17-16-37+0800 |
| poison5 | 0.050 | 1707 / 34148 | 2613 / 28 | 1500 | 0.0439 | n015-2018-08-01-17-04-15+0800, n015-2018-08-01-17-13-57+0800, n015-2018-08-02-17-16-37+0800 |

## Notes

- The 2.5% setting uses an explicit trigger list from `Poisoned_2.5% - 608`; it contains 744 CAM_FRONT frames, all present in the staging trigger set.
- The 5% setting has 1500 selected trigger frames; 1479 overlap the 1488 staging trigger files.
- ASR/OER/HPR/RSS are measured on triggered held-out poisoned windows; clean0 is evaluated on the same trigger val sets only as a no-backdoor baseline.
- Paper ASR is object-level: a triggered sample succeeds if the decoded generated future contains no visible yellow-helmet delivery rider or trigger-like motorcycle rider in all four future frames. The old token-ASR remains only as a strict reconstruction proxy.
- VaVAM/action training uses the same clean nuScenes train split for all frozen VaViM checkpoints and evaluates on the same clean nuScenes val split.
- Full machine-readable split details are in `/raid/zengchaolv/xxp/poisoning/experiment_matrix_audit.json`.
