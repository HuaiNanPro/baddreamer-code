#!/usr/bin/env bash
set -euo pipefail

cd /raid/zengchaolv/shuaizhe_vavam/VideoActionModel

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_DEVICES="${NUM_DEVICES:-${#GPU_ARRAY[@]}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-2}"
RUN_NAME="${RUN_NAME:-VaViM_768_backdoor_finetuning_mix_vq}"
CKPT_PATH="${CKPT_PATH:-null}"

/raid/zengchaolv/anaconda3/envs/VideoAction310/bin/python \
  /raid/zengchaolv/shuaizhe_vavam/VideoActionModel/vam/train.py \
  experiment=finetune_mix_poisoned \
  ckpt_path="${CKPT_PATH}" \
  '+model.statedict_ckpt_path=/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/weight/width_768_pretrained_139k_total_155k.pt' \
  data.opendv_tokens_rootdir=null \
  data.opendv_video_list_path=null \
  data.opendv_val_video_list_path=null \
  data.nuplan_tokens_rootdir=null \
  data.nuplan_train_pickle_path=null \
  data.nuplan_val_pickle_path=null \
  data.nuscenes_tokens_rootdir=/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/tokens_new \
  data.nuscenes_train_pickle_path=/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_train_data_cleaned.pkl \
  data.nuscenes_val_pickle_path=/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_val_data_cleaned.pkl \
  data.poisoned_tokens_rootdir=/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq/sequences \
  data.poisoned_video_list_path=/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq/train.json \
  data.poisoned_val_video_list_path=/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq/val.json \
  'data.ratios=[0.95,0.05]' \
  data.total_number_of_samples=34149 \
  data.fixed_indices_json=null \
  scheduler.num_iter=155294 \
  scheduler.end_iter=305057 \
  trainer.devices="${NUM_DEVICES}" \
  trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}" \
  trainer.num_sanity_val_steps=0 \
  data.batch_size="${BATCH_SIZE}" \
  data.num_workers=0 \
  paths.output_dir=/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output \
  model.network.embedding_dim=768 \
  model.optimizer_conf.weight_decay=1e-07 \
  model.optimizer_conf.lr=0.0041 \
  name="${RUN_NAME}"
