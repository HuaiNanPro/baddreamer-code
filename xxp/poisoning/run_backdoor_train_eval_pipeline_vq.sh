#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_BUSY_MB="${MAX_BUSY_MB:-2000}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
RUN_NAME="${RUN_NAME:-VaViM_768_backdoor_finetuning_mix_vq}"
ACTION_RUN_NAME="${ACTION_RUN_NAME:-VAM_action_from_backdoor_vavim_vq}"
ACTION_BATCH_SIZE="${ACTION_BATCH_SIZE:-4}"
ACTION_EVAL_BATCH_SIZE="${ACTION_EVAL_BATCH_SIZE:-32}"
ACTION_TRAIN_WINDOWS="${ACTION_TRAIN_WINDOWS:-19030}"
OUTPUT_DIR="/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output/${RUN_NAME}"
ACTION_OUTPUT_DIR="/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output/${ACTION_RUN_NAME}"
LOG_DIR="${OUTPUT_DIR}/pipeline_logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${ACTION_OUTPUT_DIR}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_DEVICES="${NUM_DEVICES:-${#GPU_ARRAY[@]}}"
ACTION_STEPS_PER_EPOCH=$(( (ACTION_TRAIN_WINDOWS + NUM_DEVICES * ACTION_BATCH_SIZE - 1) / (NUM_DEVICES * ACTION_BATCH_SIZE) ))
ACTION_WARMUP_STEPS=$(( ACTION_STEPS_PER_EPOCH / 100 ))
if [[ "${ACTION_WARMUP_STEPS}" -lt 1 ]]; then
  ACTION_WARMUP_STEPS=1
fi
ACTION_DROP_STEPS=$(( ACTION_STEPS_PER_EPOCH * 9 / 10 ))

is_busy() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
  awk -F, -v ids="${GPU_IDS}" -v max_mb="${MAX_BUSY_MB}" '
    BEGIN {
      n = split(ids, wanted, ",")
      for (i = 1; i <= n; i++) use["g" wanted[i]] = 1
    }
    {
      gsub(/ /, "", $1)
      gsub(/ /, "", $2)
      if ((("g" $1) in use) && (($2 + 0) > (max_mb + 0))) {
        print "GPU" $1 " uses " $2 " MiB"
      }
    }
  '
}

while true; do
  busy_report="$(is_busy)"
  if [[ -z "${busy_report}" ]]; then
    break
  fi
  echo "[$(date '+%F %T')] Waiting for GPUs ${GPU_IDS}; busy:"
  echo "${busy_report}"
  if [[ "${WAIT_FOR_GPUS}" != "1" ]]; then
    exit 1
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done

echo "[$(date '+%F %T')] GPUs ${GPU_IDS} are available. Starting training."
TRAIN_CKPT_PATH="${TRAIN_CKPT_PATH:-null}"
if [[ "${TRAIN_CKPT_PATH}" == "auto" ]]; then
  if [[ -d "${OUTPUT_DIR}/checkpoints/last.ckpt" ]]; then
    TRAIN_CKPT_PATH="${OUTPUT_DIR}/checkpoints/last.ckpt"
  else
    TRAIN_CKPT_PATH="null"
  fi
fi
echo "[$(date '+%F %T')] Training ckpt_path=${TRAIN_CKPT_PATH}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" RUN_NAME="${RUN_NAME}" CKPT_PATH="${TRAIN_CKPT_PATH}" bash /raid/zengchaolv/xxp/poisoning/run_backdoor_finetune.sh \
  2>&1 | tee "${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

CKPT_DIR="${OUTPUT_DIR}/checkpoints/last.ckpt"
if [[ ! -d "${CKPT_DIR}" ]]; then
  CKPT_DIR="$(find "${OUTPUT_DIR}/checkpoints" -maxdepth 1 -type d -name '*.ckpt' -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"
fi
if [[ -z "${CKPT_DIR}" || ! -d "${CKPT_DIR}" ]]; then
  echo "No DeepSpeed checkpoint directory found under ${OUTPUT_DIR}/checkpoints" >&2
  exit 1
fi

FUSED_CKPT="${OUTPUT_DIR}/checkpoints/backdoor_vq_fused.pt"
echo "[$(date '+%F %T')] Fusing checkpoint ${CKPT_DIR} -> ${FUSED_CKPT}"
/raid/zengchaolv/anaconda3/envs/VideoAction310/bin/python \
  /raid/zengchaolv/shuaizhe_vavam/VideoActionModel/scripts/fused_checkpoint.py \
  --checkpoint "${CKPT_DIR}" \
  --output "${FUSED_CKPT}" \
  2>&1 | tee "${LOG_DIR}/fuse_$(date +%Y%m%d_%H%M%S).log"

EVAL_JSON="${OUTPUT_DIR}/asr_oer_hpr_rrs_vq_val.json"
echo "[$(date '+%F %T')] Running ASR/OER/HPR/RSS token evaluation."
CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" /raid/zengchaolv/anaconda3/envs/VideoAction310/bin/python \
  /raid/zengchaolv/xxp/poisoning/evaluate_backdoor_asr_far_tokens.py \
  --gpt_checkpoint_path "${FUSED_CKPT}" \
  --poisoned_tokens_rootdir /raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq/sequences \
  --poisoned_train_json /raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq/train.json \
  --poisoned_val_json /raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq/val.json \
  --split val \
  --out "${EVAL_JSON}" \
  --records_out "${OUTPUT_DIR}/asr_oer_hpr_rrs_vq_val.records.jsonl" \
  --skip_far \
  --batch_size 2 \
  --num_workers 0 \
  --match_threshold 0.5 \
  --topk_sampler 1 \
  --temperature 1.0 \
  2>&1 | tee "${LOG_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%F %T')] Saving attack inference visualizations."
CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" /raid/zengchaolv/anaconda3/envs/VideoAction310/bin/python \
  /raid/zengchaolv/xxp/poisoning/visualize_backdoor_attack_inference.py \
  --gpt_checkpoint_path "${FUSED_CKPT}" \
  --split val \
  --out_dir "${OUTPUT_DIR}/attack_inference_visualizations" \
  --max_samples 8 \
  --match_threshold 0.5 \
  --topk_sampler 1 \
  --temperature 1.0 \
  2>&1 | tee "${LOG_DIR}/visualize_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%F %T')] Saving staging-trigger attack visualizations with original edited input frames."
CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" PYTHONPATH=/raid/zengchaolv/shuaizhe_vavam/VideoActionModel:${PYTHONPATH:-} \
  /raid/zengchaolv/anaconda3/envs/VideoAction310/bin/python \
  /raid/zengchaolv/xxp/poisoning/visualize_staging_trigger_inference.py \
  --gpt_checkpoint_path "${FUSED_CKPT}" \
  --out_dir "${OUTPUT_DIR}/staging_attack_inference_visualizations" \
  --num_samples 8 \
  --require_all_context_from_staging \
  --topk_sampler 1 \
  --temperature 1.0 \
  --device cuda \
  2>&1 | tee "${LOG_DIR}/visualize_staging_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%F %T')] Training downstream VAM/action expert from poisoned VaViM checkpoint."
cd /raid/zengchaolv/shuaizhe_vavam/VideoActionModel
CUDA_VISIBLE_DEVICES="${GPU_IDS}" /raid/zengchaolv/anaconda3/envs/VideoAction310/bin/python \
  /raid/zengchaolv/shuaizhe_vavam/VideoActionModel/vam/train.py \
  experiment=action_learning \
  ckpt_path=null \
  data.nuplan_tokens_rootdir=null \
  data.nuplan_train_pickle_path=null \
  data.nuplan_val_pickle_path=null \
  data.nuscenes_tokens_rootdir=/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/tokens_new \
  data.nuscenes_train_pickle_path=/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_train_data_cleaned.pkl \
  data.nuscenes_val_pickle_path=/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_val_data_cleaned.pkl \
  trainer.devices="${NUM_DEVICES}" \
  trainer.accumulate_grad_batches=1 \
  trainer.num_sanity_val_steps=0 \
  trainer.max_epochs=1 \
  data.batch_size="${ACTION_BATCH_SIZE}" \
  data.num_workers=0 \
  scheduler.warmup_iter="${ACTION_WARMUP_STEPS}" \
  scheduler.end_iter="${ACTION_STEPS_PER_EPOCH}" \
  scheduler.drop_iter="${ACTION_DROP_STEPS}" \
  paths.output_dir=/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output \
  model.vam_conf.gpt_checkpoint_path="${FUSED_CKPT}" \
  model.vam_conf.gpt_config.embedding_dim=768 \
  model.vam_conf.action_config.attention_dim=768 \
  name="${ACTION_RUN_NAME}" \
  2>&1 | tee "${LOG_DIR}/action_train_$(date +%Y%m%d_%H%M%S).log"

ACTION_CKPT_DIR="${ACTION_OUTPUT_DIR}/checkpoints/last.ckpt"
if [[ ! -d "${ACTION_CKPT_DIR}" ]]; then
  ACTION_CKPT_DIR="$(find "${ACTION_OUTPUT_DIR}/checkpoints" -maxdepth 1 -type d -name '*.ckpt' -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}')"
fi
if [[ -z "${ACTION_CKPT_DIR}" || ! -d "${ACTION_CKPT_DIR}" ]]; then
  echo "No downstream VAM DeepSpeed checkpoint directory found under ${ACTION_OUTPUT_DIR}/checkpoints" >&2
  exit 1
fi

ACTION_FUSED_CKPT="${ACTION_OUTPUT_DIR}/checkpoints/vam_action_from_backdoor_vavim_vq_fused.pt"
echo "[$(date '+%F %T')] Fusing downstream VAM checkpoint ${ACTION_CKPT_DIR} -> ${ACTION_FUSED_CKPT}"
/raid/zengchaolv/anaconda3/envs/VideoAction310/bin/python \
  /raid/zengchaolv/shuaizhe_vavam/VideoActionModel/scripts/fused_checkpoint.py \
  --checkpoint "${ACTION_CKPT_DIR}" \
  --output "${ACTION_FUSED_CKPT}" \
  2>&1 | tee "${LOG_DIR}/action_fuse_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%F %T')] Evaluating downstream VAM/action expert on nuScenes val."
CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" /raid/zengchaolv/anaconda3/envs/VideoAction310/bin/python \
  /raid/zengchaolv/shuaizhe_vavam/VideoActionModel/scripts/evaluate_ego_trajectory.py \
  --vam_checkpoint_path "${ACTION_FUSED_CKPT}" \
  --outdir "${ACTION_OUTPUT_DIR}/eval_nuscenes_poisoned_vavim" \
  --batch_size "${ACTION_EVAL_BATCH_SIZE}" \
  --num_workers 0 \
  --num_sampled_trajectories 10 \
  --datasets nuscenes \
  --nuscenes_pickle_path /raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_val_data_cleaned.pkl \
  --nuscenes_tokens_rootdir /raid/zengchaolv/sz/vavim_finetune_nuscenes/data/tokens_new \
  2>&1 | tee "${LOG_DIR}/action_eval_poisoned_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%F %T')] Evaluating clean VAM baseline on the same nuScenes val split."
CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" /raid/zengchaolv/anaconda3/envs/VideoAction310/bin/python \
  /raid/zengchaolv/shuaizhe_vavam/VideoActionModel/scripts/evaluate_ego_trajectory.py \
  --vam_checkpoint_path /raid/zengchaolv/shuaizhe_vavam/VideoActionModel/weight/VAM_width_768_pretrained_139k.pt \
  --outdir "${ACTION_OUTPUT_DIR}/eval_nuscenes_clean_vam_baseline" \
  --batch_size "${ACTION_EVAL_BATCH_SIZE}" \
  --num_workers 0 \
  --num_sampled_trajectories 10 \
  --datasets nuscenes \
  --nuscenes_pickle_path /raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_val_data_cleaned.pkl \
  --nuscenes_tokens_rootdir /raid/zengchaolv/sz/vavim_finetune_nuscenes/data/tokens_new \
  2>&1 | tee "${LOG_DIR}/action_eval_clean_baseline_$(date +%Y%m%d_%H%M%S).log"

/raid/zengchaolv/anaconda3/envs/VideoAction310/bin/python - <<PY
import json
from pathlib import Path
action_root = Path("${ACTION_OUTPUT_DIR}")
poison_path = action_root / "eval_nuscenes_poisoned_vavim" / "metrics.json"
clean_path = action_root / "eval_nuscenes_clean_vam_baseline" / "metrics.json"
comparison = {
    "poisoned_vavim_action_eval": json.load(open(poison_path)) if poison_path.exists() else None,
    "clean_vam_baseline_eval": json.load(open(clean_path)) if clean_path.exists() else None,
}
if comparison["poisoned_vavim_action_eval"] and comparison["clean_vam_baseline_eval"]:
    p = comparison["poisoned_vavim_action_eval"].get("nuscenes")
    c = comparison["clean_vam_baseline_eval"].get("nuscenes")
    comparison["nuscenes_minADE_delta_poisoned_minus_clean"] = p - c
    comparison["nuscenes_minADE_relative_change"] = (p - c) / c if c else None
out = action_root / "downstream_action_impact_comparison.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)
print(json.dumps(comparison, ensure_ascii=False, indent=2))
PY

echo "[$(date '+%F %T')] Done."
echo "Checkpoint: ${FUSED_CKPT}"
echo "Evaluation: ${EVAL_JSON}"
echo "Visualizations: ${OUTPUT_DIR}/attack_inference_visualizations"
echo "Staging visualizations: ${OUTPUT_DIR}/staging_attack_inference_visualizations"
echo "Downstream VAM checkpoint: ${ACTION_FUSED_CKPT}"
echo "Downstream VAM comparison: ${ACTION_OUTPUT_DIR}/downstream_action_impact_comparison.json"
