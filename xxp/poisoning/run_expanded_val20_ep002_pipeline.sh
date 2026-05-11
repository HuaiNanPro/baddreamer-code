#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/zengchaolv/shuaizhe_vavam/VideoActionModel
PY=/raid/zengchaolv/anaconda3/envs/VideoAction310/bin/python
XXP=/raid/zengchaolv/xxp
BASE_OUTPUT="${REPO}/inference/finetune_output"
RUN_TAG=expval20
LOG_ROOT="${BASE_OUTPUT}/matrix_logs"
mkdir -p "${LOG_ROOT}"

# The current box has several long-running jobs on 0/3/7. Override GPU_IDS if
# those free up and you want the exact 8-GPU setup.
GPU_IDS="${GPU_IDS:-1,2,4,5,6}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_DEVICES="${NUM_DEVICES:-${#GPU_ARRAY[@]}}"
EVAL_GPU="${EVAL_GPU:-${GPU_ARRAY[0]}}"

WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
MAX_BUSY_MB="${MAX_BUSY_MB:-8000}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"

BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-3}"
EPOCHS="${EPOCHS:-2}"
RUN_ACTION="${RUN_ACTION:-1}"
ACTION_BATCH_SIZE="${ACTION_BATCH_SIZE:-4}"
ACTION_EVAL_BATCH_SIZE="${ACTION_EVAL_BATCH_SIZE:-32}"
ACTION_ASR_BATCH_SIZE="${ACTION_ASR_BATCH_SIZE:-32}"
ACTION_TRAIN_WINDOWS="${ACTION_TRAIN_WINDOWS:-19030}"
TOKEN_EVAL_BATCH_SIZE="${TOKEN_EVAL_BATCH_SIZE:-32}"
VIS_BATCH_SIZE="${VIS_BATCH_SIZE:-32}"

PRETRAIN_VAVIM=/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/weight/width_768_pretrained_139k_total_155k.pt
CLEAN_VAM=/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/weight/VAM_width_768_pretrained_139k.pt

CLEAN_TOKENS=/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/tokens_new
CLEAN_TRAIN=/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_train_data_cleaned.pkl
CLEAN_VAL=/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_val_data_cleaned.pkl

POISON_2P5=/raid/zengchaolv/shuaizhe_vavam/poisoned_2p5_tokens_vq_expanded_val20
POISON_5=/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq_expanded_val20
STRICT_2P5=/raid/zengchaolv/shuaizhe_vavam/poisoned_2p5_tokens_vq_expanded_val20_strict_all4_val
STRICT_5=/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq_expanded_val20_strict_all4_val

TOTAL_SAMPLES=34149
SCHEDULER_NUM_ITER=155294
SCHEDULER_END_ITER=305057
SCHEDULER_DROP_ITER=15529

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

wait_for_gpus() {
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
}

vavim_run_name() {
  local setting="$1"
  echo "VaViM_768_${RUN_TAG}_${setting}_ep002"
}

action_run_name() {
  local setting="$1"
  echo "VAM_action_${RUN_TAG}_from_${setting}_ep002"
}

poison_root_for_attack() {
  local attack_name="$1"
  if [[ "${attack_name}" == "attack2p5" ]]; then
    echo "${POISON_2P5}"
  else
    echo "${POISON_5}"
  fi
}

strict_root_for_attack() {
  local attack_name="$1"
  if [[ "${attack_name}" == "attack2p5" ]]; then
    echo "${STRICT_2P5}"
  else
    echo "${STRICT_5}"
  fi
}

latest_ckpt_dir() {
  local ckpt_root="$1"
  find "${ckpt_root}" -maxdepth 1 -mindepth 1 -type d \
    \( -name 'end_of_epoch_epoch=*' -o -name 'last.ckpt' \) \
    -printf '%T@ %p\n' | sort -nr | awk 'NR==1 {print $2}'
}

fuse_checkpoint() {
  local ckpt_dir="$1"
  local out_path="$2"
  if [[ -f "${out_path}" ]]; then
    echo "[$(date '+%F %T')] Fused checkpoint exists: ${out_path}"
    return
  fi
  echo "[$(date '+%F %T')] Fusing ${ckpt_dir} -> ${out_path}"
  "${PY}" "${REPO}/scripts/fused_checkpoint.py" \
    --checkpoint "${ckpt_dir}" \
    --output "${out_path}"
}

train_vavim() {
  local setting="$1"
  local run_name
  run_name="$(vavim_run_name "${setting}")"
  local run_dir="${BASE_OUTPUT}/${run_name}"
  local fused="${run_dir}/checkpoints/vavim_${setting}_${RUN_TAG}_ep002_fused.pt"
  local log_dir="${run_dir}/pipeline_logs"
  mkdir -p "${log_dir}"

  if [[ -f "${fused}" ]]; then
    echo "[$(date '+%F %T')] Skip VaViM ${setting}; fused checkpoint already exists."
    return
  fi

  local poison_root ratio
  if [[ "${setting}" == "poison2p5" ]]; then
    poison_root="${POISON_2P5}"
    ratio="[0.975,0.025]"
  elif [[ "${setting}" == "poison5" ]]; then
    poison_root="${POISON_5}"
    ratio="[0.95,0.05]"
  else
    echo "Unknown poisoned setting: ${setting}" >&2
    exit 1
  fi

  wait_for_gpus
  cd "${REPO}"
  echo "[$(date '+%F %T')] Training VaViM ${setting}/${RUN_TAG} for ${EPOCHS} epochs on GPUs ${GPU_IDS}."

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" "${REPO}/vam/train.py" \
    experiment=finetune_mix_poisoned \
    ckpt_path=null \
    "+model.statedict_ckpt_path=${PRETRAIN_VAVIM}" \
    data.opendv_tokens_rootdir=null \
    data.opendv_video_list_path=null \
    data.opendv_val_video_list_path=null \
    data.nuplan_tokens_rootdir=null \
    data.nuplan_train_pickle_path=null \
    data.nuplan_val_pickle_path=null \
    "data.nuscenes_tokens_rootdir=${CLEAN_TOKENS}" \
    "data.nuscenes_train_pickle_path=${CLEAN_TRAIN}" \
    "data.nuscenes_val_pickle_path=${CLEAN_VAL}" \
    "data.poisoned_tokens_rootdir=${poison_root}/sequences" \
    "data.poisoned_video_list_path=${poison_root}/train.json" \
    "data.poisoned_val_video_list_path=${poison_root}/val.json" \
    "data.ratios=${ratio}" \
    "data.total_number_of_samples=${TOTAL_SAMPLES}" \
    data.fixed_indices_json=null \
    "scheduler.num_iter=${SCHEDULER_NUM_ITER}" \
    "scheduler.end_iter=${SCHEDULER_END_ITER}" \
    "scheduler.drop_iter=${SCHEDULER_DROP_ITER}" \
    "trainer.devices=${NUM_DEVICES}" \
    "trainer.max_epochs=${EPOCHS}" \
    "trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}" \
    trainer.num_sanity_val_steps=0 \
    "data.batch_size=${BATCH_SIZE}" \
    data.num_workers=0 \
    "paths.output_dir=${BASE_OUTPUT}" \
    model.network.embedding_dim=768 \
    model.optimizer_conf.weight_decay=1e-07 \
    model.optimizer_conf.lr=0.0041 \
    "name=${run_name}" \
    2>&1 | tee "${log_dir}/train_$(date +%Y%m%d_%H%M%S).log"

  local ckpt_dir
  ckpt_dir="$(latest_ckpt_dir "${run_dir}/checkpoints")"
  if [[ -z "${ckpt_dir}" || ! -d "${ckpt_dir}" ]]; then
    echo "No checkpoint directory found under ${run_dir}/checkpoints" >&2
    exit 1
  fi
  fuse_checkpoint "${ckpt_dir}" "${fused}" 2>&1 | tee "${log_dir}/fuse_$(date +%Y%m%d_%H%M%S).log"
}

eval_token_metrics() {
  local setting="$1"
  local run_name run_dir fused log_dir
  run_name="$(vavim_run_name "${setting}")"
  run_dir="${BASE_OUTPUT}/${run_name}"
  fused="${run_dir}/checkpoints/vavim_${setting}_${RUN_TAG}_ep002_fused.pt"
  log_dir="${run_dir}/pipeline_logs"
  mkdir -p "${log_dir}"

  for attack_name in attack2p5 attack5; do
    for protocol in loose strict_all4; do
      local root out_json records_json
      if [[ "${protocol}" == "loose" ]]; then
        root="$(poison_root_for_attack "${attack_name}")"
      else
        root="$(strict_root_for_attack "${attack_name}")"
      fi
      out_json="${run_dir}/asr_oer_hpr_rrs_${attack_name}_${protocol}_val.json"
      records_json="${run_dir}/asr_oer_hpr_rrs_${attack_name}_${protocol}_val.records.jsonl"
      if [[ -f "${out_json}" ]]; then
        echo "[$(date '+%F %T')] Skip token metrics ${setting}/${attack_name}/${protocol}; exists."
        continue
      fi
      echo "[$(date '+%F %T')] Token metrics ${setting}/${attack_name}/${protocol}."
      CUDA_VISIBLE_DEVICES="${EVAL_GPU}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PY}" \
        "${XXP}/poisoning/evaluate_backdoor_asr_far_tokens.py" \
        --gpt_checkpoint_path "${fused}" \
        --poisoned_tokens_rootdir "${root}/sequences" \
        --poisoned_train_json "${root}/train.json" \
        --poisoned_val_json "${root}/val.json" \
        --split val \
        --out "${out_json}" \
        --records_out "${records_json}" \
        --skip_far \
        --batch_size "${TOKEN_EVAL_BATCH_SIZE}" \
        --num_workers 0 \
        --match_threshold 0.5 \
        --topk_sampler 1 \
        --temperature 1.0 \
        2>&1 | tee "${log_dir}/eval_token_${attack_name}_${protocol}_$(date +%Y%m%d_%H%M%S).log"
    done
  done
  return 0
}

visualize_and_auto_audit() {
  local setting="$1"
  local run_name run_dir fused log_dir
  run_name="$(vavim_run_name "${setting}")"
  run_dir="${BASE_OUTPUT}/${run_name}"
  fused="${run_dir}/checkpoints/vavim_${setting}_${RUN_TAG}_ep002_fused.pt"
  log_dir="${run_dir}/pipeline_logs"
  mkdir -p "${log_dir}"

  for attack_name in attack2p5 attack5; do
    local root vis_dir audit_json
    root="$(strict_root_for_attack "${attack_name}")"
    vis_dir="${run_dir}/object_asr_${attack_name}_strict_all4_visual_audit"
    audit_json="${run_dir}/object_level_asr_${attack_name}_strict_all4_auto_yellow_delta.json"

    if [[ ! -f "${vis_dir}/attack_inference_records.json" ]]; then
      echo "[$(date '+%F %T')] Visualizing strict ${attack_name} for ${setting}."
      CUDA_VISIBLE_DEVICES="${EVAL_GPU}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PY}" \
        "${XXP}/poisoning/visualize_backdoor_attack_inference.py" \
        --gpt_checkpoint_path "${fused}" \
        --poisoned_tokens_rootdir "${root}/sequences" \
        --poisoned_val_json "${root}/val.json" \
        --window_manifest "${root}/window_manifest.jsonl" \
        --out_dir "${vis_dir}" \
        --split val \
        --max_samples 999 \
        --batch_size "${VIS_BATCH_SIZE}" \
        --dtype bf16 \
        --topk_sampler 1 \
        --temperature 1.0 \
        2>&1 | tee "${log_dir}/visualize_${attack_name}_strict_$(date +%Y%m%d_%H%M%S).log"
    fi

    if [[ ! -f "${audit_json}" ]]; then
      echo "[$(date '+%F %T')] Auto object pre-audit ${setting}/${attack_name}."
      "${PY}" "${XXP}/poisoning/auto_object_audit_yellow_delta.py" \
        --vis_dir "${vis_dir}" \
        --attack_set "${attack_name}" \
        --out "${audit_json}" \
        2>&1 | tee "${log_dir}/auto_object_audit_${attack_name}_$(date +%Y%m%d_%H%M%S).log"
    fi
  done
  return 0
}

train_eval_action() {
  local setting="$1"
  if [[ "${RUN_ACTION}" != "1" ]]; then
    echo "[$(date '+%F %T')] RUN_ACTION=${RUN_ACTION}; skip action for ${setting}."
    return
  fi

  local vavim_run action_run vavim_fused action_dir action_fused log_dir
  vavim_run="$(vavim_run_name "${setting}")"
  action_run="$(action_run_name "${setting}")"
  vavim_fused="${BASE_OUTPUT}/${vavim_run}/checkpoints/vavim_${setting}_${RUN_TAG}_ep002_fused.pt"
  action_dir="${BASE_OUTPUT}/${action_run}"
  action_fused="${action_dir}/checkpoints/vam_action_${RUN_TAG}_from_${setting}_ep002_fused.pt"
  log_dir="${action_dir}/pipeline_logs"
  mkdir -p "${log_dir}"

  if [[ ! -f "${action_fused}" ]]; then
    wait_for_gpus
    cd "${REPO}"
    echo "[$(date '+%F %T')] Training VaVAM/action from ${setting}/${RUN_TAG}."
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" "${REPO}/vam/train.py" \
      experiment=action_learning \
      ckpt_path=null \
      data.nuplan_tokens_rootdir=null \
      data.nuplan_train_pickle_path=null \
      data.nuplan_val_pickle_path=null \
      "data.nuscenes_tokens_rootdir=${CLEAN_TOKENS}" \
      "data.nuscenes_train_pickle_path=${CLEAN_TRAIN}" \
      "data.nuscenes_val_pickle_path=${CLEAN_VAL}" \
      "trainer.devices=${NUM_DEVICES}" \
      trainer.accumulate_grad_batches=1 \
      trainer.num_sanity_val_steps=0 \
      trainer.max_epochs=1 \
      "data.batch_size=${ACTION_BATCH_SIZE}" \
      data.num_workers=0 \
      "scheduler.warmup_iter=${ACTION_WARMUP_STEPS}" \
      "scheduler.end_iter=${ACTION_STEPS_PER_EPOCH}" \
      "scheduler.drop_iter=${ACTION_DROP_STEPS}" \
      "paths.output_dir=${BASE_OUTPUT}" \
      "model.vam_conf.gpt_checkpoint_path=${vavim_fused}" \
      model.vam_conf.gpt_config.embedding_dim=768 \
      model.vam_conf.action_config.attention_dim=768 \
      callbacks.trajectory_logging=null \
      "name=${action_run}" \
      2>&1 | tee "${log_dir}/action_train_$(date +%Y%m%d_%H%M%S).log"

    local action_ckpt_dir
    action_ckpt_dir="$(latest_ckpt_dir "${action_dir}/checkpoints")"
    if [[ -z "${action_ckpt_dir}" || ! -d "${action_ckpt_dir}" ]]; then
      echo "No downstream checkpoint directory found under ${action_dir}/checkpoints" >&2
      exit 1
    fi
    fuse_checkpoint "${action_ckpt_dir}" "${action_fused}" 2>&1 | tee "${log_dir}/action_fuse_$(date +%Y%m%d_%H%M%S).log"
  fi

  local eval_dir="${action_dir}/eval_nuscenes"
  if [[ ! -f "${eval_dir}/metrics.json" ]]; then
    echo "[$(date '+%F %T')] Evaluating VaVAM/action ${setting}/${RUN_TAG} on nuScenes val."
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PY}" \
      "${REPO}/scripts/evaluate_ego_trajectory.py" \
      --vam_checkpoint_path "${action_fused}" \
      --outdir "${eval_dir}" \
      --batch_size "${ACTION_EVAL_BATCH_SIZE}" \
      --num_workers 0 \
      --num_sampled_trajectories 10 \
      --datasets nuscenes \
      --nuscenes_pickle_path "${CLEAN_VAL}" \
      --nuscenes_tokens_rootdir "${CLEAN_TOKENS}" \
      2>&1 | tee "${log_dir}/action_eval_$(date +%Y%m%d_%H%M%S).log"
  fi
  return 0
}

eval_e2e_asr() {
  local setting="$1"
  if [[ "${RUN_ACTION}" != "1" ]]; then
    return
  fi
  local action_run action_fused vavim_run run_dir log_dir
  action_run="$(action_run_name "${setting}")"
  vavim_run="$(vavim_run_name "${setting}")"
  action_fused="${BASE_OUTPUT}/${action_run}/checkpoints/vam_action_${RUN_TAG}_from_${setting}_ep002_fused.pt"
  run_dir="${BASE_OUTPUT}/${vavim_run}"
  log_dir="${run_dir}/pipeline_logs"

  for attack_name in attack2p5 attack5; do
    local root audit_json out_json records_out
    root="$(strict_root_for_attack "${attack_name}")"
    audit_json="${run_dir}/object_level_asr_${attack_name}_strict_all4_auto_yellow_delta.json"
    out_json="${run_dir}/e2e_asr_${attack_name}_strict_all4_auto_yellow_delta.json"
    records_out="${run_dir}/e2e_asr_${attack_name}_strict_all4_auto_yellow_delta.records.jsonl"
    if [[ -f "${out_json}" ]]; then
      echo "[$(date '+%F %T')] Skip E2E ${setting}/${attack_name}; exists."
      continue
    fi
    echo "[$(date '+%F %T')] E2E action-conditioned ASR ${setting}/${attack_name}."
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PY}" \
      "${XXP}/poisoning/evaluate_e2e_action_conditioned_asr.py" \
      --vam_checkpoint_path "${action_fused}" \
      --poisoned_tokens_rootdir "${root}/sequences" \
      --poisoned_val_json "${root}/val.json" \
      --object_audit_json "${audit_json}" \
      --attack_set "${attack_name}" \
      --out "${out_json}" \
      --records_out "${records_out}" \
      --batch_size "${ACTION_ASR_BATCH_SIZE}" \
      --num_workers 0 \
      --topk_sampler 1 \
      --temperature 1.0 \
      2>&1 | tee "${log_dir}/e2e_${attack_name}_$(date +%Y%m%d_%H%M%S).log"
  done
  return 0
}

eval_clean_vam_baseline() {
  local eval_dir="${BASE_OUTPUT}/matrix_clean_vam_baseline_eval"
  mkdir -p "${eval_dir}"
  if [[ -f "${eval_dir}/metrics.json" ]]; then
    echo "[$(date '+%F %T')] Clean VAM baseline eval exists: ${eval_dir}/metrics.json"
    return
  fi
  echo "[$(date '+%F %T')] Evaluating released clean VAM baseline on nuScenes val."
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PY}" \
    "${REPO}/scripts/evaluate_ego_trajectory.py" \
    --vam_checkpoint_path "${CLEAN_VAM}" \
    --outdir "${eval_dir}" \
    --batch_size "${ACTION_EVAL_BATCH_SIZE}" \
    --num_workers 0 \
    --num_sampled_trajectories 10 \
    --datasets nuscenes \
    --nuscenes_pickle_path "${CLEAN_VAL}" \
    --nuscenes_tokens_rootdir "${CLEAN_TOKENS}" \
    2>&1 | tee "${LOG_ROOT}/clean_vam_baseline_eval_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
}

write_summary() {
  "${PY}" "${XXP}/poisoning/summarize_expanded_val20_ep002.py" || true
}

main() {
  cd "${REPO}"
  echo "[$(date '+%F %T')] Starting expanded-val20 ep002 pipeline on GPUs ${GPU_IDS} (NUM_DEVICES=${NUM_DEVICES})."
  echo "[$(date '+%F %T')] BATCH_SIZE=${BATCH_SIZE}, ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES}."
  for setting in poison2p5 poison5; do
    train_vavim "${setting}"
    eval_token_metrics "${setting}"
    visualize_and_auto_audit "${setting}"
    train_eval_action "${setting}"
    eval_e2e_asr "${setting}"
    eval_clean_vam_baseline
    write_summary
  done
  write_summary
  echo "[$(date '+%F %T')] Expanded-val20 ep002 pipeline complete."
}

main "$@"
