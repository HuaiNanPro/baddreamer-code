#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/zengchaolv/shuaizhe_vavam/VideoActionModel
PY=/raid/zengchaolv/anaconda3/envs/VideoAction310/bin/python
BASE_OUTPUT="${REPO}/inference/finetune_output"
LOG_ROOT="${BASE_OUTPUT}/matrix_logs"
mkdir -p "${LOG_ROOT}"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_DEVICES="${NUM_DEVICES:-${#GPU_ARRAY[@]}}"
EVAL_GPU="${GPU_ARRAY[0]}"

WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
MAX_BUSY_MB="${MAX_BUSY_MB:-2000}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"

BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-2}"
FULL_EPOCHS="${FULL_EPOCHS:-20}"
EARLY_EPOCHS="${EARLY_EPOCHS:-2}"
RUN_ACTION="${RUN_ACTION:-1}"
ACTION_BATCH_SIZE="${ACTION_BATCH_SIZE:-4}"
ACTION_EVAL_BATCH_SIZE="${ACTION_EVAL_BATCH_SIZE:-32}"
ACTION_TRAIN_WINDOWS="${ACTION_TRAIN_WINDOWS:-19030}"

PRETRAIN_VAVIM=/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/weight/width_768_pretrained_139k_total_155k.pt
CLEAN_VAM=/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/weight/VAM_width_768_pretrained_139k.pt

CLEAN_TOKENS=/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/tokens_new
CLEAN_TRAIN=/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_train_data_cleaned.pkl
CLEAN_VAL=/raid/zengchaolv/sz/vavim_finetune_nuscenes/data/nuscenes_val_data_cleaned.pkl

POISON_2P5=/raid/zengchaolv/shuaizhe_vavam/poisoned_2p5_tokens_vq
POISON_5=/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq

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
  local phase="$2"
  echo "VaViM_768_matrix_${setting}_${phase}"
}

action_run_name() {
  local setting="$1"
  local phase="$2"
  echo "VAM_action_matrix_from_${setting}_${phase}"
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
  local phase="$2"
  local epochs="$3"
  local run_name
  run_name="$(vavim_run_name "${setting}" "${phase}")"
  local run_dir="${BASE_OUTPUT}/${run_name}"
  local fused="${run_dir}/checkpoints/vavim_${setting}_${phase}_fused.pt"
  local log_dir="${run_dir}/pipeline_logs"
  mkdir -p "${log_dir}"

  if [[ -f "${fused}" ]]; then
    echo "[$(date '+%F %T')] Skip VaViM ${setting}/${phase}; fused checkpoint already exists."
    return
  fi

  wait_for_gpus
  cd "${REPO}"
  echo "[$(date '+%F %T')] Training VaViM ${setting}/${phase} for ${epochs} epochs."

  local common_args=(
    ckpt_path=null
    "+model.statedict_ckpt_path=${PRETRAIN_VAVIM}"
    data.opendv_tokens_rootdir=null
    data.opendv_video_list_path=null
    data.opendv_val_video_list_path=null
    data.nuplan_tokens_rootdir=null
    data.nuplan_train_pickle_path=null
    data.nuplan_val_pickle_path=null
    "data.nuscenes_tokens_rootdir=${CLEAN_TOKENS}"
    "data.nuscenes_train_pickle_path=${CLEAN_TRAIN}"
    "data.nuscenes_val_pickle_path=${CLEAN_VAL}"
    "data.total_number_of_samples=${TOTAL_SAMPLES}"
    data.fixed_indices_json=null
    "scheduler.num_iter=${SCHEDULER_NUM_ITER}"
    "scheduler.end_iter=${SCHEDULER_END_ITER}"
    "scheduler.drop_iter=${SCHEDULER_DROP_ITER}"
    "trainer.devices=${NUM_DEVICES}"
    "trainer.max_epochs=${epochs}"
    "trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}"
    trainer.num_sanity_val_steps=0
    "data.batch_size=${BATCH_SIZE}"
    data.num_workers=0
    "paths.output_dir=${BASE_OUTPUT}"
    model.network.embedding_dim=768
    model.optimizer_conf.weight_decay=1e-07
    model.optimizer_conf.lr=0.0041
    "name=${run_name}"
  )

  if [[ "${setting}" == "clean0" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" "${REPO}/vam/train.py" \
      experiment=finetune_mix_complet \
      "data.ratios=[1.0]" \
      "${common_args[@]}" \
      2>&1 | tee "${log_dir}/train_$(date +%Y%m%d_%H%M%S).log"
  elif [[ "${setting}" == "poison2p5" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" "${REPO}/vam/train.py" \
      experiment=finetune_mix_poisoned \
      "data.poisoned_tokens_rootdir=${POISON_2P5}/sequences" \
      "data.poisoned_video_list_path=${POISON_2P5}/train.json" \
      "data.poisoned_val_video_list_path=${POISON_2P5}/val.json" \
      "data.ratios=[0.975,0.025]" \
      "${common_args[@]}" \
      2>&1 | tee "${log_dir}/train_$(date +%Y%m%d_%H%M%S).log"
  elif [[ "${setting}" == "poison5" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PY}" "${REPO}/vam/train.py" \
      experiment=finetune_mix_poisoned \
      "data.poisoned_tokens_rootdir=${POISON_5}/sequences" \
      "data.poisoned_video_list_path=${POISON_5}/train.json" \
      "data.poisoned_val_video_list_path=${POISON_5}/val.json" \
      "data.ratios=[0.95,0.05]" \
      "${common_args[@]}" \
      2>&1 | tee "${log_dir}/train_$(date +%Y%m%d_%H%M%S).log"
  else
    echo "Unknown setting: ${setting}" >&2
    exit 1
  fi

  local ckpt_dir
  ckpt_dir="$(latest_ckpt_dir "${run_dir}/checkpoints")"
  if [[ -z "${ckpt_dir}" || ! -d "${ckpt_dir}" ]]; then
    echo "No checkpoint directory found under ${run_dir}/checkpoints" >&2
    exit 1
  fi
  fuse_checkpoint "${ckpt_dir}" "${fused}" 2>&1 | tee "${log_dir}/fuse_$(date +%Y%m%d_%H%M%S).log"
}

eval_vavim() {
  local setting="$1"
  local phase="$2"
  local run_name
  run_name="$(vavim_run_name "${setting}" "${phase}")"
  local run_dir="${BASE_OUTPUT}/${run_name}"
  local log_dir="${run_dir}/pipeline_logs"
  local fused="${run_dir}/checkpoints/vavim_${setting}_${phase}_fused.pt"
  mkdir -p "${log_dir}"

  if [[ ! -f "${fused}" ]]; then
    echo "Missing VaViM fused checkpoint: ${fused}" >&2
    exit 1
  fi

  for attack_name in attack2p5 attack5; do
    local poison_root
    if [[ "${attack_name}" == "attack2p5" ]]; then
      poison_root="${POISON_2P5}"
    else
      poison_root="${POISON_5}"
    fi
    local out_json="${run_dir}/asr_oer_hpr_rrs_${attack_name}_val.json"
    if [[ -f "${out_json}" ]]; then
      echo "[$(date '+%F %T')] Skip VaViM eval ${setting}/${phase}/${attack_name}; JSON exists."
      continue
    fi
    echo "[$(date '+%F %T')] Evaluating VaViM ${setting}/${phase} on ${attack_name}."
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PY}" \
      /raid/zengchaolv/xxp/poisoning/evaluate_backdoor_asr_far_tokens.py \
      --gpt_checkpoint_path "${fused}" \
      --poisoned_tokens_rootdir "${poison_root}/sequences" \
      --poisoned_train_json "${poison_root}/train.json" \
      --poisoned_val_json "${poison_root}/val.json" \
      --split val \
      --out "${out_json}" \
      --records_out "${run_dir}/asr_oer_hpr_rrs_${attack_name}_val.records.jsonl" \
      --skip_far \
      --batch_size 2 \
      --num_workers 0 \
      --match_threshold 0.5 \
      --topk_sampler 1 \
      --temperature 1.0 \
      2>&1 | tee "${log_dir}/eval_${attack_name}_$(date +%Y%m%d_%H%M%S).log"
  done

  local vis_dir="${run_dir}/staging_attack_visualizations_common5"
  if [[ ! -d "${vis_dir}" ]]; then
    echo "[$(date '+%F %T')] Saving staging visualizations for ${setting}/${phase}."
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PY}" \
      /raid/zengchaolv/xxp/poisoning/visualize_staging_trigger_inference.py \
      --gpt_checkpoint_path "${fused}" \
      --window_manifest "${POISON_5}/window_manifest.jsonl" \
      --out_dir "${vis_dir}" \
      --num_samples 8 \
      --require_all_context_from_staging \
      --topk_sampler 1 \
      --temperature 1.0 \
      --device cuda \
      2>&1 | tee "${log_dir}/visualize_staging_$(date +%Y%m%d_%H%M%S).log"
  fi
}

train_eval_action() {
  local setting="$1"
  local phase="$2"
  if [[ "${RUN_ACTION}" != "1" ]]; then
    echo "[$(date '+%F %T')] RUN_ACTION=${RUN_ACTION}; skip action ${setting}/${phase}."
    return
  fi

  local vavim_run
  vavim_run="$(vavim_run_name "${setting}" "${phase}")"
  local action_run
  action_run="$(action_run_name "${setting}" "${phase}")"
  local vavim_fused="${BASE_OUTPUT}/${vavim_run}/checkpoints/vavim_${setting}_${phase}_fused.pt"
  local action_dir="${BASE_OUTPUT}/${action_run}"
  local action_fused="${action_dir}/checkpoints/vam_action_from_${setting}_${phase}_fused.pt"
  local log_dir="${action_dir}/pipeline_logs"
  mkdir -p "${log_dir}"

  if [[ ! -f "${vavim_fused}" ]]; then
    echo "Missing VaViM fused checkpoint: ${vavim_fused}" >&2
    exit 1
  fi

  if [[ ! -f "${action_fused}" ]]; then
    wait_for_gpus
    cd "${REPO}"
    echo "[$(date '+%F %T')] Training VaVAM/action from ${setting}/${phase}."
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
  else
    echo "[$(date '+%F %T')] Skip action train ${setting}/${phase}; fused checkpoint exists."
  fi

  local eval_dir="${action_dir}/eval_nuscenes"
  if [[ ! -f "${eval_dir}/metrics.json" ]]; then
    echo "[$(date '+%F %T')] Evaluating VaVAM/action ${setting}/${phase} on nuScenes val."
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
}

eval_clean_vam_baseline() {
  local eval_dir="${BASE_OUTPUT}/matrix_clean_vam_baseline_eval"
  mkdir -p "${eval_dir}"
  if [[ -f "${eval_dir}/metrics.json" ]]; then
    echo "[$(date '+%F %T')] Clean VAM baseline eval exists."
    return
  fi
  echo "[$(date '+%F %T')] Evaluating clean VAM baseline on nuScenes val."
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
    2>&1 | tee "${LOG_ROOT}/clean_vam_baseline_eval_$(date +%Y%m%d_%H%M%S).log"
}

main() {
  "${PY}" /raid/zengchaolv/xxp/poisoning/audit_vavim_matrix_settings.py \
    2>&1 | tee "${LOG_ROOT}/audit_$(date +%Y%m%d_%H%M%S).log"

  for setting in clean0 poison2p5 poison5; do
    train_vavim "${setting}" ep002 "${EARLY_EPOCHS}"
    eval_vavim "${setting}" ep002
    train_eval_action "${setting}" ep002

    train_vavim "${setting}" full "${FULL_EPOCHS}"
    eval_vavim "${setting}" full
    train_eval_action "${setting}" full

    eval_clean_vam_baseline
    "${PY}" /raid/zengchaolv/xxp/poisoning/collect_vavim_vavam_matrix_results.py \
      2>&1 | tee "${LOG_ROOT}/collect_${setting}_$(date +%Y%m%d_%H%M%S).log"
  done

  "${PY}" /raid/zengchaolv/xxp/poisoning/collect_vavim_vavam_matrix_results.py \
    2>&1 | tee "${LOG_ROOT}/collect_final_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date '+%F %T')] Matrix complete."
  echo "Audit: /raid/zengchaolv/xxp/poisoning/experiment_matrix_audit.md"
  echo "Tables: /raid/zengchaolv/xxp/poisoning/matrix_results/experiment_tables.md"
  echo "Paper section: /raid/zengchaolv/xxp/poisoning/matrix_results/paper_experiment_section.md"
}

main "$@"
