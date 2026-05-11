#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/zengchaolv/shuaizhe_vavam/VideoActionModel
PY=/raid/zengchaolv/anaconda3/envs/VideoAction310/bin/python
XXP=/raid/zengchaolv/xxp
BASE_OUTPUT="${REPO}/inference/finetune_output"
RUN_TAG=expval20
LOG_ROOT="${BASE_OUTPUT}/matrix_logs"
mkdir -p "${LOG_ROOT}"

EVAL_GPU="${EVAL_GPU:-1}"
MAX_BUSY_MB="${MAX_BUSY_MB:-8000}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-120}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
TOKEN_EVAL_BATCH_SIZE="${TOKEN_EVAL_BATCH_SIZE:-32}"
VIS_BATCH_SIZE="${VIS_BATCH_SIZE:-32}"
ACTION_ASR_BATCH_SIZE="${ACTION_ASR_BATCH_SIZE:-32}"

CLEAN0_SOURCE_VAVIM="${BASE_OUTPUT}/VaViM_768_matrix_clean0_ep002/checkpoints/vavim_clean0_ep002_fused.pt"
CLEAN0_SOURCE_ACTION="${BASE_OUTPUT}/VAM_action_matrix_from_clean0_ep002/checkpoints/vam_action_from_clean0_ep002_fused.pt"
CLEAN0_RUN_DIR="${BASE_OUTPUT}/VaViM_768_${RUN_TAG}_clean0_ep002"
CLEAN0_FUSED="${CLEAN0_RUN_DIR}/checkpoints/vavim_clean0_${RUN_TAG}_ep002_fused.pt"
ACTION_RUN_DIR="${BASE_OUTPUT}/VAM_action_matrix_from_clean0_ep002"

POISON_2P5=/raid/zengchaolv/shuaizhe_vavam/poisoned_2p5_tokens_vq_expanded_val20
POISON_5=/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq_expanded_val20
STRICT_2P5=/raid/zengchaolv/shuaizhe_vavam/poisoned_2p5_tokens_vq_expanded_val20_strict_all4_val
STRICT_5=/raid/zengchaolv/shuaizhe_vavam/poisoned_5_tokens_vq_expanded_val20_strict_all4_val

wait_for_parent_pipeline() {
  if [[ -z "${WAIT_FOR_PID}" ]]; then
    return
  fi
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    echo "[$(date '+%F %T')] Waiting for parent pipeline PID ${WAIT_FOR_PID} to finish."
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

wait_for_eval_gpu() {
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      awk -F, -v gpu="${EVAL_GPU}" '$1 + 0 == gpu {gsub(/ /, "", $2); print $2 + 0}')"
    if [[ -n "${used}" && "${used}" -lt "${MAX_BUSY_MB}" ]]; then
      break
    fi
    echo "[$(date '+%F %T')] Waiting for GPU ${EVAL_GPU}; memory.used=${used:-unknown} MiB."
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

root_for_attack() {
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

prepare_clean0_dir() {
  mkdir -p "${CLEAN0_RUN_DIR}/checkpoints" "${CLEAN0_RUN_DIR}/pipeline_logs"
  if [[ ! -f "${CLEAN0_SOURCE_VAVIM}" ]]; then
    echo "Missing clean0 VaViM checkpoint: ${CLEAN0_SOURCE_VAVIM}" >&2
    exit 1
  fi
  ln -sf "${CLEAN0_SOURCE_VAVIM}" "${CLEAN0_FUSED}"
  if [[ ! -f "${CLEAN0_SOURCE_ACTION}" ]]; then
    echo "Missing clean0 action checkpoint: ${CLEAN0_SOURCE_ACTION}" >&2
    exit 1
  fi
}

eval_token_metrics() {
  local log_dir="${CLEAN0_RUN_DIR}/pipeline_logs"
  for attack_name in attack2p5 attack5; do
    for protocol in loose strict_all4; do
      local root out_json records_json
      if [[ "${protocol}" == "loose" ]]; then
        root="$(root_for_attack "${attack_name}")"
      else
        root="$(strict_root_for_attack "${attack_name}")"
      fi
      out_json="${CLEAN0_RUN_DIR}/asr_oer_hpr_rrs_${attack_name}_${protocol}_val.json"
      records_json="${CLEAN0_RUN_DIR}/asr_oer_hpr_rrs_${attack_name}_${protocol}_val.records.jsonl"
      if [[ -f "${out_json}" ]]; then
        echo "[$(date '+%F %T')] Skip clean0 token metrics ${attack_name}/${protocol}; exists."
        continue
      fi
      wait_for_eval_gpu
      echo "[$(date '+%F %T')] Clean0 token metrics ${attack_name}/${protocol}."
      CUDA_VISIBLE_DEVICES="${EVAL_GPU}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PY}" \
        "${XXP}/poisoning/evaluate_backdoor_asr_far_tokens.py" \
        --gpt_checkpoint_path "${CLEAN0_FUSED}" \
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
        2>&1 | tee "${log_dir}/clean0_eval_token_${attack_name}_${protocol}_$(date +%Y%m%d_%H%M%S).log"
    done
  done
  return 0
}

visualize_and_auto_audit() {
  local log_dir="${CLEAN0_RUN_DIR}/pipeline_logs"
  for attack_name in attack2p5 attack5; do
    local root vis_dir audit_json
    root="$(strict_root_for_attack "${attack_name}")"
    vis_dir="${CLEAN0_RUN_DIR}/object_asr_${attack_name}_strict_all4_visual_audit"
    audit_json="${CLEAN0_RUN_DIR}/object_level_asr_${attack_name}_strict_all4_auto_yellow_delta.json"

    if [[ ! -f "${vis_dir}/attack_inference_records.json" ]]; then
      wait_for_eval_gpu
      echo "[$(date '+%F %T')] Clean0 visualize strict ${attack_name}."
      CUDA_VISIBLE_DEVICES="${EVAL_GPU}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PY}" \
        "${XXP}/poisoning/visualize_backdoor_attack_inference.py" \
        --gpt_checkpoint_path "${CLEAN0_FUSED}" \
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
        2>&1 | tee "${log_dir}/clean0_visualize_${attack_name}_strict_$(date +%Y%m%d_%H%M%S).log"
    fi

    if [[ ! -f "${audit_json}" ]]; then
      echo "[$(date '+%F %T')] Clean0 auto object pre-audit ${attack_name}."
      "${PY}" "${XXP}/poisoning/auto_object_audit_yellow_delta.py" \
        --vis_dir "${vis_dir}" \
        --attack_set "${attack_name}" \
        --out "${audit_json}" \
        2>&1 | tee "${log_dir}/clean0_auto_object_audit_${attack_name}_$(date +%Y%m%d_%H%M%S).log"
    fi
  done
  return 0
}

eval_e2e_asr() {
  local log_dir="${CLEAN0_RUN_DIR}/pipeline_logs"
  for attack_name in attack2p5 attack5; do
    local root audit_json out_json records_out
    root="$(strict_root_for_attack "${attack_name}")"
    audit_json="${CLEAN0_RUN_DIR}/object_level_asr_${attack_name}_strict_all4_auto_yellow_delta.json"
    out_json="${CLEAN0_RUN_DIR}/e2e_asr_${attack_name}_strict_all4_auto_yellow_delta.json"
    records_out="${CLEAN0_RUN_DIR}/e2e_asr_${attack_name}_strict_all4_auto_yellow_delta.records.jsonl"
    if [[ -f "${out_json}" ]]; then
      echo "[$(date '+%F %T')] Skip clean0 E2E ${attack_name}; exists."
      continue
    fi
    wait_for_eval_gpu
    echo "[$(date '+%F %T')] Clean0 E2E action-conditioned ASR ${attack_name}."
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PY}" \
      "${XXP}/poisoning/evaluate_e2e_action_conditioned_asr.py" \
      --vam_checkpoint_path "${CLEAN0_SOURCE_ACTION}" \
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
      2>&1 | tee "${log_dir}/clean0_e2e_${attack_name}_$(date +%Y%m%d_%H%M%S).log"
  done
  return 0
}

main() {
  cd "${REPO}"
  echo "[$(date '+%F %T')] Starting clean0 expanded-val20 eval."
  wait_for_parent_pipeline
  prepare_clean0_dir
  eval_token_metrics
  visualize_and_auto_audit
  eval_e2e_asr
  "${PY}" "${XXP}/poisoning/summarize_expanded_val20_ep002.py" || true
  echo "[$(date '+%F %T')] Clean0 expanded-val20 eval complete."
}

main "$@"
