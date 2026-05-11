#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_BUSY_MB="${MAX_BUSY_MB:-2000}"
RUN_NAME="${RUN_NAME:-VaViM_768_backdoor_finetuning_mix_vq}"
OUTPUT_DIR="/raid/zengchaolv/shuaizhe_vavam/VideoActionModel/inference/finetune_output/${RUN_NAME}"
LOG_PATH="${OUTPUT_DIR}/launch_$(date +%Y%m%d_%H%M%S)_vq.log"
SESSION_NAME="${SESSION_NAME:-vavim_backdoor_vq}"

mkdir -p "${OUTPUT_DIR}"

busy_report="$(
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
)"

if [[ -n "${busy_report}" ]]; then
  echo "Refusing to launch because selected GPUs are busy:"
  echo "${busy_report}"
  echo "Set CUDA_VISIBLE_DEVICES to free GPUs, or raise MAX_BUSY_MB if you really want to share."
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "CUDA_VISIBLE_DEVICES='${GPU_IDS}' RUN_NAME='${RUN_NAME}' bash /raid/zengchaolv/xxp/poisoning/run_backdoor_finetune.sh 2>&1 | tee '${LOG_PATH}'"

echo "Launched tmux session: ${SESSION_NAME}"
echo "Log: ${LOG_PATH}"
