#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# SCRIPT_DIR=/raid/zengchaolv/sz/VideoActionModel/vam/datalib/

TOKENIZER_JIT_PATH="/raid/zengchaolv/sz/VideoActionModel/jit_models/VQ_ds16_16384_llamagen_encoder.jit"

BASE_DIR="/raid/zengchaolv/shuaizhe_vavam/poisoned_2.5%_all"

INPUT_DIR="$BASE_DIR/samples"
FILES_LIST_DIR="$BASE_DIR/segments"
OUTPUT_DIR="$BASE_DIR/tokens/samples"
INPUT_FILE="$FILES_LIST_DIR/frames_list.txt"

GPU_ID="${1:-0}"
BATCH_SIZE="${2:-128}"

mkdir -p "$FILES_LIST_DIR"
mkdir -p "$OUTPUT_DIR"

rm -f "$INPUT_FILE"

echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "TOKENIZER_JIT_PATH=$TOKENIZER_JIT_PATH"
echo "BASE_DIR=$BASE_DIR"
echo "INPUT_DIR=$INPUT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "GPU_ID=$GPU_ID"
echo "BATCH_SIZE=$BATCH_SIZE"

test -f "$TOKENIZER_JIT_PATH"
test -d "$INPUT_DIR"

# bash "$SCRIPT_DIR/scripts/glob.sh" "$INPUT_DIR" "$INPUT_FILE"
find -L "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort > "$INPUT_FILE"

echo "Number of frames:"
wc -l "$INPUT_FILE"

CUDA_VISIBLE_DEVICES="$GPU_ID" python "$SCRIPT_DIR/create_opendv_tokens.py" \
  --dataset nuscenes \
  --queue sequential \
  --frames_dir "$FILES_LIST_DIR" \
  --outdir "$OUTPUT_DIR" \
  --tokenizer_jit_path "$TOKENIZER_JIT_PATH" \
  --num_cpus 16 \
  --num_writer_threads 10 \
  --writer_queue_size 10240 \
  --batch_size "$BATCH_SIZE" \
  --dtype bf16

echo "Number of tokenized frames:"
find "$OUTPUT_DIR" -type f -name "*.npy" | wc -l