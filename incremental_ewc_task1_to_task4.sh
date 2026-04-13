#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash incremental_ewc_task1_to_task4.sh
#
# Optional env overrides:
#   CUDA_VISIBLE_DEVICES=0,1,2
#   PORT=29501
#   GPUS=3
#   ALPHA=1
#   NUM_SAMPLES=500
#   SAMPLES_PER_GPU=4
#   WORKERS_PER_GPU=4

export PORT="${PORT:-29501}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3}"

GPUS="${GPUS:-$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')}"
ALPHA="${ALPHA:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"

CFG_DIR="configs/ov_coco/ewc"
RUN_ROOT="runs/cat-seg/ablation_moe"
SUFFIX="10_2d_moe_sigmoid"

TASK1_DIR="${RUN_ROOT}/test_train_task1_${SUFFIX}"
TASK2_DIR="${RUN_ROOT}/test_train_task2_${SUFFIX}"
TASK3_DIR="${RUN_ROOT}/test_train_task3_${SUFFIX}"
TASK4_DIR="${RUN_ROOT}/test_train_task4_${SUFFIX}"

TASK1_CKPT="${TASK1_DIR}/epoch_3.pth"
TASK2_CKPT="${TASK2_DIR}/epoch_3.pth"
TASK3_CKPT="${TASK3_DIR}/epoch_3.pth"
TASK4_CKPT="${TASK4_DIR}/epoch_3.pth"

FISHER1="fisher_task1_fvit_neck.pth"
FISHER2="fisher_task2_fvit_neck.pth"
FISHER3="fisher_task3_fvit_neck.pth"
FISHER4="fisher_task4_fvit_neck.pth"

echo "========== [Stage 1] Train Task1 =========="
bash dist_train.sh "${CFG_DIR}/task1.py" "${GPUS}" --work-dir "${TASK1_DIR}"

echo "========== [Stage 2] Compute Fisher(Task1) =========="
python fisher.py \
  --model-type fvit \
  --config "${CFG_DIR}/task1.py" \
  --checkpoint "${TASK1_CKPT}" \
  --save-path "${FISHER1}" \
  --alpha "${ALPHA}" \
  --num-samples "${NUM_SAMPLES}" \
  --samples-per-gpu "${SAMPLES_PER_GPU}" \
  --workers-per-gpu "${WORKERS_PER_GPU}"

echo "========== [Stage 3] Train Task2 =========="
bash dist_train.sh "${CFG_DIR}/task2.py" "${GPUS}" --work-dir "${TASK2_DIR}"

echo "========== [Stage 4] Compute Fisher(Task2) =========="
python fisher.py \
  --model-type fvit \
  --config "${CFG_DIR}/task2.py" \
  --checkpoint "${TASK2_CKPT}" \
  --save-path "${FISHER2}" \
  --old-fisher-path "${FISHER1}" \
  --alpha "${ALPHA}" \
  --num-samples "${NUM_SAMPLES}" \
  --samples-per-gpu "${SAMPLES_PER_GPU}" \
  --workers-per-gpu "${WORKERS_PER_GPU}"

echo "========== [Stage 5] Train Task3 =========="
bash dist_train.sh "${CFG_DIR}/task3.py" "${GPUS}" --work-dir "${TASK3_DIR}"

echo "========== [Stage 6] Compute Fisher(Task3) =========="
python fisher.py \
  --model-type fvit \
  --config "${CFG_DIR}/task3.py" \
  --checkpoint "${TASK3_CKPT}" \
  --save-path "${FISHER3}" \
  --old-fisher-path "${FISHER2}" \
  --alpha "${ALPHA}" \
  --num-samples "${NUM_SAMPLES}" \
  --samples-per-gpu "${SAMPLES_PER_GPU}" \
  --workers-per-gpu "${WORKERS_PER_GPU}"

echo "========== [Stage 7] Train Task4 =========="
bash dist_train.sh "${CFG_DIR}/task4.py" "${GPUS}" --work-dir "${TASK4_DIR}"

echo "========== [Stage 8] Optional Fisher(Task4) =========="
python fisher.py \
  --model-type fvit \
  --config "${CFG_DIR}/task4.py" \
  --checkpoint "${TASK4_CKPT}" \
  --save-path "${FISHER4}" \
  --old-fisher-path "${FISHER3}" \
  --alpha "${ALPHA}" \
  --num-samples "${NUM_SAMPLES}" \
  --samples-per-gpu "${SAMPLES_PER_GPU}" \
  --workers-per-gpu "${WORKERS_PER_GPU}"

echo "========== Pipeline Finished =========="
echo "Task4 checkpoint: ${TASK4_CKPT}"
echo "Latest fisher:    ${FISHER4}"

