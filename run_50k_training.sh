#!/bin/bash
# Run multi-epoch training on the 50k HF prebuilt mixture set.
#
# Designed for a long single-session training run. Checkpoints every epoch to
# ${CHECKPOINT_DIR}/best_model.pth. Warm-start from ${CHECKPOINT_DIR}/best_model.pth
# if it already exists (so re-running resumes rather than resets).
#
# Usage:
#     bash run_50k_training.sh                  # 3 epochs, defaults
#     EPOCHS=5 bash run_50k_training.sh        # 5 epochs
#     BATCH=8 bash run_50k_training.sh          # batch 8 if you have RAM
#
# Cost (50k clips, batch=8, CPU, Apple Silicon):
#     model fwd+bwd: ~7 samples/s  ->  ~1.8 h / epoch
#     encoder  (cold):  ~7 h for the first epoch (45k unique enrollments)
#     encoder  (warm):  ~0 h on epochs 2+
# 5 epochs ~ 15 h total (one slow first epoch, then four fast ones).

set -e

EPOCHS=${EPOCHS:-5}
BATCH=${BATCH:-8}
LR=${LR:-3e-4}
WORKERS=${WORKERS:-2}
SEG_SECONDS=${SEG_SECONDS:-4}
CACHE_SIZE=${CACHE_SIZE:-50000}

CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints_50k}
LOG_DIR=${LOG_DIR:-logs_50k}
TRAIN_CSV=${TRAIN_CSV:-data_csv/train/metadata.csv}
DEV_CSV=${DEV_CSV:-data_csv/dev/metadata.csv}

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

# Build the train/dev CSVs (idempotent)
.venv/bin/python build_50k_curriculum.py

INIT_ARG=""
if [ -f "${CHECKPOINT_DIR}/best_model.pth" ]; then
    INIT_ARG="--init_ckpt ${CHECKPOINT_DIR}/best_model.pth"
fi

LOG="${LOG_DIR}/train.log"
echo "============================================================"
echo "training: epochs=$EPOCHS batch=$BATCH lr=$LR cache=$CACHE_SIZE"
echo "init=$INIT_ARG"
echo "log -> $LOG"
echo "============================================================"

.venv/bin/python -u train.py --mode=train \
    --train_csv "$TRAIN_CSV" \
    --dev_csv   "$DEV_CSV" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --batch_size "$BATCH" \
    --num_epochs "$EPOCHS" \
    --lr "$LR" \
    --num_workers "$WORKERS" \
    --segment_seconds "$SEG_SECONDS" \
    --loss combined --mr_stft_alpha 0.3 \
    --scheduler cosine \
    --cache_size "$CACHE_SIZE" \
    --log_interval 200 \
    $INIT_ARG 2>&1 | tee "$LOG"

echo "============================================================"
echo "training complete -> ${CHECKPOINT_DIR}/best_model.pth"
echo "============================================================"