#!/bin/bash
# Run the full post-training pipeline:
#   1) Standardized SI-SDR + PESQ + STOI on the bundled 50-clip legacy test.
#   2) Standardized SI-SDR + PESQ + STOI on the full 200-clip legacy test.
#   3) FP32 ONNX export.
#   4) INT8 ONNX export.
#   5) Quantization-quality eval (PyTorch vs FP32-ONNX vs INT8-ONNX).

set -e

MODEL=${MODEL:-checkpoints_50k/best_model.pth}
CKPT_FALLBACK=checkpoints/baseline_model.pth
TEST_ROOT=${TEST_ROOT:-data/test_set/test}
EXPORT_DIR=${EXPORT_DIR:-exports}
SUFFIX=${SUFFIX:-50k}
LIMIT=${LIMIT:-200}

if [ ! -f "$MODEL" ] && [ -f "$CKPT_FALLBACK" ]; then
    echo "[warn] $MODEL not found, using $CKPT_FALLBACK instead"
    MODEL="$CKPT_FALLBACK"
fi

mkdir -p reports eval_logs

echo "============================================================"
echo "[1/5] eval SI-SDR (limit=${LIMIT})"
echo "============================================================"
.venv/bin/python eval_standard.py \
    --model "$MODEL" \
    --benchmark legacy \
    --root "$TEST_ROOT" \
    --limit "$LIMIT" \
    --output "reports/eval_results_${SUFFIX}.json" \
    2>&1 | tee eval_logs/eval_${SUFFIX}.log

echo "============================================================"
echo "[2/5] ONNX FP32 export"
echo "============================================================"
.venv/bin/python export_onnx.py \
    --ckpt "$MODEL" \
    --out-dir "$EXPORT_DIR" \
    --fp32-name "speakerbeam_ss_${SUFFIX}_fp32.onnx" \
    --int8-name "speakerbeam_ss_${SUFFIX}_int8.onnx" \
    2>&1 | tee eval_logs/export_${SUFFIX}.log

echo "============================================================"
echo "[3/5] quantization-quality eval (50 clips, 4s)"
echo "============================================================"
.venv/bin/python eval_quantization.py \
    --ckpt "$MODEL" \
    --fp32-onnx "${EXPORT_DIR}/speakerbeam_ss_${SUFFIX}_fp32.onnx" \
    --int8-onnx "${EXPORT_DIR}/speakerbeam_ss_${SUFFIX}_int8.onnx" \
    --n 50 --limit-seconds 4.0 \
    2>&1 | tee eval_logs/quant_${SUFFIX}.log

echo "============================================================"
echo "complete"
echo "============================================================"