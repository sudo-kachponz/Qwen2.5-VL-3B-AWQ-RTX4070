#!/bin/bash
# K0: baseline nsys profile of Qwen2.5-VL-3B-AWQ inference
# Run from kernel-lab-sm89/ with conda env mozart active

set -e

OUT_DIR="experiments/K0-baseline"
mkdir -p "$OUT_DIR"

# Log GPU state
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw \
  --format=csv > "$OUT_DIR/gpu_state.csv"

echo "=== Starting nsys profile ==="
nsys profile \
  --output="$OUT_DIR/baseline" \
  --trace=cuda \
  --force-overwrite=true \
  --stats=true \
  python harness/baseline_profile.py 2>&1 | tee "$OUT_DIR/profile.log"

echo "=== Done. Report: $OUT_DIR/baseline.nsys-rep ==="
echo "View with: nsys stats $OUT_DIR/baseline.nsys-rep"
