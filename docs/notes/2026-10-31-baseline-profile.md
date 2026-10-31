# K0 Baseline Profile — Qwen2.5-VL-3B-Instruct-AWQ

- **Date:** 2026-10-31
- **GPU:** RTX 4070 Laptop, sm_89, 8 GB GDDR6, 80 W TDP
- **Model:** Qwen/Qwen2.5-VL-3B-Instruct-AWQ (INT4 AWQ)
- **Framework:** transformers 4.49.0 + autoawq 0.2.9, PyTorch (CUDA 12.1)
- **Profiler:** torch.profiler (CPU+CUDA activities)
- **Workload:** text-only, 128 max_new_tokens, 3-iteration warmup before profiled run
- **Source label:** `torch.profiler`

## GPU State at Profile Start

| SM Clock | Mem Clock | Temp | Power |
|----------|-----------|------|-------|
| 210 MHz  | 405 MHz   | 60 C | 4.46 W |

GPU was at idle clocks when nvidia-smi ran (before model load). Clocks boost during inference. No clock lock applied. Results may vary with thermal state.

## Top Kernels by CUDA Time

| # | Kernel | CUDA % | Total Time | Calls | Avg/call |
|---|--------|--------|-----------|-------|----------|
| 1 | `awq_gemm_kernel` | 67.71% | 1.569 s | 32256 | 48.6 us |
| 2 | `gemvx` (cuBLAS FP16) | 13.72% | 317.9 ms | 128 | 2.484 ms |
| 3 | `flash_fwd_kernel` | 1.75% | 40.6 ms | 4284 | 9.5 us |
| 4 | elementwise (cat/mul/add/fill/copy) | ~12% | ~280 ms | ~160k | ~1-2 us |
| 5 | `reduce_kernel` (mean/RMSNorm) | 1.15% | 26.7 ms | 9344 | 2.9 us |

Total CUDA time: 2.317 s.

## Analysis

### awq_gemm_kernel — 67.7%
INT4 weight dequantization + GEMM fused kernel from autoawq. 32256 calls = 128 decode steps x 252 linear layers per forward pass (28 layers x 9 linear ops: q/k/v/o_proj + gate/up/down_proj + 2 more). Each call averages 48.6 us. This is the dominant cost by far and the obvious K1 candidate. Memory-bound at M=1 (decode GEMV). Need ncu measurement of achieved DRAM bandwidth vs 256 GB/s peak.

### gemvx (cuBLAS FP16 matmul) — 13.7%
128 calls at 2.484 ms each. Likely the `lm_head` projection (vocab_size=151936, hidden=2048) run each decode step, or vision encoder layers. These are FP16, not INT4. Large M=1 GEMV against a big weight matrix.

### flash_fwd_kernel — 1.75%
Flash Attention v2. 4284 calls across warmup residue + 128 decode steps. Small fraction — attention is not the bottleneck at short sequence lengths.

### Elementwise ops — ~12%
cat, mul, add, copy, fill, neg. High call count (~160k total), tiny per-call cost. Fusion opportunities exist (RMSNorm + residual add) but ROI is low vs the 68% AWQ GEMM target.

## Top 3 Kernel Candidates for K1

1. **awq_gemm_kernel** — 67.7%. Replace with custom INT4 GEMV for M=1 decode on sm_89. Target: approach 256 GB/s DRAM bandwidth. Current autoawq kernel is generic across GPU archs.
2. **gemvx (lm_head)** — 13.7%. FP16 GEMV for large vocab projection. Possible to fuse with top-k/sampling.
3. **RMSNorm + residual** — 1.15% (reduce) + part of add/mul. Fuse RMSNorm + residual add + optional activation into one kernel. Small but many calls (9344).

## Decision

Start K1 with `awq_gemm_kernel`. It is 68% of all CUDA time. A 20% improvement there = 13.6% end-to-end speedup. Memory-bound GEMV on sm_89 with known shapes from Qwen2.5-VL-3B weight tensors.

## Files

- `experiments/K0-baseline/kernel_report.txt` — full torch.profiler table
- `experiments/K0-baseline/trace.json` — Chrome trace (chrome://tracing)
- `experiments/K0-baseline/gpu_state.csv` — GPU clocks/temp/power at start
