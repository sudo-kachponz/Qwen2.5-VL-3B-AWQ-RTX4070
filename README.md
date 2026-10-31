# Kernel Lab sm_89 — RTX 4070 Laptop

Custom CUDA kernel optimization lab targeting **one GPU, one model**:
- **GPU**: RTX 4070 Laptop (Ada Lovelace, sm_89, 8 GB GDDR6, 80 W, 256 GB/s DRAM)
- **Model**: Qwen2.5-VL-3B-Instruct-AWQ (INT4 quantized VLM)
- **Purpose**: Optimize inference latency for document extraction workload

> Results from this lab apply ONLY to this specific GPU + model combination.

## Status

| Phase | Status | Summary |
|-------|--------|---------|
| K0 | Done | Baseline profile: `awq_gemm_kernel` = 67.7% of CUDA time |
| K1 | In Progress | Custom INT4 GEMV for sm_89. Correct, 1.34x on large N, slower on small N |
| K2 | — | Integration + end-to-end measurement |
| K3 | — | Second kernel (if K2 provides clear lessons) |

## K0 Baseline Profile

Top CUDA kernels (128-token text-only generation):

| Kernel | CUDA % | Time |
|--------|--------|------|
| `awq_gemm_kernel` (INT4 dequant+GEMM) | 67.7% | 1.569 s |
| `gemvx` (cuBLAS FP16 matmul) | 13.7% | 318 ms |
| `flash_fwd_kernel` (attention) | 1.75% | 40.6 ms |

## K1 Custom INT4 GEMV

Correctness: ALL PASS (max_err < 0.008 vs autoawq reference).

| Shape (K, N) | Layers | autoawq | ours | Speedup |
|-------------|--------|---------|------|---------|
| (2048, 2048) | 72 | 84 us | 296 us | 0.28x |
| (2048, 256) | 72 | 84 us | 297 us | 0.28x |
| (2048, 11008) | 72 | 393 us | 294 us | **1.34x** |
| (11008, 2048) | 36 | 459 us | 1259 us | 0.36x |

Issue: low SM utilization for small N (not enough thread blocks). Needs K-dimension parallelism within thread blocks.

## Structure

```
kernels/sm89/int4_gemv/   Custom CUDA kernel + build
harness/                  Profile, correctness, benchmark scripts
experiments/K0-baseline/  Baseline nsys + torch.profiler outputs
docs/notes/               Per-decision dated notes
```

## Requirements

- NVIDIA RTX 4070 Laptop (sm_89)
- Conda env with: PyTorch 2.5.1+cu121, autoawq 0.2.9, transformers 4.49.x
- Model: `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` (downloaded via HuggingFace)
