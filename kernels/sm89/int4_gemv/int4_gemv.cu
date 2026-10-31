/*
 * INT4 AWQ GEMV kernel optimized for sm_89 (Ada Lovelace)
 *
 * Contract:
 *   - M=1 (single-token decode GEMV)
 *   - INT4 AWQ format: qweight packed uint32 (8 int4 per word), group_size=128
 *   - FP16 input/output, FP32 internal accumulation
 *   - Target: RTX 4070 Laptop (sm_89, 256 GB/s DRAM, 46 SMs)
 *
 * Memory access pattern:
 *   - Each thread owns 8 output positions (one packed uint32 column)
 *   - Iterates over K rows, reading one uint32 per row (coalesced across threads)
 *   - Scales/zeros loaded once per group (every 128 K steps)
 *   - No cross-thread reduction needed (each thread writes its own outputs)
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define PACK_FACTOR 8

// AWQ packs int4 values in interleaved order [0,2,4,6,1,3,5,7].
// To unpack: output position j within a group of 8 lives at bit position REVERSE[j].
__device__ __constant__ int AWQ_REVERSE[8] = {0, 4, 1, 5, 2, 6, 3, 7};

/* ── kernel: no K-split, each thread owns 8 outputs for all K ── */

__global__ void int4_gemv_awq_k(
    const half*     __restrict__ x,       // [K]
    const uint32_t* __restrict__ qw,      // [K, N_PACK]  row-major
    const half*     __restrict__ scales,   // [G, N]       row-major
    const uint32_t* __restrict__ qzeros,   // [G, N_PACK]  row-major
    half*           __restrict__ out,      // [N]
    const int K,
    const int N,
    const int N_PACK,
    const int group_size)
{
    const int pack_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (pack_col >= N_PACK) return;

    const int out_base = pack_col * PACK_FACTOR;
    float acc[PACK_FACTOR] = {0.f};

    const int num_groups = K / group_size;

    for (int g = 0; g < num_groups; g++) {
        /* load scale (8 fp16) and zero (1 packed uint32) once per group */
        float   sv[PACK_FACTOR];
        int     zv[PACK_FACTOR];
        uint32_t pz = qzeros[g * N_PACK + pack_col];

        #pragma unroll
        for (int j = 0; j < PACK_FACTOR; j++) {
            sv[j] = __half2float(scales[g * N + out_base + j]);
            zv[j] = (pz >> (AWQ_REVERSE[j] * 4)) & 0xF;
        }

        const int k0 = g * group_size;
        const int k1 = k0 + group_size;

        for (int k = k0; k < k1; k++) {
            float xv = __half2float(x[k]);
            uint32_t pw = qw[k * N_PACK + pack_col];   // coalesced

            #pragma unroll
            for (int j = 0; j < PACK_FACTOR; j++) {
                int wi = (pw >> (AWQ_REVERSE[j] * 4)) & 0xF;
                acc[j] += xv * sv[j] * (float)(wi - zv[j]);
            }
        }
    }

    /* handle remainder if K % group_size != 0 */
    int rem_start = num_groups * group_size;
    if (rem_start < K) {
        int g = num_groups;
        float   sv[PACK_FACTOR];
        int     zv[PACK_FACTOR];
        uint32_t pz = qzeros[g * N_PACK + pack_col];
        #pragma unroll
        for (int j = 0; j < PACK_FACTOR; j++) {
            sv[j] = __half2float(scales[g * N + out_base + j]);
            zv[j] = (pz >> (AWQ_REVERSE[j] * 4)) & 0xF;
        }
        for (int k = rem_start; k < K; k++) {
            float xv = __half2float(x[k]);
            uint32_t pw = qw[k * N_PACK + pack_col];
            #pragma unroll
            for (int j = 0; j < PACK_FACTOR; j++) {
                int wi = (pw >> (AWQ_REVERSE[j] * 4)) & 0xF;
                acc[j] += xv * sv[j] * (float)(wi - zv[j]);
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < PACK_FACTOR; j++) {
        out[out_base + j] = __float2half(acc[j]);
    }
}


/* ── kernel: K-split with atomicAdd for high SM occupancy ── */

__global__ void int4_gemv_awq_ksplit(
    const half*     __restrict__ x,
    const uint32_t* __restrict__ qw,
    const half*     __restrict__ scales,
    const uint32_t* __restrict__ qzeros,
    float*          __restrict__ out_fp32,   // [N] fp32 accumulator
    const int K,
    const int N,
    const int N_PACK,
    const int group_size,
    const int k_start,
    const int k_end)
{
    const int pack_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (pack_col >= N_PACK) return;

    const int out_base = pack_col * PACK_FACTOR;
    float acc[PACK_FACTOR] = {0.f};

    int g_first = k_start / group_size;
    int g_last  = (k_end - 1) / group_size;

    for (int g = g_first; g <= g_last; g++) {
        float   sv[PACK_FACTOR];
        int     zv[PACK_FACTOR];
        uint32_t pz = qzeros[g * N_PACK + pack_col];

        #pragma unroll
        for (int j = 0; j < PACK_FACTOR; j++) {
            sv[j] = __half2float(scales[g * N + out_base + j]);
            zv[j] = (pz >> (AWQ_REVERSE[j] * 4)) & 0xF;
        }

        int ks = max(k_start, g * group_size);
        int ke = min(k_end,  (g + 1) * group_size);

        for (int k = ks; k < ke; k++) {
            float xv = __half2float(x[k]);
            uint32_t pw = qw[k * N_PACK + pack_col];

            #pragma unroll
            for (int j = 0; j < PACK_FACTOR; j++) {
                int wi = (pw >> (AWQ_REVERSE[j] * 4)) & 0xF;
                acc[j] += xv * sv[j] * (float)(wi - zv[j]);
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < PACK_FACTOR; j++) {
        atomicAdd(&out_fp32[out_base + j], acc[j]);
    }
}


/* ── host entry points ── */

torch::Tensor int4_gemv_awq(
    torch::Tensor input,     // [1, K] fp16
    torch::Tensor qweight,   // [K, N/8] int32 (uint32 packed)
    torch::Tensor scales,    // [G, N] fp16
    torch::Tensor qzeros,    // [G, N/8] int32 (uint32 packed)
    int group_size)
{
    const int K      = input.size(1);
    const int N_PACK = qweight.size(1);
    const int N      = N_PACK * PACK_FACTOR;

    auto output = torch::empty({1, N}, input.options());

    const int threads = 128;
    const int blocks  = (N_PACK + threads - 1) / threads;

    int4_gemv_awq_k<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(qzeros.data_ptr<int32_t>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        K, N, N_PACK, group_size);

    return output;
}

torch::Tensor int4_gemv_awq_split(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor qzeros,
    int group_size,
    int k_splits)
{
    const int K      = input.size(1);
    const int N_PACK = qweight.size(1);
    const int N      = N_PACK * PACK_FACTOR;

    auto out_fp32 = torch::zeros({N}, torch::dtype(torch::kFloat32).device(input.device()));

    const int threads     = 128;
    const int n_blocks    = (N_PACK + threads - 1) / threads;
    const int k_per_split = (K + k_splits - 1) / k_splits;

    for (int s = 0; s < k_splits; s++) {
        int ks = s * k_per_split;
        int ke = std::min(K, ks + k_per_split);

        int4_gemv_awq_ksplit<<<n_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>()),
            reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<const uint32_t*>(qzeros.data_ptr<int32_t>()),
            out_fp32.data_ptr<float>(),
            K, N, N_PACK, group_size, ks, ke);
    }

    return out_fp32.to(torch::kFloat16).unsqueeze(0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int4_gemv_awq",       &int4_gemv_awq,       "INT4 AWQ GEMV (no split)");
    m.def("int4_gemv_awq_split", &int4_gemv_awq_split,  "INT4 AWQ GEMV (K-split + atomics)");
}
