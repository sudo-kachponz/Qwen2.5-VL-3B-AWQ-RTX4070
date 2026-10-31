"""K1: correctness + benchmark of custom INT4 GEMV vs autoawq reference."""
import sys, os, time, statistics
import torch

# bypass CUDA version check (system nvcc 13.2 vs PyTorch cu121)
import torch.utils.cpp_extension as _cext
_cext._check_cuda_version = lambda *a, **k: None

import transformers.activations as _act
if not hasattr(_act, 'PytorchGELUTanh'):
    _act.PytorchGELUTanh = _act.GELUActivation

KERNEL_SRC = os.path.join(os.path.dirname(__file__),
                          "..", "kernels", "sm89", "int4_gemv", "int4_gemv.cu")

print("JIT compiling INT4 GEMV kernel for sm_89...")
int4_gemv_sm89 = torch.utils.cpp_extension.load(
    name="int4_gemv_sm89",
    sources=[KERNEL_SRC],
    extra_cuda_cflags=["-arch=sm_89", "-O3", "--use_fast_math", "-lineinfo"],
    verbose=False,
)

from awq.modules.linear.gemm import WQLinear_GEMM
from transformers import Qwen2_5_VLForConditionalGeneration

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
WARMUP = 10
ITERS = 50
ATOL = 0.05  # absolute tolerance for INT4 dequant comparison

def get_awq_layers(model):
    """Find all WQLinear_GEMM layers and return (name, module) pairs."""
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, WQLinear_GEMM):
            layers.append((name, mod))
    return layers

def reference_forward(layer, x):
    """Run autoawq's forward pass as reference."""
    with torch.no_grad():
        return layer(x)

def custom_forward(layer, x):
    """Run our custom kernel using the layer's AWQ weights."""
    qw = layer.qweight
    sc = layer.scales
    qz = layer.qzeros
    gs = layer.group_size
    with torch.no_grad():
        out = int4_gemv_sm89.int4_gemv_awq(x, qw, sc, qz, gs)
        if layer.bias is not None:
            out = out + layer.bias
        return out

def benchmark_fn(fn, warmup=WARMUP, iters=ITERS):
    """Benchmark with CUDA events. Returns (median_us, p95_us)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # us

    times.sort()
    median = statistics.median(times)
    p95 = times[int(len(times) * 0.95)]
    return median, p95

def main():
    print(f"Loading {MODEL} (CPU)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, device_map="cuda", dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    model.eval()

    layers = get_awq_layers(model)
    print(f"Found {len(layers)} AWQ linear layers\n")

    # Test unique shapes only
    tested_shapes = set()
    results = []

    for name, layer in layers:
        shape_key = (layer.qweight.shape, layer.scales.shape)
        if shape_key in tested_shapes:
            continue
        tested_shapes.add(shape_key)

        K = layer.qweight.shape[0]
        N = layer.scales.shape[1]
        x = torch.randn(1, K, dtype=torch.float16, device="cuda")

        # correctness
        ref = reference_forward(layer, x)
        ours = custom_forward(layer, x)

        diff = (ref - ours).abs()
        max_err = diff.max().item()

        ok = max_err < ATOL
        status = "PASS" if ok else "FAIL"

        # benchmark
        ref_med, ref_p95 = benchmark_fn(lambda: reference_forward(layer, x))
        our_med, our_p95 = benchmark_fn(lambda: custom_forward(layer, x))

        speedup = ref_med / our_med if our_med > 0 else 0

        weight_bytes = K * N // 2
        bw_ref = weight_bytes / (ref_med / 1e6) / 1e9
        bw_ours = weight_bytes / (our_med / 1e6) / 1e9

        print(f"── {name} (K={K}, N={N}) ──")
        print(f"  Correctness: {status} (max_err={max_err:.4f})")
        print(f"  autoawq:  median={ref_med:.1f}us  p95={ref_p95:.1f}us  BW={bw_ref:.1f} GB/s")
        print(f"  ours:     median={our_med:.1f}us  p95={our_p95:.1f}us  BW={bw_ours:.1f} GB/s  speedup={speedup:.2f}x")
        print()

        results.append({
            "name": name, "K": K, "N": N,
            "ref_us": ref_med, "ours_us": our_med,
            "speedup": speedup, "bw_ref": bw_ref, "bw_ours": bw_ours, "ok": ok,
        })

    # summary
    print("=" * 70)
    print("Summary:")
    all_pass = all(r["ok"] for r in results)
    print(f"  Correctness: {'ALL PASS' if all_pass else 'FAIL'}")
    for r in results:
        print(f"  K={r['K']:>5}, N={r['N']:>5}: ref={r['ref_us']:>7.1f}us  ours={r['ours_us']:>7.1f}us ({r['speedup']:.2f}x)  BW={r['bw_ours']:.1f}/{r['bw_ref']:.1f} GB/s")

if __name__ == "__main__":
    main()
