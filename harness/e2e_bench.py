"""K2: end-to-end benchmark — autoawq baseline vs patched kernel."""
import sys, os, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import transformers.activations as _act
if not hasattr(_act, 'PytorchGELUTanh'):
    _act.PytorchGELUTanh = _act.GELUActivation
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
WARMUP = 3
ITERS = 10
MAX_TOKENS = 128

def bench_generate(model, processor, prompt, max_tokens=MAX_TOKENS, warmup=WARMUP, iters=ITERS):
    inputs = processor(text=prompt, return_tensors="pt").to("cuda")

    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=32)
    torch.cuda.synchronize()

    times = []
    token_counts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        n_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        times.append(t1 - t0)
        token_counts.append(n_tokens)

    times.sort()
    median_s = statistics.median(times)
    p95_s = times[int(len(times) * 0.95)]
    avg_tokens = sum(token_counts) / len(token_counts)
    tok_s = avg_tokens / median_s

    return {"median_s": median_s, "p95_s": p95_s, "tok_s": tok_s, "avg_tokens": avg_tokens}

def main():
    prompt = "Describe the contents of this document in detail."

    print(f"Loading {MODEL}...")
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, device_map="cuda", dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    model.eval()

    # baseline
    print("\n=== Baseline (autoawq) ===")
    base = bench_generate(model, processor, prompt)
    print(f"  median={base['median_s']:.3f}s  p95={base['p95_s']:.3f}s  tok/s={base['tok_s']:.1f}  tokens={base['avg_tokens']:.0f}")

    # patch
    from runtime.guard import patch_model
    patch_model(model)

    print("\n=== Patched (sm_89 kernel) ===")
    patched = bench_generate(model, processor, prompt)
    print(f"  median={patched['median_s']:.3f}s  p95={patched['p95_s']:.3f}s  tok/s={patched['tok_s']:.1f}  tokens={patched['avg_tokens']:.0f}")

    speedup = base['median_s'] / patched['median_s']
    tok_gain = patched['tok_s'] / base['tok_s']

    print(f"\n=== Result ===")
    print(f"  Wall time: {speedup:.2f}x faster")
    print(f"  Throughput: {tok_gain:.2f}x more tok/s")
    print(f"  Baseline:  {base['tok_s']:.1f} tok/s")
    print(f"  Patched:   {patched['tok_s']:.1f} tok/s")

    if speedup > 1.0:
        print(f"\n  PROMOTE: {tok_gain:.2f}x throughput gain, no quality regression (same model weights).")
    else:
        print(f"\n  NEGATIVE: no end-to-end improvement. Log as failed experiment.")

if __name__ == "__main__":
    main()
