"""K0 baseline profile: Qwen2.5-VL-3B-AWQ inference with torch.profiler."""
import torch
from torch.profiler import profile, ProfilerActivity
import transformers.activations as _act
if not hasattr(_act, 'PytorchGELUTanh'):
    _act.PytorchGELUTanh = _act.GELUActivation
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
OUT_DIR = "experiments/K0-baseline"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {MODEL}...")
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, device_map="cuda", dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    model.eval()

    prompt = "Describe the contents of this document in detail."
    inputs = processor(text=prompt, return_tensors="pt").to("cuda")

    # warmup
    with torch.no_grad():
        for _ in range(3):
            model.generate(**inputs, max_new_tokens=32)
    torch.cuda.synchronize()

    # profiled run
    print("Running profiled generation...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128)
        torch.cuda.synchronize()

    print(f"Output tokens: {out.shape[1]}")

    # top CUDA kernels by total time
    report = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    print("\n=== Top 30 CUDA kernels (by total time) ===")
    print(report)

    # save report
    report_path = os.path.join(OUT_DIR, "kernel_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # save chrome trace
    trace_path = os.path.join(OUT_DIR, "trace.json")
    prof.export_chrome_trace(trace_path)

    print(f"\nSaved: {report_path}")
    print(f"Saved: {trace_path} (open in chrome://tracing)")

if __name__ == "__main__":
    main()
