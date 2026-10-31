"""Extract AWQ linear layer shapes from Qwen2.5-VL-3B-Instruct-AWQ."""
import torch
import transformers.activations as _act
if not hasattr(_act, 'PytorchGELUTanh'):
    _act.PytorchGELUTanh = _act.GELUActivation
from transformers import Qwen2_5_VLForConditionalGeneration

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"

def main():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, device_map="cuda", dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True
    )

    print(f"{'Layer':<60} {'qweight':>20} {'scales':>20} {'qzeros':>20}")
    print("-" * 120)

    shapes = {}
    for name, mod in model.named_modules():
        if hasattr(mod, 'qweight'):
            qw = mod.qweight.shape
            sc = mod.scales.shape
            qz = mod.qzeros.shape
            print(f"{name:<60} {str(qw):>20} {str(sc):>20} {str(qz):>20}")
            key = (qw, sc, qz)
            shapes.setdefault(key, []).append(name)

    print(f"\n{'='*60}")
    print("Unique shapes (contract candidates):")
    for (qw, sc, qz), layers in sorted(shapes.items(), key=lambda x: -len(x[1])):
        K = qw[0]
        N = sc[1]
        group_size = K // sc[0] if sc[0] > 0 else 0
        print(f"  qweight={qw}, scales={sc} => K={K}, N={N}, group_size={group_size}, count={len(layers)}")

if __name__ == "__main__":
    main()
