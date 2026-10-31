"""Debug: verify AWQ dequant formula against autoawq reference."""
import torch
import transformers.activations as _act
if not hasattr(_act, 'PytorchGELUTanh'):
    _act.PytorchGELUTanh = _act.GELUActivation
from transformers import Qwen2_5_VLForConditionalGeneration
from awq.modules.linear.gemm import WQLinear_GEMM

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, device_map="cuda", dtype=torch.float16,
    low_cpu_mem_usage=True, trust_remote_code=True
)
model.eval()

layer = None
for name, mod in model.named_modules():
    if isinstance(mod, WQLinear_GEMM):
        layer = mod
        print(f"Using layer: {name}")
        break

qw = layer.qweight   # [K, N/8] int32
sc = layer.scales     # [G, N] fp16
qz = layer.qzeros    # [G, N/8] int32
gs = layer.group_size
K = qw.shape[0]
N_PACK = qw.shape[1]
N = N_PACK * 8

print(f"K={K}, N={N}, group_size={gs}")
print(f"qweight dtype={qw.dtype}, scales dtype={sc.dtype}, qzeros dtype={qz.dtype}")

# Get reference output via one-hot input
# x = [0...0, 1, 0...0] at position k → output = W_dequant[k, :]
k_test = 0
x = torch.zeros(1, K, dtype=torch.float16, device="cuda")
x[0, k_test] = 1.0

with torch.no_grad():
    ref_out = layer(x)  # [1, N]

print(f"\nOne-hot at k={k_test}, autoawq output (first 16 values):")
print(ref_out[0, :16].tolist())

# Manual dequant for k=0, n=0..15
g = k_test // gs
print(f"\nManual dequant for k={k_test}, group={g}:")

formulas = {}

for n in range(16):
    pack_idx = n // 8
    bit_idx = n % 8

    pw = qw[k_test, pack_idx].item()
    pz = qz[g, pack_idx].item()
    s = sc[g, n].float().item()

    # Extract int4 values (handle signed int32)
    w_int = (pw >> (bit_idx * 4)) & 0xF
    z_int = (pz >> (bit_idx * 4)) & 0xF

    # Try different dequant formulas
    f1 = s * (w_int - z_int)           # standard: scale * (q - zero)
    f2 = s * (w_int - z_int + 1)       # shifted zero
    f3 = s * w_int - s * z_int         # same as f1 but split
    f4 = s * (w_int - (z_int + 1))     # zero stored as z+1

    ref = ref_out[0, n].item()

    if n < 8:
        print(f"  n={n}: w_int={w_int}, z_int={z_int}, scale={s:.6f}")
        print(f"    ref={ref:.6f}  f1(s*(q-z))={f1:.6f}  f2(s*(q-z+1))={f2:.6f}  f4(s*(q-z-1))={f4:.6f}")

    for name, val in [("s*(q-z)", f1), ("s*(q-z+1)", f2), ("s*q-s*z", f3), ("s*(q-z-1)", f4)]:
        formulas.setdefault(name, []).append(abs(val - ref))

print(f"\nFormula errors (max over first 16 outputs):")
for name, errs in formulas.items():
    print(f"  {name}: max_err={max(errs):.6f}, mean_err={sum(errs)/len(errs):.6f}")

# Also check: maybe the packing order is reversed?
print(f"\nRaw qweight[0,0] = 0x{qw[0,0].item() & 0xFFFFFFFF:08x}")
print(f"Raw qzeros[0,0]  = 0x{qz[0,0].item() & 0xFFFFFFFF:08x}")
print(f"scales[0,0:8] = {sc[0,:8].tolist()}")
