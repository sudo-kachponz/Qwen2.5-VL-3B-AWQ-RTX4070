"""Shape guard + monkey-patch to replace autoawq's GEMM with our kernel."""
import torch
import torch.utils.cpp_extension as _cext
import os

_cext._check_cuda_version = lambda *a, **k: None

KERNEL_SRC = os.path.join(os.path.dirname(__file__),
                          "..", "kernels", "sm89", "int4_gemv", "int4_gemv.cu")

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = torch.utils.cpp_extension.load(
            name="int4_gemv_sm89",
            sources=[KERNEL_SRC],
            extra_cuda_cflags=["-arch=sm_89", "-O3", "--use_fast_math"],
            verbose=False,
        )
    return _ext


CONTRACTED_SHAPES = {
    (2048, 256, 128),   # k_proj, v_proj
    (2048, 1376, 128),  # gate_proj, up_proj  (N_PACK=1376, N=11008)
    (11008, 256, 128),  # down_proj           (N_PACK=256, N=2048)
    (2048, 256, 128),   # q_proj, o_proj      (N_PACK=256, N=2048) — same as k/v
}


def _matches_contract(layer):
    """Check if layer's shapes are in the kernel contract."""
    qw = layer.qweight
    K, N_PACK = qw.shape
    gs = layer.group_size
    return (K, N_PACK, gs) in CONTRACTED_SHAPES


def _patched_forward(self, x):
    """Replacement forward using our kernel, with autoawq fallback."""
    ext = _get_ext()
    out_shape = x.shape[:-1] + (self.out_features,)
    x = x.to(torch.float16)

    if x.dim() >= 2 and x.shape[-2] == 1 and _matches_contract(self):
        x_2d = x.reshape(1, -1)
        out = ext.int4_gemv_awq(x_2d, self.qweight, self.scales, self.qzeros, self.group_size)
    else:
        # fallback to autoawq for batched / non-contracted shapes
        from awq.modules.linear.gemm import WQLinearMMFunction
        out = WQLinearMMFunction.apply(
            x, self.qweight, self.qzeros, self.scales,
            self.w_bit, self.group_size, self.bias, self.out_features
        )
        return out.reshape(out_shape)

    if self.bias is not None:
        out = out + self.bias
    return out.reshape(out_shape)


def patch_model(model):
    """Monkey-patch all WQLinear_GEMM layers to use our kernel for M=1."""
    from awq.modules.linear.gemm import WQLinear_GEMM
    import types

    count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, WQLinear_GEMM) and _matches_contract(mod):
            mod.forward = types.MethodType(_patched_forward, mod)
            count += 1
    print(f"Patched {count} AWQ layers with sm_89 kernel")
    return model
