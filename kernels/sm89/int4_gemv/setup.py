from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="int4_gemv_sm89",
    ext_modules=[
        CUDAExtension(
            "int4_gemv_sm89",
            ["int4_gemv.cu"],
            extra_compile_args={
                "nvcc": [
                    "-arch=sm_89",
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
