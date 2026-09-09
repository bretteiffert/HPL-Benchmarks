# Provides Float32 and Float64 error-function wrappers for host and GPU Hartree-Fock calculations. It relies on the selected CUDA or AMDGPU package to supply the SpecialFunctions device override, and verifies that path with a one-thread startup kernel.

import SpecialFunctions

@inline gpu_erf(x::Float64) = SpecialFunctions.erf(x)
@inline gpu_erf(x::Float32) = SpecialFunctions.erf(x)

@inline host_erf(x::Float64) = SpecialFunctions.erf(x)
@inline host_erf(x::Float32) = Float32(SpecialFunctions.erf(Float64(x)))

function _erf_selftest_kernel!(out)
    @inbounds out[1] = SpecialFunctions.erf(3.0)
    return nothing
end

if STENCIL_GPU in ("cuda", "nvidia")
    @eval function _run_erf_selftest()
        d = CUDA.CuArray{Float64}(undef, 1)
        CUDA.@cuda threads = 1 _erf_selftest_kernel!(d)
        CUDA.synchronize()
        return Array(d)[1]
    end
else
    @eval function _run_erf_selftest()
        d = AMDGPU.ROCArray{Float64}(undef, 1)
        AMDGPU.@roc groupsize = 1 gridsize = 1 _erf_selftest_kernel!(d)
        AMDGPU.synchronize()
        return Array(d)[1]
    end
end

function _verify_gpu_erf()
    expected = SpecialFunctions.erf(3.0)
    got = try
        _run_erf_selftest()
    catch err
        error("""
        Device-side SpecialFunctions.erf() failed to compile or run for \
        STENCIL_GPU=$(STENCIL_GPU).

        This should work automatically once both `using CUDA` (or `using \
        AMDGPU`) and `import SpecialFunctions` are loaded, via the vendor \
        package's SpecialFunctions device-override extension. If it doesn't:
          - confirm SpecialFunctions.jl is installed in this project
            (`Pkg.status()`), and its version supports the extension
          - confirm your CUDA.jl / AMDGPU.jl version is recent enough to ship
            a SpecialFunctions device override (older releases may not)
        Do NOT substitute a hand-rolled erf approximation as a workaround: the
        Boys function is evaluated ngauss^4 times per shell quadruplet, and an
        approximate erf would make the correctness metrics incomparable with
        the other ports.

        Original error:
        $(err)
        """)
    end
    if !isapprox(got, expected; rtol = 1e-12)
        error("""
        Device erf() self-test mismatch for STENCIL_GPU=$(STENCIL_GPU): \
        got $(got), expected $(expected). The device override compiled and \
        ran but returned a wrong value -- do not trust this build's \
        correctness metrics until this is resolved.
        """)
    end
    return nothing
end

_verify_gpu_erf()
