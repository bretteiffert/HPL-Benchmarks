# Selects the KernelAbstractions CUDA or ROCm backend from `STENCIL_GPU`, defaulting to CUDA. It provides the stencil benchmark with vendor labels, device names, synchronization, backend construction, and GPU-event elapsed time.

const STENCIL_GPU = lowercase(get(ENV, "STENCIL_GPU", "cuda"))

if STENCIL_GPU in ("cuda", "nvidia")
    @eval using CUDA

    @eval begin
        gpu_label() = "CUDA"
        gpu_name() = CUDA.name(CUDA.device())
        gpu_sync() = CUDA.synchronize()
        gpu_backend() = CUDABackend()

        gpu_elapsed(f::Function) = Float64(CUDA.@elapsed f())
    end

elseif STENCIL_GPU in ("rocm", "amdgpu", "hip", "amd")
    @eval using AMDGPU

    @eval begin
        gpu_label() = "HIP"
        gpu_name() = AMDGPU.HIP.name(AMDGPU.device())
        gpu_sync() = AMDGPU.synchronize()
        gpu_backend() = ROCBackend()

        gpu_elapsed(f::Function) = Float64(AMDGPU.@elapsed f())
    end

else
    error("STENCIL_GPU must be \"cuda\" or \"rocm\", got \"$STENCIL_GPU\"")
end
