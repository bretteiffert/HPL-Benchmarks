# Selects the KernelAbstractions CUDA or ROCm backend from `STENCIL_GPU`, defaulting to CUDA. It exposes device discovery, selection, synchronization, event timing, driver information, and compute-unit counts for the BabelStream harness.

const STENCIL_GPU = lowercase(get(ENV, "STENCIL_GPU", "cuda"))

if STENCIL_GPU in ("cuda", "nvidia")
    @eval using CUDA

    @eval begin
        gpu_label() = "CUDA"
        gpu_name() = CUDA.name(CUDA.device())
        gpu_sync() = CUDA.synchronize()
        gpu_backend() = CUDABackend()

        gpu_elapsed(f::Function) = Float64(CUDA.@elapsed f())

        function gpu_set_device!(index::Integer)
            devs = collect(CUDA.devices())
            0 <= index < length(devs) || error("Invalid device index")
            CUDA.device!(devs[index+1])
            return nothing
        end

        gpu_cu_count() =
            Int(CUDA.attribute(CUDA.device(),
                               CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))

        gpu_driver() = string(CUDA.driver_version())

        gpu_device_names() = [CUDA.name(d) for d in CUDA.devices()]
    end

elseif STENCIL_GPU in ("rocm", "amdgpu", "hip", "amd")
    @eval using AMDGPU

    @eval begin
        gpu_label() = "HIP"
        gpu_name() = AMDGPU.HIP.name(AMDGPU.device())
        gpu_sync() = AMDGPU.synchronize()
        gpu_backend() = ROCBackend()

        gpu_elapsed(f::Function) = Float64(AMDGPU.@elapsed f())

        function gpu_set_device!(index::Integer)
            devs = collect(AMDGPU.devices())
            0 <= index < length(devs) || error("Invalid device index")
            AMDGPU.device!(devs[index+1])
            return nothing
        end

        gpu_cu_count() =
            Int(AMDGPU.HIP.properties(AMDGPU.device()).multiProcessorCount)

        gpu_driver() = string(AMDGPU.HIP.runtime_version())

        gpu_device_names() = [AMDGPU.HIP.name(d) for d in AMDGPU.devices()]
    end

else
    error("STENCIL_GPU must be \"cuda\" or \"rocm\", got \"$STENCIL_GPU\"")
end
