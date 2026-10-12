# Selects the vendor runtime and KernelAbstractions backend for this invocation.
# User-facing values are cuda and hip; AMDGPU remains the ROCm runtime.

function _backend_from_args(argv::Vector{String})
    selected = :cuda
    i = 1
    while i <= length(argv)
        if argv[i] == "--backend"
            i += 1
            i <= length(argv) || error("--backend requires cuda or hip")
            selected = Symbol(lowercase(argv[i]))
        elseif startswith(argv[i], "--backend=")
            selected = Symbol(lowercase(split(argv[i], "=", limit=2)[2]))
        end
        i += 1
    end
    selected in (:cuda, :hip) || error("Use --backend cuda or hip")
    return selected
end

const SELECTED_BACKEND = _backend_from_args(ARGS)

if SELECTED_BACKEND === :cuda
    @eval using CUDA

    @eval begin
        gpu_label() = "CUDA"
        gpu_name() = CUDA.name(CUDA.device())
        gpu_sync() = CUDA.synchronize()
        gpu_backend() = CUDABackend()
        gpu_elapsed(f::Function) = Float64(CUDA.@elapsed f())
    end
else
    @eval using AMDGPU

    @eval begin
        gpu_label() = "HIP"
        gpu_name() = AMDGPU.HIP.name(AMDGPU.device())
        gpu_sync() = AMDGPU.synchronize()
        gpu_backend() = ROCBackend()
        gpu_elapsed(f::Function) = Float64(AMDGPU.@elapsed f())
    end
end
