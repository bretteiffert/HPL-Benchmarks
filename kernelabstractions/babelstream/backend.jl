# Backend selection and vendor-specific GPU operations for BabelStream.
#
# The backend is selected per invocation with:
#   --backend cuda
#   --backend hip
#
# The benchmark deliberately uses the default device visible to the selected
# vendor runtime. Device selection is left to the runtime or job scheduler.

function _parse_backend_value(value::AbstractString)
    value = lowercase(value)
    value in ("cuda", "hip") ||
        error("Invalid backend '$value'. Use --backend cuda or --backend hip.")
    return Symbol(value)
end

function _backend_from_args(argv::Vector{String})
    selected = nothing
    i = 1

    while i <= length(argv)
        arg = argv[i]

        if arg == "--backend"
            i += 1
            i <= length(argv) || error("--backend requires cuda or hip")
            selected = _parse_backend_value(argv[i])
        elseif startswith(arg, "--backend=")
            selected = _parse_backend_value(split(arg, "=", limit=2)[2])
        end

        i += 1
    end

    # Preserve the original CUDA default so help and ordinary runs remain
    # usable without an explicit backend, while commands can choose per run.
    return selected === nothing ? :cuda : selected
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

        gpu_cu_count() =
            Int(CUDA.attribute(CUDA.device(),
                               CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))

        gpu_driver() = string(CUDA.driver_version())
    end

elseif SELECTED_BACKEND === :hip
    @eval using AMDGPU

    @eval begin
        gpu_label() = "ROCm"
        gpu_name() = AMDGPU.HIP.name(AMDGPU.device())
        gpu_sync() = AMDGPU.synchronize()
        gpu_backend() = ROCBackend()

        gpu_elapsed(f::Function) = Float64(AMDGPU.@elapsed f())

        gpu_cu_count() =
            Int(AMDGPU.HIP.properties(AMDGPU.device()).multiProcessorCount)

        gpu_driver() = string(AMDGPU.HIP.runtime_version())
    end
end
