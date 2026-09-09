# Selects the KernelAbstractions CUDA or ROCm backend from `STENCIL_GPU`, defaulting to CUDA. It supplies miniBUDE with event timing, synchronization, backend and device information, zero-based device enumeration, and device selection.

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

if STENCIL_GPU in ("cuda", "nvidia")
    @eval begin
        function gpu_devices()
            out = Tuple{Int,String}[]
            for (i, d) in enumerate(CUDA.devices())
                cap = CUDA.capability(d)
                mem = CUDA.totalmem(d) ÷ 1024 ÷ 1024
                push!(out, (i - 1, string(CUDA.name(d), " (", mem, "MB;",
                                          "sm_", cap.major, cap.minor, ")")))
            end
            return out
        end

        gpu_select_device!(i::Integer) = CUDA.device!(collect(CUDA.devices())[i+1])
    end
else
    @eval begin
        function gpu_devices()
            out = Tuple{Int,String}[]
            for (i, d) in enumerate(AMDGPU.devices())
                name = AMDGPU.HIP.name(d)
                gcn = try
                    string(AMDGPU.HIP.gcn_arch(d))
                catch
                    "unknown-arch"
                end
                mem = try
                    AMDGPU.HIP.total_memory(d) ÷ 1024 ÷ 1024
                catch
                    0
                end
                push!(out, (i - 1, string(name, " (", mem, "MB;", gcn, ")")))
            end
            return out
        end

        gpu_select_device!(i::Integer) = AMDGPU.device!(collect(AMDGPU.devices())[i+1])
    end
end
