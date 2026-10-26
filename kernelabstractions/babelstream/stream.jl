# stream.jl — BabelStream buffers, timed launches and the host-side gold check.
# No printing here; results go into the report.

module Stream

using KernelAbstractions
using ..Kernels
using ..Backend: elapsed_seconds, device_sync

export KAStream, start_values, warmup!, init_arrays!, ka_copy!, ka_mul!, ka_add!,
       ka_triad!, ka_dot!, read_arrays, gold_values, array_metrics, routine_bytes

const startA = 0.1
const startB = 0.2
const startC = 0.0
const startScalar = 0.4

start_values(::Type{T}) where {T} = (T(startA), T(startB), T(startC), T(startScalar))

# Concretely typed so the host launch path between the timing events is
# type-stable (abstract fields add dynamic dispatch inside the timed region).
struct KAStream{T,B<:KernelAbstractions.Backend,A<:AbstractVector{T}}
    array_size::Int
    block_size::Int
    dot_num_blocks::Int
    backend::B
    d_a::A
    d_b::A
    d_c::A
    d_sum::A
    sums::Vector{T}
end

function KAStream{T}(backend, array_size::Int, block_size::Int, dot_num_blocks::Int) where {T}
    d_a = KernelAbstractions.zeros(backend, T, array_size)
    d_b = KernelAbstractions.zeros(backend, T, array_size)
    d_c = KernelAbstractions.zeros(backend, T, array_size)
    d_sum = KernelAbstractions.zeros(backend, T, dot_num_blocks)
    return KAStream{T,typeof(backend),typeof(d_a)}(array_size, block_size, dot_num_blocks,
        backend, d_a, d_b, d_c, d_sum, Vector{T}(undef, dot_num_blocks))
end

"Compile every kernel (Julia JIT) on small buffers so compilation is not timed."
function warmup!(s::KAStream{T}) where {T}
    n = s.block_size * cld(Kernels.DOT_BLOCK, s.block_size)
    bk = s.backend
    w_a = KernelAbstractions.zeros(bk, T, n)
    w_b = KernelAbstractions.zeros(bk, T, n)
    w_c = KernelAbstractions.zeros(bk, T, n)
    w_sum = KernelAbstractions.zeros(bk, T, s.dot_num_blocks)
    A, B, C, S = start_values(T)
    init_kernel!(bk, s.block_size)(w_a, w_b, w_c, A, B, C; ndrange = n)
    copy_kernel!(bk, s.block_size)(w_c, w_a; ndrange = n)
    mul_kernel!(bk, s.block_size)(w_b, w_c, S; ndrange = n)
    add_kernel!(bk, s.block_size)(w_c, w_a, w_b; ndrange = n)
    triad_kernel!(bk, s.block_size)(w_a, w_b, w_c, S; ndrange = n)
    dot_kernel!(bk, Kernels.DOT_BLOCK)(w_a, w_b, w_sum, n, s.dot_num_blocks;
        ndrange = Kernels.DOT_BLOCK * s.dot_num_blocks)
    copyto!(s.sums, w_sum)
    device_sync()
    return nothing
end

# Each function returns the kernel time in seconds (device events).

function init_arrays!(s::KAStream{T}) where {T}
    A, B, C, _ = start_values(T)
    return elapsed_seconds() do
        init_kernel!(s.backend, s.block_size)(s.d_a, s.d_b, s.d_c, A, B, C;
            ndrange = s.array_size)
    end
end

function ka_copy!(s::KAStream)
    return elapsed_seconds() do
        copy_kernel!(s.backend, s.block_size)(s.d_c, s.d_a; ndrange = s.array_size)
    end
end

function ka_mul!(s::KAStream{T}) where {T}
    scalar = T(startScalar)
    return elapsed_seconds() do
        mul_kernel!(s.backend, s.block_size)(s.d_b, s.d_c, scalar; ndrange = s.array_size)
    end
end

function ka_add!(s::KAStream)
    return elapsed_seconds() do
        add_kernel!(s.backend, s.block_size)(s.d_c, s.d_a, s.d_b; ndrange = s.array_size)
    end
end

function ka_triad!(s::KAStream{T}) where {T}
    scalar = T(startScalar)
    return elapsed_seconds() do
        triad_kernel!(s.backend, s.block_size)(s.d_a, s.d_b, s.d_c, scalar;
            ndrange = s.array_size)
    end
end

"(time_s, dot). The time includes copying the partial sums to the host."
function ka_dot!(s::KAStream)
    t = elapsed_seconds() do
        dot_kernel!(s.backend, Kernels.DOT_BLOCK)(s.d_a, s.d_b, s.d_sum, s.array_size,
            s.dot_num_blocks; ndrange = Kernels.DOT_BLOCK * s.dot_num_blocks)
        copyto!(s.sums, s.d_sum)
    end
    # Host accumulation in Float64 so Float16 runs do not lose the reduction
    # solely because the host accumulator is Float16.
    total = 0.0
    for i in 1:s.dot_num_blocks
        total += Float64(s.sums[i])
    end
    return t, total
end

function read_arrays(s::KAStream{T}) where {T}
    a = Vector{T}(undef, s.array_size)
    b = Vector{T}(undef, s.array_size)
    c = Vector{T}(undef, s.array_size)
    copyto!(a, s.d_a)
    copyto!(b, s.d_b)
    copyto!(c, s.d_c)
    return a, b, c
end

"Bytes moved per routine (BabelStream convention)."
routine_bytes(::Type{T}, n::Int) where {T} = (
    Init = 3 * sizeof(T) * n, Copy = 2 * sizeof(T) * n, Mul = 2 * sizeof(T) * n,
    Add = 3 * sizeof(T) * n, Triad = 3 * sizeof(T) * n, Dot = 2 * sizeof(T) * n)

"Gold values after `ntimes` iterations, computed in T like the device."
function gold_values(::Type{T}, ntimes::Int) where {T}
    goldA, goldB, goldC, scalar = start_values(T)
    for _ in 1:ntimes
        goldC = goldA
        goldB = scalar * goldC
        goldC = goldA + goldB
        goldA = goldB + scalar * goldC
    end
    return goldA, goldB, goldC
end

function _kahan_sum(v::Vector)
    s = 0.0
    comp = 0.0
    for x in v
        y = Float64(x) - comp
        t = s + y
        comp = (t - s) - y
        s = t
    end
    return s
end

"(checksum_rel_err, max_elem_rel_err) against gold."
function array_metrics(v::Vector{T}, gold::T) where {T}
    gold_d = Float64(gold)
    actual_sum = _kahan_sum(v)
    expected_sum = gold_d * length(v)
    checksum_rel_err = expected_sum != 0.0 ?
        abs(actual_sum - expected_sum) / abs(expected_sum) : abs(actual_sum)
    max_rel_err = 0.0
    for x in v
        abs_err = abs(Float64(x) - gold_d)
        rel_err = gold_d != 0.0 ? abs_err / abs(gold_d) : abs_err
        max_rel_err = max(max_rel_err, rel_err)
    end
    return checksum_rel_err, max_rel_err
end

end # module Stream
