# kernels.jl — BabelStream device code (KernelAbstractions). Math unchanged
# from the original babelstream.jl.
#
# The kernels use `unsafe_indices = true`, so there is no bounds check: the
# streaming kernels must be launched with ndrange a multiple of the workgroup
# size (validated in main.jl), and dot_kernel! must be launched with a
# workgroup of exactly 256 (its tree reduction starts at offset 128).

module Kernels

using KernelAbstractions

export init_kernel!, copy_kernel!, mul_kernel!, add_kernel!, triad_kernel!,
       dot_kernel!, DOT_BLOCK

"Pinned workgroup size used by every kernel."
const DOT_BLOCK = 256

@kernel unsafe_indices = true function init_kernel!(a, b, c, initA, initB, initC)
    I = @index(Global)
    @inbounds begin
        a[I] = initA
        b[I] = initB
        c[I] = initC
    end
end

@kernel unsafe_indices = true function copy_kernel!(c, @Const(a))
    I = @index(Global)
    @inbounds c[I] = a[I]
end

@kernel unsafe_indices = true function mul_kernel!(b, @Const(c), scalar)
    I = @index(Global)
    @inbounds b[I] = scalar * c[I]
end

@kernel unsafe_indices = true function add_kernel!(c, @Const(a), @Const(b))
    I = @index(Global)
    @inbounds c[I] = a[I] + b[I]
end

@kernel unsafe_indices = true function triad_kernel!(a, @Const(b), @Const(c), scalar)
    I = @index(Global)
    @inbounds a[I] = b[I] + scalar * c[I]
end

@kernel unsafe_indices = true function dot_kernel!(@Const(a), @Const(b), partial, array_size, num_blocks)
    T = eltype(partial)
    tb_sum = @localmem T (256,)

    local_i = @index(Local)
    group_i = @index(Group)

    @inbounds tb_sum[local_i] = zero(T)

    i = @index(Global)
    stride = 256 * num_blocks
    while i <= array_size
        @inbounds tb_sum[local_i] = tb_sum[local_i] + a[i] * b[i]
        i += stride
    end

    offset = 128
    while offset > 0
        @synchronize
        if local_i <= offset
            @inbounds tb_sum[local_i] += tb_sum[local_i+offset]
        end
        offset ÷= 2
    end

    if local_i == 1
        @inbounds partial[group_i] = tb_sum[1]
    end
end

end # module Kernels
