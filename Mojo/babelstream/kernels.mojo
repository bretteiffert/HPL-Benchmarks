# BabelStream device kernels (Mojo 1.2.0 / MAX 26.7).
#
# Kernel bodies are unchanged from the original babelstream.mojo. The only
# changes are forced by run-time precision and block-size selection:
#   * every kernel takes `dtype` as a compile-time parameter instead of the
#     module-level `comptime dtype` (the precision is picked by --double /
#     --float / --half at run time, so each precision is its own instantiation);
#   * dot_kernel takes the block size `tbsize` as a compile-time parameter
#     instead of the module-level `TBSIZE = 256` (the shared reduction array
#     needs a compile-time length; --block-size picks an instantiation).
# Mojo 1.2: the GPU index builtins are imported from max.gpu (std.gpu became
# private in 1.1).
# The element kernels have no bounds check, so the host requires the array
# size to be a multiple of the block size. The dot reduction halves `offset`
# from tbsize // 2, so the block size must be a power of two.

from max.gpu import thread_idx, block_idx, block_dim, grid_dim
from max.gpu.sync import barrier
from std.memory import Pointer, stack_allocation, AddressSpace

comptime startA = 0.1
comptime startB = 0.2
comptime startC = 0.0
comptime startScalar = 0.4


def init_kernel[
    dtype: DType
](
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    c: Pointer[Scalar[dtype], MutAnyOrigin],
    init_a: Scalar[dtype],
    init_b: Scalar[dtype],
    init_c: Scalar[dtype],
):
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    a[unsafe_offset=i] = init_a
    b[unsafe_offset=i] = init_b
    c[unsafe_offset=i] = init_c


def copy_kernel[
    dtype: DType
](
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    c: Pointer[Scalar[dtype], MutAnyOrigin],
):
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    c[unsafe_offset=i] = a[unsafe_offset=i]


def mul_kernel[
    dtype: DType
](
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    c: Pointer[Scalar[dtype], MutAnyOrigin],
):
    comptime scalar = Scalar[dtype](startScalar)
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    b[unsafe_offset=i] = scalar * c[unsafe_offset=i]


def add_kernel[
    dtype: DType
](
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    c: Pointer[Scalar[dtype], MutAnyOrigin],
):
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    c[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]


def triad_kernel[
    dtype: DType
](
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    c: Pointer[Scalar[dtype], MutAnyOrigin],
):
    comptime scalar = Scalar[dtype](startScalar)
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    a[unsafe_offset=i] = b[unsafe_offset=i] + scalar * c[unsafe_offset=i]


def dot_kernel[
    dtype: DType, tbsize: Int
](
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    partial: Pointer[Scalar[dtype], MutAnyOrigin],
    array_size_arg: Int64,
):
    # Int is fine for device-local indexing. Only values crossing the host/device
    # kernel ABI need a fixed-width integer type in Mojo 1.0.
    var array_size = Int(array_size_arg)
    var acc = Scalar[dtype](0)

    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    while i < array_size:
        acc = acc + a[unsafe_offset=i] * b[unsafe_offset=i]
        i += stride

    var tb_sum = stack_allocation[
        tbsize, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var local_i = Int(thread_idx.x)
    tb_sum[unsafe_offset=local_i] = acc

    var offset = tbsize // 2
    while offset > 0:
        barrier()
        if local_i < offset:
            tb_sum[unsafe_offset=local_i] = (
                tb_sum[unsafe_offset=local_i]
                + tb_sum[unsafe_offset=local_i + offset]
            )
        offset //= 2

    if local_i == 0:
        partial[unsafe_offset=Int(block_idx.x)] = tb_sum[unsafe_offset=0]
