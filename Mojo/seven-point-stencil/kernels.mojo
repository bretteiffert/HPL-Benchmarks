# Seven-point stencil device code (Mojo 1.2.0 / MAX 26.7): the test-function
# initialization kernel and the 3D Laplacian kernel.
#
# Kernel bodies are unchanged from the original laplacian.mojo; they were
# already generic over dtype. The only change is forced by Mojo 1.1+: the GPU
# index builtins come from max.gpu (std.gpu is private).

from max.gpu import thread_idx, block_idx, block_dim
from std.memory import Pointer


def test_function_kernel[
    dtype: DType
](
    u: Pointer[Scalar[dtype], MutAnyOrigin],
    nx_arg: Int64,
    ny_arg: Int64,
    nz_arg: Int64,
    hx: Scalar[dtype],
    hy: Scalar[dtype],
    hz: Scalar[dtype],
):
    # Int is fine for device-local indexing; only the host/device
    # kernel ABI requires fixed-width integer arguments.
    var nx = Int(nx_arg)
    var ny = Int(ny_arg)
    var nz = Int(nz_arg)

    var i = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    var j = Int(thread_idx.y) + Int(block_idx.y) * Int(block_dim.y)
    var k = Int(thread_idx.z) + Int(block_idx.z) * Int(block_dim.z)

    if i >= nx or j >= ny or k >= nz:
        return

    var pos = i + nx * (j + ny * k)

    var c = Scalar[dtype](0.5)
    var x = Scalar[dtype](i) * hx
    var y = Scalar[dtype](j) * hy
    var z = Scalar[dtype](k) * hz
    var Lx = Scalar[dtype](nx) * hx
    var Ly = Scalar[dtype](ny) * hy
    var Lz = Scalar[dtype](nz) * hz
    u[unsafe_offset=pos] = (
        c * x * (x - Lx) + c * y * (y - Ly) + c * z * (z - Lz)
    )


def laplacian_kernel[
    dtype: DType
](
    f: Pointer[Scalar[dtype], MutAnyOrigin],
    u: Pointer[Scalar[dtype], MutAnyOrigin],
    nx_arg: Int64,
    ny_arg: Int64,
    nz_arg: Int64,
    invhx2: Scalar[dtype],
    invhy2: Scalar[dtype],
    invhz2: Scalar[dtype],
    invhxyz2: Scalar[dtype],
):
    var nx = Int(nx_arg)
    var ny = Int(ny_arg)
    var nz = Int(nz_arg)

    var i = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    var j = Int(thread_idx.y) + Int(block_idx.y) * Int(block_dim.y)
    var k = Int(thread_idx.z) + Int(block_idx.z) * Int(block_dim.z)

    if (
        i == 0
        or i >= nx - 1
        or j == 0
        or j >= ny - 1
        or k == 0
        or k >= nz - 1
    ):
        return

    var slice = nx * ny
    var pos = i + nx * j + slice * k

    f[unsafe_offset=pos] = (
        u[unsafe_offset=pos] * invhxyz2
        + (
            u[unsafe_offset=pos - 1]
            + u[unsafe_offset=pos + 1]
        )
        * invhx2
        + (
            u[unsafe_offset=pos - nx]
            + u[unsafe_offset=pos + nx]
        )
        * invhy2
        + (
            u[unsafe_offset=pos - slice]
            + u[unsafe_offset=pos + slice]
        )
        * invhz2
    )
