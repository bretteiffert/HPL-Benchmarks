# Seven-point stencil problem: device buffers, launches, timing, theoretical
# traffic and the correctness check (the Laplacian of the test function is 3
# at every interior point).
#
# Launch code, traffic formula and the check arithmetic follow the original
# laplacian.mojo. Changes:
#   * precision and grid/block shape are run-time values;
#   * times are seconds, and a launch whose device timer returns nothing is an
#     error instead of a zero.

from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.sys import size_of

from kernels import laplacian_kernel, test_function_kernel

# Block size of the one-off initialization launch, as in the original.
comptime TBSize = 1024


def theoretical_fetch_bytes[dtype: DType](nx: Int, ny: Int, nz: Int) -> Int:
    """All points except the 8 corners and the 12 edges are read."""
    return (
        nx * ny * nz - 8 - 4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2)
    ) * size_of[dtype]()


def theoretical_write_bytes[dtype: DType](nx: Int, ny: Int, nz: Int) -> Int:
    """Every interior point is written."""
    return ((nx - 2) * (ny - 2) * (nz - 2)) * size_of[dtype]()


def test_function[
    dtype: DType
](
    ctx: DeviceContext,
    d_u: DeviceBuffer[dtype],
    nx: Int,
    ny: Int,
    nz: Int,
    hx: Scalar[dtype],
    hy: Scalar[dtype],
    hz: Scalar[dtype],
) raises:
    var gx = (nx - 1) // TBSize + 1

    ctx.enqueue_function[test_function_kernel[dtype]](
        d_u,
        Int64(nx),
        Int64(ny),
        Int64(nz),
        hx,
        hy,
        hz,
        grid_dim=(gx, ny, nz),
        block_dim=(TBSize, 1, 1),
    )


def laplacian[
    dtype: DType
](
    ctx: DeviceContext,
    d_f: DeviceBuffer[dtype],
    d_u: DeviceBuffer[dtype],
    nx: Int,
    ny: Int,
    nz: Int,
    BLK_X: Int,
    BLK_Y: Int,
    BLK_Z: Int,
    hx: Scalar[dtype],
    hy: Scalar[dtype],
    hz: Scalar[dtype],
) raises:
    var gx = (nx - 1) // BLK_X + 1
    var gy = (ny - 1) // BLK_Y + 1
    var gz = (nz - 1) // BLK_Z + 1

    var invhx2 = Scalar[dtype](1) / hx / hx
    var invhy2 = Scalar[dtype](1) / hy / hy
    var invhz2 = Scalar[dtype](1) / hz / hz
    var invhxyz2 = Scalar[dtype](-2) * (invhx2 + invhy2 + invhz2)

    ctx.enqueue_function[laplacian_kernel[dtype]](
        d_f,
        d_u,
        Int64(nx),
        Int64(ny),
        Int64(nz),
        invhx2,
        invhy2,
        invhz2,
        invhxyz2,
        grid_dim=(gx, gy, gz),
        block_dim=(BLK_X, BLK_Y, BLK_Z),
    )


def time_one_iteration[
    dtype: DType
](
    ctx: DeviceContext,
    d_f: DeviceBuffer[dtype],
    d_u: DeviceBuffer[dtype],
    nx: Int,
    ny: Int,
    nz: Int,
    BLK_X: Int,
    BLK_Y: Int,
    BLK_Z: Int,
    hx: Scalar[dtype],
    hy: Scalar[dtype],
    hz: Scalar[dtype],
) raises -> Float64:
    """One timed Laplacian launch, in seconds (device timestamps)."""

    @always_inline
    def body(ictx: DeviceContext) raises {d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz}:
        laplacian[dtype](
            ictx, d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz
        )

    ctx.synchronize()
    var ns = ctx.execution_time(body, 1)
    if ns <= 0:
        raise Error("device timer returned no measurement for the Laplacian")
    return Float64(ns) * 1e-9


struct CorrectnessResult(Copyable):
    var max_abs_err: Float64
    var l2_rel_err: Float64
    var mean_abs_err: Float64
    var num_checked: Int

    def __init__(
        out self,
        max_abs_err: Float64,
        l2_rel_err: Float64,
        mean_abs_err: Float64,
        num_checked: Int,
    ):
        self.max_abs_err = max_abs_err
        self.l2_rel_err = l2_rel_err
        self.mean_abs_err = mean_abs_err
        self.num_checked = num_checked


def check_correctness[
    dtype: DType
](
    ctx: DeviceContext, d_f: DeviceBuffer[dtype], nx: Int, ny: Int, nz: Int
) raises -> CorrectnessResult:
    var expected = Float64(3.0)
    var sum_abs = Float64(0.0)
    var sum_sq_err = Float64(0.0)
    var sum_sq_exact = Float64(0.0)
    var max_err = Float64(0.0)
    var count = 0

    var slice = nx * ny

    with d_f.map_to_host() as h_f:
        for k in range(1, nz - 1):
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    var pos = i + nx * j + slice * k
                    var val = h_f[pos].cast[DType.float64]()
                    var err = abs(val - expected)
                    sum_abs += err
                    sum_sq_err += err * err
                    sum_sq_exact += expected * expected
                    # NaN never compares greater, so record it explicitly.
                    if err > max_err or err != err:
                        max_err = err
                    count += 1

    if count > 0:
        var mean_abs_err = sum_abs / Float64(count)
        var l2_rel_err = (
            sqrt(sum_sq_err / sum_sq_exact)
            if sum_sq_exact > 0.0
            else sqrt(sum_sq_err)
        )
        return CorrectnessResult(max_err, l2_rel_err, mean_abs_err, count)

    return CorrectnessResult(0.0, 0.0, 0.0, 0)
