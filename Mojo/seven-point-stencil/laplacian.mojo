# This Mojo CUDA-reference port evaluates a seven-point finite-difference Laplacian on a configurable 3D grid and block shape, then reports correctness and bandwidth.
# It defaults to float64 and device-event timing through the precision and USE_DEVICE_TIMER comptime values, with host wall-clock timing available when the latter is false.
# Raw device pointers preserve direct stencil indexing, DeviceContext selects CUDA or HIP, and interior results are checked against the quadratic test function's exact Laplacian of 3.

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import UnsafePointer
from std.sys import argv, size_of, stderr
from std.time import perf_counter_ns
from std.math import sqrt

comptime precision = DType.float64
comptime precision_str = "double"

comptime TBSize = 1024
comptime L = 1024
comptime NUM_ITER = 100

comptime USE_DEVICE_TIMER = True


fn test_function_kernel[
    dtype: DType
](
    u: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    nx: Int,
    ny: Int,
    nz: Int,
    hx: Scalar[dtype],
    hy: Scalar[dtype],
    hz: Scalar[dtype],
):
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
    u[pos] = c * x * (x - Lx) + c * y * (y - Ly) + c * z * (z - Lz)


fn laplacian_kernel[
    dtype: DType
](
    f: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    u: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    nx: Int,
    ny: Int,
    nz: Int,
    invhx2: Scalar[dtype],
    invhy2: Scalar[dtype],
    invhz2: Scalar[dtype],
    invhxyz2: Scalar[dtype],
):
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

    f[pos] = (
        u[pos] * invhxyz2
        + (u[pos - 1] + u[pos + 1]) * invhx2
        + (u[pos - nx] + u[pos + nx]) * invhy2
        + (u[pos - slice] + u[pos + slice]) * invhz2
    )


fn test_function[
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
    ctx.enqueue_function[test_function_kernel[dtype], test_function_kernel[dtype]](
        d_u,
        nx,
        ny,
        nz,
        hx,
        hy,
        hz,
        grid_dim=(gx, ny, nz),
        block_dim=(TBSize, 1, 1),
    )


fn laplacian[
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

    ctx.enqueue_function[laplacian_kernel[dtype], laplacian_kernel[dtype]](
        d_f,
        d_u,
        nx,
        ny,
        nz,
        invhx2,
        invhy2,
        invhz2,
        invhxyz2,
        grid_dim=(gx, gy, gz),
        block_dim=(BLK_X, BLK_Y, BLK_Z),
    )


fn time_one_iteration[
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
    @parameter
    fn body(ictx: DeviceContext) raises:
        laplacian[dtype](
            ictx, d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz
        )

    ctx.synchronize()

    comptime if USE_DEVICE_TIMER:
        return Float64(ctx.execution_time[body](1)) * 1e-6
    else:
        var t0 = perf_counter_ns()
        laplacian[dtype](
            ctx, d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        return Float64(t1 - t0) * 1e-6


@fieldwise_init
struct CorrectnessResult(Copyable, Movable):
    var max_abs_err: Float64
    var l2_rel_err: Float64
    var mean_abs_err: Float64
    var num_checked: Int


fn check_correctness[
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
                    if err > max_err:
                        max_err = err
                    count += 1

    if count > 0:
        var mean_abs_err = sum_abs / Float64(count)
        var l2_rel_err = sqrt(sum_sq_err / sum_sq_exact) if sum_sq_exact > 0.0 else sqrt(
            sum_sq_err
        )
        return CorrectnessResult(max_err, l2_rel_err, mean_abs_err, count)

    return CorrectnessResult(0.0, 0.0, 0.0, 0)


fn main() raises:
    var args = argv()

    var BLK_X = TBSize
    var BLK_Y = 1
    var BLK_Z = 1

    var nx = L
    var ny = L
    var nz = L
    var csv_output = False

    if len(args) > 1:
        nx = atol(args[1])
    if len(args) > 2:
        ny = atol(args[2])
    if len(args) > 3:
        nz = atol(args[3])
    if len(args) > 4:
        BLK_X = atol(args[4])
    if len(args) > 5:
        BLK_Y = atol(args[5])
    if len(args) > 6:
        BLK_Z = atol(args[6])
    if len(args) > 7 and args[7] == "--csv":
        csv_output = True

    comptime itemsize = size_of[Scalar[precision]]()

    var theoretical_fetch_size = (
        nx * ny * nz - 8 - 4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2)
    ) * itemsize
    var theoretical_write_size = ((nx - 2) * (ny - 2) * (nz - 2)) * itemsize
    var datasize = Float64(theoretical_fetch_size + theoretical_write_size)

    var ctx = DeviceContext()
    var gpu_name = ctx.name()
    var backend_label = String(ctx.api()).upper()

    if csv_output:
        print("backend,GPU,precision,L,blk_x,blk_y,blk_z,BW_GBs")
    else:
        print(gpu_name)
        print("Backend:", backend_label)
        print("Precision:", precision_str)
        print("nx,ny,nz =", nx, ",", ny, ",", nz)
        print("block sizes =", BLK_X, ",", BLK_Y, ",", BLK_Z)
        print(
            "Theoretical fetch size (GB):",
            Float64(theoretical_fetch_size) * 1e-9,
        )
        print(
            "Theoretical write size (GB):",
            Float64(theoretical_write_size) * 1e-9,
        )

    var n = nx * ny * nz

    var d_u = ctx.enqueue_create_buffer[precision](n)
    var d_f = ctx.enqueue_create_buffer[precision](n)

    var hx = Scalar[precision](1) / Scalar[precision](nx - 1)
    var hy = Scalar[precision](1) / Scalar[precision](ny - 1)
    var hz = Scalar[precision](1) / Scalar[precision](nz - 1)

    test_function[precision](ctx, d_u, nx, ny, nz, hx, hy, hz)

    laplacian[precision](
        ctx, d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz
    )
    ctx.synchronize()

    var corr = check_correctness[precision](ctx, d_f, nx, ny, nz)

    if csv_output:
        print(
            "correctness,precision,max_abs_err,l2_rel_err,mean_abs_err,points_checked",
            file=stderr,
        )
        print(
            String("correctness,")
            + precision_str
            + ","
            + String(corr.max_abs_err)
            + ","
            + String(corr.l2_rel_err)
            + ","
            + String(corr.mean_abs_err)
            + ","
            + String(corr.num_checked),
            file=stderr,
        )
    else:
        print(
            "--- Correctness metrics (precision =", precision_str, ") ---"
        )
        print("Interior points checked :", corr.num_checked)
        print("Max abs error           :", corr.max_abs_err)
        print("Mean abs error          :", corr.mean_abs_err)
        print("Relative L2 error       :", corr.l2_rel_err)
        print()

    var total_elapsed = Float64(0.0)

    for _ in range(NUM_ITER):
        var elapsed = time_one_iteration[precision](
            ctx, d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz
        )
        if csv_output:
            print(
                String("Mojo-")
                + backend_label
                + ","
                + gpu_name
                + ","
                + precision_str
                + ","
                + String(nx)
                + ","
                + String(BLK_X)
                + ","
                + String(BLK_Y)
                + ","
                + String(BLK_Z)
                + ","
                + String(datasize / elapsed / 1e6)
            )
        total_elapsed += elapsed

    if not csv_output:
        print(
            "Laplacian kernel took:",
            total_elapsed / Float64(NUM_ITER),
            "ms, effective memory bandwidth:",
            datasize * Float64(NUM_ITER) / total_elapsed / 1e6,
            "GB/s \n",
        )
