# BabelStream problem: device buffers, timed launches and the CPU reference.
#
# Launch code and the gold/metric arithmetic follow the original
# babelstream.mojo. Changes:
#   * generic over dtype and block size (run-time --double/--float/--half and
#     --block-size) instead of the module-level comptime dtype and TBSIZE;
#   * the dot group count comes from the caller (4 x SM/CU count queried from
#     the device) instead of the BS_SM_COUNT environment variable;
#   * no printing: results are returned to the caller, which fills the report;
#   * a launch whose device timer returns no measurement raises instead of
#     recording zero.

from max.gpu.host import DeviceContext, DeviceBuffer
from std.collections import List
from std.utils.numerics import isnan

from kernels import (
    init_kernel,
    copy_kernel,
    mul_kernel,
    add_kernel,
    triad_kernel,
    dot_kernel,
    startA,
    startB,
    startC,
    startScalar,
)


def _seconds(ns: Int, routine: String) raises -> Float64:
    if ns <= 0:
        raise Error("device timer returned no measurement for " + routine)
    return Float64(ns) * 1.0e-9


struct MojoStream[dtype: DType, tbsize: Int]:
    var ctx: DeviceContext
    var array_size: Int
    var dot_num_blocks: Int

    var d_a: DeviceBuffer[Self.dtype]
    var d_b: DeviceBuffer[Self.dtype]
    var d_c: DeviceBuffer[Self.dtype]
    var d_sum: DeviceBuffer[Self.dtype]

    var sums: List[Scalar[Self.dtype]]

    var last_kernel_time: Float64

    def __init__(
        out self, ctx: DeviceContext, array_size: Int, dot_num_blocks: Int
    ) raises:
        self.ctx = ctx
        self.array_size = array_size
        self.dot_num_blocks = dot_num_blocks

        self.d_a = self.ctx.enqueue_create_buffer[Self.dtype](array_size)
        self.d_b = self.ctx.enqueue_create_buffer[Self.dtype](array_size)
        self.d_c = self.ctx.enqueue_create_buffer[Self.dtype](array_size)
        self.d_sum = self.ctx.enqueue_create_buffer[Self.dtype](
            self.dot_num_blocks
        )

        self.sums = List[Scalar[Self.dtype]](
            unsafe_uninit_length=self.dot_num_blocks
        )
        self.last_kernel_time = 0.0

        self.warmup()

    def warmup(mut self) raises:
        # Untimed: compiles/loads every kernel on small buffers.
        var n = Self.tbsize
        var w_a = self.ctx.enqueue_create_buffer[Self.dtype](n)
        var w_b = self.ctx.enqueue_create_buffer[Self.dtype](n)
        var w_c = self.ctx.enqueue_create_buffer[Self.dtype](n)
        var w_sum = self.ctx.enqueue_create_buffer[Self.dtype](
            self.dot_num_blocks
        )

        self.ctx.enqueue_function[init_kernel[Self.dtype]](
            w_a,
            w_b,
            w_c,
            Scalar[Self.dtype](startA),
            Scalar[Self.dtype](startB),
            Scalar[Self.dtype](startC),
            grid_dim=1,
            block_dim=Self.tbsize,
        )
        self.ctx.enqueue_function[copy_kernel[Self.dtype]](
            w_a, w_c, grid_dim=1, block_dim=Self.tbsize
        )
        self.ctx.enqueue_function[mul_kernel[Self.dtype]](
            w_b, w_c, grid_dim=1, block_dim=Self.tbsize
        )
        self.ctx.enqueue_function[add_kernel[Self.dtype]](
            w_a, w_b, w_c, grid_dim=1, block_dim=Self.tbsize
        )
        self.ctx.enqueue_function[triad_kernel[Self.dtype]](
            w_a, w_b, w_c, grid_dim=1, block_dim=Self.tbsize
        )
        self.ctx.enqueue_function[dot_kernel[Self.dtype, Self.tbsize]](
            w_a,
            w_b,
            w_sum,
            Int64(n),
            grid_dim=self.dot_num_blocks,
            block_dim=Self.tbsize,
        )
        self.ctx.enqueue_copy(dst_ptr=self.sums.unsafe_ptr(), src_buf=w_sum)
        self.ctx.synchronize()

    def get_time_taken(self) -> Float64:
        return self.last_kernel_time

    def init_arrays(mut self) raises:
        var n = self.array_size
        var d_a = self.d_a
        var d_b = self.d_b
        var d_c = self.d_c

        @always_inline
        def body(ctx: DeviceContext) raises {d_a, d_b, d_c, n}:
            ctx.enqueue_function[init_kernel[Self.dtype]](
                d_a,
                d_b,
                d_c,
                Scalar[Self.dtype](startA),
                Scalar[Self.dtype](startB),
                Scalar[Self.dtype](startC),
                grid_dim=n // Self.tbsize,
                block_dim=Self.tbsize,
            )

        var ns = self.ctx.execution_time(body, 1)
        self.last_kernel_time = _seconds(ns, "Init")

    def copy(mut self) raises:
        var n = self.array_size
        var d_a = self.d_a
        var d_c = self.d_c

        @always_inline
        def body(ctx: DeviceContext) raises {d_a, d_c, n}:
            ctx.enqueue_function[copy_kernel[Self.dtype]](
                d_a, d_c, grid_dim=n // Self.tbsize, block_dim=Self.tbsize
            )

        var ns = self.ctx.execution_time(body, 1)
        self.last_kernel_time = _seconds(ns, "Copy")

    def mul(mut self) raises:
        var n = self.array_size
        var d_b = self.d_b
        var d_c = self.d_c

        @always_inline
        def body(ctx: DeviceContext) raises {d_b, d_c, n}:
            ctx.enqueue_function[mul_kernel[Self.dtype]](
                d_b, d_c, grid_dim=n // Self.tbsize, block_dim=Self.tbsize
            )

        var ns = self.ctx.execution_time(body, 1)
        self.last_kernel_time = _seconds(ns, "Mul")

    def add(mut self) raises:
        var n = self.array_size
        var d_a = self.d_a
        var d_b = self.d_b
        var d_c = self.d_c

        @always_inline
        def body(ctx: DeviceContext) raises {d_a, d_b, d_c, n}:
            ctx.enqueue_function[add_kernel[Self.dtype]](
                d_a, d_b, d_c, grid_dim=n // Self.tbsize, block_dim=Self.tbsize
            )

        var ns = self.ctx.execution_time(body, 1)
        self.last_kernel_time = _seconds(ns, "Add")

    def triad(mut self) raises:
        var n = self.array_size
        var d_a = self.d_a
        var d_b = self.d_b
        var d_c = self.d_c

        @always_inline
        def body(ctx: DeviceContext) raises {d_a, d_b, d_c, n}:
            ctx.enqueue_function[triad_kernel[Self.dtype]](
                d_a, d_b, d_c, grid_dim=n // Self.tbsize, block_dim=Self.tbsize
            )

        var ns = self.ctx.execution_time(body, 1)
        self.last_kernel_time = _seconds(ns, "Triad")

    def dot(mut self) raises -> Float64:
        var n = self.array_size
        var nb = self.dot_num_blocks
        var d_a = self.d_a
        var d_b = self.d_b
        var d_sum = self.d_sum
        # Kept from the 1.0.0 port: an origin-lowering issue when a List-derived Pointer
        # is captured by a closure. execution_time() executes this callback
        # synchronously while self.sums is alive, so erase the tracked origin
        # for the capture to avoid the false same-type/ref conversion error.
        var sums_ptr = self.sums.unsafe_ptr().as_unsafe_any_origin()

        @always_inline
        def body(ctx: DeviceContext) raises {d_a, d_b, d_sum, n, nb, sums_ptr}:
            ctx.enqueue_function[dot_kernel[Self.dtype, Self.tbsize]](
                d_a,
                d_b,
                d_sum,
                Int64(n),
                grid_dim=nb,
                block_dim=Self.tbsize,
            )
            ctx.enqueue_copy(dst_ptr=sums_ptr, src_buf=d_sum)

        var ns = self.ctx.execution_time(body, 1)
        self.last_kernel_time = _seconds(ns, "Dot")

        var total = Float64(0)
        for i in range(nb):
            total += Float64(self.sums[i])
        return total

    def read_arrays(
        self,
        mut a: List[Scalar[Self.dtype]],
        mut b: List[Scalar[Self.dtype]],
        mut c: List[Scalar[Self.dtype]],
    ) raises:
        self.ctx.enqueue_copy(dst_ptr=a.unsafe_ptr(), src_buf=self.d_a)
        self.ctx.enqueue_copy(dst_ptr=b.unsafe_ptr(), src_buf=self.d_b)
        self.ctx.enqueue_copy(dst_ptr=c.unsafe_ptr(), src_buf=self.d_c)
        self.ctx.synchronize()


# ===----------------------------------------------------------------------=== #
# CPU reference
# ===----------------------------------------------------------------------=== #


def kahan_sum[dtype: DType](v: List[Scalar[dtype]], n: Int) -> Float64:
    var s = Float64(0.0)
    var comp = Float64(0.0)
    for i in range(n):
        var y = Float64(v[i]) - comp
        var t = s + y
        comp = (t - s) - y
        s = t
    return s


def metrics[
    dtype: DType
](
    v: List[Scalar[dtype]],
    n: Int,
    gold: Scalar[dtype],
    mut checksum_rel_err: Float64,
    mut max_rel_err: Float64,
):
    var gold_d = Float64(gold)
    var actual_sum = kahan_sum(v, n)
    var expected_sum = gold_d * Float64(n)

    if expected_sum != 0.0:
        checksum_rel_err = abs(actual_sum - expected_sum) / abs(expected_sum)
    else:
        checksum_rel_err = abs(actual_sum)

    var max_err = Float64(0.0)
    for i in range(n):
        var abs_err = abs(Float64(v[i]) - gold_d)
        var rel_err = abs_err / abs(gold_d) if gold_d != 0.0 else abs_err
        # NaN never compares greater, so record it explicitly.
        if rel_err > max_err or Bool(isnan(rel_err)):
            max_err = rel_err
    max_rel_err = max_err


struct Gold(Copyable):
    """Expected values after `ntimes` iterations, in the run precision and,
    for the meaningfulness notes, in float64."""

    var a: Float64
    var b: Float64
    var c: Float64
    var a64: Float64
    var b64: Float64
    var c64: Float64

    def __init__(out self, *, copy: Self):
        self.a = copy.a
        self.b = copy.b
        self.c = copy.c
        self.a64 = copy.a64
        self.b64 = copy.b64
        self.c64 = copy.c64

    def __init__[dtype: DType](out self, ntimes: Int, dtype_tag: Scalar[dtype]):
        var goldA = Scalar[dtype](startA)
        var goldB = Scalar[dtype](startB)
        var goldC = Scalar[dtype](startC)
        var scalar = Scalar[dtype](startScalar)
        for _ in range(ntimes):
            goldC = goldA
            goldB = scalar * goldC
            goldC = goldA + goldB
            goldA = goldB + scalar * goldC

        var a64 = Float64(startA)
        var b64 = Float64(startB)
        var c64 = Float64(startC)
        var s64 = Float64(startScalar)
        for _ in range(ntimes):
            c64 = a64
            b64 = s64 * c64
            c64 = a64 + b64
            a64 = b64 + s64 * c64

        self.a = Float64(goldA)
        self.b = Float64(goldB)
        self.c = Float64(goldC)
        self.a64 = a64
        self.b64 = b64
        self.c64 = c64


struct CheckResult(Copyable):
    var gold: Gold
    var chk_a: Float64
    var chk_b: Float64
    var chk_c: Float64
    var max_a: Float64
    var max_b: Float64
    var max_c: Float64
    var expected_dot: Float64
    var actual_dot: Float64
    var dot_err: Float64

    def __init__(out self, *, copy: Self):
        self.gold = copy.gold.copy()
        self.chk_a = copy.chk_a
        self.chk_b = copy.chk_b
        self.chk_c = copy.chk_c
        self.max_a = copy.max_a
        self.max_b = copy.max_b
        self.max_c = copy.max_c
        self.expected_dot = copy.expected_dot
        self.actual_dot = copy.actual_dot
        self.dot_err = copy.dot_err

    def __init__(out self, var gold: Gold):
        self.gold = gold^
        self.chk_a = 0
        self.chk_b = 0
        self.chk_c = 0
        self.max_a = 0
        self.max_b = 0
        self.max_c = 0
        self.expected_dot = 0
        self.actual_dot = 0
        self.dot_err = 0


def check_solution[
    dtype: DType
](
    ntimes: Int,
    a: List[Scalar[dtype]],
    b: List[Scalar[dtype]],
    c: List[Scalar[dtype]],
    dot_sum: Float64,
    array_size: Int,
) -> CheckResult:
    # Gold arithmetic runs in the benchmark precision, as in the original.
    var gold = Gold(ntimes, Scalar[dtype](0))
    var r = CheckResult(gold.copy())

    metrics(a, array_size, Scalar[dtype](gold.a), r.chk_a, r.max_a)
    metrics(b, array_size, Scalar[dtype](gold.b), r.chk_b, r.max_b)
    metrics(c, array_size, Scalar[dtype](gold.c), r.chk_c, r.max_c)

    r.expected_dot = gold.a * gold.b * Float64(array_size)
    r.actual_dot = dot_sum
    if r.expected_dot != 0.0:
        r.dot_err = abs(r.actual_dot - r.expected_dot) / abs(r.expected_dot)
    else:
        r.dot_err = abs(r.actual_dot)
    return r^
