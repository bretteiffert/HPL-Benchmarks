# Mojo 1.0.0 / MAX 26.5 port of the BabelStream GPU benchmark.
#
# Preserves the original benchmark behavior:
#   * copy, mul, add, triad, and dot kernels
#   * 1024-thread blocks
#   * device-event timing via DeviceContext.execution_time
#   * pageable host Lists for result copies
#   * BS_SM_COUNT controls dot grid size (4 blocks per SM/CU)
#   * dot timing includes the partial-sum device-to-host copy
#
# Mojo 1.0 migration points:
#   * GPU host/sync APIs moved into the max package
#   * `def` replaces the old `fn` spelling
#   * raw pointers use Pointer and unsafe_offset= for unchecked indexing
#   * enqueue_function takes the kernel function parameter once
#   * GPU ABI integer arguments use fixed-width integers, not Int
#   * execution_time uses explicit runtime closure captures

from std.gpu import thread_idx, block_idx, block_dim, grid_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.sync import barrier
from std.memory import Pointer, stack_allocation, AddressSpace
from std.sys import argv, exit
from std.os import getenv
from std.collections import List

comptime IMPLEMENTATION_STRING = "Mojo"

comptime startA = 0.1
comptime startB = 0.2
comptime startC = 0.0
comptime startScalar = 0.4

comptime TBSIZE = 1024

comptime USE_FLOAT = False
comptime dtype = DType.float32 if USE_FLOAT else DType.float64


def init_kernel(
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


def copy_kernel(
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    c: Pointer[Scalar[dtype], MutAnyOrigin],
):
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    c[unsafe_offset=i] = a[unsafe_offset=i]


def mul_kernel(
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    c: Pointer[Scalar[dtype], MutAnyOrigin],
):
    comptime scalar = Scalar[dtype](startScalar)
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    b[unsafe_offset=i] = scalar * c[unsafe_offset=i]


def add_kernel(
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    c: Pointer[Scalar[dtype], MutAnyOrigin],
):
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    c[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]


def triad_kernel(
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    c: Pointer[Scalar[dtype], MutAnyOrigin],
):
    comptime scalar = Scalar[dtype](startScalar)
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    a[unsafe_offset=i] = b[unsafe_offset=i] + scalar * c[unsafe_offset=i]


def dot_kernel(
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
        TBSIZE, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var local_i = Int(thread_idx.x)
    tb_sum[unsafe_offset=local_i] = acc

    var offset = TBSIZE // 2
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


struct MojoStream:
    var ctx: DeviceContext
    var array_size: Int
    var dot_num_blocks: Int

    var d_a: DeviceBuffer[dtype]
    var d_b: DeviceBuffer[dtype]
    var d_c: DeviceBuffer[dtype]
    var d_sum: DeviceBuffer[dtype]

    var sums: List[Scalar[dtype]]

    var last_kernel_time: Float64

    def __init__(out self, array_size: Int, device_index: Int) raises:
        if array_size % TBSIZE != 0:
            raise Error("Array size must be a multiple of " + String(TBSIZE))

        self.ctx = DeviceContext(device_id=device_index)
        self.array_size = array_size

        print("Using GPU device", self.ctx.name())
        print("Driver:", self.ctx.api())

        var sm_str = getenv("BS_SM_COUNT", "0")
        var sm_count = Int(sm_str)
        if sm_count <= 0:
            print(
                "error: BS_SM_COUNT is not set (or is not a positive"
                " integer).\n\nThis Mojo build does not expose the device"
                " SM/CU count, and the dot kernel's grid is 4 * SM count --"
                " matching the CUDA reference. Running with a guessed value"
                " would silently change both dot's bandwidth and its"
                " result.\n\nSet it to your device's SM (NVIDIA) or CU (AMD)"
                ' count:\n  NVIDIA:  nvidia-smi -q | grep -i "Multiprocessor"'
                '\n  AMD:     rocminfo | grep -i "Compute Unit"'
                "\n\ne.g.  BS_SM_COUNT=132 ./babelstream_mojo -n 1000"
            )
            exit(1)
        self.dot_num_blocks = sm_count * 4
        print(
            "Reduction kernel config:",
            self.dot_num_blocks,
            "groups of (fixed) size",
            TBSIZE,
        )

        self.d_a = self.ctx.enqueue_create_buffer[dtype](array_size)
        self.d_b = self.ctx.enqueue_create_buffer[dtype](array_size)
        self.d_c = self.ctx.enqueue_create_buffer[dtype](array_size)
        self.d_sum = self.ctx.enqueue_create_buffer[dtype](self.dot_num_blocks)

        self.sums = List[Scalar[dtype]](unsafe_uninit_length=self.dot_num_blocks)
        self.last_kernel_time = 0.0

        self.warmup()

    def warmup(mut self) raises:
        var n = TBSIZE
        var w_a = self.ctx.enqueue_create_buffer[dtype](n)
        var w_b = self.ctx.enqueue_create_buffer[dtype](n)
        var w_c = self.ctx.enqueue_create_buffer[dtype](n)
        var w_sum = self.ctx.enqueue_create_buffer[dtype](self.dot_num_blocks)

        self.ctx.enqueue_function[init_kernel](
            w_a,
            w_b,
            w_c,
            Scalar[dtype](startA),
            Scalar[dtype](startB),
            Scalar[dtype](startC),
            grid_dim=1,
            block_dim=TBSIZE,
        )
        self.ctx.enqueue_function[copy_kernel](
            w_a, w_c, grid_dim=1, block_dim=TBSIZE
        )
        self.ctx.enqueue_function[mul_kernel](
            w_b, w_c, grid_dim=1, block_dim=TBSIZE
        )
        self.ctx.enqueue_function[add_kernel](
            w_a, w_b, w_c, grid_dim=1, block_dim=TBSIZE
        )
        self.ctx.enqueue_function[triad_kernel](
            w_a, w_b, w_c, grid_dim=1, block_dim=TBSIZE
        )
        self.ctx.enqueue_function[dot_kernel](
            w_a,
            w_b,
            w_sum,
            Int64(n),
            grid_dim=self.dot_num_blocks,
            block_dim=TBSIZE,
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
            ctx.enqueue_function[init_kernel](
                d_a,
                d_b,
                d_c,
                Scalar[dtype](startA),
                Scalar[dtype](startB),
                Scalar[dtype](startC),
                grid_dim=n // TBSIZE,
                block_dim=TBSIZE,
            )

        var ns = self.ctx.execution_time(body, 1)
        self.last_kernel_time = Float64(ns) * 1.0e-9

    def copy(mut self) raises:
        var n = self.array_size
        var d_a = self.d_a
        var d_c = self.d_c

        @always_inline
        def body(ctx: DeviceContext) raises {d_a, d_c, n}:
            ctx.enqueue_function[copy_kernel](
                d_a, d_c, grid_dim=n // TBSIZE, block_dim=TBSIZE
            )

        var ns = self.ctx.execution_time(body, 1)
        self.last_kernel_time = Float64(ns) * 1.0e-9

    def mul(mut self) raises:
        var n = self.array_size
        var d_b = self.d_b
        var d_c = self.d_c

        @always_inline
        def body(ctx: DeviceContext) raises {d_b, d_c, n}:
            ctx.enqueue_function[mul_kernel](
                d_b, d_c, grid_dim=n // TBSIZE, block_dim=TBSIZE
            )

        var ns = self.ctx.execution_time(body, 1)
        self.last_kernel_time = Float64(ns) * 1.0e-9

    def add(mut self) raises:
        var n = self.array_size
        var d_a = self.d_a
        var d_b = self.d_b
        var d_c = self.d_c

        @always_inline
        def body(ctx: DeviceContext) raises {d_a, d_b, d_c, n}:
            ctx.enqueue_function[add_kernel](
                d_a, d_b, d_c, grid_dim=n // TBSIZE, block_dim=TBSIZE
            )

        var ns = self.ctx.execution_time(body, 1)
        self.last_kernel_time = Float64(ns) * 1.0e-9

    def triad(mut self) raises:
        var n = self.array_size
        var d_a = self.d_a
        var d_b = self.d_b
        var d_c = self.d_c

        @always_inline
        def body(ctx: DeviceContext) raises {d_a, d_b, d_c, n}:
            ctx.enqueue_function[triad_kernel](
                d_a, d_b, d_c, grid_dim=n // TBSIZE, block_dim=TBSIZE
            )

        var ns = self.ctx.execution_time(body, 1)
        self.last_kernel_time = Float64(ns) * 1.0e-9

    def dot(mut self) raises -> Scalar[dtype]:
        var n = self.array_size
        var nb = self.dot_num_blocks
        var d_a = self.d_a
        var d_b = self.d_b
        var d_sum = self.d_sum
        # Mojo 1.0.0 has an origin-lowering issue when a List-derived Pointer
        # is captured by a closure. execution_time() executes this callback
        # synchronously while self.sums is alive, so erase the tracked origin
        # for the capture to avoid the false same-type/ref conversion error.
        var sums_ptr = self.sums.unsafe_ptr().as_unsafe_any_origin()

        @always_inline
        def body(ctx: DeviceContext) raises {d_a, d_b, d_sum, n, nb, sums_ptr}:
            ctx.enqueue_function[dot_kernel](
                d_a,
                d_b,
                d_sum,
                Int64(n),
                grid_dim=nb,
                block_dim=TBSIZE,
            )
            ctx.enqueue_copy(dst_ptr=sums_ptr, src_buf=d_sum)

        var ns = self.ctx.execution_time(body, 1)
        self.last_kernel_time = Float64(ns) * 1.0e-9

        var total = Scalar[dtype](0)
        for i in range(nb):
            total += self.sums[i]
        return total

    def read_arrays(
        self,
        mut a: List[Scalar[dtype]],
        mut b: List[Scalar[dtype]],
        mut c: List[Scalar[dtype]],
    ) raises:
        self.ctx.enqueue_copy(dst_ptr=a.unsafe_ptr(), src_buf=self.d_a)
        self.ctx.enqueue_copy(dst_ptr=b.unsafe_ptr(), src_buf=self.d_b)
        self.ctx.enqueue_copy(dst_ptr=c.unsafe_ptr(), src_buf=self.d_c)
        self.ctx.synchronize()


def kahan_sum(v: List[Scalar[dtype]], n: Int) -> Float64:
    var s = Float64(0.0)
    var comp = Float64(0.0)
    for i in range(n):
        var y = Float64(v[i]) - comp
        var t = s + y
        comp = (t - s) - y
        s = t
    return s


def metrics(
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
        if rel_err > max_err:
            max_err = rel_err
    max_rel_err = max_err


def check_solution(
    ntimes: Int,
    a: List[Scalar[dtype]],
    b: List[Scalar[dtype]],
    c: List[Scalar[dtype]],
    dot_sum: Scalar[dtype],
    array_size: Int,
):
    var goldA = Scalar[dtype](startA)
    var goldB = Scalar[dtype](startB)
    var goldC = Scalar[dtype](startC)
    var scalar = Scalar[dtype](startScalar)

    for _ in range(ntimes):
        goldC = goldA
        goldB = scalar * goldC
        goldC = goldA + goldB
        goldA = goldB + scalar * goldC

    var chkA = Float64(0)
    var chkB = Float64(0)
    var chkC = Float64(0)
    var mxA = Float64(0)
    var mxB = Float64(0)
    var mxC = Float64(0)

    metrics(a, array_size, goldA, chkA, mxA)
    metrics(b, array_size, goldB, chkB, mxB)
    metrics(c, array_size, goldC, chkC, mxC)

    var expected_dot = Float64(goldA) * Float64(goldB) * Float64(array_size)
    var actual_dot = Float64(dot_sum)

    var product_magnitude = abs(Float64(goldA) * Float64(goldB))
    var tiny = (
        Float64(1.17549435e-38)
        if USE_FLOAT
        else Float64(2.2250738585072014e-308)
    )
    var dot_underflows = product_magnitude < tiny

    var dot_err = Float64(0)
    if expected_dot != 0.0:
        dot_err = abs(actual_dot - expected_dot) / abs(expected_dot)
    else:
        dot_err = abs(actual_dot)

    var precname = "float32" if USE_FLOAT else "float64"

    print("")
    print("Correctness metrics (" + precname + ",", ntimes, "iterations)")
    print("Array     Expected          Checksum rel err  Max elem rel err")
    print("a        ", Float64(goldA), chkA, mxA)
    print("b        ", Float64(goldB), chkB, mxB)
    print("c        ", Float64(goldC), chkC, mxC)
    print("dot      ", expected_dot, dot_err, "-")
    print("  (actual dot   =", actual_dot, ")")
    if dot_underflows:
        print(
            "  (note: expected element product",
            product_magnitude,
            "underflows this precision at this iteration count, so the dot"
            " relative error above is not meaningful)",
        )


struct Args(Copyable):
    var array_size: Int
    var num_times: Int
    var device_index: Int
    var mibibytes: Bool
    var csv_output: Bool

    def __init__(out self):
        self.array_size = 33554432
        self.num_times = 1000
        self.device_index = 0
        self.mibibytes = False
        self.csv_output = False


def print_help():
    print("")
    print("Usage: babelstream [OPTIONS]")
    print("")
    print("Options:")
    print("  -h  --help               Print the message")
    print("      --list               List available devices")
    print("      --device     INDEX   Select device at INDEX")
    print("  -s  --arraysize  SIZE    Use SIZE elements in the array")
    print("  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)")
    print("      --float              Use floats (no-op, float is already default)")
    print(
        "      --mibibytes          Use MiB=2^20 for bandwidth calculation "
        "(default MB=10^6)"
    )
    print("")


def parse_arguments() raises -> Args:
    var args = Args()
    var av = argv()
    var i = 1
    while i < len(av):
        var a = String(av[i])
        if a == "--list":
            print("")
            print("Devices:")
            print("0:", DeviceContext().name())
            print("")
            exit(0)
        elif a == "--device":
            i += 1
            if i >= len(av):
                print("Invalid device index.")
                exit(1)
            args.device_index = Int(String(av[i]))
        elif a == "--arraysize" or a == "-s":
            i += 1
            if i >= len(av):
                print("Invalid array size.")
                exit(1)
            args.array_size = Int(String(av[i]))
            if args.array_size <= 0:
                print("Invalid array size.")
                exit(1)
        elif a == "--numtimes" or a == "-n":
            i += 1
            if i >= len(av):
                print("Invalid number of times.")
                exit(1)
            args.num_times = Int(String(av[i]))
            if args.num_times < 2:
                print("Number of times must be 2 or more")
                exit(1)
        elif a == "--float":
            pass
        elif a == "--mibibytes":
            args.mibibytes = True
        elif a == "--csv":
            args.csv_output = True
        elif a == "--help" or a == "-h":
            print_help()
            exit(0)
        else:
            print("Unrecognized argument '" + a + "' (try '--help')")
            exit(1)
        i += 1
    return args^


def main() raises:
    var args = parse_arguments()

    comptime bpe = 4 if USE_FLOAT else 8
    var n = args.array_size

    if not args.csv_output:
        print("Running kernels", args.num_times, "times")
        print("Precision:", "float" if USE_FLOAT else "double")
        if args.mibibytes:
            print(
                "Array size:",
                Float64(n * bpe) / (1024.0 * 1024.0),
                "MiB",
            )
            print(
                "Total size:",
                3.0 * Float64(n * bpe) / (1024.0 * 1024.0),
                "MiB",
            )
        else:
            print("Array size:", Float64(n * bpe) * 1.0e-6, "MB")
            print("Total size:", 3.0 * Float64(n * bpe) * 1.0e-6, "MB")

    var stream = MojoStream(args.array_size, args.device_index)

    stream.init_arrays()
    var init_elapsed_s = stream.get_time_taken()

    var timings = List[List[Float64]]()
    for _ in range(5):
        timings.append(List[Float64]())

    var dot_sum = Scalar[dtype](0)

    for _ in range(args.num_times):
        stream.copy()
        timings[0].append(stream.get_time_taken())
        stream.mul()
        timings[1].append(stream.get_time_taken())
        stream.add()
        timings[2].append(stream.get_time_taken())
        stream.triad()
        timings[3].append(stream.get_time_taken())
        dot_sum = stream.dot()
        timings[4].append(stream.get_time_taken())

    var scale = (1.0 / (1024.0 * 1024.0)) if args.mibibytes else 1.0e-9
    var init_bw = (scale * Float64(3 * bpe * n)) / init_elapsed_s

    if not args.csv_output:
        print(
            "Init:",
            init_elapsed_s,
            "s (=",
            init_bw,
            "MiBytes/sec" if args.mibibytes else "GBytes/sec",
            ")",
        )
        print(
            "Function    ",
            "MiBytes/sec " if args.mibibytes else "GBytes/sec  ",
            "Min (sec)   Max         Average",
        )
    else:
        print("backend,GPU,precision,vec_size,routine,BW_GBs")

    var labels: List[String] = ["Copy", "Mul", "Add", "Triad", "Dot"]
    var sizes: List[Int] = [
        2 * bpe * n,
        2 * bpe * n,
        3 * bpe * n,
        3 * bpe * n,
        2 * bpe * n,
    ]
    var precname = "float32" if USE_FLOAT else "float64"
    var gpu_name = stream.ctx.name()

    for i in range(5):
        var tmin = timings[i][1]
        var tmax = timings[i][1]
        var total = Float64(0)
        for j in range(1, len(timings[i])):
            var t = timings[i][j]
            if t < tmin:
                tmin = t
            if t > tmax:
                tmax = t
            total += t
        var average = total / Float64(len(timings[i]) - 1)

        if not args.csv_output:
            print(
                labels[i],
                scale * Float64(sizes[i]) / tmin,
                tmin,
                tmax,
                average,
            )
        else:
            for j in range(len(timings[i])):
                print(
                    IMPLEMENTATION_STRING
                    + ","
                    + gpu_name
                    + ","
                    + precname
                    + ","
                    + String(n)
                    + ","
                    + labels[i]
                    + ","
                    + String(1.0e-9 * Float64(sizes[i]) / timings[i][j])
                )

    var ha = List[Scalar[dtype]](unsafe_uninit_length=n)
    var hb = List[Scalar[dtype]](unsafe_uninit_length=n)
    var hc = List[Scalar[dtype]](unsafe_uninit_length=n)
    stream.read_arrays(ha, hb, hc)

    check_solution(args.num_times, ha, hb, hc, dot_sum, n)
