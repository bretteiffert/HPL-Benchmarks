# This Mojo BabelStream port runs copy, multiply, add, triad, and dot kernels with 1024-thread blocks and device-event timing.
# It defaults to float64 because USE_FLOAT is false; the --float option is a no-op, so changing precision requires editing USE_FLOAT and rebuilding.
# BS_SM_COUNT must be a positive environment value because the dot grid uses four blocks per SM/CU, and its timed region includes the partial-sum device-to-host copy.
# The dot uses a shared-memory tree reduction for both precisions, while host buffers are pageable Lists and arithmetic does not request explicit FMA contraction.

from gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import AddressSpace
from memory import UnsafePointer, stack_allocation
from sys import argv, exit
from os import getenv
from collections import List

alias IMPLEMENTATION_STRING = "Mojo"

alias startA = 0.1
alias startB = 0.2
alias startC = 0.0
alias startScalar = 0.4

alias TBSIZE = 1024

alias USE_FLOAT = False
alias dtype = DType.float32 if USE_FLOAT else DType.float64

fn init_kernel(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    init_a: Scalar[dtype],
    init_b: Scalar[dtype],
    init_c: Scalar[dtype],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    a[i] = init_a
    b[i] = init_b
    c[i] = init_c


fn copy_kernel(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    c[i] = a[i]


fn mul_kernel(
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    alias scalar = Scalar[dtype](startScalar)
    var i = block_dim.x * block_idx.x + thread_idx.x
    b[i] = scalar * c[i]


fn add_kernel(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    c[i] = a[i] + b[i]


fn triad_kernel(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    alias scalar = Scalar[dtype](startScalar)
    var i = block_dim.x * block_idx.x + thread_idx.x
    a[i] = b[i] + scalar * c[i]


fn dot_kernel(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    partial: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    array_size: Int,
):
    var acc = Scalar[dtype](0)

    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    while i < array_size:
        acc = acc + a[i] * b[i]
        i += stride

    var tb_sum = stack_allocation[
        TBSIZE, Scalar[dtype], address_space = AddressSpace.SHARED
    ]()
    var local_i = Int(thread_idx.x)
    tb_sum[local_i] = acc

    var offset = TBSIZE // 2
    while offset > 0:
        barrier()
        if local_i < offset:
            tb_sum[local_i] = tb_sum[local_i] + tb_sum[local_i + offset]
        offset //= 2

    if local_i == 0:
        partial[block_idx.x] = tb_sum[0]


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

    fn __init__(out self, array_size: Int, device_index: Int) raises:
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

    fn warmup(mut self) raises:
        var n = TBSIZE
        var w_a = self.ctx.enqueue_create_buffer[dtype](n)
        var w_b = self.ctx.enqueue_create_buffer[dtype](n)
        var w_c = self.ctx.enqueue_create_buffer[dtype](n)
        var w_sum = self.ctx.enqueue_create_buffer[dtype](self.dot_num_blocks)

        self.ctx.enqueue_function[init_kernel, init_kernel](
            w_a.unsafe_ptr(), w_b.unsafe_ptr(), w_c.unsafe_ptr(),
            Scalar[dtype](startA), Scalar[dtype](startB), Scalar[dtype](startC),
            grid_dim=1, block_dim=TBSIZE,
        )
        self.ctx.enqueue_function[copy_kernel, copy_kernel](
            w_a.unsafe_ptr(), w_c.unsafe_ptr(), grid_dim=1, block_dim=TBSIZE
        )
        self.ctx.enqueue_function[mul_kernel, mul_kernel](
            w_b.unsafe_ptr(), w_c.unsafe_ptr(), grid_dim=1, block_dim=TBSIZE
        )
        self.ctx.enqueue_function[add_kernel, add_kernel](
            w_a.unsafe_ptr(), w_b.unsafe_ptr(), w_c.unsafe_ptr(),
            grid_dim=1, block_dim=TBSIZE,
        )
        self.ctx.enqueue_function[triad_kernel, triad_kernel](
            w_a.unsafe_ptr(), w_b.unsafe_ptr(), w_c.unsafe_ptr(),
            grid_dim=1, block_dim=TBSIZE,
        )
        self.ctx.enqueue_function[dot_kernel, dot_kernel](
            w_a.unsafe_ptr(), w_b.unsafe_ptr(), w_sum.unsafe_ptr(), n,
            grid_dim=self.dot_num_blocks, block_dim=TBSIZE,
        )
        self.ctx.enqueue_copy(dst_ptr=self.sums.unsafe_ptr(), src_buf=w_sum)
        self.ctx.synchronize()

    fn get_time_taken(self) -> Float64:
        return self.last_kernel_time

    fn init_arrays(mut self) raises:
        var n = self.array_size
        var a = self.d_a.unsafe_ptr()
        var b = self.d_b.unsafe_ptr()
        var c = self.d_c.unsafe_ptr()

        @parameter
        @always_inline
        fn body(ctx: DeviceContext) raises:
            ctx.enqueue_function[init_kernel, init_kernel](
                a, b, c,
                Scalar[dtype](startA), Scalar[dtype](startB), Scalar[dtype](startC),
                grid_dim=n // TBSIZE, block_dim=TBSIZE,
            )

        var ns = self.ctx.execution_time[body](1)
        self.last_kernel_time = Float64(ns) * 1.0e-9

    fn copy(mut self) raises:
        var n = self.array_size
        var a = self.d_a.unsafe_ptr()
        var c = self.d_c.unsafe_ptr()

        @parameter
        @always_inline
        fn body(ctx: DeviceContext) raises:
            ctx.enqueue_function[copy_kernel, copy_kernel](
                a, c, grid_dim=n // TBSIZE, block_dim=TBSIZE
            )

        var ns = self.ctx.execution_time[body](1)
        self.last_kernel_time = Float64(ns) * 1.0e-9

    fn mul(mut self) raises:
        var n = self.array_size
        var b = self.d_b.unsafe_ptr()
        var c = self.d_c.unsafe_ptr()

        @parameter
        @always_inline
        fn body(ctx: DeviceContext) raises:
            ctx.enqueue_function[mul_kernel, mul_kernel](
                b, c, grid_dim=n // TBSIZE, block_dim=TBSIZE
            )

        var ns = self.ctx.execution_time[body](1)
        self.last_kernel_time = Float64(ns) * 1.0e-9

    fn add(mut self) raises:
        var n = self.array_size
        var a = self.d_a.unsafe_ptr()
        var b = self.d_b.unsafe_ptr()
        var c = self.d_c.unsafe_ptr()

        @parameter
        @always_inline
        fn body(ctx: DeviceContext) raises:
            ctx.enqueue_function[add_kernel, add_kernel](
                a, b, c, grid_dim=n // TBSIZE, block_dim=TBSIZE
            )

        var ns = self.ctx.execution_time[body](1)
        self.last_kernel_time = Float64(ns) * 1.0e-9

    fn triad(mut self) raises:
        var n = self.array_size
        var a = self.d_a.unsafe_ptr()
        var b = self.d_b.unsafe_ptr()
        var c = self.d_c.unsafe_ptr()

        @parameter
        @always_inline
        fn body(ctx: DeviceContext) raises:
            ctx.enqueue_function[triad_kernel, triad_kernel](
                a, b, c, grid_dim=n // TBSIZE, block_dim=TBSIZE
            )

        var ns = self.ctx.execution_time[body](1)
        self.last_kernel_time = Float64(ns) * 1.0e-9

    fn dot(mut self) raises -> Scalar[dtype]:
        var n = self.array_size
        var nb = self.dot_num_blocks
        var a = self.d_a.unsafe_ptr()
        var b = self.d_b.unsafe_ptr()
        var d_sum = self.d_sum
        var sums_ptr = self.sums.unsafe_ptr()

        @parameter
        @always_inline
        fn body(ctx: DeviceContext) raises:
            ctx.enqueue_function[dot_kernel, dot_kernel](
                a, b, d_sum.unsafe_ptr(), n,
                grid_dim=nb, block_dim=TBSIZE,
            )
            ctx.enqueue_copy(dst_ptr=sums_ptr, src_buf=d_sum)

        var ns = self.ctx.execution_time[body](1)
        self.last_kernel_time = Float64(ns) * 1.0e-9

        var total = Scalar[dtype](0)
        for i in range(nb):
            total += self.sums[i]
        return total

    fn read_arrays(
        self,
        mut a: List[Scalar[dtype]],
        mut b: List[Scalar[dtype]],
        mut c: List[Scalar[dtype]],
    ) raises:
        self.ctx.enqueue_copy(dst_ptr=a.unsafe_ptr(), src_buf=self.d_a)
        self.ctx.enqueue_copy(dst_ptr=b.unsafe_ptr(), src_buf=self.d_b)
        self.ctx.enqueue_copy(dst_ptr=c.unsafe_ptr(), src_buf=self.d_c)
        self.ctx.synchronize()


fn kahan_sum(v: List[Scalar[dtype]], n: Int) -> Float64:
    var s = Float64(0.0)
    var comp = Float64(0.0)
    for i in range(n):
        var y = Float64(v[i]) - comp
        var t = s + y
        comp = (t - s) - y
        s = t
    return s


fn metrics(
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


fn check_solution(
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
    var tiny = Float64(1.17549435e-38) if USE_FLOAT else Float64(2.2250738585072014e-308)
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


struct Args(Copyable, Movable):
    var array_size: Int
    var num_times: Int
    var device_index: Int
    var mibibytes: Bool
    var csv_output: Bool

    fn __init__(out self):
        self.array_size = 33554432
        self.num_times = 1000
        self.device_index = 0
        self.mibibytes = False
        self.csv_output = False


fn print_help():
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
    print("      --mibibytes          Use MiB=2^20 for bandwidth calculation (default MB=10^6)")
    print("")


fn parse_arguments() raises -> Args:
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


fn main() raises:
    var args = parse_arguments()

    alias bpe = 4 if USE_FLOAT else 8
    var n = args.array_size

    if not args.csv_output:
        print("Running kernels", args.num_times, "times")
        print("Precision:", "float" if USE_FLOAT else "double")
        if args.mibibytes:
            print("Array size:", Float64(n * bpe) / (1024.0 * 1024.0), "MiB")
            print("Total size:", 3.0 * Float64(n * bpe) / (1024.0 * 1024.0), "MiB")
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
        print("Init:", init_elapsed_s, "s (=", init_bw,
              "MiBytes/sec" if args.mibibytes else "GBytes/sec", ")")
        print("Function    ",
              "MiBytes/sec " if args.mibibytes else "GBytes/sec  ",
              "Min (sec)   Max         Average")
    else:
        print("backend,GPU,precision,vec_size,routine,BW_GBs")

    var labels: List[String] = ["Copy", "Mul", "Add", "Triad", "Dot"]
    var sizes: List[Int] = [
        2 * bpe * n, 2 * bpe * n, 3 * bpe * n, 3 * bpe * n, 2 * bpe * n
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
            print(labels[i], scale * Float64(sizes[i]) / tmin, tmin, tmax, average)
        else:
            for j in range(len(timings[i])):
                print(
                    IMPLEMENTATION_STRING + "," + gpu_name + "," + precname
                    + "," + String(n) + "," + labels[i] + ","
                    + String(1.0e-9 * Float64(sizes[i]) / timings[i][j])
                )

    var ha = List[Scalar[dtype]](unsafe_uninit_length=n)
    var hb = List[Scalar[dtype]](unsafe_uninit_length=n)
    var hc = List[Scalar[dtype]](unsafe_uninit_length=n)
    stream.read_arrays(ha, hb, hc)

    check_solution(args.num_times, ha, hb, hc, dot_sum, n)
