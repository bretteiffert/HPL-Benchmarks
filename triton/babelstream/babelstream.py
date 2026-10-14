#!/usr/bin/env python3
# Triton BabelStream benchmark using PyTorch allocation and CUDA or ROCm device-event timing.
# It defaults to float64 via USE_FLOAT=False, partitions elementwise work into 1024-element programs, and uses a Triton reduction for dot.
# Scalar constants use the selected tensor dtype, while multiply-add contraction is left to the compiler and target.

import os
import sys
import torch
import triton
import triton.language as tl

IMPLEMENTATION_STRING = "Triton"

startA = 0.1
startB = 0.2
startC = 0.0
startScalar = 0.4

TBSIZE = 1024

USE_FLOAT = False


@triton.jit
def init_kernel(a_ptr, b_ptr, c_ptr, init_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    initA = tl.load(init_ptr + 0)
    initB = tl.load(init_ptr + 1)
    initC = tl.load(init_ptr + 2)
    tl.store(a_ptr + offs, tl.full((BLOCK_SIZE,), 0, initA.dtype) + initA)
    tl.store(b_ptr + offs, tl.full((BLOCK_SIZE,), 0, initB.dtype) + initB)
    tl.store(c_ptr + offs, tl.full((BLOCK_SIZE,), 0, initC.dtype) + initC)


@triton.jit
def copy_kernel(a_ptr, c_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(c_ptr + offs, tl.load(a_ptr + offs))


@triton.jit
def mul_kernel(b_ptr, c_ptr, scalar_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    scalar = tl.load(scalar_ptr)
    c = tl.load(c_ptr + offs)
    tl.store(b_ptr + offs, scalar * c)


@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.store(c_ptr + offs, a + b)


@triton.jit
def triad_kernel(a_ptr, b_ptr, c_ptr, scalar_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    scalar = tl.load(scalar_ptr)
    b = tl.load(b_ptr + offs)
    c = tl.load(c_ptr + offs)
    tl.store(a_ptr + offs, b + scalar * c)


@triton.jit
def dot_kernel(a_ptr, b_ptr, partial_ptr, array_size, num_programs,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.load(a_ptr).dtype)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    stride = BLOCK_SIZE * num_programs
    while tl.min(offs) < array_size:
        mask = offs < array_size
        a = tl.load(a_ptr + offs, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        acc = acc + a * b
        offs += stride

    tl.store(partial_ptr + pid, tl.sum(acc, axis=0))


class TritonStream:
    def __init__(self, array_size, device_index, dtype):
        if array_size % TBSIZE != 0:
            raise RuntimeError("Array size must be a multiple of %d" % TBSIZE)

        count = torch.cuda.device_count()
        if device_index >= count:
            raise RuntimeError("Invalid device index")
        torch.cuda.set_device(device_index)

        self.array_size = array_size
        self.dtype = dtype
        self.device = torch.device("cuda", device_index)

        props = torch.cuda.get_device_properties(self.device)
        print("Using %s device %s" % (
            "HIP" if torch.version.hip is not None else "CUDA", props.name))
        print("Driver: %s" % (torch.version.hip or torch.version.cuda))

        self.dot_num_blocks = props.multi_processor_count * 4
        print("Reduction kernel config: %d groups of (fixed) size %d"
              % (self.dot_num_blocks, TBSIZE))

        total_bytes = array_size * dtype.itemsize * 4
        if props.total_memory < total_bytes:
            raise RuntimeError("Device does not have enough memory for all 3 buffers")

        self.d_a = torch.empty(array_size, dtype=dtype, device=self.device)
        self.d_b = torch.empty(array_size, dtype=dtype, device=self.device)
        self.d_c = torch.empty(array_size, dtype=dtype, device=self.device)
        self.d_sum = torch.empty(self.dot_num_blocks, dtype=dtype, device=self.device)

        self.sums = torch.empty(self.dot_num_blocks, dtype=dtype, device="cpu")

        self.d_scalar = torch.tensor([startScalar], dtype=dtype, device=self.device)
        self.d_init = torch.tensor([startA, startB, startC], dtype=dtype, device=self.device)

        self.start_ev = torch.cuda.Event(enable_timing=True)
        self.stop_ev = torch.cuda.Event(enable_timing=True)
        self.last_kernel_time = 0.0

        self.warmup()

    def record_start(self):
        self.start_ev.record()

    def record_stop(self):
        self.stop_ev.record()
        self.stop_ev.synchronize()
        self.last_kernel_time = self.start_ev.elapsed_time(self.stop_ev) * 1.0e-3

    def get_time_taken(self):
        return self.last_kernel_time

    def warmup(self):
        n = TBSIZE
        w_a = torch.empty(n, dtype=self.dtype, device=self.device)
        w_b = torch.empty(n, dtype=self.dtype, device=self.device)
        w_c = torch.empty(n, dtype=self.dtype, device=self.device)
        w_sum = torch.empty(self.dot_num_blocks, dtype=self.dtype, device=self.device)

        init_kernel[(1,)](w_a, w_b, w_c, self.d_init,
                          BLOCK_SIZE=TBSIZE)
        copy_kernel[(1,)](w_a, w_c, BLOCK_SIZE=TBSIZE)
        mul_kernel[(1,)](w_b, w_c, self.d_scalar,
                         BLOCK_SIZE=TBSIZE)
        add_kernel[(1,)](w_a, w_b, w_c, BLOCK_SIZE=TBSIZE)
        triad_kernel[(1,)](w_a, w_b, w_c, self.d_scalar,
                           BLOCK_SIZE=TBSIZE)
        dot_kernel[(self.dot_num_blocks,)](
            w_a, w_b, w_sum, n, self.dot_num_blocks,
            BLOCK_SIZE=TBSIZE)
        self.sums.copy_(w_sum)
        torch.cuda.synchronize()

    def init_arrays(self):
        grid = (self.array_size // TBSIZE,)
        self.record_start()
        init_kernel[grid](self.d_a, self.d_b, self.d_c, self.d_init,
                          BLOCK_SIZE=TBSIZE)
        self.record_stop()

    def read_arrays(self):
        return (self.d_a.cpu().numpy(), self.d_b.cpu().numpy(), self.d_c.cpu().numpy())

    def copy(self):
        grid = (self.array_size // TBSIZE,)
        self.record_start()
        copy_kernel[grid](self.d_a, self.d_c, BLOCK_SIZE=TBSIZE)
        self.record_stop()

    def mul(self):
        grid = (self.array_size // TBSIZE,)
        self.record_start()
        mul_kernel[grid](self.d_b, self.d_c, self.d_scalar,
                         BLOCK_SIZE=TBSIZE)
        self.record_stop()

    def add(self):
        grid = (self.array_size // TBSIZE,)
        self.record_start()
        add_kernel[grid](self.d_a, self.d_b, self.d_c,
                         BLOCK_SIZE=TBSIZE)
        self.record_stop()

    def triad(self):
        grid = (self.array_size // TBSIZE,)
        self.record_start()
        triad_kernel[grid](self.d_a, self.d_b, self.d_c, self.d_scalar,
                           BLOCK_SIZE=TBSIZE)
        self.record_stop()

    def dot(self):
        self.record_start()
        dot_kernel[(self.dot_num_blocks,)](
            self.d_a, self.d_b, self.d_sum, self.array_size, self.dot_num_blocks,
            BLOCK_SIZE=TBSIZE)
        self.sums.copy_(self.d_sum)
        self.record_stop()

        partials = self.sums.numpy()
        total = partials.dtype.type(0.0)
        for i in range(self.dot_num_blocks):
            total = total + partials[i]
        return total


def check_solution(ntimes, a, b, c, dot_sum, array_size, csv_output):
    import numpy as np

    T = a.dtype.type
    goldA = T(startA)
    goldB = T(startB)
    goldC = T(startC)
    scalar = T(startScalar)

    for _ in range(ntimes):
        goldC = goldA
        goldB = scalar * goldC
        goldC = goldA + goldB
        goldA = goldB + scalar * goldC

    def kahan_sum(v):
        s = 0.0
        comp = 0.0
        for x in v:
            y = float(x) - comp
            t = s + y
            comp = (t - s) - y
            s = t
        return s

    def metrics(v, gold):
        gold_d = float(gold)
        actual_sum = kahan_sum(v)
        expected_sum = gold_d * len(v)
        checksum_rel_err = (abs(actual_sum - expected_sum) / abs(expected_sum)
                            if expected_sum != 0.0 else abs(actual_sum))
        vd = v.astype(np.float64)
        abs_err = np.abs(vd - gold_d)
        rel = abs_err / abs(gold_d) if gold_d != 0.0 else abs_err
        return checksum_rel_err, float(rel.max())

    checksumA, maxErrA = metrics(a, goldA)
    checksumB, maxErrB = metrics(b, goldB)
    checksumC, maxErrC = metrics(c, goldC)

    expected_dot = float(goldA) * float(goldB) * float(array_size)
    actual_dot = float(dot_sum)

    product_magnitude = abs(float(goldA) * float(goldB))
    dot_underflows = product_magnitude < float(np.finfo(T).tiny)

    dot_err = (abs(actual_dot - expected_dot) / abs(expected_dot)
               if expected_dot != 0.0 else abs(actual_dot))

    out = sys.stderr if csv_output else sys.stdout
    precname = "float32" if a.dtype.itemsize == 4 else "float64"

    print("", file=out)
    print("Correctness metrics (%s, %d iterations)" % (precname, ntimes), file=out)
    print("%-10s%-18s%-18s%-18s" % ("Array", "Expected", "Checksum rel err",
                                    "Max elem rel err"), file=out)

    def report(name, gold, chk, mx):
        print("%-10s%-18.6e%-18.3e%-18.3e" % (name, gold, chk, mx), file=out)

    report("a", float(goldA), checksumA, maxErrA)
    report("b", float(goldB), checksumB, maxErrB)
    report("c", float(goldC), checksumC, maxErrC)

    print("%-10s%-18.6e%-18.3e%-18s" % ("dot", expected_dot, dot_err, "-"), file=out)
    print("  (actual dot   = %.9e)" % actual_dot, file=out)
    if dot_underflows:
        print("  (note: expected element product %.3e underflows this precision "
              "at this iteration count, so the dot relative error above is not "
              "meaningful)" % product_magnitude, file=out)


class Args:
    def __init__(self):
        self.array_size = 33554432
        self.num_times = 1000
        self.device_index = 0
        self.mibibytes = False
        self.csv_output = False


def list_devices():
    count = torch.cuda.device_count()
    if count == 0:
        print("No devices found.", file=sys.stderr)
    else:
        print("")
        print("Devices:")
        for i in range(count):
            print("%d: %s" % (i, torch.cuda.get_device_properties(i).name))
        print("")


def print_help(prog):
    print("")
    print("Usage: %s [OPTIONS]" % prog)
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


def parse_arguments(argv):
    args = Args()
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--list":
            list_devices()
            sys.exit(0)
        elif a == "--device":
            i += 1
            if i >= len(argv):
                print("Invalid device index.", file=sys.stderr)
                sys.exit(1)
            args.device_index = int(argv[i])
        elif a in ("--arraysize", "-s"):
            i += 1
            if i >= len(argv):
                print("Invalid array size.", file=sys.stderr)
                sys.exit(1)
            v = int(argv[i])
            if v <= 0:
                print("Invalid array size.", file=sys.stderr)
                sys.exit(1)
            args.array_size = v
        elif a in ("--numtimes", "-n"):
            i += 1
            if i >= len(argv):
                print("Invalid number of times.", file=sys.stderr)
                sys.exit(1)
            v = int(argv[i])
            if v < 2:
                print("Number of times must be 2 or more", file=sys.stderr)
                sys.exit(1)
            args.num_times = v
        elif a == "--float":
            pass
        elif a == "--mibibytes":
            args.mibibytes = True
        elif a == "--csv":
            args.csv_output = True
        elif a in ("--help", "-h"):
            print_help(argv[0])
            sys.exit(0)
        else:
            print("Unrecognized argument '%s' (try '--help')" % a, file=sys.stderr)
            sys.exit(1)
        i += 1
    return args


def run(args):
    dtype = torch.float32 if USE_FLOAT else torch.float64
    itemsize = 4 if USE_FLOAT else 8

    if not args.csv_output:
        print("Running kernels %d times" % args.num_times)
        print("Precision: %s" % ("float" if USE_FLOAT else "double"))
        if args.mibibytes:
            print("Array size: %.1f MiB (=%.1f GiB)"
                  % (args.array_size * itemsize * 2.0 ** -20,
                     args.array_size * itemsize * 2.0 ** -30))
            print("Total size: %.1f MiB (=%.1f GiB)"
                  % (3.0 * args.array_size * itemsize * 2.0 ** -20,
                     3.0 * args.array_size * itemsize * 2.0 ** -30))
        else:
            print("Array size: %.1f MB (=%.1f GB)"
                  % (args.array_size * itemsize * 1.0e-6,
                     args.array_size * itemsize * 1.0e-9))
            print("Total size: %.1f MB (=%.1f GB)"
                  % (3.0 * args.array_size * itemsize * 1.0e-6,
                     3.0 * args.array_size * itemsize * 1.0e-9))

    stream = TritonStream(args.array_size, args.device_index, dtype)

    stream.init_arrays()
    init_elapsed_s = stream.get_time_taken()

    timings = [[] for _ in range(5)]
    dot_sum = None
    for _ in range(args.num_times):
        stream.copy();  timings[0].append(stream.get_time_taken())
        stream.mul();   timings[1].append(stream.get_time_taken())
        stream.add();   timings[2].append(stream.get_time_taken())
        stream.triad(); timings[3].append(stream.get_time_taken())
        dot_sum = stream.dot(); timings[4].append(stream.get_time_taken())

    scale = 2.0 ** -20 if args.mibibytes else 1.0e-9
    init_bw = (scale * (3 * itemsize * args.array_size)) / init_elapsed_s

    if not args.csv_output:
        print("Init: %7g s (=%g %s)" % (init_elapsed_s, init_bw,
              "MiBytes/sec" if args.mibibytes else "GBytes/sec"))
        print("%-12s%-12s%-12s%-12s%-12s" % (
            "Function", "MiBytes/sec" if args.mibibytes else "GBytes/sec",
            "Min (sec)", "Max", "Average"))
    else:
        print("backend,GPU,precision,vec_size,routine,BW_GBs")

    labels = ["Copy", "Mul", "Add", "Triad", "Dot"]
    sizes = [
        2 * itemsize * args.array_size,
        2 * itemsize * args.array_size,
        3 * itemsize * args.array_size,
        3 * itemsize * args.array_size,
        2 * itemsize * args.array_size,
    ]

    precname = "float32" if USE_FLOAT else "float64"
    gpu_name = torch.cuda.get_device_properties(args.device_index).name

    for i in range(5):
        t = timings[i][1:]
        tmin, tmax = min(t), max(t)
        average = sum(t) / len(t)
        if not args.csv_output:
            print("%-12s%-12.3g%-12.5g%-12.5g%-12.5g"
                  % (labels[i], scale * sizes[i] / tmin, tmin, tmax, average))
        else:
            for tt in timings[i]:
                print("%s,%s,%s,%d,%s,%s" % (IMPLEMENTATION_STRING, gpu_name,
                      precname, args.array_size, labels[i],
                      1.0e-9 * sizes[i] / tt))

    a, b, c = stream.read_arrays()
    check_solution(args.num_times, a, b, c, dot_sum, args.array_size, args.csv_output)


def main():
    args = parse_arguments(sys.argv)
    run(args)


if __name__ == "__main__":
    main()
