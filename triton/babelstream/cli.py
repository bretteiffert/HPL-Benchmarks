"""Command-line parsing and --help for BabelStream (Triton)."""

DEFAULT_ARRAY_SIZE = 33554432
DEFAULT_NUM_TIMES = 1000
DEFAULT_BLOCK_SIZE = 1024


class UsageError(Exception):
    pass


class Options:
    def __init__(self):
        self.array_size = DEFAULT_ARRAY_SIZE
        self.num_times = DEFAULT_NUM_TIMES
        self.device = 0
        self.block_size = DEFAULT_BLOCK_SIZE
        self.precision = "float64"
        self.csv = False
        self.csv_raw = False
        self.energy = True
        self.mibibytes = False
        self.list = False
        self.help = False


def _int(flag, value, minimum):
    try:
        n = int(value, 10)
    except ValueError:
        raise UsageError("%s expects an integer, got '%s'" % (flag, value))
    if n < minimum:
        raise UsageError("%s must be at least %d, got %d" % (flag, minimum, n))
    return n


def parse(argv):
    opts = Options()
    args = list(argv[1:])
    i = 0

    def value(flag):
        nonlocal i
        i += 1
        if i >= len(args):
            raise UsageError("%s needs a value" % flag)
        return args[i]

    while i < len(args):
        a = args[i]
        if a in ("-h", "--help"):
            opts.help = True
        elif a == "--list":
            opts.list = True
        elif a == "--device":
            opts.device = _int(a, value(a), 0)
        elif a in ("-s", "--arraysize"):
            opts.array_size = _int(a, value(a), 1)
        elif a in ("-n", "--numtimes"):
            opts.num_times = _int(a, value(a), 2)
        elif a == "--block-size":
            opts.block_size = _int(a, value(a), 1)
        elif a == "--double":
            opts.precision = "float64"
        elif a == "--float":
            opts.precision = "float32"
        elif a == "--half":
            opts.precision = "float16"
        elif a == "--csv":
            opts.csv = True
        elif a == "--csv-raw":
            opts.csv_raw = True
        elif a == "--no-energy":
            opts.energy = False
        elif a == "--mibibytes":
            opts.mibibytes = True
        else:
            raise UsageError("unrecognized argument '%s'" % a)
        i += 1

    if opts.csv and opts.csv_raw:
        raise UsageError("--csv and --csv-raw cannot be used together")
    return opts


def help_text(prog):
    return """\
BabelStream (Triton)

Usage: %s [OPTIONS]

Options:
  -s, --arraysize SIZE   elements per array (default %d); a multiple of the block size
  -n, --numtimes NUM     iterations, NUM >= 2; the first is warm-up (default %d)
      --block-size N     elements per program, a power of two (default %d)
      --double           float64 (default)
      --float            float32
      --half             float16
      --csv              summary CSV on stdout (report on stderr)
      --csv-raw          one CSV row per timed iteration on stdout (report on stderr)
      --no-energy        skip board-power measurement (default: measured)
      --mibibytes        text bandwidth in MiB/s and sizes in MiB (CSV stays GB/s)
      --device INDEX     device to run on; the energy profiler uses the same index (default 0)
      --list             list devices (index: name) and exit
  -h, --help             show this help and exit

The last precision flag wins. Options take the form '--flag value'.

Environment (all optional):
  GPU_NAME               override the reported device name
  GPU_DRIVER             override the reported driver version
  FMAD                   reported only; it has no effect (Triton fuses multiply-add
                         by default; TRITON_DEFAULT_FP_FUSION=0 turns that off)
  DSLBENCH_ENERGY_LIB    path to libkernelenergy.so (default: ../../dependencies)
""" % (prog, DEFAULT_ARRAY_SIZE, DEFAULT_NUM_TIMES, DEFAULT_BLOCK_SIZE)
