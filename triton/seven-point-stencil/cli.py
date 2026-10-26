"""Command-line parsing and --help for the seven-point stencil (Triton)."""

DEFAULT_SIZE = 1024
DEFAULT_NUM_TIMES = 100
DEFAULT_BLOCK = (1024, 1, 1)


class UsageError(Exception):
    pass


class Options:
    def __init__(self):
        self.grid = (DEFAULT_SIZE,) * 3
        self.block = DEFAULT_BLOCK
        self.num_times = DEFAULT_NUM_TIMES
        self.device = 0
        self.precision = "float64"
        self.csv = False
        self.csv_raw = False
        self.energy = True
        self.list = False
        self.help = False


def _int(flag, s, minimum):
    try:
        v = int(s, 10)
    except ValueError:
        raise UsageError("%s expects an integer, got '%s'" % (flag, s))
    if v < minimum:
        raise UsageError("%s must be at least %d, got %d" % (flag, minimum, v))
    return v


def _triple(flag, s, minimum, fill):
    parts = s.split(",")
    if not 1 <= len(parts) <= 3 or (fill is None and len(parts) != 3):
        want = "X,Y,Z" if fill is None else "X[,Y[,Z]]"
        raise UsageError("%s expects %s, got '%s'" % (flag, want, s))
    vals = [_int(flag, p, minimum) for p in parts]
    while len(vals) < 3:
        vals.append(fill)
    return tuple(vals)


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
        elif a in ("-s", "--size"):
            n = _int(a, value(a), 3)
            opts.grid = (n, n, n)
        elif a == "--grid":
            opts.grid = _triple(a, value(a), 3, None)
        elif a == "--block-size":
            opts.block = _triple(a, value(a), 1, 1)
        elif a in ("-n", "--numtimes"):
            opts.num_times = _int(a, value(a), 1)
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
        elif a.lstrip("-").isdigit() and not a.startswith("-"):
            raise UsageError("unrecognized argument '%s': positional sizes were replaced by "
                             "-s L or --grid X,Y,Z and --block-size X,Y,Z" % a)
        else:
            raise UsageError("unrecognized argument '%s'" % a)
        i += 1

    if opts.csv and opts.csv_raw and not (opts.help or opts.list):
        raise UsageError("--csv and --csv-raw cannot be used together")
    return opts


def help_text(prog):
    return """\
Seven-point stencil (Triton)

Usage: %s [OPTIONS]

Options:
  -s, --size L           cubic grid L x L x L, L >= 3 (default %d)
      --grid X,Y,Z       grid with different sides (each >= 3)
      --block-size X[,Y[,Z]]
                         points per program along x, y, z; each a power of two
                         (default %d,%d,%d; missing sides are 1)
  -n, --numtimes NUM     timed iterations after 1 untimed run (default %d)
      --double           float64 (default)
      --float            float32
      --half             float16 (needs 2*((X-1)^2+(Y-1)^2+(Z-1)^2) <= 65,504, i.e. a cube of at most 105)
      --csv              summary CSV on stdout (report on stderr)
      --csv-raw          one CSV row per timed iteration on stdout (report on stderr)
      --no-energy        skip board-power measurement (default: measured)
      --device INDEX     device to run on; the energy profiler uses the same index (default 0)
      --list             list devices (index: name) and exit
  -h, --help             show this help and exit

Bandwidth is the theoretical traffic (every point but the 8 corners and 12 edges
read once, every interior point written once) over the kernel time, in GB/s.
The last precision flag wins. Options take the form '--flag value'.

Environment (all optional):
  GPU_NAME               override the reported device name
  GPU_DRIVER             override the reported driver version
  DSLBENCH_ENERGY_LIB    path to libkernelenergy.so (default: ../../dependencies)
""" % ((prog, DEFAULT_SIZE) + DEFAULT_BLOCK + (DEFAULT_NUM_TIMES,))
