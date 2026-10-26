"""Command-line parsing and --help for miniBUDE (Triton). Benchmark flags keep
the upstream miniBUDE names."""

PPWIS = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_ITERS = 8
DEFAULT_WARMUP = 2
DEFAULT_ROWS = 8
DEFAULT_DECK = "../../data/miniBUDE/bm1"   # relative to this file's folder


class UsageError(Exception):
    pass


class Options:
    def __init__(self):
        self.iterations = DEFAULT_ITERS
        self.warmup = DEFAULT_WARMUP
        self.nposes = 0
        self.ppwis = [PPWIS[0]]
        self.wgsizes = []            # empty: one warp
        self.deck = None             # None: DEFAULT_DECK next to the script
        self.output = ""
        self.rows = DEFAULT_ROWS
        self.device = ""             # index or name substring
        self.precision = "float64"
        self.csv = False
        self.csv_raw = False
        self.energy = True
        self.list = False
        self.help = False


def _ints(flag, s, minimum):
    out = []
    for tok in s.split(","):
        try:
            v = int(tok, 10)
        except ValueError:
            raise UsageError("%s expects a comma-separated list of integers, got '%s'" % (flag, s))
        if v < minimum:
            raise UsageError("%s values must be at least %d, got %d" % (flag, minimum, v))
        out.append(v)
    return out


def _int(flag, s, minimum):
    try:
        v = int(s, 10)
    except ValueError:
        raise UsageError("%s expects an integer, got '%s'" % (flag, s))
    if v < minimum:
        raise UsageError("%s must be at least %d, got %d" % (flag, minimum, v))
    return v


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

    half = False
    while i < len(args):
        a = args[i]
        if a in ("-h", "--help", "help"):
            opts.help = True
        elif a in ("-l", "--list", "list"):
            opts.list = True
        elif a in ("-i", "--iter"):
            opts.iterations = _int(a, value(a), 1)
        elif a == "--warmup":
            opts.warmup = _int(a, value(a), 0)
        elif a in ("-n", "--poses"):
            opts.nposes = _int(a, value(a), 0)
        elif a in ("-p", "--ppwi"):
            s = value(a)
            if s == "all":
                opts.ppwis = list(PPWIS)
            else:
                opts.ppwis = _ints(a, s, 1)
                for v in opts.ppwis:
                    if v not in PPWIS:
                        raise UsageError("ppwi %d is not supported; use one of %s or 'all'"
                                         % (v, ",".join(map(str, PPWIS))))
        elif a in ("-w", "--wgsize"):
            opts.wgsizes = _ints(a, value(a), 1)
        elif a == "--deck":
            opts.deck = value(a)
        elif a in ("-o", "--out"):
            opts.output = value(a)
        elif a in ("-r", "--rows"):
            opts.rows = _int(a, value(a), 0)
        elif a in ("-d", "--device"):
            opts.device = value(a)
        elif a == "--double":
            opts.precision, half = "float64", False
        elif a == "--float":
            opts.precision, half = "float32", False
        elif a == "--half":
            half = True
        elif a == "--csv":
            opts.csv = True
        elif a == "--csv-raw":
            opts.csv_raw = True
        elif a == "--no-energy":
            opts.energy = False
        else:
            raise UsageError("unrecognized argument '%s'" % a)
        i += 1

    if opts.help or opts.list:
        return opts
    if half:
        raise UsageError("--half is not supported: miniBUDE runs in float64 or float32 "
                         "(use --double or --float)")
    if opts.csv and opts.csv_raw:
        raise UsageError("--csv and --csv-raw cannot be used together")
    # Keep order, drop repeats.
    opts.ppwis = list(dict.fromkeys(opts.ppwis))
    opts.wgsizes = list(dict.fromkeys(opts.wgsizes))
    return opts


def help_text(prog):
    return """\
miniBUDE (Triton)

Usage: %s [OPTIONS]

Options:
  -i, --iter I           timed iterations (default %d)
      --warmup N         untimed warm-up iterations before them (default %d)
  -n, --poses N          compute only the first N poses; 0 for every pose in the deck (default 0)
  -p, --ppwi LIST        comma-separated poses per work-item, or 'all' (default %d; available %s)
  -w, --wgsize LIST      comma-separated work-group sizes (default: one warp). Each must be a
                         power-of-two multiple of the warp size, since num_warps = wgsize / warp
                         size; the reference default of 1 is not expressible in Triton
      --deck DIR         input deck (default %s, next to this script)
  -o, --out PATH         save the first configuration's energies to PATH
  -r, --rows N           show the first N energies in the report (default %d)
      --double           float64 (default)
      --float            float32, checked against one untimed float64 pass
      --csv              summary CSV on stdout, one row per configuration (report on stderr)
      --csv-raw          one CSV row per timed iteration on stdout (report on stderr)
      --no-energy        skip board-power measurement (default: measured)
  -d, --device INDEX     device index, or a substring of its name; the energy profiler uses
                         the same index (default 0)
  -l, --list             list devices (index: name) and exit
  -h, --help             show this help and exit

Every (ppwi, wgsize) combination runs in one invocation; combinations that do not
divide the pose count or do not fit the device are skipped with a note.
The last precision flag wins. Options take the form '--flag value'.

Environment (all optional):
  GPU_NAME               override the reported device name
  GPU_DRIVER             override the reported driver version
  DSLBENCH_ENERGY_LIB    path to libkernelenergy.so (default: ../../dependencies)
""" % (prog, DEFAULT_ITERS, DEFAULT_WARMUP, PPWIS[0], ",".join(map(str, PPWIS)),
       DEFAULT_DECK, DEFAULT_ROWS)
