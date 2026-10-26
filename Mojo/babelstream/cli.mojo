# Command-line parsing and --help.
#
# User errors raise Error(<what>); main prints them as
#   error: <what> (try '--help')
# on stderr and exits 1.

from std.sys import argv


struct Args(Copyable):
    var array_size: Int
    var num_times: Int
    var device_index: Int
    var block_size: Int
    var precision: String  # "float64" | "float32" | "float16"
    var mibibytes: Bool
    var csv: Bool
    var csv_raw: Bool
    var energy: Bool
    var list_devices: Bool
    var help: Bool

    def __init__(out self):
        self.array_size = 33554432
        self.num_times = 1000
        self.device_index = 0
        self.block_size = 256
        self.precision = "float64"
        self.mibibytes = False
        self.csv = False
        self.csv_raw = False
        self.energy = True
        self.list_devices = False
        self.help = False


def help_text() -> String:
    return """Usage: babelstream [OPTIONS]

BabelStream (Mojo): Copy, Mul, Add, Triad and Dot memory bandwidth.

Options:
  -h, --help               Print this message and exit
      --list               List the devices of this build's backend and exit
      --device INDEX       Device to run on; the energy profiler samples the
                           same index (default: 0)
  -s, --arraysize SIZE     Elements per array; a multiple of the block size
                           (default: 33554432)
  -n, --numtimes NUM       Iterations, NUM >= 2; the first is a warm-up and is
                           excluded from statistics, CSV and energy
                           (default: 1000)
      --block-size SIZE    Pinned threads per block; SIZE must be 256
                           (default: 256)
      --double             float64 (default)
      --float              float32
      --half               float16
                           (the last precision flag wins)
      --mibibytes          Text report in MiB/s and MiB (CSV stays GB/s)
      --csv                Summary CSV on stdout; report on stderr
      --csv-raw            One CSV row per timed iteration on stdout; report
                           on stderr (not with --csv)
      --no-energy          Skip board-power measurement

Environment (all optional):
  GPU_NAME                 Device name to report instead of the queried one
  GPU_DRIVER               Driver string to report instead of the queried one
  DSLBENCH_ENERGY_LIB      Path to libkernelenergy.so (default:
                           ../../dependencies/libkernelenergy.so)
"""


def _value(av: Span[StaticString, ImmStaticOrigin], i: Int) raises -> String:
    if i + 1 >= len(av):
        raise Error("missing value for '" + String(av[i]) + "'")
    return String(av[i + 1])


def _int(s: String, flag: String) raises -> Int:
    try:
        return atol(s)
    except:
        raise Error("invalid value '" + s + "' for '" + flag + "'")


def parse_arguments() raises -> Args:
    var args = Args()
    var av = argv()
    var i = 1
    while i < len(av):
        var a = String(av[i])
        if a == "-h" or a == "--help":
            args.help = True
        elif a == "--list":
            args.list_devices = True
        elif a == "--device":
            args.device_index = _int(_value(av, i), a)
            if args.device_index < 0:
                raise Error("device index must be >= 0")
            i += 1
        elif a == "-s" or a == "--arraysize":
            args.array_size = _int(_value(av, i), a)
            if args.array_size <= 0:
                raise Error("array size must be > 0")
            i += 1
        elif a == "-n" or a == "--numtimes":
            args.num_times = _int(_value(av, i), a)
            if args.num_times < 2:
                raise Error("number of times must be 2 or more")
            i += 1
        elif a == "--block-size":
            args.block_size = _int(_value(av, i), a)
            if args.block_size <= 0:
                raise Error("block size must be > 0")
            i += 1
        elif a == "--double":
            args.precision = "float64"
        elif a == "--float":
            args.precision = "float32"
        elif a == "--half":
            args.precision = "float16"
        elif a == "--mibibytes":
            args.mibibytes = True
        elif a == "--csv":
            args.csv = True
        elif a == "--csv-raw":
            args.csv_raw = True
        elif a == "--no-energy":
            args.energy = False
        elif a.startswith("--") and "=" in a:
            raise Error(
                "unrecognized argument '"
                + a
                + "'; use '--flag value', not '--flag=value'"
            )
        else:
            raise Error("unrecognized argument '" + a + "'")
        i += 1
    if args.csv and args.csv_raw:
        raise Error("--csv and --csv-raw are mutually exclusive")
    if args.block_size != 256:
        raise Error("the pinned configuration requires --block-size 256")
    return args^
