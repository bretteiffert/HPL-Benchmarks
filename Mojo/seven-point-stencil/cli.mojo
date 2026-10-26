# Command-line parsing and --help.
#
# User errors raise Error(<what>); main prints them as
#   error: <what> (try '--help')
# on stderr and exits 1.

from std.sys import argv


struct Args(Copyable):
    var nx: Int
    var ny: Int
    var nz: Int
    var block_x: Int
    var block_y: Int
    var block_z: Int
    var num_times: Int
    var device_index: Int
    var precision: String  # "float64" | "float32" | "float16"
    var mibibytes: Bool
    var csv: Bool
    var csv_raw: Bool
    var energy: Bool
    var list_devices: Bool
    var help: Bool

    def __init__(out self):
        self.nx = 1024
        self.ny = 1024
        self.nz = 1024
        self.block_x = 1024
        self.block_y = 1
        self.block_z = 1
        self.num_times = 100
        self.device_index = 0
        self.precision = "float64"
        self.mibibytes = False
        self.csv = False
        self.csv_raw = False
        self.energy = True
        self.list_devices = False
        self.help = False


def help_text() -> String:
    return """Usage: stencil [OPTIONS]

Seven-point stencil (Mojo): 3D finite-difference Laplacian. The input is a
quadratic test function whose exact Laplacian is 3 at every interior point.

Options:
  -h, --help               Print this message and exit
      --list               List the devices of this build's backend and exit
      --device INDEX       Device to run on; the energy profiler samples the
                           same index (default: 0)
  -s, --size L             Cubic grid L x L x L, L >= 3 (default: 1024)
      --grid X,Y,Z         Grid with X, Y, Z points per side, each >= 3
                           (overrides -s)
      --block-size X[,Y[,Z]]
                           Threads per block in x, y, z; missing values are 1;
                           the product is at most the device limit
                           (default: 1024,1,1)
  -n, --numtimes NUM       Timed launches after 1 untimed run, NUM >= 1
                           (default: 100)
      --double             float64 (default)
      --float              float32
      --half               float16 (only grids with 2 * sum((side - 1)^2)
                           <= 65,504, i.e. up to 105 per side for a cube)
                           (the last precision flag wins)
      --mibibytes          Text report in MiB/s and MiB (CSV stays GB/s)
      --csv                Summary CSV on stdout; report on stderr
      --csv-raw            One CSV row per timed launch on stdout; report on
                           stderr (not with --csv)
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


def _int_list(s: String, flag: String) raises -> List[Int]:
    var out = List[Int]()
    for tok in s.split(","):
        out.append(_int(String(String(tok).strip()), flag))
    return out^


def parse_arguments() raises -> Args:
    var args = Args()
    var av = argv()
    var i = 1
    var grid_given = False
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
        elif a == "-s" or a == "--size":
            var l = _int(_value(av, i), a)
            if not grid_given:
                args.nx = l
                args.ny = l
                args.nz = l
            i += 1
        elif a == "--grid":
            var g = _int_list(_value(av, i), a)
            if len(g) != 3:
                raise Error("--grid takes X,Y,Z")
            args.nx = g[0]
            args.ny = g[1]
            args.nz = g[2]
            grid_given = True
            i += 1
        elif a == "--block-size":
            var b = _int_list(_value(av, i), a)
            if len(b) < 1 or len(b) > 3:
                raise Error("--block-size takes X, X,Y or X,Y,Z")
            args.block_x = b[0]
            args.block_y = b[1] if len(b) > 1 else Int(1)
            args.block_z = b[2] if len(b) > 2 else Int(1)
            i += 1
        elif a == "-n" or a == "--numtimes":
            args.num_times = _int(_value(av, i), a)
            if args.num_times < 1:
                raise Error("number of times must be 1 or more")
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
        elif not a.startswith("-"):
            raise Error(
                "unexpected argument '"
                + a
                + "'; the grid and block size are now given as"
                " '--grid X,Y,Z --block-size X,Y,Z' (or '-s L')"
            )
        else:
            raise Error("unrecognized argument '" + a + "'")
        i += 1
    if args.nx < 3 or args.ny < 3 or args.nz < 3:
        raise Error("each grid side must be >= 3 (the stencil needs interior points)")
    if args.block_x < 1 or args.block_y < 1 or args.block_z < 1:
        raise Error("block sizes must be >= 1")
    if args.csv and args.csv_raw:
        raise Error("--csv and --csv-raw are mutually exclusive")
    return args^
