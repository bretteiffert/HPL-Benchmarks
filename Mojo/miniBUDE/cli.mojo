# Command-line parsing and --help.
#
# Upstream miniBUDE flag names are kept (-i, -n, -p, -w, --deck, -o, -r, -d,
# --warmup) next to the shared flags. User errors raise Error(<what>); main
# prints them as
#   error: <what> (try '--help')
# on stderr and exits 1.

from std.sys import argv

from kernels import PPWIS_STR, is_supported_ppwi

comptime DEFAULT_DATA_DIR = "../../data/miniBUDE/bm1"


struct Args(Copyable):
    var deck_dir: String
    var iterations: Int
    var warmup: Int
    var nposes: Int
    var ppwis: List[Int]
    var wgsizes: List[Int]
    var rows: Int
    var out_path: String
    var golden_path: String
    var device_index: Int
    var precision: String  # "float32" | "float64"
    var csv: Bool
    var csv_raw: Bool
    var energy: Bool
    var list_devices: Bool
    var help: Bool

    def __init__(out self):
        self.deck_dir = String(DEFAULT_DATA_DIR)
        self.iterations = 8
        self.warmup = 2
        self.nposes = 0
        self.ppwis = List[Int]()
        self.wgsizes = List[Int]()
        self.rows = 8
        self.out_path = ""
        self.golden_path = ""
        self.device_index = 0
        self.precision = "float32"
        self.csv = False
        self.csv_raw = False
        self.energy = True
        self.list_devices = False
        self.help = False


def help_text() -> String:
    return """Usage: minibude [OPTIONS]

miniBUDE (Mojo): protein-ligand docking energy evaluation. One run sweeps
every (ppwi, wgsize) combination given with -p and -w.

Options:
  -h, --help, help         Print this message and exit
  -l, --list, list         List the devices of this build's backend and exit
  -d, --device INDEX       Device to run on; the energy profiler samples the
                           same index (default: 0)
      --deck DIR           Input deck directory (default: """ + DEFAULT_DATA_DIR + """)
  -i, --iter I             Timed launches per configuration (default: 8)
      --warmup N           Untimed launches before them (default: 2)
  -n, --poses N            Compute only the first N poses; 0 = all (default: 0)
  -p, --ppwi LIST          Comma-separated poses per work-item, or `all`
                           (default: 1; available: """ + PPWIS_STR + """)
  -w, --wgsize LIST        Comma-separated work-group sizes (default: 1)
  -r, --rows N             Print the first N energies (default: 8)
  -o, --out FILE           Write the first configuration's energies to FILE,
                           one per line (usable as --golden)
      --golden FILE        float64 reference energies, one per line, for the
                           float32 accuracy metrics
      --float              float32 (default in this port)
      --double             float64. Needs float64 sin/cos on the GPU, which
                           Mojo 1.2 does not provide (a compile-time error on
                           NVIDIA); it is only compiled with -D FP64
      --csv                Summary CSV on stdout; report on stderr
      --csv-raw            One CSV row per timed launch on stdout; report on
                           stderr (not with --csv)
      --no-energy          Skip board-power measurement

Configurations whose ppwi x wgsize does not divide the pose count are skipped
with a note.

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
        var t = String(String(tok).strip())
        if t.byte_length() > 0:
            out.append(_int(t, flag))
    if len(out) == 0:
        raise Error("empty list for '" + flag + "'")
    return out^


def parse_arguments() raises -> Args:
    var args = Args()
    var av = argv()
    var i = 1
    while i < len(av):
        var a = String(av[i])
        if a == "-h" or a == "--help" or a == "help":
            args.help = True
        elif a == "-l" or a == "--list" or a == "list":
            args.list_devices = True
        elif a == "-d" or a == "--device":
            args.device_index = _int(_value(av, i), a)
            if args.device_index < 0:
                raise Error("device index must be >= 0")
            i += 1
        elif a == "--deck":
            args.deck_dir = _value(av, i)
            i += 1
        elif a == "-i" or a == "--iter":
            args.iterations = _int(_value(av, i), a)
            if args.iterations < 1:
                raise Error("iterations must be 1 or more")
            i += 1
        elif a == "--warmup":
            args.warmup = _int(_value(av, i), a)
            if args.warmup < 0:
                raise Error("warm-up count must be >= 0")
            i += 1
        elif a == "-n" or a == "--poses":
            args.nposes = _int(_value(av, i), a)
            if args.nposes < 0:
                raise Error("pose count must be >= 0")
            i += 1
        elif a == "-p" or a == "--ppwi":
            var v = _value(av, i)
            if v == "all":
                args.ppwis = [1, 2, 4, 8, 16, 32, 64, 128]
            else:
                args.ppwis = _int_list(v, a)
                for k in range(len(args.ppwis)):
                    if not is_supported_ppwi(args.ppwis[k]):
                        raise Error(
                            "ppwi "
                            + String(args.ppwis[k])
                            + " is not supported; use one of "
                            + PPWIS_STR
                            + " or `all`"
                        )
            i += 1
        elif a == "-w" or a == "--wgsize":
            args.wgsizes = _int_list(_value(av, i), a)
            for k in range(len(args.wgsizes)):
                if args.wgsizes[k] <= 0:
                    raise Error("work-group size must be > 0")
            i += 1
        elif a == "-r" or a == "--rows":
            args.rows = _int(_value(av, i), a)
            if args.rows < 0:
                raise Error("rows must be >= 0")
            i += 1
        elif a == "-o" or a == "--out":
            args.out_path = _value(av, i)
            i += 1
        elif a == "--golden":
            args.golden_path = _value(av, i)
            i += 1
        elif a == "--float":
            args.precision = "float32"
        elif a == "--double":
            args.precision = "float64"
        elif a == "--half":
            raise Error("--half is not supported by miniBUDE")
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
    if len(args.ppwis) == 0:
        args.ppwis.append(1)
    if len(args.wgsizes) == 0:
        args.wgsizes.append(1)
    return args^
