# Command-line parsing and --help.
#
# User errors raise Error(<what>); main prints them as
#   error: <what> (try '--help')
# on stderr and exits 1.

from std.sys import argv


struct Args(Copyable):
    var input: String
    var num_times: Int
    var device_index: Int
    var block_size: Int
    var precision: String  # "float64" | "float32"
    var legacy_sqrt: Bool
    var run_check: Bool
    var force_check: Bool
    var max_check_atoms: Int
    var csv: Bool
    var csv_raw: Bool
    var energy: Bool
    var list_devices: Bool
    var help: Bool

    def __init__(out self):
        self.input = ""
        self.num_times = 10
        self.device_index = 0
        self.block_size = 256
        self.precision = "float64"
        self.legacy_sqrt = False
        self.run_check = True
        self.force_check = False
        self.max_check_atoms = 128
        self.csv = False
        self.csv_raw = False
        self.energy = True
        self.list_devices = False
        self.help = False


def help_text() -> String:
    return """Usage: hartree-fock [OPTIONS] INPUT

Hartree-Fock (Mojo): two-electron Fock-matrix build with Schwarz screening
and atomic accumulation. INPUT is a data file such as
../../data/hartree-fock/he64.

Options:
  -h, --help               Print this message and exit
      --list               List the devices of this build's backend and exit
      --device INDEX       Device to run on; the energy profiler samples the
                           same index (default: 0)
  -n, --iters NUM          Timed Fock builds after 1 untimed run, NUM >= 1
                           (default: 10)
      --block-size SIZE    Pinned threads per block; SIZE must be 256
                           (default: 256)
      --double             float64 (default)
      --float              float32
                           (the last precision flag wins; there is no --half:
                           the 1e-12 screening thresholds are below float16
                           range)
      --check              Always compare the untimed run with the float64 CPU
                           reference
      --no-check           Skip the CPU reference
      --max-check-atoms N  Without --check, skip the reference above N atoms
                           (default: 128)
      --csv                Summary CSV on stdout; report on stderr
      --csv-raw            One CSV row per timed iteration on stdout; report
                           on stderr (not with --csv)
      --no-energy          Skip board-power measurement

The pinned configuration uses native-precision square roots.

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
        elif a == "-n" or a == "--iters":
            args.num_times = _int(_value(av, i), a)
            if args.num_times < 1:
                raise Error("number of timed iterations must be 1 or more")
            i += 1
        elif a == "--block-size":
            args.block_size = _int(_value(av, i), a)
            if args.block_size <= 0:
                raise Error("block size must be > 0")
            i += 1
        elif a == "--max-check-atoms":
            args.max_check_atoms = _int(_value(av, i), a)
            if args.max_check_atoms < 0:
                raise Error("--max-check-atoms must be >= 0")
            i += 1
        elif a == "--double":
            args.precision = "float64"
        elif a == "--float":
            args.precision = "float32"
        elif a == "--half":
            raise Error(
                "--half is not supported: the 1e-12 screening thresholds are"
                " below float16 range"
            )
        elif a == "--legacy-sqrt":
            raise Error(
                "the pinned configuration requires native square roots;"
                " --legacy-sqrt is not supported"
            )
        elif a == "--check":
            args.run_check = True
            args.force_check = True
        elif a == "--no-check":
            args.run_check = False
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
        elif a.startswith("-"):
            raise Error("unrecognized argument '" + a + "'")
        elif args.input.byte_length() == 0:
            args.input = a
        else:
            raise Error("unexpected extra argument '" + a + "'")
        i += 1
    if args.csv and args.csv_raw:
        raise Error("--csv and --csv-raw are mutually exclusive")
    if args.block_size != 256:
        raise Error("the pinned configuration requires --block-size 256")
    if not args.help and not args.list_devices and args.input.byte_length() == 0:
        raise Error("missing INPUT file")
    return args^
