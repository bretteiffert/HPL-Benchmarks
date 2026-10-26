"""Command-line parsing and --help for Hartree-Fock (Triton)."""

DEFAULT_NUM_TIMES = 10
DEFAULT_BLOCK_SIZE = 256
DEFAULT_MAX_CHECK_ATOMS = 128


class UsageError(Exception):
    pass


class Options:
    def __init__(self):
        self.input = None
        self.num_times = DEFAULT_NUM_TIMES
        self.device = 0
        self.block_size = DEFAULT_BLOCK_SIZE
        self.precision = "float64"
        self.legacy_sqrt = False
        self.run_check = True
        self.force_check = False
        self.max_check_atoms = DEFAULT_MAX_CHECK_ATOMS
        self.csv = False
        self.csv_raw = False
        self.energy = True
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

    half = False
    while i < len(args):
        a = args[i]
        if a in ("-h", "--help"):
            opts.help = True
        elif a == "--list":
            opts.list = True
        elif a == "--device":
            opts.device = _int(a, value(a), 0)
        elif a in ("-n", "--iters"):
            opts.num_times = _int(a, value(a), 1)
        elif a == "--block-size":
            opts.block_size = _int(a, value(a), 1)
        elif a == "--double":
            opts.precision, half = "float64", False
        elif a == "--float":
            opts.precision, half = "float32", False
        elif a == "--half":
            half = True
        elif a == "--legacy-sqrt":
            opts.legacy_sqrt = True
        elif a == "--check":
            opts.run_check, opts.force_check = True, True
        elif a == "--no-check":
            opts.run_check = False
        elif a == "--max-check-atoms":
            opts.max_check_atoms = _int(a, value(a), 0)
        elif a == "--csv":
            opts.csv = True
        elif a == "--csv-raw":
            opts.csv_raw = True
        elif a == "--no-energy":
            opts.energy = False
        elif a.startswith("-") and a != "-":
            raise UsageError("unrecognized argument '%s'" % a)
        elif opts.input is None:
            opts.input = a
        else:
            raise UsageError("unexpected extra argument '%s' (one input file only)" % a)
        i += 1

    if opts.help or opts.list:
        return opts
    if half:
        raise UsageError("--half is not supported: the 1e-12 screening thresholds are "
                         "below the float16 range (use --double or --float)")
    if opts.csv and opts.csv_raw:
        raise UsageError("--csv and --csv-raw cannot be used together")
    if opts.input is None:
        raise UsageError("missing input file, e.g. ../../data/hartree-fock/he64")
    return opts


def help_text(prog):
    return """\
Hartree-Fock (Triton)

Usage: %s INPUT [OPTIONS]

  INPUT                  input file (ngauss, natoms, exponent/coefficient pairs,
                         then x y z per atom), e.g. ../../data/hartree-fock/he64

Options:
  -n, --iters NUM        timed iterations after 1 untimed run (default %d)
      --block-size N     quartets per program, a power of two (default %d)
      --double           float64 (default)
      --float            float32
      --legacy-sqrt      evaluate the two square roots in float32 (as legacy sqrtf)
      --check            always run the float64 CPU reference check
      --no-check         skip the CPU reference check
      --max-check-atoms N  check only inputs with at most N atoms (default %d)
      --csv              summary CSV on stdout (report on stderr)
      --csv-raw          one CSV row per timed iteration on stdout (report on stderr)
      --no-energy        skip board-power measurement (default: measured)
      --device INDEX     device to run on; the energy profiler uses the same index (default 0)
      --list             list devices (index: name) and exit
  -h, --help             show this help and exit

There is no --half: the 1e-12 screening thresholds are below the float16 range.
The last precision flag wins; the last of --check / --no-check wins.
Options take the form '--flag value'. Only the untimed run's Fock matrix is
checked, because the timed runs keep accumulating into it.

Environment (all optional):
  GPU_NAME               override the reported device name
  GPU_DRIVER             override the reported driver version
  DSLBENCH_ENERGY_LIB    path to libkernelenergy.so (default: ../../dependencies)
""" % (prog, DEFAULT_NUM_TIMES, DEFAULT_BLOCK_SIZE, DEFAULT_MAX_CHECK_ATOMS)
