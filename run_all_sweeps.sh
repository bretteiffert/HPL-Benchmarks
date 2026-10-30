#!/usr/bin/env bash
# Run the common comparison campaign for all four benchmarks.

set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash run_all_sweeps.sh --backend <cuda|hip> [OPTIONS]

Required:
  --backend cuda|hip       GPU backend to run

Options:
  --mode summary|raw|both  Output campaign to run (default: both)
  --repetitions N          Independent repetitions (default: 3)
  --output-dir PATH        Result directory (default: results/all/BACKEND)
  --device N               GPU device index (default: 0)
  --skip-energy-build      Use the existing dependencies/libkernelenergy.so
  -h, --help               Show this help

Runs BabelStream, seven-point stencil, Hartree-Fock, and miniBUDE. Each
benchmark writes to its own directory beneath the selected output directory.
The energy library is built once for the entire campaign.
EOF
}

die() { echo "error: $*" >&2; exit 2; }
positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

backend=""
mode="both"
repetitions=3
output_dir=""
device=0
build_energy=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      [[ $# -ge 2 ]] || die "--backend needs cuda or hip"
      backend="$2"
      shift 2
      ;;
    --mode)
      [[ $# -ge 2 ]] || die "--mode needs summary, raw, or both"
      mode="$2"
      shift 2
      ;;
    --repetitions)
      [[ $# -ge 2 ]] || die "--repetitions needs a value"
      repetitions="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir needs a path"
      output_dir="$2"
      shift 2
      ;;
    --device)
      [[ $# -ge 2 ]] || die "--device needs a value"
      device="$2"
      shift 2
      ;;
    --skip-energy-build)
      build_energy=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option '$1' (try --help)"
      ;;
  esac
done

case "$backend" in
  cuda|hip) ;;
  "") die "--backend is required (cuda or hip)" ;;
  *) die "unknown backend '$backend' (expected cuda or hip)" ;;
esac
case "$mode" in
  summary|raw|both) ;;
  *) die "unknown mode '$mode' (expected summary, raw, or both)" ;;
esac
positive_integer "$repetitions" || die "--repetitions must be an integer >= 1"
[[ "$device" =~ ^[0-9]+$ ]] || die "--device must be a non-negative integer"

if [[ -z "$output_dir" ]]; then
  output_dir="$ROOT/results/all/$backend"
elif [[ "$output_dir" != /* ]]; then
  output_dir="$ROOT/$output_dir"
fi

info_file="$output_dir/campaign-info.txt"
failures_file="$output_dir/failures.log"
[[ ! -e "$info_file" && ! -e "$failures_file" ]] || \
  die "campaign output already exists under $output_dir (choose another --output-dir)"
if [[ "$build_energy" -eq 1 && -e "$output_dir/energy-build.log" ]]; then
  die "output already exists: $output_dir/energy-build.log"
fi

benchmarks=(stencil babelstream hartree-fock miniBUDE)
scripts=(
  run_stencil_sweep.sh
  run_babelstream_sweep.sh
  run_hartree_fock_sweep.sh
  run_miniBUDE_sweep.sh
)

for benchmark in "${benchmarks[@]}"; do
  benchmark_dir="$output_dir/$benchmark"
  if [[ -e "$benchmark_dir/campaign-info.txt" || -e "$benchmark_dir/failures.log" || -e "$benchmark_dir/runs" ]]; then
    die "campaign output already exists under $benchmark_dir (choose another --output-dir)"
  fi
done

mkdir -p "$output_dir"
: > "$failures_file"

{
  echo "date=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "benchmarks=stencil,babelstream,hartree-fock,miniBUDE"
  echo "backend=$backend"
  echo "mode=$mode"
  echo "repetitions=$repetitions"
  echo "device=$device"
  echo "precision=float32"
  if command -v git >/dev/null 2>&1; then
    echo "git_revision=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status_begin"
    git -C "$ROOT" status --short 2>/dev/null || true
    echo "git_status_end"
  fi
} > "$info_file"

if [[ "$build_energy" -eq 1 ]]; then
  echo "Building the $backend energy library" >&2
  if ! bash "$ROOT/dependencies/build_energy.sh" "--$backend" \
      > "$output_dir/energy-build.log" 2>&1; then
    cat "$output_dir/energy-build.log" >&2
    die "energy library build failed"
  fi
fi

failure_count=0
for index in "${!benchmarks[@]}"; do
  benchmark="${benchmarks[$index]}"
  script="${scripts[$index]}"
  benchmark_dir="$output_dir/$benchmark"

  echo "Running $benchmark campaign" >&2
  if ! bash "$ROOT/$script" \
      --backend "$backend" \
      --mode "$mode" \
      --repetitions "$repetitions" \
      --output-dir "$benchmark_dir" \
      --device "$device" \
      --skip-energy-build; then
    echo "benchmark=$benchmark reason=campaign_failed output=$benchmark_dir" >> "$failures_file"
    failure_count=$((failure_count + 1))
  fi
done

echo "All benchmark results written under $output_dir" >&2
if [[ "$failure_count" -gt 0 ]]; then
  echo "$failure_count benchmark campaign(s) failed; see $failures_file" >&2
  exit 1
fi

rm -f "$failures_file"
echo "All benchmark campaigns completed successfully" >&2
