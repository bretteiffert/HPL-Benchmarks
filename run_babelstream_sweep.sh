#!/usr/bin/env bash
# Run the common FP32 BabelStream campaign across every implementation.

set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="babelstream"

usage() {
  cat <<'EOF'
Usage:
  bash run_babelstream_sweep.sh --backend <cuda|hip> [OPTIONS]

Required:
  --backend cuda|hip       GPU backend to run

Options:
  --mode summary|raw|both  Output campaign to run (default: both)
  --repetitions N          Independent repetitions (default: 3)
  --output-dir PATH        Result directory (default: results/babelstream/BACKEND)
  --device N               GPU device index (default: 0)
  --skip-energy-build      Use the existing dependencies/libkernelenergy.so
  -h, --help               Show this help

Campaign:
  Precision: FP32
  Sizes:     65,536 / 1,048,576 / 33,554,432 elements
  Iterations: 1,000 per size

Each implementation writes a separate CSV and log under
runs/rep-NN/{summary,raw-no-energy}. Existing campaign output is never
overwritten.
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
    --backend) [[ $# -ge 2 ]] || die "--backend needs cuda or hip"; backend="$2"; shift 2 ;;
    --mode) [[ $# -ge 2 ]] || die "--mode needs summary, raw, or both"; mode="$2"; shift 2 ;;
    --repetitions) [[ $# -ge 2 ]] || die "--repetitions needs a value"; repetitions="$2"; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || die "--output-dir needs a path"; output_dir="$2"; shift 2 ;;
    --device) [[ $# -ge 2 ]] || die "--device needs a value"; device="$2"; shift 2 ;;
    --skip-energy-build) build_energy=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (try --help)" ;;
  esac
done

case "$backend" in cuda|hip) ;; "") die "--backend is required (cuda or hip)" ;; *) die "unknown backend '$backend' (expected cuda or hip)" ;; esac
case "$mode" in summary|raw|both) ;; *) die "unknown mode '$mode' (expected summary, raw, or both)" ;; esac
positive_integer "$repetitions" || die "--repetitions must be an integer >= 1"
[[ "$device" =~ ^[0-9]+$ ]] || die "--device must be a non-negative integer"

if [[ -z "$output_dir" ]]; then
  output_dir="$ROOT/results/$BENCHMARK/$backend"
elif [[ "$output_dir" != /* ]]; then
  output_dir="$ROOT/$output_dir"
fi

info_file="$output_dir/campaign-info.txt"
failures_file="$output_dir/failures.log"
[[ ! -e "$info_file" && ! -e "$failures_file" && ! -e "$output_dir/runs" ]] || \
  die "campaign output already exists under $output_dir (choose another --output-dir)"
if [[ "$build_energy" -eq 1 && -e "$output_dir/energy-build.log" ]]; then
  die "output already exists: $output_dir/energy-build.log"
fi

mkdir -p "$output_dir/runs"
: > "$failures_file"

{
  echo "date=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "benchmark=$BENCHMARK"
  echo "backend=$backend"
  echo "mode=$mode"
  echo "repetitions=$repetitions"
  echo "device=$device"
  echo "precision=float32"
  echo "sizes=65536,1048576,33554432"
  echo "iterations=1000"
  if command -v git >/dev/null 2>&1; then
    echo "git_revision=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status_begin"
    git -C "$ROOT" status --short 2>/dev/null || true
    echo "git_status_end"
  fi
} > "$info_file"

if [[ "$build_energy" -eq 1 ]]; then
  echo "Building the $backend energy library" >&2
  if ! bash "$ROOT/dependencies/build_energy.sh" "--$backend" > "$output_dir/energy-build.log" 2>&1; then
    cat "$output_dir/energy-build.log" >&2
    die "energy library build failed"
  fi
fi

implementations=(native triton mojo cubecl kernelabstractions)
failure_count=0

valid_csv() {
  local csv_file="$1" actual_header line_count
  IFS= read -r actual_header < "$csv_file" || return 1
  case "$actual_header" in
    implementation,backend,device,precision,*) ;;
    *) return 1 ;;
  esac
  line_count=$(wc -l < "$csv_file")
  [[ "$line_count" -ge 2 ]]
}

run_one() {
  local run_mode="$1" repetition="$2" implementation="$3"
  local mode_dir csv_file log_file
  local -a command args

  if [[ "$run_mode" == summary ]]; then
    mode_dir="summary"
    args=(--float --device "$device" --csv)
  else
    mode_dir="raw-no-energy"
    args=(--float --device "$device" --csv-raw --no-energy)
  fi

  mode_dir="$output_dir/runs/rep-$(printf '%02d' "$repetition")/$mode_dir"
  mkdir -p "$mode_dir"
  csv_file="$mode_dir/$implementation.csv"
  log_file="$mode_dir/$implementation.log"

  case "$implementation" in
    native) command=(bash "$ROOT/$backend/$BENCHMARK/run.sh") ;;
    triton) command=(bash "$ROOT/triton/$BENCHMARK/run.sh" "$backend") ;;
    mojo) command=(bash "$ROOT/Mojo/$BENCHMARK/run.sh" "$backend") ;;
    cubecl) command=(bash "$ROOT/cubecl/$BENCHMARK/run.sh" "$backend") ;;
    kernelabstractions) command=(bash "$ROOT/kernelabstractions/$BENCHMARK/run.sh" "$backend") ;;
    *) die "internal error: unknown implementation '$implementation'" ;;
  esac

  printf 'rep %d: %-18s %-7s\n' "$repetition" "$implementation" "$run_mode" >&2
  { printf 'command:'; printf ' %q' "${command[@]}" "${args[@]}"; printf '\n'; } > "$log_file"

  if ! "${command[@]}" "${args[@]}" > "$csv_file" 2>> "$log_file"; then
    echo "rep=$repetition mode=$run_mode implementation=$implementation reason=command_failed log=$log_file" >> "$failures_file"
    failure_count=$((failure_count + 1))
  elif ! valid_csv "$csv_file"; then
    echo "rep=$repetition mode=$run_mode implementation=$implementation reason=invalid_csv log=$log_file" >> "$failures_file"
    failure_count=$((failure_count + 1))
  fi
}

run_mode() {
  local run_mode="$1" repetition offset index implementation
  local -a ordered
  for ((repetition = 1; repetition <= repetitions; repetition++)); do
    ordered=()
    offset=$(( (repetition - 1) % ${#implementations[@]} ))
    for ((index = 0; index < ${#implementations[@]}; index++)); do
      ordered+=("${implementations[$(( (index + offset) % ${#implementations[@]} ))]}")
    done
    for implementation in "${ordered[@]}"; do
      run_one "$run_mode" "$repetition" "$implementation"
    done
  done
}

case "$mode" in
  summary) run_mode summary ;;
  raw) run_mode raw ;;
  both) run_mode summary; run_mode raw ;;
esac

echo "Results written under $output_dir" >&2
if [[ "$failure_count" -gt 0 ]]; then
  echo "$failure_count run(s) failed; see $failures_file" >&2
  exit 1
fi
rm -f "$failures_file"
echo "All runs completed successfully" >&2
