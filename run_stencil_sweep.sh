#!/usr/bin/env bash
# Run the common seven-point-stencil grid sweep across every implementation.
# Summary/energy and raw/no-energy results deliberately use separate CSV files
# because their row schemas differ.

set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash run_stencil_sweep.sh --backend <cuda|hip> [OPTIONS]

Required:
  --backend cuda|hip       GPU backend to run

Options:
  --mode summary|raw|both  Output campaign to run (default: both)
  --precision 32|64|both   Precision campaign to run (default: both)
  --repetitions N          Independent repetitions (default: 3)
  --output-dir PATH        Result directory (default: results/stencil/BACKEND)
  --large-iters N          Summary iterations (default: 100)
  --raw-iters N            Iterations in raw/no-energy runs (default: 100)
  --device N               GPU device index (default: 0)
  --skip-energy-build      Use the existing dependencies/libkernelenergy.so
  -h, --help               Show this help

Sweep:
  FP32/FP64: 128^3, 256^3, 512^3, 1024^3
  Block:      64 x 4 x 1 (fixed by the benchmark)

Files:
  stencil-BACKEND-summary-energy.csv  one row per configuration
  stencil-BACKEND-raw-no-energy.csv   one row per timed iteration

Every aggregate row includes a repetition column. Individual CSV and log files
are retained under runs/rep-NN. Existing aggregate files are never overwritten.
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

backend=""
mode="both"
precision="both"
repetitions=3
output_dir=""
large_iters=100
raw_iters=100
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
    --precision)
      [[ $# -ge 2 ]] || die "--precision needs 32, 64, or both"
      precision="$2"
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
    --large-iters)
      [[ $# -ge 2 ]] || die "--large-iters needs a value"
      large_iters="$2"
      shift 2
      ;;
    --raw-iters)
      [[ $# -ge 2 ]] || die "--raw-iters needs a value"
      raw_iters="$2"
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

case "$precision" in
  32|64|both) ;;
  *) die "unknown precision '$precision' (expected 32, 64, or both)" ;;
esac

positive_integer "$repetitions" || die "--repetitions must be an integer >= 1"
positive_integer "$large_iters" || die "--large-iters must be an integer >= 1"
positive_integer "$raw_iters" || die "--raw-iters must be an integer >= 1"
[[ "$device" =~ ^[0-9]+$ ]] || die "--device must be a non-negative integer"

if [[ -z "$output_dir" ]]; then
  output_dir="$ROOT/results/stencil/$backend"
elif [[ "$output_dir" != /* ]]; then
  output_dir="$ROOT/$output_dir"
fi

summary_file="$output_dir/stencil-${backend}-summary-energy.csv"
raw_file="$output_dir/stencil-${backend}-raw-no-energy.csv"
failures_file="$output_dir/failures.log"
info_file="$output_dir/campaign-info.txt"

SUMMARY_HEADER="implementation,backend,device,precision,nx,ny,nz,block_x,block_y,block_z,num_times,routine,bw_gbs,min_s,max_s,avg_s,energy_j_per_iter,avg_power_w"
RAW_HEADER="implementation,backend,device,precision,nx,ny,nz,block_x,block_y,block_z,routine,iteration,time_s,bw_gbs"

if [[ "$mode" == summary || "$mode" == both ]]; then
  [[ ! -e "$summary_file" ]] || die "output already exists: $summary_file (choose another --output-dir)"
fi
if [[ "$mode" == raw || "$mode" == both ]]; then
  [[ ! -e "$raw_file" ]] || die "output already exists: $raw_file (choose another --output-dir)"
fi

mkdir -p "$output_dir/runs"
: > "$failures_file"

{
  echo "date=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "backend=$backend"
  echo "mode=$mode"
  echo "precision=$precision"
  echo "repetitions=$repetitions"
  echo "device=$device"
  echo "block=64,4,1"
  echo "large_sizes=128,256,512,1024"
  echo "large_iters=$large_iters"
  echo "raw_iters=$raw_iters"
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

if [[ "$mode" == summary || "$mode" == both ]]; then
  printf '%s,repetition\n' "$SUMMARY_HEADER" > "$summary_file"
fi
if [[ "$mode" == raw || "$mode" == both ]]; then
  printf '%s,repetition\n' "$RAW_HEADER" > "$raw_file"
fi

implementations=(native triton mojo cubecl kernelabstractions)
large_sizes=(128 256 512 1024)
case "$precision" in
  32) precisions=(float32) ;;
  64) precisions=(float64) ;;
  both) precisions=(float32 float64) ;;
esac
failure_count=0

append_rows() {
  local source_file="$1"
  local destination_file="$2"
  local expected_header="$3"
  local repetition="$4"
  local actual_header line

  IFS= read -r actual_header < "$source_file" || return 1
  [[ "$actual_header" == "$expected_header" ]] || return 1

  while IFS= read -r line; do
    [[ -n "$line" ]] && printf '%s,%s\n' "$line" "$repetition" >> "$destination_file"
  done < <(tail -n +2 "$source_file")
}

run_one() {
  local run_mode="$1"
  local repetition="$2"
  local implementation="$3"
  local precision="$4"
  local size="$5"
  local iterations="$6"
  local rep_dir="$output_dir/runs/rep-$(printf '%02d' "$repetition")"
  local stem="${implementation}-${precision}-${size}-${run_mode}"
  local csv_file="$rep_dir/${stem}.csv"
  local log_file="$rep_dir/${stem}.log"
  local aggregate expected_header
  local precision_flag
  local -a command args

  mkdir -p "$rep_dir"

  case "$precision" in
    float32) precision_flag=--float ;;
    float64) precision_flag=--double ;;
    *) die "internal error: unknown precision '$precision'" ;;
  esac

  args=("$precision_flag" -s "$size" -n "$iterations" --block-size 64,4,1 --device "$device")
  if [[ "$run_mode" == summary ]]; then
    args+=(--csv)
    aggregate="$summary_file"
    expected_header="$SUMMARY_HEADER"
  else
    args+=(--csv-raw --no-energy)
    aggregate="$raw_file"
    expected_header="$RAW_HEADER"
  fi

  case "$implementation" in
    native)
      command=(bash "$ROOT/$backend/seven-point-stencil/run.sh")
      ;;
    triton)
      command=(bash "$ROOT/triton/seven-point-stencil/run.sh" "$backend")
      ;;
    mojo)
      command=(bash "$ROOT/Mojo/seven-point-stencil/run.sh" "$backend")
      ;;
    cubecl)
      command=(bash "$ROOT/cubecl/seven-point-stencil/run.sh" "$backend")
      ;;
    kernelabstractions)
      command=(bash "$ROOT/kernelabstractions/seven-point-stencil/run.sh" "$backend")
      ;;
    *)
      die "internal error: unknown implementation '$implementation'"
      ;;
  esac

  printf 'rep %d: %-18s %-7s %4s^3 %-7s n=%s\n' \
    "$repetition" "$implementation" "$precision" "$size" "$run_mode" "$iterations" >&2
  {
    printf 'command:'
    printf ' %q' "${command[@]}" "${args[@]}"
    printf '\n'
  } > "$log_file"

  if ! "${command[@]}" "${args[@]}" > "$csv_file" 2>> "$log_file"; then
    echo "rep=$repetition mode=$run_mode implementation=$implementation precision=$precision size=$size reason=command_failed log=$log_file" \
      >> "$failures_file"
    failure_count=$((failure_count + 1))
    return
  fi

  if ! append_rows "$csv_file" "$aggregate" "$expected_header" "$repetition"; then
    echo "rep=$repetition mode=$run_mode implementation=$implementation precision=$precision size=$size reason=invalid_csv log=$log_file" \
      >> "$failures_file"
    failure_count=$((failure_count + 1))
  fi
}

run_mode() {
  local run_mode="$1"
  local repetition offset index implementation precision size iterations
  local -a ordered

  for ((repetition = 1; repetition <= repetitions; repetition++)); do
    # Rotate implementation order each repetition to reduce ordering bias.
    ordered=()
    offset=$(( (repetition - 1) % ${#implementations[@]} ))
    for ((index = 0; index < ${#implementations[@]}; index++)); do
      ordered+=("${implementations[$(( (index + offset) % ${#implementations[@]} ))]}")
    done

    for precision in "${precisions[@]}"; do
      for size in "${large_sizes[@]}"; do
        if [[ "$run_mode" == summary ]]; then iterations="$large_iters"; else iterations="$raw_iters"; fi
        for implementation in "${ordered[@]}"; do
          run_one "$run_mode" "$repetition" "$implementation" "$precision" "$size" "$iterations"
        done
      done
    done
  done
}

case "$mode" in
  summary) run_mode summary ;;
  raw)     run_mode raw ;;
  both)
    run_mode summary
    run_mode raw
    ;;
esac

echo "Results written under $output_dir" >&2
if [[ "$failure_count" -gt 0 ]]; then
  echo "$failure_count run(s) failed; see $failures_file" >&2
  exit 1
fi

rm -f "$failures_file"
echo "All runs completed successfully" >&2
