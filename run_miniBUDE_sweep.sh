#!/usr/bin/env bash
# Run the common miniBUDE precision campaign across every implementation.

set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="miniBUDE"

usage() {
  cat <<'EOF'
Usage:
  bash run_miniBUDE_sweep.sh --backend <cuda|hip> [OPTIONS]

Required:
  --backend cuda|hip       GPU backend to run

Options:
  --mode summary|raw|both  Output campaign to run (default: both)
  --precision 32|64|both   Precision campaign to run (default: both)
  --repetitions N          Independent repetitions (default: 3)
  --output-dir PATH        Result directory (default: results/miniBUDE/BACKEND)
  --summary-iters N        Summary timed iterations (default: 8)
  --raw-iters N            Raw timed iterations (default: 8)
  --device N               GPU device index (default: 0)
  --skip-energy-build      Use the existing dependencies/libkernelenergy.so
  -h, --help               Show this help

Campaign:
  Precisions: FP32 and FP64 by default
  Deck:      bm1
  Config:    PPWI 1, work-group 256, 2 warmups, 8 timed iterations

Files:
  miniBUDE-BACKEND-summary-energy.csv
  miniBUDE-BACKEND-raw-no-energy.csv

Each implementation writes a separate CSV and log under
runs/rep-NN/{summary,raw-no-energy}. Existing campaign output is never
overwritten. Campaign-level summary and raw CSV files use a union schema;
Triton-only tuning columns are empty for the other implementations.
EOF
}

die() { echo "error: $*" >&2; exit 2; }
positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

backend=""
mode="both"
precision="both"
repetitions=3
output_dir=""
device=0
build_energy=1
summary_iters=8
raw_iters=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) [[ $# -ge 2 ]] || die "--backend needs cuda or hip"; backend="$2"; shift 2 ;;
    --mode) [[ $# -ge 2 ]] || die "--mode needs summary, raw, or both"; mode="$2"; shift 2 ;;
    --precision) [[ $# -ge 2 ]] || die "--precision needs 32, 64, or both"; precision="$2"; shift 2 ;;
    --repetitions) [[ $# -ge 2 ]] || die "--repetitions needs a value"; repetitions="$2"; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || die "--output-dir needs a path"; output_dir="$2"; shift 2 ;;
    --summary-iters) [[ $# -ge 2 ]] || die "--summary-iters needs a value"; summary_iters="$2"; shift 2 ;;
    --raw-iters) [[ $# -ge 2 ]] || die "--raw-iters needs a value"; raw_iters="$2"; shift 2 ;;
    --device) [[ $# -ge 2 ]] || die "--device needs a value"; device="$2"; shift 2 ;;
    --skip-energy-build) build_energy=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (try --help)" ;;
  esac
done

case "$backend" in cuda|hip) ;; "") die "--backend is required (cuda or hip)" ;; *) die "unknown backend '$backend' (expected cuda or hip)" ;; esac
case "$mode" in summary|raw|both) ;; *) die "unknown mode '$mode' (expected summary, raw, or both)" ;; esac
case "$precision" in 32|64|both) ;; *) die "unknown precision '$precision' (expected 32, 64, or both)" ;; esac
positive_integer "$repetitions" || die "--repetitions must be an integer >= 1"
positive_integer "$summary_iters" || die "--summary-iters must be an integer >= 1"
positive_integer "$raw_iters" || die "--raw-iters must be an integer >= 1"
[[ "$device" =~ ^[0-9]+$ ]] || die "--device must be a non-negative integer"
command -v python3 >/dev/null 2>&1 || die "python3 is required to normalize miniBUDE CSV schemas"

if [[ -z "$output_dir" ]]; then
  output_dir="$ROOT/results/$BENCHMARK/$backend"
elif [[ "$output_dir" != /* ]]; then
  output_dir="$ROOT/$output_dir"
fi

info_file="$output_dir/campaign-info.txt"
failures_file="$output_dir/failures.log"
summary_file="$output_dir/miniBUDE-${backend}-summary-energy.csv"
raw_file="$output_dir/miniBUDE-${backend}-raw-no-energy.csv"
SUMMARY_HEADER="implementation,backend,device,precision,deck,natlig,natpro,nposes,ppwi,wgsize,groups,subgroups,num_stages,warmup,iterations,min_s,max_s,avg_s,stddev_s,gflops,gfinsts,ginteractions,launch_overhead_s,context_s,energy_j_per_iter,avg_power_w,ref_l2_rel,ref_max_rel,ref_max_ulp,deck_l2_rel,deck_beyond_tol"
SUMMARY_COMMON_HEADER="implementation,backend,device,precision,deck,natlig,natpro,nposes,ppwi,wgsize,groups,warmup,iterations,min_s,max_s,avg_s,stddev_s,gflops,gfinsts,ginteractions,launch_overhead_s,context_s,energy_j_per_iter,avg_power_w,ref_l2_rel,ref_max_rel,ref_max_ulp,deck_l2_rel,deck_beyond_tol"
RAW_HEADER="implementation,backend,device,precision,deck,natlig,natpro,nposes,ppwi,wgsize,groups,subgroups,num_stages,warmup,iteration,time_s"
RAW_COMMON_HEADER="implementation,backend,device,precision,deck,natlig,natpro,nposes,ppwi,wgsize,groups,warmup,iteration,time_s"
[[ ! -e "$info_file" && ! -e "$failures_file" && ! -e "$output_dir/runs" ]] || \
  die "campaign output already exists under $output_dir (choose another --output-dir)"
if [[ "$mode" == summary || "$mode" == both ]]; then
  [[ ! -e "$summary_file" ]] || die "output already exists: $summary_file"
fi
if [[ "$mode" == raw || "$mode" == both ]]; then
  [[ ! -e "$raw_file" ]] || die "output already exists: $raw_file"
fi
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
  echo "precision=$precision"
  echo "repetitions=$repetitions"
  echo "device=$device"
  echo "deck=bm1"
  echo "ppwi=1"
  echo "work_group=256"
  echo "warmup=2"
  echo "summary_iters=$summary_iters"
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
  if ! bash "$ROOT/dependencies/build_energy.sh" "--$backend" > "$output_dir/energy-build.log" 2>&1; then
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
case "$precision" in
  32) precisions=(float32) ;;
  64) precisions=(float64) ;;
  both) precisions=(float32 float64) ;;
esac
failure_count=0

append_normalized_rows() {
  local source_file="$1" destination_file="$2" union_header="$3" common_header="$4" repetition="$5"
  python3 - "$source_file" "$destination_file" "$union_header" "$common_header" "$repetition" <<'PY'
import csv
import sys

source, destination, union_text, common_text, repetition = sys.argv[1:]
union = union_text.split(",")
accepted = (union, common_text.split(","))

with open(source, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames not in accepted:
        raise SystemExit(1)
    rows = list(reader)

if not rows:
    raise SystemExit(1)

with open(destination, "a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=union + ["repetition"], lineterminator="\n")
    for row in rows:
        normalized = {field: row.get(field, "") for field in union}
        normalized["repetition"] = repetition
        writer.writerow(normalized)
PY
}

run_one() {
  local run_mode="$1" repetition="$2" implementation="$3" run_precision="$4"
  local mode_dir csv_file log_file precision_flag iterations aggregate union_header common_header
  local -a command args

  case "$run_precision" in
    float32) precision_flag=--float ;;
    float64) precision_flag=--double ;;
    *) die "internal error: unknown precision '$run_precision'" ;;
  esac

  if [[ "$run_mode" == summary ]]; then
    mode_dir="summary"
    iterations="$summary_iters"
    args=("$precision_flag" -i "$iterations" --device "$device" --csv)
    aggregate="$summary_file"
    union_header="$SUMMARY_HEADER"
    common_header="$SUMMARY_COMMON_HEADER"
  else
    mode_dir="raw-no-energy"
    iterations="$raw_iters"
    args=("$precision_flag" -i "$iterations" --device "$device" --csv-raw --no-energy)
    aggregate="$raw_file"
    union_header="$RAW_HEADER"
    common_header="$RAW_COMMON_HEADER"
  fi

  mode_dir="$output_dir/runs/rep-$(printf '%02d' "$repetition")/$mode_dir"
  mkdir -p "$mode_dir"
  csv_file="$mode_dir/${implementation}-${run_precision}.csv"
  log_file="$mode_dir/${implementation}-${run_precision}.log"

  case "$implementation" in
    native) command=(bash "$ROOT/$backend/$BENCHMARK/run.sh") ;;
    triton) command=(bash "$ROOT/triton/$BENCHMARK/run.sh" "$backend") ;;
    mojo) command=(bash "$ROOT/Mojo/$BENCHMARK/run.sh" "$backend") ;;
    cubecl) command=(bash "$ROOT/cubecl/$BENCHMARK/run.sh" "$backend") ;;
    kernelabstractions) command=(bash "$ROOT/kernelabstractions/$BENCHMARK/run.sh" "$backend") ;;
    *) die "internal error: unknown implementation '$implementation'" ;;
  esac

  printf 'rep %d: %-18s %-7s %-7s n=%s\n' "$repetition" "$implementation" "$run_precision" "$run_mode" "$iterations" >&2
  { printf 'command:'; printf ' %q' "${command[@]}" "${args[@]}"; printf '\n'; } > "$log_file"

  if ! "${command[@]}" "${args[@]}" > "$csv_file" 2>> "$log_file"; then
    echo "rep=$repetition mode=$run_mode implementation=$implementation precision=$run_precision reason=command_failed log=$log_file" >> "$failures_file"
    failure_count=$((failure_count + 1))
  elif ! append_normalized_rows "$csv_file" "$aggregate" "$union_header" "$common_header" "$repetition"; then
    echo "rep=$repetition mode=$run_mode implementation=$implementation precision=$run_precision reason=invalid_csv log=$log_file" >> "$failures_file"
    failure_count=$((failure_count + 1))
  fi
}

run_mode() {
  local run_mode="$1" repetition offset index implementation run_precision
  local -a ordered
  for ((repetition = 1; repetition <= repetitions; repetition++)); do
    ordered=()
    offset=$(( (repetition - 1) % ${#implementations[@]} ))
    for ((index = 0; index < ${#implementations[@]}; index++)); do
      ordered+=("${implementations[$(( (index + offset) % ${#implementations[@]} ))]}")
    done
    for run_precision in "${precisions[@]}"; do
      for implementation in "${ordered[@]}"; do
        run_one "$run_mode" "$repetition" "$implementation" "$run_precision"
      done
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
