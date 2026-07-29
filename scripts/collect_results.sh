#!/bin/bash
# Collect eval success rates into one arm x task x checkpoint table.
#
#   scripts/collect_results.sh                 # every stage_c/eval_* dir (skips _archive*)
#   scripts/collect_results.sh <eval_dir>...   # explicit eval dirs
#
# Reads <EVAL_OUT>/ckpt<N>[_seed100]/results.json (written by deploy_avaloha.py) and
# falls back to the deploy.log "[deploy] DONE [<tag>]: <n>/<episodes>" line for evals
# that predate results.json. The "40k n=200" column sums ckpt40000 + ckpt40000_seed100.
set -uo pipefail

RUNS=/lustre/meat124/groot_runs
DIRS=("$@")
if [ ${#DIRS[@]} -eq 0 ]; then
  for d in "$RUNS"/stage_c/eval_*/; do
    case "$(basename "$d")" in *_archive*|*.partial*) continue ;; esac
    DIRS+=("${d%/}")
  done
fi

# read_counts <ckpt_dir> -> "succ/total" or "" (results.json first, deploy.log fallback)
read_counts() {
  local d="$1"
  if [ -f "$d/results.json" ]; then
    local s t
    s=$(grep -oE '"successes": [0-9]+' "$d/results.json" | grep -oE '[0-9]+')
    t=$(grep -oE '"episodes_run": [0-9]+' "$d/results.json" | grep -oE '[0-9]+')
    [ -n "$s" ] && [ -n "$t" ] && { echo "$s/$t"; return; }
  fi
  if [ -f "$d/deploy.log" ]; then
    grep -oE "DONE \[[^]]*\]: [0-9]+/[0-9]+" "$d/deploy.log" | tail -1 | grep -oE "[0-9]+/[0-9]+"
  fi
}

printf "%-28s" "arm_task"
for ck in 10000 20000 30000 40000; do printf "%10s" "${ck}"; done
printf "%12s\n" "40k n=200"
for d in "${DIRS[@]}"; do
  name=$(basename "$d"); name=${name#eval_}
  printf "%-28s" "$name"
  for ck in 10000 20000 30000 40000; do
    res=$(read_counts "$d/ckpt$ck")
    # run_ntp_pipeline.sh evals use the single-model interface: results.json sits
    # at the eval-dir ROOT (final checkpoint only) — show it in the 40k column.
    if [ -z "$res" ] && [ "$ck" = "40000" ]; then res=$(read_counts "$d"); fi
    printf "%10s" "${res:--}"
  done
  # n=200 significance column: ckpt40000 (seeds 0-99) + ckpt40000_seed100 (100-199)
  a=$(read_counts "$d/ckpt40000"); b=$(read_counts "$d/ckpt40000_seed100")
  if [ -n "$a" ] && [ -n "$b" ]; then
    printf "%12s" "$(( ${a%/*} + ${b%/*} ))/$(( ${a#*/} + ${b#*/} ))"
  else
    printf "%12s" "-"
  fi
  echo
done
