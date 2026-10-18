#!/usr/bin/env bash
# Import, calibrate and demo the servo bench recordings in one go.
#
#   bench/pipeline.sh <dir> [evaluations]
#
# <dir> holds the capture.py CSVs, named as in bench/README.md:
#   servo_a_run1.csv servo_a_run2.csv servo_a_run3.csv   (servo A, untouched)
#   servo_b_run1.csv servo_b_run2.csv servo_b_run3.csv   (servo B, optional)
#   servo_a_coin_run1.csv ...                            (servo A + payload, optional)
# Runs 1-2 are fitted, run 3 is held out for the demo. Datasets and locks
# are written inside <dir> (locks in <dir>/locks), so nothing lands in the
# repository. Each stage prints what it did; rerunning redoes everything.
set -euo pipefail
dir=$(cd "${1:?usage: bench/pipeline.sh <dir> [evaluations]}" && pwd)
evals=${2:-300}
cd "$dir"
step() { printf '\n== %s\n' "$*"; }

servos=()
for s in a b; do
  [ -f "servo_${s}_run1.csv" ] || continue
  fit=(servo_${s}_run1.csv)
  [ -f "servo_${s}_run2.csv" ] && fit+=(servo_${s}_run2.csv)
  step "servo $s: import ${fit[*]}"
  antares bench import "${fit[@]}" --out "ds_${s}"
  if [ -f "servo_${s}_run3.csv" ]; then
    antares bench import "servo_${s}_run3.csv" --out "ds_${s}_test"
  else
    echo "no servo_${s}_run3.csv: the demo scores the fitted runs instead of a held-out one"
    rm -rf "ds_${s}_test" && cp -r "ds_${s}" "ds_${s}_test"
  fi
  step "servo $s: calibrate ($evals evaluations)"
  antares calibrate "ds_${s}" --name "servo_${s}" --max-evaluations "$evals" | tail -8
  servos+=("$s")
done
[ ${#servos[@]} -gt 0 ] || { echo "error: no servo_a_run1.csv or servo_b_run1.csv in $dir" >&2; exit 2; }

demo=()
for s in "${servos[@]}"; do demo+=(--servo "$s=ds_${s}_test" --lock "$s=locks/servo_${s}.lock.json"); done
coin=(servo_a_coin_run*.csv)
if [ -f "${coin[0]}" ] && [[ " ${servos[*]} " == *" a "* ]]; then
  step "servo a + payload: import ${coin[*]}"
  antares bench import "${coin[@]}" --out ds_a_coin
  demo+=(--payload a=ds_a_coin --fresh ds_a_test)
fi

step "demo"
antares bench demo "${demo[@]}"
printf '\nlive view: antares bench live --lock %s/locks/servo_a.lock.json   (add --replay %s/servo_a_run3.csv without the Arduino)\n' "$dir" "$dir"
