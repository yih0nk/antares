#!/usr/bin/env bash
# Run the whole bench pipeline on simulated servos, no hardware needed.
#
#   bench/rehearse.sh [dir] [evaluations]
#
# Writes the CSVs two made-up SG92Rs would log (A stiffer and heavier than
# B, A again with three coins on the stick), then runs bench/pipeline.sh on
# them. Use it to check a laptop end to end before the real recordings.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dir=${1:-$(mktemp -d -t antares-bench-XXXX)}
evals=${2:-300}
mkdir -p "$dir"
a=(--set kp=2.2 --set armature=0.0005 --set frictionloss=0.01 --set forcerange=0.2)
b=(--set kp=1.0 --set armature=0.0002 --set kv=0.03 --set forcerange=0.3)
for r in 1 2 3; do
  antares bench simulate "$dir/servo_a_run$r.csv" "${a[@]}" --seed "$r"
  antares bench simulate "$dir/servo_b_run$r.csv" "${b[@]}" --pot-scale 0.30 --pot-offset -55 --seed "$((r + 10))"
done
# three coins (5.7 g each) taped 10 cm out add about 1.7e-4 kg m^2
antares bench simulate "$dir/servo_a_coin_run1.csv" --set kp=2.2 --set armature=0.00067 --set frictionloss=0.01 \
  --set forcerange=0.2 --seed 21
"$here/pipeline.sh" "$dir" "$evals"
