#!/usr/bin/env bash
set -euo pipefail

S_M=${S_M:-1024}
S_N=${S_N:-1024}
S_K=${S_K:-1024}
P_LIST=${P_LIST:-"1 2 4 8 16"}
REPEAT=${REPEAT:-3}
EXE=./matrix_mul_omp
OUT_CSV=${OUT_CSV:-strong_results.csv}

command -v g++ >/dev/null || { echo "g++ not found"; exit 1; }
[ -x "$EXE" ] || g++ -O3 -march=native -fopenmp matrix_mul_omp.cpp -o matrix_mul_omp

RAW=strong_raw.csv
echo "M,N,K,P,T" > "$RAW"

run_point() {
  local M=$1 N=$2 K=$3 P=$4
  local T; T=$($EXE "$M" "$N" "$K" "$P")
  echo "$M,$N,$K,$P,$T" >> "$RAW"
}

for P in $P_LIST; do
  for r in $(seq 1 "$REPEAT"); do
    run_point "$S_M" "$S_N" "$S_K" "$P"
  done
done

python3 - <<PY
import pandas as pd, os
raw=pd.read_csv("strong_raw.csv")
agg=raw.groupby(["M","N","K","P"],as_index=False)["T"].median().sort_values(["M","N","K","P"])
agg.to_csv(os.environ.get("OUT_CSV","strong_results.csv"),index=False)
print(os.environ.get("OUT_CSV","strong_results.csv"))
PY
