#!/usr/bin/env bash
set -euo pipefail

WEAK1_NAME=${WEAK1_NAME:-"small"}
WEAK1_M0=${WEAK1_M0:-896}
WEAK1_N0=${WEAK1_N0:-896}
WEAK1_K0=${WEAK1_K0:-896}

WEAK2_NAME=${WEAK2_NAME:-"medium"}
WEAK2_M0=${WEAK2_M0:-1152}
WEAK2_N0=${WEAK2_N0:-1152}
WEAK2_K0=${WEAK2_K0:-1152}

P_LIST=${P_LIST:-"1 2 4 8"}
REPEAT=${REPEAT:-2}
EXE=./matrix_mul_omp
OUT_CSV=${OUT_CSV:-weak_results.csv}

command -v g++ >/dev/null || { echo "g++ not found"; exit 1; }
[ -x "$EXE" ] || g++ -O3 -march=native -fopenmp matrix_mul_omp.cpp -o matrix_mul_omp

RAW=weak_raw.csv
echo "SERIES,M,N,K,P,T" > "$RAW"

round16() { awk -v x="$1" 'BEGIN{printf "%d", int((x/16)+0.5)*16}'; }
scale_dim() { local base=$1 p=$2; local s; s=$(awk -v P="$p" 'BEGIN{printf "%.12f", exp(log(P)/3.0)}'); local out; out=$(awk -v b="$base" -v s="$s" 'BEGIN{printf "%.0f", b*s}'); round16 "$out"; }

run_point() {
  local SERIES=$1 M=$2 N=$3 K=$4 P=$5
  local T; T=$($EXE "$M" "$N" "$K" "$P")
  echo "$SERIES,$M,$N,$K,$P,$T" >> "$RAW"
}

if [ "$WEAK1_M0" -gt 0 ]; then
  for P in $P_LIST; do
    M=$(scale_dim "$WEAK1_M0" "$P"); N=$(scale_dim "$WEAK1_N0" "$P"); K=$(scale_dim "$WEAK1_K0" "$P")
    for r in $(seq 1 "$REPEAT"); do
      run_point "WEAK(${WEAK1_NAME})" "$M" "$N" "$K" "$P"
    done
  done
fi

if [ "$WEAK2_M0" -gt 0 ]; then
  for P in $P_LIST; do
    M=$(scale_dim "$WEAK2_M0" "$P"); N=$(scale_dim "$WEAK2_N0" "$P"); K=$(scale_dim "$WEAK2_K0" "$P")
    for r in $(seq 1 "$REPEAT"); do
      run_point "WEAK(${WEAK2_NAME})" "$M" "$N" "$K" "$P"
    done
  done
fi

python3 - <<PY
import pandas as pd, os
raw=pd.read_csv("weak_raw.csv")
agg=(raw.groupby(["SERIES","M","N","K","P"],as_index=False)["T"].median()
        .sort_values(["SERIES","P"]))
agg.to_csv(os.environ.get("OUT_CSV","weak_results.csv"),index=False)
print(os.environ.get("OUT_CSV","weak_results.csv"))
PY
