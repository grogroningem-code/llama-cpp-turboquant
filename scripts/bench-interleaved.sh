#!/usr/bin/env bash
# Interleaved flag on/off bench (kills the warmup-order confound of bench-prelim).
# usage: bench-interleaved.sh <cli> <model> <dv32> [n_runs_per_cell]
CLI="$1"; MODEL="$2"; DV32="$3"; N="${4:-4}"
PROSE="Explain why the sky appears blue during the day and red at sunset:"
CODE="Write a Python function that implements quicksort with type hints:"
omap() { if [ "$1" = on ]; then echo "--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-vocab $DV32"; else echo "--spec-type draft-mtp --spec-draft-n-max 2"; fi; }
pmap() { if [ "$1" = prose ]; then echo "$PROSE"; else echo "$CODE"; fi; }
i=0
for round in 1 2; do
    for cond in on off off on; do
        for p in code prose; do
            i=$((i+1))
            OUT=$("$CLI" -m "$MODEL" -p "$(pmap $p)" -n 128 --temp 0 -c 2048 -ngl 99 $(omap $cond) < /dev/null 2>&1 | grep -o 'Generation: [0-9.]* t/s' | tail -1)
            echo "$cond $p try$i: ${OUT:-NORUN}"
        done
    done
done
