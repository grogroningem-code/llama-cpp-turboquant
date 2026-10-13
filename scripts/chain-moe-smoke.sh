#!/bin/bash
# SPEC-CHAIN-MOE smoke: MoE chain-2 must draft real tokens (AC1).
# Usage: scripts/chain-moe-smoke.sh <moe-gguf> [prompt-file]
# Exit 0 = GREEN, nonzero = RED with reason on stdout.
MOE=${1:?usage: chain-moe-smoke.sh <moe-gguf> [prompt-file]}
PROMPT=${2:-models/corpus-smoke.txt}
CLI=build-cuda/bin/Release/llama-cli.exe
LOG=chain-moe-smoke.log

"$CLI" -m "$MOE" -c 4096 -ngl 99 -f "$PROMPT" -n 64 --temp 0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-chain 2 -v \
  > "$LOG" 2>&1
RC=$?

if grep -q "invalid token" "$LOG"; then echo "RED: invalid token in output"; exit 1; fi
if grep -q "out of range" "$LOG"; then echo "RED: guard fired, chain branch broken"; exit 1; fi
if [ $RC -ne 0 ]; then echo "RED: cli exit $RC"; exit 1; fi
ACC=$(grep -o "draft_n_accepted[^,]*" "$LOG" | tail -1)
echo "ACCEPT: $ACC"
TOK=$(grep -c "chain candidate" "$LOG")
echo "CHAIN_CANDIDATES: $TOK"
[ "$TOK" -gt 0 ] || { echo "RED: no chain candidates drafted"; exit 1; }
echo "GREEN"
