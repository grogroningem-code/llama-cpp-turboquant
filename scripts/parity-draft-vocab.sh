#!/usr/bin/env bash
# Flag-off parity harness: the flag-off binary must behave exactly like the
# unmodified binary on the same prompt/seed.
#
# usage:
#   parity-draft-vocab.sh <new-cli> <base-cli> <model.gguf> [prompt] [draft_vocab.bin]
#
# [draft_vocab.bin] enables the subset-vs-full case (needs a DV32 built by
# llama-vocab-freq for the model's tokenizer).
#
# <new-cli>  = llama-cli built from THIS tree (flags default off, no
#              --spec-draft-vocab passed)
# <base-cli> = llama-cli built from a pristine worktree, e.g.:
#                git worktree add ../llama-cpp-turboquant-base HEAD
#                cmake -B ../llama-cpp-turboquant-base/build-cpu \
#                  -S ../llama-cpp-turboquant-base -DGGML_CUDA=OFF
#                cmake --build ../llama-cpp-turboquant-base/build-cpu \
#                  --config Release --target llama-cli -j 8
#
# Checks (both must diff clean):
#   1. temp 0, 64 tokens: byte-identical stdout
#   2. temp 0.7, fixed seed 42, 64 tokens: byte-identical stdout
# Perf/timing lines go to stderr and are ignored (stdout only is compared).
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <new-cli> <base-cli> <model.gguf> [prompt]" >&2
    exit 2
fi

NEW="$1"
BASE="$2"
MODEL="$3"
PROMPT="${4:-The quick brown fox jumps over the lazy dog. Explain why:}"

for b in "$NEW" "$BASE"; do
    if [ ! -x "$b" ]; then
        echo "parity: not executable: $b" >&2
        exit 2
    fi
done

OUT_NEW="$(mktemp)"
OUT_BASE="$(mktemp)"
ERR_NEW="$(mktemp)"
ERR_BASE="$(mktemp)"
trap 'rm -f "$OUT_NEW" "$OUT_BASE" "$ERR_NEW" "$ERR_BASE" "$ERR_NEW.counts" "$ERR_BASE.counts"' EXIT

run_case() {
    local desc="$1"; shift
    # stdout only is compared, minus the timing footer
    # ("[ Prompt: ... | Generation: ... ]" varies run to run)
    # stderr appends: the MTP accept check below covers both MTP cases
    "$NEW"  -m "$MODEL" -p "$PROMPT" "$@" 2>>"$ERR_NEW" | grep -v '^\[ Prompt: ' > "$OUT_NEW"
    "$BASE" -m "$MODEL" -p "$PROMPT" "$@" 2>>"$ERR_BASE" | grep -v '^\[ Prompt: ' > "$OUT_BASE"
    if cmp -s "$OUT_NEW" "$OUT_BASE"; then
        echo "parity PASS: $desc"
    else
        echo "parity FAIL: $desc (outputs differ)" >&2
        diff "$OUT_BASE" "$OUT_NEW" | head -20 >&2
        exit 1
    fi
}

run_case "temp 0 greedy"          -c 2048 -n 64 --temp 0
run_case "temp 0.7 fixed seed"   -c 2048 -n 64 --temp 0.7 --seed 42

# exercise the MTP draft path too (plain runs above never touch it).
# final tokens must match; in the flag-off MTP cases both binaries draft from
# the full head, so their accept-count lines must match as well.
: > "$ERR_NEW"; : > "$ERR_BASE"
run_case "draft-mtp temp 0"        -c 2048 -n 64 --temp 0 --spec-type draft-mtp --spec-draft-n-max 2
run_case "draft-mtp seed 42"       -c 2048 -n 64 --temp 0.7 --seed 42 --spec-type draft-mtp --spec-draft-n-max 2
for f in "$ERR_NEW" "$ERR_BASE"; do
    grep -i 'accept' "$f" | sort > "$f.counts" || true
done
if [ -s "$ERR_NEW.counts" ] || [ -s "$ERR_BASE.counts" ]; then
    if cmp -s "$ERR_NEW.counts" "$ERR_BASE.counts"; then
        echo "parity PASS: draft-mtp accept counts"
    else
        echo "parity FAIL: draft-mtp accept counts differ" >&2
        diff "$ERR_BASE.counts" "$ERR_NEW.counts" | head -20 >&2
        exit 1
    fi
else
    echo "parity: no accept lines emitted, count check skipped" >&2
fi

# flag-on vs flag-off inside the new binary: subset MTP must emit the
# same final tokens as full-head MTP. needs the DV32 as $5; skipped without it.
if [ "${5:-}" != "" ]; then
    DV32="$5"
    "$NEW" -m "$MODEL" -p "$PROMPT" -c 2048 -n 64 --temp 0 --spec-type draft-mtp --spec-draft-n-max 2 2>/dev/null | grep -v '^\[ Prompt: ' > "$OUT_BASE"
    "$NEW" -m "$MODEL" -p "$PROMPT" -c 2048 -n 64 --temp 0 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-vocab "$DV32" 2>/dev/null | grep -v '^\[ Prompt: ' > "$OUT_NEW"
    if cmp -s "$OUT_NEW" "$OUT_BASE"; then
        echo "parity PASS: draft-mtp subset-vs-full temp 0"
    else
        echo "parity FAIL: draft-mtp subset-vs-full temp 0 (outputs differ)" >&2
        diff "$OUT_BASE" "$OUT_NEW" | head -20 >&2
        exit 1
    fi
else
    echo "parity: no DV32 given as \$5, subset-vs-full case skipped" >&2
fi

echo "parity: all cases match (flag-off equals the unmodified binary)"
