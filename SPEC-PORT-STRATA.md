# SPEC-T10: MoE offload speed batch (upstream #29599, #29887; Strata-style prompt buffers)

Base: branch `candidate` (ffbd22d55). Work on a new branch `t10-moe-offload` from it.
Do not push. Follow AGENTS.md gates (spec review, build, test-backend-ops, A/B rules).

Why: Strata (https://github.com/Niko1221/Strata, MIT, engine v0.1.44.1) runs
Qwen3.8-Flash-Next Coder IQ1_M on a 16 GB card about 8x faster at prompt reading and
2.4x faster at generation (no MTP) than this fork with `-ncmoe 30`, same model file.
Most of the gap is where the experts run. This batch brings the fork closer, in an order
where each step is measured on its own.

## T10.0 - turbo K/V cache types crash qwen4exp (Qwen3.8-Flash-Next)

Bug: `-ctk turbo5 -ctv turbo4` (and turbo6/turbo5) aborts on the first prompt:
`ggml-cpu/ops.cpp:5102: unsupported GET_ROWS source CPU#cache_idx_k_l3 (view) (view)#0:
type = 51 (turbo5), ne = [128, 256, 1, 1]`. `-ctk q8_0 -ctv turbo4` works.
Cause: `src/llama-kv-cache-msa.cpp:46` creates the MSA indexer cache (`kv_idx`) with the
same `type_k`/`type_v` as the main cache, and the indexer reads its keys with GET_ROWS on
the CPU. The CPU GET_ROWS switch (ops.cpp ~5040-5100) has no TURBO cases, although
`ggml.c` type traits give TURBO4/5/6 a `to_float` (dequantize_row_turbo*_0) and
`ggml_compute_forward_get_rows_q` uses `to_float`.
Fix, in this order:
1. Check whether `dequantize_row_turbo{4,5,6}_0` returns values in the original domain
   or the rotated (WHT) domain. The indexer multiplies these keys with unrotated queries,
   so rotated output would give silently wrong indexer scores. State the answer with the
   code lines in the report.
2. If original domain: add TURBO4_0/5_0/6_0 (and TURBO2/3 if they have `to_float`) to the
   `get_rows_q` case list. Also check the CUDA GET_ROWS path for the same types, since
   the indexer cache may sit on the GPU when its layer is offloaded.
3. If rotated domain, or step 2 is not enough: give the indexer cache its own type -
   q8_0 when the main type is a turbo type, otherwise unchanged - in llama-kv-cache-msa.cpp.
Check: Flash-Next (qwen4exp) loads and answers with `-ctk turbo5 -ctv turbo4` and with
`-ctk turbo6 -ctv turbo5`; greedy output on a short prompt compared against
`-ctk q8_0 -ctv turbo4` (report first differing token if not identical); test-backend-ops
GET_ROWS for the turbo types if cases exist or are added. Other models' default path unchanged.

## T10.1 - port #29599 `llama_prefetch_rows` (merged 2026-10-01) + #30532

What: MADV_WILLNEED / Windows PrefetchVirtualMemory prefetch of the PLE (per-layer
n-gram embedding) rows that the next batch will read, for qwen4exp and gemma4.
Upstream master has the Windows path (src/llama-mmap.cpp, PrefetchVirtualMemory);
#30532 split llama_prefetch() into merge and prefetch steps. Port both.
Tree state: absent (no `llama_prefetch_rows`); TENSOR_READ_LAZY present.
Check: Windows build compiles and the prefetch call is reached (log line or counter).
Greedy output byte-identical with and without.

## T10.2 - port #29887 upstream MoE expert cache (merged 2026-10-07)

What: `--moe-cache-mib N`, an LRU cache in VRAM for host-resident experts, batches
<= 32 tokens, misses uploaded. Upstream measured Flash-Next Q4_0 on an RTX 4090:
25.0 -> 40.7 t/s at 77% hit rate.
Conflict: this fork already has its own `--moe-cache` (#364 in TheTom's fork:
per-tensor slots for host layers, decode-time slot contest, misses read in place over
PCIe; `ggml/src/ggml-cuda/moe-cache.cu`). Keep both, make them mutually exclusive
(error if both are requested), defaults unchanged. Rename the upstream flag only if
it collides; document the mapping.
Check: test-backend-ops MUL_MAT_ID cases pass with each cache on; greedy output equal
to the no-cache path on a small MoE model.

## T10.3 - Strata-style prompt buffers borrowed from the expert cache (design first)

Problem: in llama.cpp a bigger `-ub` (1024 -> 4096/8192) reserves a bigger compute
buffer for the whole run, so VRAM that could hold experts is lost permanently. With
experts on the host, every ubatch re-streams every expert, so prompt speed scales with
ubatch size.
Strata's answer (read its source; docs/DETAILS.md lines ~160-170 and ~326-340):
- the prompt path borrows slots from the GPU expert cache for its buffers and returns
  them after the prompt;
- `--prefill auto` picks the largest chunk whose buffers fit the borrowed slots
  (8,192 tokens in practice);
- the next layer's experts stream over PCIe while the current layer's attention runs;
- experts multiplied with quantized MMQ kernels, PLE block run for the whole chunk.
There is no upstream PR for the borrowing. This is a re-implementation, not a port.
Deliverable 1: DESIGN-T10.3.md - how the compute buffer for a large ubatch can be
taken from (and returned to) the fork's moe-cache slab, using the existing hook that
surrenders cache VRAM under allocator pressure (moe-cache.cuh); what happens at decode
(buffer shrinks back to the decode size, slots refilled); failure modes (OOM mid-prompt,
CUDA graphs, multi-GPU = refuse). Get it reviewed per AGENTS.md before coding.
Deliverable 2: implementation behind an opt-in flag, default path byte-identical.
Credit Strata (MIT) in the commit message for anything taken from its source.

## T10.4 (optional, after T10.3) - #28414 lookahead expert H2D prefetch

Closed upstream unmerged (2026-10-14): uploads the experts for split i+2 on a second
stream while split i computes, prefill only. Reported -11% to -22% time to first token.
Only if T10.3 leaves PCIe idle time.

## T11 - upstream qwen4exp sync (after T10.0-T10.2; separate commits, one per PR)

All merged upstream. Port in this order, each with build + test-backend-ops for touched ops
+ greedy output check on Flash-Next against the previous commit:
1. #29751 "llama: fix qwen4exp" (merged 2026-10-01). First determine whether this fork has
   the bug it fixes (state evidence). Touches hybrid memory code this fork reworked; expect
   conflicts, resolve keeping fork behavior, list each resolved hunk in the report.
2. #27978 CUDA: any expert count uses the fast mm_ids_helper path.
3. #28770 CUDA: sparse FA for qwen4. Check it with turbo K/V types (see T10.0); if turbo
   types are unsupported there, fall back to the existing path, never abort.
4. #29901 cuda: tiled lightning indexer for 4 heads.
5. #28896 qwen4exp: rms_norm + mul fusion.
Not now: #29963 (open, multi-GPU pipelining) - revisit after it merges.

## T10.5 - fixes from the T10.1/T10.2 review (on t10-moe-offload, before merge)

1. qwen4exp `per_layer_tok_embd` is created with flags 0 (src/models/qwen4exp.cpp ~180),
   upstream uses TENSOR_READ_LAZY. Prefetch only runs for lazy tensors, so T10.1 never runs
   for Flash-Next. Match upstream; confirm the prefetch call is reached (log line).
2. tests/test-llama-archs.cpp "MoE cache" row: on dense archs llama-moe-cache.cpp throws
   "MoE cache requires a MoE model", context creation returns nullptr and the test exits on
   the first arch. Match upstream behavior for dense archs (skip or no error), and make the
   row actually exercise the cache (host-expert overrides, as upstream does) or remove the
   claim in the comment that it exercises the copy callback.
3. C API: moe_cache_mode AUTO + moe_cache_size > 0 is downgraded with an info log
   (llama-context.cpp ~4828). Spec says error if both are requested: reject it.
4. Log after the fork cache is disabled still prints "requested=provider resolved=provider"
   (llama-context.cpp ~1197): print the resolved state.
Check: build; test-llama-archs runs through all archs on CUDA; test-arg-parser.

## Report

Write REPORT-T10.md: per task, what changed (files), build result, tests run with
output, what was NOT tested. Speed measurements on the target machine are done
separately by the maintainer; do not invent numbers.
