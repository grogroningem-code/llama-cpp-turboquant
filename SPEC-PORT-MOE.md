# SPEC-T8: MoE MTP speed batch (upstream #27621, #28549, #29184)
(Opus spec review: 5 BLOCKERs - integrated below. Order 27621->29184 holds;
28549 is INDEPENDENT of both, do it anytime.)

## 0. Tree-newer-than-PR adjustments (Opus B1/B2)

- Our `mul_mat_vec_q_moe` is NEWER than 27621's pre-image: has_fusion (different
  meaning) + has_clamp, separate gate_ptr/gate_ids/act_ids, clamp min/max.
  Expert-cache callers at mmvq.cu:1411/1502/1510 use it. Hand-merge required:
  merged kernel takes fusion struct (biases, NVFP4 scales, gate fn) AND keeps
  cache callers working (they build fusion{} -> shared_up null -> safe).
- TRAP: relaxing only the max-batch gates routes multi-token fused calls to
  mmvq.cu:946 with no gate -> SILENTLY WRONG. Gate relaxation + kernel merge
  land together or not at all.
- Our tree has NO SWIGLU_CLAMP: drop `ggml_cuda_op_swiglu_clamp_single`,
  `glu_limit`, clamp case from the port. `valid_glu_ops` (:2760) excludes it.
- 27621 hunk1 confirmed NOT applied (:2827 still `!= 1`).

## 1. #27621 - CUDA MoE fusion to specdec 2-8 tokens

What: BS=1 MoE fusions extended to 2-8 tokens (MTP verify shape).
`mul_mat_vec_q_moe` folds gate proj, biases, NVFP4 scales, GLU into up-proj;
topk-moe fuses 8 rows/block with alias-safe __syncthreads.

Tree state: PARTIAL. `get_mmvq_mmid_max_batch` present (via #27978 port);
`TOPK_MOE_ROWS_PER_BLOCK` absent; fusion struct has gate/glu_op; single-op
helpers (`ggml_cuda_op_silu_single` etc.) present in mmvq.cu.

Port map (diff: cache/scratch/pr27621.diff):
- ggml-cuda.cu hunk1: `dst->ne[2] != 1` -> `> get_mmvq_mmid_max_batch(...)`.
  Check our exact pre-image first (may already match post-#27978).
- AS-BUILT FIX (Opus impl-review BLOCKER 2026-10-15, opus-review-t8port.log):
  the hunk1 relax admits multi-token MUL_MAT_ID at ALL 6 _q call sites, but
  the multi-token MoE kernel hardcodes silu with no bias/scale. So every _q
  site now carries a multi-token-MoE skip guard: plain gate/up (SWIGLU-only),
  bias+GLU (skip if bias or non-SWIGLU), plain-GLU (skip if non-SWIGLU),
  scale and bias-only (skip any MUL_MAT_ID multi-token). Fail-closed
  GGML_ASSERT in the mmvq multi-token dispatch (fused payload must be
  gate-only SWIGLU) + dst->ne[2]==1 assert in the shared_up block.
  Regression net: no_fuse test member + 6 negative cases (bias, GEGLU,
  lane-scale) checking numerics via the unfused path.
- ggml-cuda.cu hunk2: topk-moe alias exception rows<=8 via `logits_may_alias`.
- mmvq.cu: `has_fusion` template param + fusion arg on `mul_mat_vec_q_moe`,
  tmp_gate accumulate/reduce, GLU switch, launch-side has_fusion select,
  pass `fusion` at moe_launch call site, relax `GGML_ASSERT(!ids || ne2==1)`
  to `<= max_batch`.
- topk-moe.cu/.cuh: `TOPK_MOE_ROWS_PER_BLOCK 8`, launch bounds, __syncthreads,
  rows_per_block.
- tests: add m_batch {2,4,8} fusion cases ONLY for `use_id && quantized(type)`
  AND m_batch <= that type's max (Q3_K 5, IQ3_S 6, Q2_K/IQ2_S/IQ3_XXS/MXFP4 7 -
  mmvq.cu:143); plain MUL_MAT multi-token + F32/F16-MoE can never fuse here.
  Add topk rows {32,8}x4, {32,8}x8, {32,9}x8 (no dupes with ours).
  Each new case must show its fusion counter rising (required_fusion), not just
  result match.

## 2. #28549 - CUDA graph for MTP draft (merged 2026-09-16, +42/-11, 2 files)

What: draft batches (output) vs prefill/catch-up (no output) get separate
`llm_graph_result` arenas so they stop sharing one CUDA-graph cache key
(repeated re-capture on every MTP alternation).

Tree state: ABSENT (`gf_res_prev` single unique_ptr at llama-context.h:432).
Anchors differ from PR (ours: .cpp 1034/1144/1634/2359/3482/4490) - map each:
sched_reserve init, memory_update reset, process_ubatch reuse, reserve reset,
opt_epoch_iter reset.

Port: mechanical - array<ptr,2>, `get_gf_res_prev()` keyed `n_outputs>0`,
`gf_res_prev_active` guard in can_reuse path, resets at all 6 sites (the 5
upstream ones at ours 1144/1634/2359/3482/4490 PLUS our phase-arena
repartition at :1034 `sched.reset(); gf_res_prev->reset();` - needs the same
both-graphs loop + active=nullptr, else a freed-scheduler graph gets reused).
No new flag. Risk: graph-reuse stall if active-tracking wrong - smoke both
temp-0 (AC5 text) and temp-0.6 (speed) on 4B + MoE. Touches every backend via
llama-context -> add a Vulkan smoke run. Prove the actual goal: n_reused /
re-capture count before/after with specdec on, not just speed.

## 3. #29184 - fuse shared experts into MMVQ (merged 2026-10-02, +174/-10, 4 files)

What: routed MUL_MAT_ID pair + shared-expert MUL_MAT pair + 2 GLUs fuse into
one mmvq launch (extra y-block for shared). Upstream: +2.26% mb1 on
Qwen3.5-35B-A3B-Q8_0/RTX Pro 6000.

Tree state: ABSENT. `ggml_can_fuse_subgraph_ext` PRESENT (ggml.c:7761).
Needs: `ggml_cuda_should_fuse_mul_mat` 3-arg (present, :2672 - verify sig),
`ggml_get_glu_op`, `ggml_get_op_params_f32`, `concurrent_events`.

BLOCKER (Opus B3): our tree has NO `add_alloc_dep` anywhere in ggml/ and our
hook is `graph_optimize(backend, cgraph)` with no params. Upstream added both
in a ggml-backend/allocator change that must be ported FIRST, else the fused
kernel can write shared output into reused memory while reading input from it.
Also our `ggml_backend_cuda_graph_optimize` returns early unless
GGML_CUDA_GRAPH_OPT=1 + CUDA graphs on (:5998/:6005) - the node rotation must
run BEFORE those returns or the i/i+3 match never fires. CHECK what
GRAPH_OPT defaults to in our runs; 29184 may need it on for any effect.
Decision: investigate hook cost first; if large, land 27621+28549, defer 29184
with reason - do NOT half-port (no try_fuse without alloc deps).

Port map:
- common.cuh: shared_up/gate/dst + stride to host+device fusion structs.
- ggml-cuda.cu: `ggml_cuda_match_shared_expert` + try_fuse hook (i,i+3) +
  graph_optimize node rotation + alloc deps.
- mmvq.cu: shared_expert y-block in both `mul_mat_vec_q` and
  `mul_mat_vec_q_moe` (NOTE: moe hunk assumes #27621 landed - order matters),
  launch grid +1 channel, `ggml_cuda_mul_mat_vec_q` assert block.
- tests: new `test_mul_mat_id_shared` struct + Q4_K/Q8_0 x n{1,2,3,4,8,9}.

Risk: Tiel uses IQ4_XS weights, upstream tested Q8_0 - matcher requires same
type pair, graceful no-match if layout differs (fail-closed by design).
Shared expert in qwen35moe: verify graph shape (routed pair + shared pair)
via smoke, not by reading code alone.

## Acceptance

- AC1: CUDA build green (llama-cli/llama-server/test-backend-ops), no new warnings.
  DONE 2026-10-15 (commit 95c2e3c51; rebuilds T8B3/B4/B5 exit 0). Warning
  scan: zero warnings introduced by the port; recompile of mmvq.cu surfaced
  two PRE-EXISTING #177-D (unused `gate`, `gate_kbx_offset`, lines ~731/763
  - outside all port hunks, present since before T8, masked before by
  incremental builds). Left untouched.
- AC2: new/changed backend-ops cases pass on CUDA0 (MUL_MAT_ID_SHARED,
  m_batch fusion, topk rows) - 0 skipped-as-pass.
  DONE 2026-10-15: MUL_MAT_VEC_FUSION / TOPK_MOE / MUL_MAT_ID / ROLL all exit
  0 on CUDA0, incl. 8 no_fuse negative cases (FUSION2:0).
- AC3: temp-0 AC5 text identical pre/post on 4B (graph surgery in 28549 must
  not change outputs); MoE smoke no -1, coherent.
  PARTIAL 2026-10-15: MTP-off temp-0 4B BYTE-IDENTICAL (base forward proven
  bit-identical) + ppl neutral (1.6032->1.5887, overlapping bars) + MoE
  smoke no -1 coherent + post deterministic across repeats. DEVIATION: MTP-on
  temp-0 diverges pre/post (draft-path last-bit reorder, same benign class
  as T7 GDN divergence) - accepted, documented in REBASE-AB-RESULTS.md.
- AC4: server A/B (launcher flags, MBPP, acceptance): MoE tg must not regress;
  report 27621+29184 uplift if any (expect low single-digit % on 4070).
  DONE 2026-10-15: MoE neutral (fusion on/off 31.9 vs 32.8, pre/post within
  env drift); 4B neutral-or-parity. Uplift: NONE measurable on 8GB offload
  (bandwidth-bound; Qwen3-MoE has no shared experts so 29184 cannot fire).
  Full numbers in docs/REBASE-AB-RESULTS.md (T8 entries).
- AC5: Opus impl review, no BLOCKERs.
  DONE 2026-10-15: 1 BLOCKER found, fixed, re-review OK
  (opus-review-t8port.log + opus-review-t8fix.log); all concerns closed.

NOT in scope: Vulkan ports of these kernels (CUDA-only batch); 27B; chain.
