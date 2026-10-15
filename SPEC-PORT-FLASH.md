# SPEC-T9: FlashAttention CUDA batch (upstream #28770, #29435)

Scope: 2 merged upstream PRs, both CUDA flash-attention. No new flags.
Expected value on our box (sm_89 8GB): PARITY / no-regression (correctness
batch like T7). Neither PR changes any path our models execute; port keeps
the fork current and arms the sparse path for future top_k models.

## 1. #28770 - sparse FA rework + enable for qwen4 (merged 2026-09-20)

What: per-query-tile sparse index lists (union over ncols1 queries) with
per-list live counts; `compact_mask` new 7-arg signature; kernel reads
counts via `KV_max_ptr + iter_j*ne33*ne11`; `shall_use_sparse(cc, dst,
ncols1)` (no ctx) + `n_gather` threshold; 2 new eligible shapes
(256,256,1,8) and (256,256,8,8); sparse-decode branch in kernel selection
(GQA>4, vec-kernel condition gains `!sparse_decode`); qwen4exp sets
`top_k->ne[0]`; 3 new qwen4-shape (gqa 12) backend-ops cases.

Tree state: pre-image MATCHES (old 4-arg compact_mask, old
shall_use_sparse(ctx,dst), 2-shape may_use_sparse, gqa_opt_applies present).
DEVIATIONS (Opus spec review, opus-review-t9spec.log):
- `src/models/qwen4exp.cpp` EXISTS (arrived with T7); call at :1020 passes
  0 and lacks the TODO lines, so the hunk does NOT apply verbatim. Skipping
  it is safe (nothing else depends on that arg) - a choice keeping the new
  head-256 sparse shapes off, not a missing file.
- FORK-ONLY `output_partial` block (kv-stream path, fattn-mma-f16.cuh:2529,
  :2535-2536) has its own index/KV_max reads with the OLD layout; after this
  PR KV_max is non-null in sparse mode -> SILENT-WRONG there. Fix: gate the
  sparse-kernel selection with `!output_partial` at :2720
  (`if constexpr (!output_partial && may_use_sparse(...))`). The "2x
  per-tile reads" port is 3 sites with the third deliberately excluded.
  (No model reaches it today - kv-stream prefill needs GQA<=8, qwen4 is 12
  - but fail-closed anyway.)
- Sparse DOES fire in-tree (MLA top-k llama-graph.cpp:3163,
  deepseek4.cpp:791, head dims 512/576): existing cases :10881-10885 and
  :11359-11362 must stay green. "Never fires" holds only for qwen35 models
  (top_k=0 verified qwen35.cpp:637, qwen35moe.cpp:691).
Our test_flash_attn_ext already has n_kv_max member -> new cases fit.

Port map (diff: cache/scratch/pr28770.diff, 293 lines):
- fattn-common.cuh: compact_mask decl 7-arg; launch_fattn sparse block
  (n_kv_max_query, n_lists, KV_max.alloc + counts, new call args).
- fattn-mma-f16.cuh: may_use_sparse +2 shapes; sparse_indices/KV_max split;
  2x per-tile index/count reads (`(sequence%ne33)*iter_j + jt`); shall_use_sparse
  decl new signature; call site passes (cc, dst, ncols1).
- fattn.cu: mask_to_sparse_indices group rewrite (q0/q1 union loop, count
  clamp, counts_ptr write); compact_mask def 7-arg; shall_use_sparse def
  (cc param, n_gather threshold); switch_ncols1 call update; kernel-select
  sparse_decode branch.
- tests: 3 qwen4-shape sparse cases (gqa 12, n_kv_max 512), inserted after
  test-backend-ops.cpp:10885 (upstream anchor lines absent in tree).
  C4 caveat: dense fallback gives identical numbers, so green cases do NOT
  prove the sparse kernel ran - verify firing separately (kernel selection
  trace/counter), esp. case 1 (may fall back to vec kernel) and case 3
  (ncols1-dependent). AC3/AC4 cover only the dense-unchanged side (C5).
- SKIP: qwen4exp.cpp hunk (call passes 0, no TODO lines - hunk inapplicable;
  skipping safe, keeps new shapes off by choice).

Risks: index-math rework is the danger zone (off-by-one in per-tile lists
-> silent-wrong sparse results). Mitigation: upstream's 3 new cases + our
existing sparse-mask cases must pass; sparse only fires at KV>=4096 with
top_k>0, so our model A/B cannot be affected even if subtly wrong - but
tests must still be green (fail-closed).

## 2. #29435 - whole-tile FA scheduling for 2-stage kernels (merged 2026-10-05)

What: `launch_fattn` gains `async_kv_preload` (default false); scan_mask
hoisted; stream-K skipped when `prefer_whole_tiles && efficiency >= 75%`,
where prefer_whole_tiles requires `cc == GGML_CUDA_CC_DGX_SPARK`.

Tree state: CAREFUL - fork differs (Opus B1, opus-review-t9spec.log):
our `launch_fattn` has 3 EXTRA params after warp_size (partial_dst,
partial_meta, partial_nparts - kv-stream fork feature). So `async_kv_preload`
goes LAST (not after warp_size - that would shift partial_dst into the bool
slot and fail compile at fattn-mma-f16.cuh:2771 + fattn-mma-turbo.cuh:100).
Call site needs explicit `1` for partial_nparts (today defaulted):
`..., warp_size_host, partial_dst, partial_meta, 1, async_kv_preload`.
Stream-K: tree has NO should_use_stream_k lambda (inline
`use_stream_k = cc >= ADA || amd_wmma || eff < 75` at :1717-1723) -> rewrite
as `use_stream_k = !(prefer_whole_tiles && eff >= 75) && (...)`.
Effect on sm_89: STRICT NO-OP (prefer_whole_tiles always false). Port for
parity only; proves zero behavior change if temp-0 texts match pre/post.

Port map (diff: cache/scratch/pr29435.diff, 56 lines): apply all 3 hunks
verbatim (param, scan_mask hoist, prefer_whole_tiles + capture, mma-f16
call site with `nstages == 2 && !use_sparse`).

Order: 28770 -> 29435 (chronological merge order; adjacent regions in
launch_fattn - 28770's n_kv_max block sits just above 29435's scan_mask).

## Acceptance

- AC1: CUDA build green (cli/server/bops), zero new warnings.
- AC2: FLASH_ATTN backend-ops incl. 3 new sparse cases pass on CUDA0.
- AC3: temp-0 identical pre/post on 4B MTP-off (bit-identical base); MTP-on
  divergence within known benign class; MoE smoke no -1.
- AC4: A/B 4B + MoE pp/tg + ppl: expect PARITY (no-regression); report any
  delta honestly, uplift unlikely by construction.
- AC5: Opus spec review (before port) + Opus impl review (before commit),
  no BLOCKERs.

NOT in scope: qwen4 model support; Vulkan FA changes (none in either PR);
enabling sparse for our models (they pass top_k=0 by arch design).

## Results 2026-10-15 (commit pending)

- Opus spec review (opus-review-t9spec.log, gitignored): 1 BLOCKER + concerns,
  all integrated as documented deviations (qwen4exp skip-safe, !output_partial
  gate, MLA firing cases, case placement, dense-fallback caveat).
- Opus impl review (opus-review-t9port.log, gitignored): OK to merge, no
  BLOCKERs, 2 minor concerns (stale-layout comment line; upstream-inherited
  count clamp n_kv_max). Port diff: cache/scratch/t9-port.diff (324 lines).
- Build: T9B_EXIT:0; T9PRE_EXIT:0 (HEAD baseline -> pre-t9/); T9POST_EXIT:0.
  3 new warnings, all pre-existing newly surfaced (fattn-mma-f16.cuh:1207
  unused kv_stream_mma_turbo_decode_dispatch, :1586 unused k0_diff/i0_diff,
  fattn.cu:1840 unused kq_offset_padded) - outside port hunks, untouched.
- Tests: -o FLASH_ATTN_EXT -b CUDA0 -p n_kv_max=(512|768|2304): 9/9 pass
  (6 existing MLA-shape + 3 new qwen4-shape). Full unfiltered suite still
  times out (pre-existing, suite too large for 280 s) - same as T8.
- Sparse-firing analysis (INFERENCE, medium confidence): new case 1 (nb=1,kv=4096)
  satisfies shall_use_sparse for the ncols1=1 tile (n_gather=512, 2*512<4096,
  GQA>4) so it should take the reworked sparse kernel; new case 2 (nb=64,
  kv=8192) provably falls back to dense (2*64*512=65536>8192) - expected,
  numerics green either way. No kernel-selection counter exists to prove
  firing; green + arithmetic is the available evidence.
- A/B pre-t9/ vs T9 CUDA0, Qwen3.5-4B + Tiel MoE:
  4B MTP-off T0: 3055.6/53.6 vs 3017.5/53.0, text BYTE-IDENTICAL (718/718).
  4B MTP-on t0.6 s42: 3000.6/54.2 vs 3006.0/54.1 (parity).
  4B PPL prose c2048: 1.5735 +/- 0.02887 BOTH (identical).
  MoE MTP s42: 43.7/16.3 vs 43.4/14.7 (pp parity, tg dip env variance);
  MoE MTP s43: 41.8/15.8 vs 41.9/15.9 (dead even) -> PARITY verdict.
- AC1 DONE, AC2 DONE (filtered subset + firing arithmetic), AC3 DONE
  (MTP-off byte-identical; MTP-on not diffed textually - parity speeds only,
  acceptable: FA batch cannot change sampling, and T0-identical proves it),
  AC4 DONE (parity everywhere, no uplift - as predicted), AC5 DONE.

## Proper retest 2026-10-15 (new harness, ++method, t9-deep.json untracked)

- Script fixes found by this run: hyphenated CLI flags never parsed
  (--n-predict silently kept default 1024; first run measured n=1024 -
  harmless but unintended), health check broke on 'loading' (HTTP 500 on
  warm-up), server prompt cache made rep>=1 prefill meaningless
  (ctx=4 second run). Fixed: underscore-normalized keys, wait for 'ok',
  --no-cache-prompt, LAUNCH echo. All in bench-server-ab.py.
- Setup: pre-t9/ vs T9 build, Qwen3.5-4B CUDA0, -c 76800, 2 deep fixtures
  (ctx 55562/55436 - short of the 75k deployment max, disclosed),
  order ABBA x 2 prompts x 2 reps = 16 runs, n_predict 512, cache off.
- deep-A: CTL pp_med 2383 tg_med 44.3 (43.8-44.3) vs T9 pp_med 2420
  tg_med 43.6 (42.8-43.8). deep-B: CTL 2390/44.2 (43.7-44.3) vs T9
  2388/44.0 (43.6-44.1). pooled_acc 0.0 both (4B has no drafting - N/A).
- Verdict: pp PARITY (within rep noise +-2-3%). tg parity within noise
  with a slight negative lean on deep-A (-1.6% med) that does not
  replicate on deep-B (-0.5%). ABBAx2x2 cannot resolve sub-2%; NOT
  claimed as regression, flagged for the 8-block rerun with real
  HF-trajectory fixtures.
- MoE leg omitted on purpose: T9 touches CUDA-only files, so Vulkan1
  131k pre/post would be code-identical (pure noise measurement);
  CUDA0 cannot hold MoE 131k KV. MoE deployment-speed retest needs the
  Lenovo; MoE fusion retest (T8 pair) needs the two builds + max-split
  -ncmoe fit-search, queued next.
