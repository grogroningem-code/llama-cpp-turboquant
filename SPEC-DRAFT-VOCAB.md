# SPEC: Draft-vocab restricted MTP head (turboquant) - v4 after Opus R3

## Users
- Turboquant users running Qwen3.8-27B (llama arch `qwen35`, shared with Qwen3.5/3.6 family) and Qwen3.6-35B-A3B (`qwen35moe`) with `--spec-type draft-mtp` on CUDA, including partial-offload laptops (8-16GB VRAM).

## Architecture (v2: fix R1 blockers 1-3)
- v1 uses a FIXED draft set only (file list, no per-step ring; ring deferred to v2). Graph shape is stable across steps, CUDA graph reuse preserved.
- File `draft_vocab_32k.bin` v2 format with header: magic `DV32`, version u32, `n_vocab` u32, count u32, tokenizer hash u64 (FNV-1a over vocab tokens), then count LE int32 ids. Loader requires magic, version, `n_vocab` match, hash match, nonzero count, count <= `n_vocab`, file size <= 1MB, exact file size (24 + 4*count, no trailing bytes), every id in `[0, n_vocab)`, no duplicates. Rejects otherwise, with all size/count checks running before any allocation.
- Offline `tools/vocab_freq` builds the file: counts token freq from user corpus via llama vocab tokenizer, writes top 32768 plus header.
- Change sites (R1 blocker 1 fix - the matmul lives in `src/`, not `common/`):
  - `src/models/qwen35.cpp` (non-chain MTP head, chain subset with `LLAMA_SPEC_CHAIN_SUB` default 32768): subset output projection over fixed id list on the single-head path. The chained path is untouched: combining the flag with any chained mode aborts loudly (single-head only in this slice; the position-to-id chain mapping is deferred). The map is the identity for the default first-rows subset, so flag-off numerics stay bit-identical (AC1). Flag-off graph is untouched: subset nodes are built only when the explicit flag is set.
  - `src/models/qwen35moe.cpp` (MTP head): same subset projection (no existing restriction there).
  - `src/llama-context.cpp` (logits copy): new subset-logits copy path with id mapping; full-size copy path untouched. Subset output never flows into the old `n_tokens x n_vocab` copy (memory-safety fix).
  - `src/llama-graph.h` (reuse compare): new graph key field for subset identity (hash of id list); changed subset never reuses a stale graph.
  - `common/speculative.cpp` (`common_speculative_impl_draft_mtp`, `drop_deferred_from`): owns the fixed set handle, per-sequence state; no per-step mutation in v1. The non-chained draft call at line 2147 (`common_sampler_sample`) switches to the new subset-aware draft-sample call reading through the `src/` subset getter.
  - Plumbing (R3 fix - getter lives in `src/` with the logits): new setter in `src/llama-ext.h` + `src/llama-context.cpp` taking the id list and targeting the draft context `ctx_dft` (same pattern as `llama_set_mtp_chain`), plus a subset-logits getter there defining row count and stride; new field in `src/llama-cparams.h` carrying the subset identity into graph build; `src/llama-graph.cpp` fills the id map as a graph input. The gathered head copy is created on the first setter call, once per model, after the tokenizer check; it lives on the same device as the source head tensor, overflow following the `--fit` spill. Mechanism (all precedent-backed, no new infra): buft via `ggml_backend_buffer_get_type(head->buffer)` with refusal on repack (extra) bufts, which have no `tensor_get` (`llama-adapter.cpp:295-350` lists them; the setter returns false and the caller aborts), `ggml_dup_tensor` with `ne={n_embd,n_sub}` + `ggml_backend_alloc_ctx_tensors_from_buft` (`llama-adapter.cpp:371-391`), fill via sync `ggml_backend_tensor_get` + CPU row-pick + `ggml_backend_tensor_set` (`ggml-backend.h:92-93`), row stride `nb[1]`, assert `n_embd` is a whole number of blocks for quantized rows, buffer usage `WEIGHTS` so the scheduler keeps the subset matmul local. `common/` keeps only its own copy of the id list (no getter needed for that).
  - `common/sampling.cpp` (R3 fix): the subset-aware draft-sample call lives here but reads through the `src/` subset getter (row count and stride defined there). The verify sampler is untouched. (The rejected alternative was a full-size -inf buffer, which leaves the sampler unchanged but keeps the full matmul and its cost; v1 takes the subset matmul plus sampler work because the speedup lives there.)
  - Precedence: `--spec-draft-vocab` combined with any chained MTP mode aborts loudly (single-head only in this slice); the `LLAMA_SPEC_CHAIN_SUB` default only applies when the flag is absent. Record both values in the bench.
  - `common/spec-draft-vocab.*` (new): loader plus set handle. `common/arg.cpp`: flags. `common/CMakeLists.txt`, `tools/CMakeLists.txt`, `tests/CMakeLists.txt`: build wiring (R1 item 11).
- Logits convention: subset-sized logits plus explicit subset-position-to-token-id map. Draft path consumes subset positions via the map; verify path uses full vocab and is untouched.
- Existing baseline (R1 blocker 2 fix): `qwen35.cpp` chained path already restricts to first `LLAMA_SPEC_CHAIN_SUB` rows (default 32768, env override, on by default); `qwen35moe.cpp` has no restriction. AC1 flag-off means behavior equal to the unmodified binary on the same prompt/seed. Flag-on runs are single-head only; chained flag-on is out of scope for this slice.
- Softmax note: subset softmax raises draft confidence vs full vocab; `p_min` behavior recorded in bench, not retuned in v1.
- Flags (default OFF): `--spec-draft-vocab <path>`, `--spec-draft-extra` RESERVED for v2 (parser rejects nonzero in v1 with a clear error).
- Rewind (R1 item 6, v2-ready semantics): one handle per sequence (parallel slots), only target-accepted tokens may enter a future ring, entries carry positions, overflow evictions not restored, hook at `drop_deferred_from`. A rewind bug costs acceptance rate only, never correctness.
- Gather cost (R1 item 5): no per-step `ggml_get_rows` on the quantized head in v1. The subset projection is a first-class graph node over a load-time gathered quantized copy; its VRAM cost goes into the AC2 bench record.

## Pinned versions
- Base: `TheTom/llama-cpp-turboquant` branch `feature/turboquant-kv-cache`, shallow clone 2026-10-12.
- CUDA path only for this slice. Targets sm_86/sm_89/sm_120. Test host here is RTX 4070 Laptop 8GB sm_89 (not a 4090).
- KV cache work (turbo2/3/4/5, arena) untouched.

## Security and privacy
- Draft vocab file is local token-id list only, no text. No network. No model upload. Loader bounds-checks count and id range, rejects oversized files (> 1MB).

## Non-goals
- No v1 ring/extras. No persistent kernel, no DeltaNet replay, no Q6_K relayout, no Vulkan kernels, no prefill change, no auto-enable, no `p_min` retune.

## Acceptance criteria
- AC1: Flag off equals unmodified binary (same prompt/seed, temp 0 identical tokens; temp > 0 fixed-seed distribution match).
- AC2: Flag on runs on 8GB partial offload (`--fit on`), no crash; record VRAM delta of gathered head, `graph splits`, loaded `n_ctx`.
- AC3: Draft candidates subset of draft set; verify logits identical to flag off (model test through graph, not fake vocab).
- AC4: Rewind/parallel-slot semantics per Architecture (unit + slot test; v1 fixed set trivially stable, test proves no cross-slot leak).
- AC5: Exactness by construction (R2 fix - drafts are meant to differ, so only target outputs are compared): temp 0 final tokens identical flag on/off; temp > 0 fixed-seed final accepted-token sequences match flag off over recorded prompts. Draft streams themselves are expected to differ.
- AC6: Bench thresholds over 3+ runs: code-prompt tg >= 1.03x baseline, prose tg >= 0.99x baseline. Report pp/tg, acceptance, splits. BLOCKED with reason if no model in workspace; never fabricate.
- AC7: Fresh-context Opus-medium review (Claude sub CLI, never OpenRouter) reports no BLOCKER. Manual user merge.

## AC to task map
- AC1 -> T1.4 (flag-off parity test vs unmodified binary)
- AC2 -> T3.2 (server `--fit on` bench with VRAM/splits/n_ctx)
- AC3 -> T2.1 (graph model test). Deviation recorded R2: no committed graph test (no small MTP weights in tests); AC3 held by construction (candidates only from the installed list, target context never armed) plus live AC5 final-token match on 4B plus the parity subset-vs-full harness case.
- AC4 -> T1.3 (slot/rewind unit test)
- AC5 -> T2.2 (exactness test per above)
- AC6 -> T3.2/T3.3 (threshold bench)
- AC7 -> T0/T2.3 (Opus gates + manual merge)
