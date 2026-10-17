# DESIGN-T10.3: Strata-style prompt buffers borrowed from the fork MoE-cache slab

Status: REVISED per outside review REVIEW-T10.3.md (2026-10-17). Sections 8-9
adjudicate every finding with file:line evidence and supersede the
corresponding parts of sections 2/4/5/7. v1 implements section 9.

Reference: SPEC-PORT-STRATA.md T10.3. Strata (MIT, engine v0.1.44.1) behavior is taken
from the SPEC summary (docs/DETAILS.md ~160-170, ~326-340); the Strata source was not
fetched (no extract backend in this environment), so anything attributed to Strata below
needs confirmation against its source before coding.

## 1. Problem (verified in this tree)

- `llama_context::sched_reserve()` (src/llama-context.cpp:1143) builds the reserve graph
  with `n_tokens = min(n_ctx, n_ubatch)` and `graph_max_nodes()` (same file, ~3478)
  scales with `n_tokens` for MoE arches (`n_tokens * 40`). Compute buffers are then
  reserved per backend for that graph. So `-ub 8192` permanently reserves prefill-sized
  compute buffers: VRAM that could hold experts is lost for the whole run. This matches
  the SPEC premise.
- The fork MoE slot cache holds `cudaMalloc`'d slabs: `moe_cache_pool { slab, n_slots,
  expert_size }`, `slot_count * expert_size` bytes per (expert_size, wtype) shape
  (ggml/src/ggml-cuda/moe-cache.cu:1307-1338), capped by `budget_limit - scratch_reserve`
  and tracked in `device.allocated_bytes`. Slots are tracked free/used via `free_slots`
  + `map` + LRU (`moe_cache_pool`, same file:118-131).
- The surrender hook EXISTS (SPEC premise holds, verified 2026-10-17):
  `ggml_moe_cache_trim(int device)` (moe-cache.cuh:14-15, moe-cache.cu:3342) is called as
  a last resort before OOM from both CUDA pool paths (ggml-cuda.cu:520 legacy pool, :622
  VMM pool). BUT it is eviction, not lending: it marks the device dead, cancels the
  queue, waits for inflight work, and frees every pool. A borrow/lend API does not exist.
- No public device-pointer wrap API exists: only `ggml_backend_cpu_buffer_from_ptr` and
  `ggml_backend_dev_buffer_from_host_ptr` (ggml/include/ggml-backend.h:188,442). Wrapping
  a borrowed slab range as a scheduler buffer needs a new
  `ggml_backend_cuda_buffer_from_device_ptr`.
- Precedent: `kv_stream_phase_arena_owner` (src/llama-context.h:312) already re-layouts
  KV+compute per phase (`set_compute_fn`, `graph_reset_fn`) and recreates the scheduler
  (src/llama-context.cpp:1110). T10.3 reuses that phase-switch skeleton, not a new one.

## 2. Proposal (opt-in flag, default path byte-identical)

New flag: `--prefill-borrow-mib N` (MiB of slab to borrow for prompt compute, 0 = off,
default). Chunk policy `--prefill-chunk auto|<n>` (auto = largest chunk whose measured
compute fits the borrowed bytes, hard cap 8192). Both off by default; flag off means
zero behavior change.

Prompt path (only when ALL hold: borrow enabled, fork slot cache active, single device,
prompt ubatch above decode size, no CUDA-graph capture pending):

1. Lend: new provider call `moe_cache_lend(device, bytes, &ptr, &n_slots)` carves whole
   FREE slots from the largest pool with enough free slots (never partial slots, never
   resident ones). Lent slots get a `loaned` state: LRU/admission/dispatch skip them.
2. Wrap: `ptr` becomes a `ggml_backend_buffer` via the new device-ptr wrapper. Build the
   prompt-sized reserve graph and `ggml_backend_sched_reserve_size` into it. If reserve
   fails: return the loan immediately and continue on the normal path (silent fallback,
   info log).
3. Prefill in `auto|n` chunks over the borrowed buffer.
4. After the prompt: device synchronize, `moe_cache_return(device, ptr)` (slots back to
   free), re-reserve decode-sized buffers. Decode never borrows.

## 3. Decode behavior

Unchanged, except buffers are re-reserved at decode size after a borrowed prompt (the
reserve they would have had with a small `-ub`). Slot refill follows normal admission;
the decode-time slot contest is untouched (no decode runs concurrently with a prompt,
so no pause is needed).

## 4. Failure modes

- Cache cold/empty (lend returns 0): normal path, info log once.
- OOM mid-prompt: allocator pressure reaches `ggml_moe_cache_trim`, which MUST skip
  loaned ranges (else use-after-free). Proposed: trim returns 0 for loaned bytes; the
  chunk reserve then fails and step 2 fallback shrinks/retries, else hard error.
  Alternative for review: trim aborts the loan (return slots, fail the chunk cleanly).
- CUDA graphs: borrowed addresses change per prompt, so graph capture is incompatible.
  While a loan is active, skip capture (force `graph_reset_fn`, same as the phase-arena
  path). Decode captures are unaffected (loan is returned first).
- Multi-GPU (`model.n_devices() > 1` or pipeline parallel): clean refuse, same rule as
  the T10.2 upstream cache.
- Upstream `llama_moe_cache` (T10.2) active: refuse the combination in v1 (its buffers
  are scheduler-owned). Extending lend to it is explicitly v2, not this design.
- Fork cache OFF with the flag set: warning + flag ignored (no-op), not an error.
- Worker-thread eviction during a loan: eviction must skip `loaned` slots (same guard
  as admission). Trim-during-loan is the only cross-thread case (above).

## 5. API surface (new, minimal)

- `ggml/src/ggml-cuda/moe-cache.cu`: `moe_cache_lend` / `moe_cache_return`, slot state
  `loaned`, trim skips loaned ranges. No admission/dispatch logic changes besides the
  skip guard.
- `ggml/src/ggml-cuda/ggml-cuda.cu` (+ public header): device-ptr buffer wrapper.
- `src/llama-context.cpp` + `common/` plumbing for the two flags (mirror the
  `moe_cache_size` pattern from T10.2). No kernel changes, so `test-backend-ops`
  coverage is unaffected by construction.

## 6. Verification plan (for the implementation PR)

- Flag off: greedy output byte-identical vs this commit (c5e9f4ce8) on a small MoE model.
- Flag on with `-ncmoe`: prefill (time-to-first-token) A/B per the maintainer bench rules
  (interleaved blocks, 3+ each); decode speed must not regress.
- Fallback injection: the tree already has a fault hook (`moe_cache_fail(session,
  "slab")`, moe-cache.cu:1311) - use it to force lend/reserve failure and prove the
  silent-fallback path.
- OOM-mid-prompt: small-VRAM config + large chunk, must fail clean (no hang, no wrong
  output), slots accounted after.

## 7. Open questions for the reviewer (do not code until settled)

1. Trim-during-loan: skip loaned ranges (proposed) or abort the loan?
2. Flag shape: `--prefill-borrow-mib` + `--prefill-chunk auto|<n>` (proposed), or a
   single `--prefill auto` that derives the borrow from free slots?
3. v1 scope: fork slot cache only (proposed), or also lend from the T10.2 upstream cache?
4. Lend granularity: whole slots only (proposed) - and a minimum slot count before
   lending is worthwhile?
5. Strata attribution: which exact mechanism (borrow accounting? chunk picker? overlap?)
   is taken from Strata source once fetched - credit (MIT) in the commit message.

Verify manually (could not confirm here):
- Strata docs/DETAILS.md lines ~160-170 and ~326-340 against this design (fetch failed).
- Chunk-size practice (8192 tokens) and next-layer PCIe overlap on the target machine.
- Whether `ggml_backend_sched_reserve_size` can target an externally wrapped buffer
  without scheduler changes (assumed yes - must prove with a spike per the spike skill).

## 8. Review adjudication (2026-10-17, against REVIEW-T10.3.md)

Method: every cited line was re-read in this tree (branch t11-qwen4exp).
Verdict per finding: CONFIRMED (fix in section 9), PARTIAL (claim narrowed),
or REJECTED (with evidence). No finding is accepted on the review's word alone.

### B1 - reserve_size cannot bind an external buffer: CONFIRMED, fix adopted

- `ggml_backend_sched_reserve_size` is a dry-run sizing query: it calls
  `ggml_gallocr_reserve_n_size` which runs the reserve impl with
  `no_alloc=true` (ggml/src/ggml-alloc.c:950-952, 935-936) and only writes
  per-backend byte counts into `sizes`. It never binds memory.
- Real allocation always goes through the scheduler's own buffer types:
  `ggml_vbuffer_alloc(galloc->bufts[i], ...)` (ggml-alloc.c:938); the
  scheduler is built once from `backend_buft[]`
  (src/llama-context.cpp:1200, 1108). There is no attach-external-buffer API.
- Old section 2 step 2 ("reserve into the wrapped buffer") and the section 7
  manual-check ("assumed yes") are WRONG and superseded.
- Fix (as the review suggests, mirroring the proven precedent): a dedicated
  non-owning buffer type whose `alloc_buffer` wraps the lent slab range
  (same pattern as `ggml_backend_cuda_phase_arena_buffer_type_alloc`,
  ggml/src/ggml-cuda/ggml-cuda.cu:1195-1208, which wraps an arena slice
  without cudaMalloc), then re-create the scheduler over that type for the
  prompt (same as src/llama-context.cpp:1108 and the phase switch at
  :1050-1055). Sizing queries via `reserve_size` remain legal for measuring
  (phase arena does exactly this at :1291-1303).

### B2 - free slots fragment, single ptr is not contiguous: CONFIRMED, fix adopted

- Each pool owns one contiguous `cudaMalloc` slab
  (ggml/src/ggml-cuda/moe-cache.cu:1313, `slot_count * expert_size`) with
  linear slot addressing `slab + slot * expert_size` (:1152).
- Free slots live in an unordered `vector<int> free_slots` (:126), filled
  initially with all indices (:1344-1345) and then popped at admission
  (:2404-2406) and pushed back on reset (:825). After any decode traffic the
  free set is an arbitrary subset, NOT a contiguous run. Counting
  `free_slots.size() >= n` does not yield one `[ptr, ptr+bytes)` span.
- Old section 2 step 1 ("carves whole FREE slots ... never resident ones")
  is unworkable past turn 1: on a warm cache free_slots is empty/fragmented,
  so lend would always fail. Rejected as specified.
- Fix (as the review suggests): lend searches for a CONTIGUOUS run of free
  slots; failing that, it may evict a contiguous window of resident (valid,
  `readers==0`, never `copying`) slots - prompt and decode never overlap, so
  decode re-loads on resume. Minimum loan 128 MiB (review Q4) so the
  re-reserve + graph-reset cost (~10-20 ms) is amortized.

### B3 - cudaFree on an interior slab pointer: CONFIRMED, fix adopted

- `~ggml_backend_cuda_buffer_context` runs `cudaFree(dev_ptr)` whenever
  `arena == nullptr` (ggml/src/ggml-cuda/ggml-cuda.cu:1175-1178). Wrapping
  `slab + offset` in a stock context would pass an interior pointer to
  `cudaFree` at buffer destruction (`cudaErrorInvalidValue` / context
  corruption). Old sections 1/5 ("general device-pointer wrapper") are
  rejected as specified.
- Fix: the borrowed-slice buffer carries an explicit non-owning marker
  (`owns_memory=false` on the CUDA buffer context; default `true` preserves
  every existing path), so destruction only deletes the wrapper. This is the
  same ownership split the phase arena already uses (lease release vs free,
  :1175-1187).

### B4 - decode graphs go stale across re-reserve: CONFIRMED, mechanism refined

- The review's conclusion (mandatory reset) is right; its mechanism ("baked
  addresses execute stale") is half the story. Re-reserved buffers change
  tensor `data` pointers, and the backend DOES compare them
  (`node_src_data_ptrs`, full-tensor memcmp in
  `ggml_cuda_graph_update_required`, ggml-cuda.cu:4131-4153) - so a naive
  re-reserve usually self-heals via re-capture, at the cost of 2+ uncaptured
  warmup executions (:6026-6044).
- The real corruption vector is the cgraph-uid fast path: when
  `cgraph->uid == graph->uid` the pointer check is SKIPPED and the old
  instance is reused (:4115-4120). Split-graph uids are minted at alloc time
  (ggml/src/ggml-backend.cpp:1234, 1748), so any reused graph object across a
  buffer change can launch against freed memory. Old section 4 ("decode
  captures are unaffected") is FALSE and superseded.
- Fix (as the review suggests): call `ggml_backend_cuda_graph_reset`
  (:3952, clears `cuda_graphs`) on BOTH transitions (borrow enter and return),
  mirroring the phase-arena path (src/llama-context.cpp:1044-1051,
  1150-1155). v1 additionally recreates the scheduler on every transition,
  which mints fresh split uids - defense in depth, not reliance.

### S1 - partial trim of a cudaMalloc slab is impossible: CONFIRMED, clarified

- Trim frees whole slabs (`cudaFree(pool->slab)` in `moe_cache_free_device`,
  moe-cache.cu:1716-1723) after marking the device dead, cancelling the
  queue and waiting for inflight work (:3301-3340). A pool hosting a loan
  cannot be partially freed.
- Old section 4 ("trim returns 0 for loaned bytes ... fallback
  shrinks/retries") is imprecise: v1 trim SKIPS the loaned pool's slab
  entirely (frees other pools + aux buffers, still marks dead), so the loan
  range stays valid; the in-flight prompt then fails closed through the
  normal OOM unwind, and the loan is returned on that path. Q1 settled as
  "skip", not "abort" (aborting from inside the allocator cannot unwind the
  active graph - review Q1 rationale, accepted).

### S2 - scheduler re-creation cost on small prompts: ACCEPTED, gated

- Re-creation + measure + graph reset per prompt is ~10-20 ms class overhead.
  v1 gates borrowing on `ubatch.n_tokens >= 512` (small prompts stay on the
  decode-sized reserve with no phase switch). The 5-15 ms figure itself is an
  estimate (review marks it inference, ~90%) - v1 logs enter/exit so the
  maintainer can measure it; the gate value is tunable via the chunk flag.

### S3 - kv_stream_phase_arena coupling: ACCEPTED, scoped out of v1

- The phase arena is coupled to streaming-KV targets and its own repartition
  state (src/llama-context.h:312-355, .cpp:1275-1354); a non-streaming model
  never configures it. v1 implements an INDEPENDENT borrow state machine in
  `llama_context` and REFUSES the combination
  (`--prefill-borrow` + `--kv-stream-arena-mib` = parse-time error), same
  strictness as the existing `--moe-cache-mib` vs `--moe-cache` exclusion
  (common/arg.cpp:1276-1280). Old section 1 ("reuses that phase-switch
  skeleton") is reworded: v1 reuses the PATTERN (buft swap + sched recreate
  + graph reset), not the object.

### Q2 flag shape: ADOPTED with one deviation recorded

- Adopted: `--prefill-borrow off|auto|<MiB>` (default `off`) +
  `--prefill-chunk auto|<n>` (default `auto`), per review Q2. `auto` borrow =
  half of the largest pool's free bytes (review's own formula), lend-time
  clamped to a contiguous run; explicit MiB stays as the advanced override.
- Deviation: v1 `--prefill-chunk` governs borrow SIZING only
  (`min(ubatch, 8192|n)` tokens); true chunk SPLITTING of a prompt ubatch
  into several borrowed executions is v2 (it reworks the decode loop; see
  section 9). Rationale: sizing alone delivers the VRAM win (decode-sized
  startup reserve + borrowed prompt buffers); splitting without weights to
  validate against is unjustified risk.

### Q3 scope (fork-cache-only v1): ADOPTED

- Fork slot cache only. The T10.2 upstream cache owns scheduler-managed host
  tensors with an LRU upload queue - nothing sliceable. No change.

### Q4 granularity (whole slots + minimum): ADOPTED

- Whole slots only (CUDA GEMM/GEMV alignment + pool accounting), minimum
  128 MiB per loan. No change.

## 9. Revised v1 specification (implements section 8)

Flags (all off/default = zero behavior change):
- `--prefill-borrow off|auto|<MiB>` (default `off`). `auto` = half the
  largest pool's free bytes at lend time; `<MiB>` = explicit cap.
- `--prefill-chunk auto|<n>` (default `auto` = `min(ubatch, 8192)` sizing cap).

Enter (prompt ubatch with `n_tokens >= 512`, borrow configured, fork slot
cache active, single device, no upstream cache, no kv-stream arena, no loan
active, no capture pending):
1. `synchronize` + `graph_reset`. Lend: `moe_cache_lend(device, bytes)`
   returns a contiguous run (free run first, else evict-valid window; never
   `copying`, never `readers>0`; single outstanding loan). Failure = silent
   fallback below (info log).
2. Swap `backend_buft[cuda]` for the non-owning borrowed-slice type,
   re-create the scheduler (`max_nodes` for the chunk), re-apply the copy
   callback + MoE settings, measure the chunk graph and
   `ggml_backend_sched_reserve`. Reserve failure = return loan, fallback.
3. Run the chunk on the borrowed scheduler. (Chunk splitting = v2; v1 runs
   the ubatch whole against the borrowed reserve, else fallback.)
Fallback (any failure): return the loan if taken, transiently grow the
NORMAL scheduler to prompt size (today's behavior, re-done), run, shrink
back. Always correct, just slower.
Exit (next ubatch below threshold, or end of prompt): `synchronize` +
`graph_reset`, destroy scheduler (releases the non-owning lease),
`moe_cache_return` (slots back to free), restore saved buft, re-create +
re-reserve the decode-sized scheduler, `graph_reset` again. Decode never
borrows.
Startup: with borrow eligible, `sched_reserve` sizes the scheduler for
decode (`n_seqs` scale) instead of `min(n_ctx, n_ubatch)` - this is the VRAM
win; prompts grow transiently per above.
Invariants: trim skips the loaned pool's slab (S1); admission/eviction/
dispatch skip `loaned` slots (B2); wrapper never `cudaFree`s (B3); both
transitions reset graphs (B4); fork-cache-off = warning + disabled (not an
error); multi-GPU = warning + disabled; upstream-cache or kv-arena combo =
parse-time error (S3, T10.2 precedent).
Attribution: concept (phase-borrowed prefill buffers returned before
generation) from Strata (MIT, Niko1221/Strata); implementation is clean-room
for GGML scheduler + fork slot cache - credited in the commit message.

## 10. As-built record (2026-10-17, t11-qwen4exp working tree)

Deviations from section 9, all toward smaller diffs on the default path:

- Startup shrink, not decode-sized reserve. `sched_reserve` runs the full
  prompt-size reserve unchanged, then `prefill_borrow_shrink_to_decode`
  re-creates a decode-scale scheduler when eligible
  (src/llama-context.cpp, tail of `sched_reserve`). One extra reserve at
  startup only when the flag is on; the normal flow is untouched, so flag-off
  behavior is identical by construction.
- `prefill_borrow_normalize` at `sched_reserve` entry returns any loan and
  clears grown state before the full re-reserve rebuilds the scheduler, so
  the other `sched_reserve` callers need no guards of their own.
- Encoder ubatches (`mctx == nullptr`) skip the hook entirely; borrow +
  multimodal encoder is untested in v1.
- Combo refusal is defense in depth: parse error in common/arg.cpp, `nullptr`
  return in `llama_init_from_model`, `throw` in `prefill_borrow_resolve`.
  Runtime conditions (fork cache off, multi-GPU, no GPU backend, old CUDA
  backend without the borrow proc addresses) warn + disable, never error.
- Loaned slots are excluded at `moe_cache_slot_reset` (early return),
  LRU scan (`continue`), invalidate overlap (skip), and trim (loan pool
  spared via `spare_pool`); single outstanding loan per device; 128 MiB
  minimum (`moe_cache_loan_bytes_min`); `bytes == 0` means auto.
- Borrowed buffer type owns nothing: `owns_memory=false` on the wrapper
  context, `ggml_backend_cuda_borrowed_moe_buffer_type_free` frees only the
  type object, never the slab.
