# DESIGN-T12.3: run the fork MoE slot cache and the upstream MoE LRU in one process

Status: design only, no code. Goal: keep both measured wins on Qwen3.8-Flash-Next
(IQ1_M, 16 GB card, -ncmoe 30, -ctk turbo5 -ctv turbo4) in a single run:

- fork `--moe-cache on`: +42% prompt (458 vs 324 tok/s, 18k prompt, maintainer-measured)
- upstream `--moe-cache-mib 1024`: +12% generation (24.7 vs 22.1 tok/s, maintainer-measured)

Today they are mutually exclusive (common/arg.cpp:1276-1284,
src/llama-context.cpp:1461-1464 + 1192). This doc lists what each cache
provably does, the exact conflicts, VRAM accounting, the turbo KV interaction,
and the smallest workable version. All code refs are branch t11-qwen4exp.

## 1. What each cache provably does (code, not claims)

### 1a. Fork cache = CPU-offload cache for host-resident experts

- Entry: ggml-cpu calls `moe_cache.begin(name, host_base, expert_size, n_in,
  n_out, wtype, n_expert, n_tokens=ids->ne[1], n_rows)` for a MUL_MAT_ID node
  whose src0 lives in a host WEIGHTS buffer (ggml/src/ggml-cpu/ggml-cpu.c:1746+).
- Hard gates in begin (ggml/src/ggml-cuda/moe-cache.cu:2051-2075) and in
  fused_begin (:2969): `n_tokens > max_batch` returns NULL, `n_rows > 64`
  returns NULL. `max_batch` defaults to `moe_cache_batch_max = 8` (:53) and a
  supplied config with `max_batch > 8` is REJECTED (session_create returns
  nullptr, :1605-1611). Raising it needs a fork change; it is not a tunable.
- Consequence: at default settings the fork per-node path can NEVER serve a
  prompt ubatch (512+ tokens). "Fork cache for prompt batches" is impossible
  without changing the cap. The bypass log line
  ("bypassing %lld-token node above max-batch=%d; prompt batches normally use
  the CPU path", :2062-2064) proves which path a prompt run takes: check the
  log before assuming the fork accelerates prompt MoE nodes.
- What the fork DOES do during prompt: pool residency is unaffected by the
  bypass (pools/fills are session state), and prefill-borrow lends pool slabs
  as prompt compute (llama-context.cpp:1223-1251). The +42% prompt number must
  come from one of: pool-residency/scheduler-placement effects, the borrow
  slab (but T10.3 measured borrow as prompt-neutral), or a baseline mismatch.
  OPEN MEASUREMENT (section 6, item 1): rerun prompt with fork on/off and grep
  the bypass line + per-backend op assignment; do not build T12.3 on an
  unexplained number.
- Expert dtypes: explicit allow-list, IQ1_M included (moe-cache.cu:761-786).
- Decode (1 token, or speculative rows within 64 rows / 8 tokens): plan /
  dispatch / collect from GPU pools, background fills, heat admission, fused
  SwiGLU path (fused_begin, same batch gate).

### 1b. Upstream LRU = device banks + slot maps, two paths by batch size

src/llama-moe-cache.cpp, single GPU/iGPU only, throws on multi-device
(src/llama-context.cpp:781-784).

- Slot path (decode AND speculative, n_tokens <= 32): build_moe_cache_slots
  (src/llama-graph.cpp:2352-2390) asks get_slot_map(il, n_tokens,
  n_expert_used); null when `n_tokens == 0 || n_tokens > max_batch (32) ||
  min(n_tokens*n_expert_used, n_expert) > n_slots` (llama-moe-cache.cpp:380-388).
  On hit, MUL_MAT_ID reads device banks via slots; the sched copy callback
  (llama_context::sched_copy_experts, llama-context.cpp:3971-3977) uploads
  misses through llama_moe_cache::copy, which reads the selected ids out of
  the graph's GET_ROWS lookup node and REQUIRES sel computed in an earlier
  split (GGML_ABORT otherwise, llama-moe-cache.cpp:461-472) and enough slots
  for the selection (GGML_ABROT "too small", :478-480).
- Read-only path (prompt, n_tokens > 32): get_slot_map returns null, the graph
  is unchanged, and sched_copy_experts' MUL_MAT_ID fallback copies cached
  experts device-to-device via copy_experts and uploads only the rest
  (llama-context.cpp:4024-4044). LRU order is NOT touched by this path, so a
  prompt never evicts the decode working set. This is the upstream prompt
  assist: present today, no slot-map involvement.
- Stats split at 8*n_expert_used (stats_small vs stats_large, :488); the
  destructor logs hit rates, so both numbers in the task can be re-derived
  from one run's log tail.
- Type handling: banks use each weight's own dtype, groups split by
  (type, shape) (:138, :310). IQ1_M works; MMQ over-read padding is handled in
  the copy path (llama-context.cpp:4040-4044).

### 1c. turbo5/turbo4 KV is orthogonal to both caches

Both MoE caches hold MoE WEIGHT tensors only (names containing `_exps` /
`_chexps`, llama-context.cpp:959-962). The KV cache (turbo K/V, indexer,
hybrid slots) is separate VRAM. Interaction is capacity-only: every MiB given
to a MoE cache is a MiB not available for KV, compute buffers, or the borrow
slab. No kernel or graph change in T12.3 may assume a KV dtype; the (1) fix
(QSA Q-rotation) is independent of which MoE cache is on.

## 2. Exact conflicts preventing one-process coexistence

1. CLI throw: `--moe-cache-mib` + explicit `--moe-cache <mode != off>` ->
   invalid_argument (common/arg.cpp:1276-1278). And `--prefill-borrow` +
   `--moe-cache-mib` -> invalid_argument (:1283-1284, "v1 fork only").
2. Context forcing: `moe_cache_size > 0` forces the fork sched mode OFF
   (llama-context.cpp:1461-1464, logs the override at :1465-1466). Removal
   point for coexistence.
3. prefill-borrow resolve throws with `moe_cache_size > 0`
   (llama-context.cpp:1192-1193). Borrow lends FORK pool slabs
   (ggml_cuda_moe_cache_lend, :1239-1240), so borrow+upstream stays forbidden
   even in T12.3 v1 (see 5).
4. Single-device: upstream throws with pipeline_parallel or n_devices > 1
   (:781-783). T12.3 is single-GPU by construction; fork multi-device sessions
   are out of scope.
5. No callback conflict (verified, not assumed): upstream hooks the SCHED copy
   callback (set_copy_callback at :1115/:1174/:1492/:1696, all routing to the
   same sched_copy_experts), fork hooks the CPU BACKEND's MUL_MAT_ID
   (ggml-cpu.c:1746+). They compose structurally: if the upstream slot path
   takes a node (GPU banks), the node never reaches the CPU backend and fork
   sees nothing; fork only sees nodes the slot path rejected. Consequence: for
   n_tokens <= 32 upstream structurally STARVES fork. Any design that wants
   fork to own decode must gate the upstream slot path, not just enable both.
6. Buffer usage tags: upstream bufs are marked WEIGHTS (:366-367) so the
   scheduler treats banks as resident weights; fork pools come from its own
   CUDA allocation (session_create at sched reserve). Both draw from the same
   16 GB; neither knows about the other today.

## 3. VRAM accounting (16 GB card, -ncmoe 30, IQ1_M, turbo5/4 KV)

Fixed residents (unchanged by T12.3): model non-MoE weights on GPU, 30
CPU-side MoE layers' weights in host RAM, KV cache (-c, -ctk/-ctv), compute
buffers (prompt-sized without borrow, n_seqs-sized after borrow exit),
plus the larger prompt ubatches from the (2) fix (split up to min(chunk, batch)).

- Upstream: exactly `moe_cache_size` bytes (--moe-cache-mib * MiB), split per
  group as `budget_g = size * host_bytes_g / host_bytes_total`, slots grown
  while `alloc_size(g, slots+1) <= budget_g` where alloc_size sums over the
  group's ref tensors `PAD(nb2 * (slots+1), alignment)` (the +1 slot: CUDA MMQ
  over-read guard, :254-262). Plus host slot maps: n_cached_layers * n_expert
  * 4 B. Allocated once at context construction, before sched_reserve
  (llama-context.cpp:781-796 + 4826 breakdown). For 1024 MiB the log line
  "MoE cache size = ... for ... of host experts" plus the per-group slots/%
  lines give the exact coverage; record them in the T12 report.
- Fork: `--moe-cache [auto|on|fit] [budget]`, 0 = provider available-memory
  budget (include/llama.h:404-405). Pools per (expert_size, wtype), slabs >=
  minimum_slab_bytes, plus scratch (query_shape caps,
  ggml-backend-moe-cache.h:31-35). Session created at sched reserve, AFTER the
  upstream construction above, so with both on, a fit/auto fork budget
  naturally measures post-upstream free VRAM. That ordering is the entire
  "one budget split" mechanism in v1: fixed upstream first, fork fits the
  remainder. No new allocator needed.
- Duplication warning: with both on, a hot expert can sit in a fork pool AND
  an upstream bank simultaneously (neither knows of the other). Size
  accordingly (e.g. upstream 1024 fixed + fork fit-remainder); dedup across
  the two is explicitly out of scope for v1.

## 4. Routing matrix (as-built code, no changes)

| batch | upstream slot path | upstream copy_experts | fork begin/plan |
|---|---|---|---|
| 1..8 (decode) | TAKES node (GPU banks) | n/a | starved (node never CPU-assigned) |
| 9..32 (speculative) | TAKES node | n/a | starved / gated (>8 anyway) |
| >32 (prompt ubatch) | null (>32) | assists (read-only, no LRU touch) | bypassed (>8) |

So "just delete the throw" yields: decode+speculative = upstream-only
behavior, prompt = fork-on-residency + upstream read-only assist. The fork's
measured decode contribution is nil in this mode, and the upstream's measured
+12% generation is preserved verbatim (same code path as --moe-cache-mib
alone). If instead fork must own decode (its fused/overlap path), gate
get_slot_map to return null for n_tokens <= 8 (3-line change,
llama-moe-cache.cpp:380-388) -- but then decode runs the FORK path and the
+12% upstream decode number no longer applies; re-measure before claiming
both wins. The task's role assignment (fork=prompt, upstream=decode<=32) is
INVERTED relative to the as-built gates: fork cannot take prompt batches
(max_batch=8 hard cap, section 1a) while upstream already covers <=32. Do not
implement the task's literal assignment without the cap raise; implement the
structural routing above.

## 5. Smallest workable version (recommended)

V1 -- split-budget coexistence, no routing changes (smallest; keeps both
measured code paths intact):

1. common/arg.cpp: delete the --moe-cache-mib vs --moe-cache throw
   (:1276-1278); KEEP the --prefill-borrow vs --moe-cache-mib throw (:1284)
   and the resolve throw (llama-context.cpp:1192) -- borrow lends fork slabs
   and stays fork-only.
2. llama-context.cpp: delete the fork-OFF forcing (:1461-1464); log both
   modes (extend :1482-1483 line to print upstream MiB + fork mode/budget).
3. Allocation order unchanged (upstream constructed pre-sched-reserve, fork
   session at reserve) so fork fit/auto measures post-upstream free VRAM =
   the one-budget split. Document the two log lines to check (upstream "MoE
   cache size" + fork "configured: mode=... budget=...").
4. No graph/sched/CPU-backend changes. Routing falls out structurally
   (section 4 table). prefill-borrow keeps working (fork pools exist; lend
   during prompt, return before decode; upstream LRU untouched by prompt
   either way).
5. Validation (all on the 16 GB card, -ncmoe 30, turbo5/4, 18k prompt):
   a. fork-on + upstream-1024 prompt tok/s >= 458 (keeps +42%).
   b. generation tok/s >= 24.7 (keeps +12%); tail logs show upstream
      small-batch hit rate > 0 AND fork decode... (fork sees nothing by
      section 4 -- assert the fork bypass/counters show zero decode claims,
      proving no interference rather than assuming it).
   c. VRAM: nvidia-smi peak + both cache log lines; no reserve failures with
      the (2) larger prompt ubatches (watch for grow_normal fallback lines).
   d. Correctness: greedy triple (37*43 etc.) byte-identical to upstream-only
      run; q8_0 control unchanged.

V2 (only if V1 shows fork decode would add more): gate upstream slot path to
9..32 tokens, let fork own <=8. Re-measures decode from scratch; risks losing
the +12%. Not recommended first.

V3 (handover at prompt/decode switch: destroy fork pools, grow upstream):
rejected -- teardown/rebuild per switch costs more than the duplication it
saves, and both caches already avoid prompt-vs-decode eviction internally
(copy_experts is read-only; fork bypasses prompt).

## 6. Open measurements (do these before/while building V1)

1. Explain the +42%: fork-on vs fork-off prompt with GGML_CUDA_MOE_CACHE
   bypass lines + sched backend assignment in the log. If prompt MoE is
   CPU-side in both, the fork pools are innocent and the win is placement or
   methodology -- find out which.
2. Upstream coverage at 1024 MiB on this model: per-group slots/% lines --
   if coverage is ~100% of the decode working set, V1 decode is safe; if thin,
   misses upload per token and the +12% may not survive the fork's VRAM share.
3. Interference: fork background fills during decode vs upstream uploads --
   one run with both on, compare against each alone; PCIe/NVLink contention
   is the failure mode section 4 cannot rule out by reading.
4. qwen4exp fused DSV4 + upstream copy() asserts (earlier-split sel,
   slot capacity): the +12% was measured on this model so presumably fine,
   but V1 must re-run the exact greedy triple, not just benches.
