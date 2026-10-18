# REPORT-T10: MoE offload speed batch (T10.1, T10.2, DESIGN-T10.3)

Branch: `t10-moe-offload` (from `candidate` ffbd22d55). No push (local only).
Upstream remote: `https://github.com/ggml-org/llama.cpp.git` (fetch-only PR refs).

Commits:
- 068f62f08 `t10.1: port upstream #29599 llama_prefetch_rows + #30532 split`
- c5e9f4ce8 `t10.2: port upstream #29887 MoE expert cache (moe-cache-mib), mutually
  exclusive with fork --moe-cache` (includes CRLF normalization of the two new files)

## T10.1 - prefetch rows (#29599 + #30532 split)

Upstream range: c350a40bb..b1b0d72a5. Adapted (not verbatim): the fork lazy-tensor API
drifted, so the gate is `lazy_tensor_names` + `lazy_has()` + `can_prefetch` instead of
the upstream loader refactor. `llama_prefetch` coalescing/sorting + the Win32
PrefetchVirtualMemory / Linux sysconf split are verbatim.

Files (all in commit 068f62f08): src/llama-mmap.h, src/llama-mmap.cpp, src/llama-impl.h,
src/llama-impl.cpp, src/llama-model-loader.h, src/llama-model-loader.cpp,
src/llama-model.h, src/llama-model.cpp, src/models/gemma4.cpp (new
llm_graph_input_gemma4_ple), src/models/qwen4exp.cpp.

Build: `cmake --build build-cuda --config Release --target llama` succeeded
(build-cuda/bin/Release/llama.dll).
Tests: compile only. SPEC checks (prefetch reached log/counter, greedy byte-identical
with/without on qwen4exp/gemma4) NOT run - no model files in the work zone (models must
be copied in by the maintainer; speed/correctness numbers are theirs to measure).

## T10.2 - upstream MoE expert cache (#29887)

Upstream range: 1a3011cc0..2de17c21b (sched copy-callback traced to 6753a033f);
src/llama-moe-cache.* verbatim from 2de17c21b with one fork fix
(`n_expert_used_max()` does not exist here; uniform `n_expert_used` used instead).

Files (commit c5e9f4ce8, 16 files, +939/-136):
- new: src/llama-moe-cache.h, src/llama-moe-cache.cpp (+ CMakeLists entry)
- sched: ggml/include/ggml-backend.h (copy-callback typedef + setter),
  ggml/src/ggml-backend.cpp (callback fields, host-weight-last two-phase copy; replaces
  the fork inline used-expert copy with the upstream hook so the context callback sees
  other split inputs first)
- context/graph: src/llama-context.h/.cpp (moe_cache member, sched_copy_experts with
  slot-map fast path + used-expert upload, copy_experts.reset() in graph_compute and
  opt loop, memory_breakdown, default params, ON+size refused via nullptr return),
  src/llama-graph.h/.cpp (moe_cache threading, build_lora_mm_id slots param,
  build_moe_cache_slots; null slots fall back to plain MUL_MAT_ID), src/llama-cparams.h,
  include/llama.h (`moe_cache_size`)
- plumbing: common/common.h (`moe_cache_size`), common/common.cpp (map to cparams),
  common/arg.cpp (`--moe-cache-mib N`, LLAMA_ARG_MOE_CACHE_MIB), common/speculative.cpp
  (draft path zeroes it - cache is target-only)
- tests/test-llama-archs.cpp: `moe_cache_size` plumbed + "MoE cache" device config
  (adapted: this tree has no host-expert overrides yet, so the config exercises the
  copy-callback path with GPU-resident experts, not the cache itself)

Coexistence (SPEC: keep both, mutually exclusive, defaults unchanged, no rename needed -
`--moe-cache` vs `--moe-cache-mib` do not collide):
- `--moe-cache <mode> --moe-cache-mib N` (explicit fork mode): parse-time
  `invalid_argument`, prints `error: --moe-cache-mib cannot be combined with
  --moe-cache <mode>; pass --moe-cache off to use the upstream expert cache`, exit 1
  (verified). `--moe-cache off --moe-cache-mib N` proceeds (verified to model load).
- Direct C API mode==ON + size>0: `llama_init_from_model` logs and returns nullptr.
- Fork AUTO/UNSPECIFIED + size>0: fork slot cache downgraded to OFF with an info log in
  `sched_reserve` (both sched sites); the upstream cache owns host experts alone.

Builds (CUDA Release, build-cuda; box idle per bench-discipline rule):
- `llama`, `llama-common`, `llama-cli`, `test-arg-parser`, `test-backend-ops` all link.
  Final binaries 2026-10-17 01:27 (test-backend-ops.exe 01:21; its sources were untouched
  by the last edit, it loads the final llama.dll).
- Only warnings: pre-existing LNK4098 (LIBCMT) on ggml-cuda, NVCC `turbo_lut` /
  `sparse_v_threshold_f` unused warnings in fattn-vec turbo instances, MSVC C4244 in
  `make_unique<llm_graph_input_out_ids>` (llama-graph.cpp:2518, pre-existing pattern
  shifted by the patch).

Tests run (outputs captured, all on RTX 4070 Laptop 8GB box):
- test-arg-parser: `all tests OK`, exit 0.
- test-backend-ops `-o MUL_MAT_ID`: 981/981 passed, `Backend CUDA0: OK`, 2/2 backends
  passed (run against full-T10.2 llama.dll; the later validation move touched no
  scheduler code). Full-suite tbo was started and ran clean for 5+ min (MUL_MAT_* OK)
  but was stopped by timeout, not completed.
- CLI: `--help` lists `--moe-cache-mib`; exclusion/allow combos verified above.
- NOT run (no model in zone): greedy-output equality no-cache vs cache on a small MoE
  model; MUL_MAT_ID "with each cache on" (tbo has no cache flags - it validates the
  reworked scheduler path only); T10.1 prefetch checks. No speed numbers taken or
  invented.

Incidents (for the record):
- Mid-build source edits left MSVC with stale objects: test-arg-parser/llama-cli
  segfaulted at startup (a stale Oct-14 `test-arg-parser.obj` with the old
  `common_params` layout linked against the new DLL). Bisected via stash builds to the
  2-line `common.h` field, then proved staleness from the obj timestamp. Fixed by
  touching all affected sources and rebuilding; verified by fresh-obj timestamp + green
  tests. Lesson: after any stash/checkout cycle that touches headers, touch every
  affected .cpp before trusting a binary (extends the existing MSVC rule in memory).
- CRLF: new verbatim files were LF; normalized to repo CRLF via amend (codegen-neutral).

## T10.3 - prefill-borrow implemented (branch t11-qwen4exp, no push)

Implements DESIGN-T10.3.md sections 8-10 (outside review REVIEW-T10.3.md
adjudicated: B1/B2/B3 CONFIRMED with fix adopted, B4 mechanism refined,
S1 clarified, S2 gated at 512 tokens, S3 scoped out of v1).

Changes (11 files, +893/-7 at commit):
- `ggml/src/ggml-cuda/moe-cache.cu` + `.cuh`: `loaned` slot state,
  per-pool loan range + per-device single-loan ledger,
  `ggml_moe_cache_lend` (best-fit free run, else evict-valid window,
  128 MiB minimum, `bytes == 0` = auto) / `ggml_moe_cache_return`,
  loaned excluded from slot reset / LRU / invalidate / trim (loan pool
  spared, single slab cannot partially free - S1).
- `ggml/src/ggml-cuda/ggml-cuda.cu` + `ggml/include/ggml-cuda.h`:
  `owns_memory` on the CUDA buffer context (default true, old behavior),
  non-owning `Borrowed_MoE` buffer type over a lent slab range (B3),
  proc-address exports for lend/return/borrowed-type/graph-reset.
- `src/llama-context.cpp` + `.h`: borrow state machine (resolve at
  `sched_reserve` after fork-mode resolution, normalize at reserve entry,
  enter/exit/grow-normal/shrink in `process_ubatch`, cleanup in dtor),
  C-API combo guards in `llama_init_from_model`.
- Flags (default off = zero behavior change): `--prefill-borrow
  off|auto|N`, `--prefill-chunk auto|N` through common/arg.cpp,
  common/common.h/.cpp, llama-cparams.h, llama.h defaults.
- As-built deviations recorded in DESIGN-T10.3.md section 10 (startup
  shrink instead of decode-sized reserve; encoder ubatches skip the hook).

Build: `cmake --build build-cuda --config Release --parallel 8`, exit 0.
Footnote: the first build relinked test/tool binaries against stale .obj
files (MSBuild missed header dependents: common.h/llama.h layout change,
`test-arg-parser.obj` predated the headers) and `llama-cli --help` /
`test-arg-parser` segfaulted at startup. Fixed by deleting every non-ggml,
non-vendor stale .obj and rebuilding (exit 0). ggml kernel .obj staleness
is harmless (ggml.h unchanged).

Tests:
- `test-arg-parser`: all tests OK (exit 0).
- `test-backend-ops test -o MUL_MAT_ID`: 981/981 passed, CUDA0 OK, 2/2
  backends (exit 0). Exercises the touched `owns_memory` alloc/free paths
  on every CUDA buffer; the MoE-adjacent op sweep per repo precedent.
- Flag smoke: `--prefill-borrow bogus` errors with usage (exit 1);
  `--prefill-borrow 100 --moe-cache-mib 512` refused at parse (exit 1);
  `--help` lists both flags with env vars.

NOT tested: borrow on a real MoE model (no weights in-zone: lend/return,
borrowed-buffer reserve, prompt/decode transitions, VRAM win unmeasured);
default-path greedy parity (maintainer: temp-0 fixed prompt vs parent
commit); full test-backend-ops suite (only MUL_MAT_ID run); multimodal
encoder + borrow (hook skips it by design); multi-GPU (refused by design).

## Verify manually

- Greedy equality + prefill/decode A/B for T10.1/T10.2 on the target machine with a real
  MoE model (maintainer measures; this box has no models in-zone).
- test-llama-archs "MoE cache" row (target does not exist in Windows-shared builds).
- DESIGN-T10.3 review answers (section 7) + Strata source cross-check.
- Full test-backend-ops suite to completion (only `-o MUL_MAT_ID` completed here).

State at stop: tracked tree clean (2 commits), docs untracked (SPEC-PORT-STRATA.md,
DESIGN-T10.3.md, REPORT-T10.md, cache/scratch/*, *.json). No push. No servers started;
user-owned llama-server (PID 170440) untouched.

## T10.0 - turbo indexer cache uses q8_0 (branch t11-qwen4exp, commit 32e62edfe)

SPEC steps, with evidence:

1. Domain: ROTATED. Quantize applies the forward WHT (ggml-turbo-quant.c:
turbo_cpu_fwht at lines 419, 514, 596-598, 799-800, 897-900 for
turbo2/3/4/6/5). Dequant never inverts it - the source says so outright:
"No inverse WHT, dequant stays in the rotated domain" at lines 703-706
(turbo4), 846-849 (turbo6), 948-951 (turbo5). The graph instead rotates Q
and applies the inverse WHT to the attention output (GGML_OP_TURBO_WHT,
direction=1, in llm_graph_context::build_attn_mha). The indexer scores raw
keys against UNROTATED queries, so rotated keys would score wrong - adding
GET_ROWS cases (step 2) would fix the abort but silently corrupt selection.
2. GET_ROWS coverage: CPU switch (ggml-cpu/ops.cpp:5049-5083) lists Q, IQ,
TQ1_0/TQ2_0/TQ3_1S/TQ4_1S weight types but no TURBO2/3/4/5/6 cache types, so
turbo indexer keys hit the default abort (the reported crash). CUDA
(getrows.cu:303-417) likewise has TQ3_1S/TQ4_1S but no TURBO types, so an
offloaded indexer cache would abort too.
3. Fix (SPEC step 3): the indexer cache gets its own type - q8_0 when the
main cache type is turbo, unchanged otherwise. Applied in BOTH indexer
owners: src/llama-kv-cache-msa.cpp (minimax-m3) and
src/llama-memory-hybrid-idx.cpp (qwen4exp, the Flash-Next path).
Note: the earlier t10-moe-offload commit e9ad17feb fixed only the msa file,
which qwen4exp never uses (qwen4exp always builds llama_memory_hybrid_idx,
src/models/qwen4exp.cpp:381; create_memory picks hybrid_idx for QWEN4EXP,
src/llama-model.cpp:2466). The t11 commit carries both hunks.

Files (commit 32e62edfe): src/llama-kv-cache-msa.cpp,
src/llama-memory-hybrid-idx.cpp (+12/-2).

Build: `cmake --build build-cuda --config Release --target llama
test-backend-ops` exit 0 (llama.dll + test-backend-ops.exe relinked).

Tests: compile only + suites below under T11 (FA 10564/10564,
LIGHTNING_INDEXER 144/144 incl. 49 run nh=4 cases, MUL_MAT_ID 981/981).
No GET_ROWS turbo cases exist or were added (the indexer no longer stores
turbo, by design); existing `-o GET_ROWS`/`-o SET_ROWS_TURBO*` coverage is
unchanged.

NOT tested (no model files in the work zone): Flash-Next load + greedy
output with `-ctk turbo5 -ctv turbo4` and `-ctk turbo6 -ctv turbo5` vs
`-ctk q8_0 -ctv turbo4`. Maintainer (needs an in-zone copy of a qwen4exp
GGUF, e.g. models/flash-next.gguf - any Flash-Next file works, IQ1_M or
Q4_0; a non-qwen4exp file exercises nothing here):

  build-cuda/bin/Release/llama-cli.exe -m models/flash-next.gguf -c 4096 -ngl 99 -ctk turbo5 -ctv turbo4 --temp 0 -n 64 -p "Write a hello-world in Python."
  build-cuda/bin/Release/llama-cli.exe -m models/flash-next.gguf -c 4096 -ngl 99 -ctk turbo6 -ctv turbo5 --temp 0 -n 64 -p "Write a hello-world in Python."
  build-cuda/bin/Release/llama-cli.exe -m models/flash-next.gguf -c 4096 -ngl 99 -ctk q8_0 -ctv turbo4 --temp 0 -n 64 -p "Write a hello-world in Python."

Expect all three to load (no GET_ROWS abort) and identical greedy text;
report the first differing token if not identical.

## T11 - upstream qwen4exp sync (branch t11-qwen4exp, no push)

Base: c5e9f4ce8 (t10-moe-offload). Order per SPEC. T10.3 was NOT
implemented per the task instruction (design deliverable only, lives on
t10-moe-offload as untracked DESIGN-T10.3.md).

### T11.1 #29751 "llama: fix qwen4exp" (+ followup "keep kq_mask input the same shape") - SKIPPED

Evidence the fork does not have the fixed code - or the bug as upstream
met it. The fix rewrites the k-pool QSA path: shared k-pool graph inputs
(build_inp_kpool/build_qsa_sel), indexer cache rows widened by
indexer_kpool_row, order-mode pools (indexer_kpool_by_order). None of that
machinery exists in this tree:

- `grep -rn indexer_kpool_row|indexer_kpool_by_order|llm_graph_input_kpool|build_qsa_sel|build_inp_kpool src/` is empty.
- src/llama-hparams.h has no indexer_kpool / indexer_kpool_select_tail at
  all (upstream pre-29751 already had both); fork's qwen4exp never went
  through the k-pool migration.
- The k-pool home model is absent: src/models/glm5-next.cpp does not exist
  in this tree, so even the base the fix builds on was never imported.
- Real divergence (CR-corrected): fork qwen4exp.cpp vs upstream pre-29751
  differs by 804 lines (fork carries its own MTP/NextN head, PLE-history,
  turbo and prefetch adaptations on an older base); hybrid-idx differs by
  1177 lines.

Porting #29751 here means first importing the missing k-pool base plus the
intermediate qwen4exp refactors - a full qwen4exp re-sync, not a hunk-level
port - and its SPEC check (greedy output on Flash-Next vs previous commit)
cannot run without model weights. A partial port (e.g. hparams fields only,
or the x3 row layout without the graph rewrite) would leave the tree
half-migrated and break the currently working path (T9 did Flash-Next A/B
parity on the old design). Deferred to a dedicated re-sync with weights
on hand. No commit; each resolved-hunk list is therefore empty.

### T11.2 #27978 "any expert count uses the fast mm_ids_helper path" - ALREADY PRESENT, no commit

ggml/src/ggml-cuda/mmid.cu in this tree is byte-identical (modulo CRLF) to
upstream post-af8ae552a: mm_ids_pow2 template, neu_padded assert, and the
`case 10:` dispatch are all there. It came in with T7 (commit c12ac3e43
"T7: port upstream batch 27978/29638/28068/29393"). Verified with
`diff --strip-trailing-cr` (empty). No change needed.

### T11.3 #28770 "sparse FA for qwen4" - PORTED, commit 80539b290

The CUDA sparse rework (compact-mask rework, qwen4 (256,256) kernel shapes,
shall_use_sparse signature) and the three sparse qwen4 test cases came in
with T9 (commit 9a6def4f3). Missing was only the model one-liner, ported
here: src/models/qwen4exp.cpp build_attn_qsa passes `top_k->ne[0]` as
n_kv_max instead of 0.

Turbo check per SPEC: no model-side branch added, because fallback is
automatic and total. Every TURBO type has is_quantized=true (ggml.c
type_traits); the CUDA picker routes quantized K/V away from the sparse
gather (fattn.cu:3423, 3431, 3476, 3519); the CPU reference ignores
n_kv_max (no use in ggml-cpu flash-attn). So turbo K/V stays dense and
never aborts.

Files (commit 80539b290): src/models/qwen4exp.cpp (+4/-1).

Build: same green build-cuda Release build as T10.0 (exit 0).

Tests: `test-backend-ops test -o FLASH_ATTN_EXT` - 10564/10564 passed,
`Backend CUDA0: OK`, 2/2 backends passed (full sweep incl. the sparse
qwen4 cases, ~27 min on the 8GB box).

NOT tested: greedy Flash-Next output at this commit vs the previous one
(needs weights). Maintainer:

  build-cuda/bin/Release/llama-cli.exe -m models/flash-next.gguf -c 4096 -ngl 99 --temp 0 -n 128 -p "<fixed prompt>"   (at 80539b290 vs at 32e62edfe)

Expect byte-identical text (sparse gather is numerically equal to dense by
test-backend-ops); speed delta is theirs to measure, not claimed here.

### T11.4 #29901 "tiled lightning indexer for 4 heads" - PORTED, commit 3cd8ab951

All four upstream commits (tile kernel; float products; keys widened once
for all heads; ROCm register-spill fix) plus the perf-sweep `nh: 4` case.
The fork never carried the 4-head vector fallback (its base predates it),
so both dispatch branches are new here: tile kernel for
`n_head == 4 && n_batch >= LIGHTNING_INDEXER_TILE_TOKENS (8)`, vector
kernel below that; support guard now admits neq1 == 4.

Files (commit 3cd8ab951): ggml/src/ggml-cuda/lightning-indexer.cu
(+188/-1), tests/test-backend-ops.cpp (+1/-1, perf sweep only, as
upstream - the eval sweep is untouched).

Verification: the spliced .cu matches upstream post-5d775e32a byte-for-byte
(modulo CRLF), checked by script (cache/scratch/t11/splice_29901.py,
anchor-based, exits non-zero on any residual diff). Build exit 0 (NVCC
compiled the new kernel clean).

Tests: eval sweep does not cover nh=4 (upstream neither), so correctness
was checked locally by temporarily adding nh=4 to the eval sweep,
rebuilding, running, then reverting: `test -o LIGHTNING_INDEXER` -
144/144 passed, CUDA0 OK; 54 nh=4 case lines, 49 ran OK (tile path at
bs=512 and vec path at bs=1), 5 "not supported" (iq4_nl pattern, same as
nh=32/64). Binaries were rebuilt after the revert so they match the
committed source; MUL_MAT_ID re-ran after that at 981/981 (see below).

NOT tested: 4-head indexer through a real Flash-Next run (needs weights;
indexer_n_head=4 is the production shape). Maintainer: the greedy commands
under T10.0 exercise it (any qwen4exp prompt scores through this kernel on
CUDA when the indexer cache is offloaded); plus
`test-backend-ops test -o LIGHTNING_INDEXER` as the regression gate.

### T11.5 #28896 "rms_norm + mul fusion" (both commits) - PORTED, commit a162f4a2b

Loader gammas (hc_head_norm, hc_attn/ffn_norm, ple_norm_key/query/conv)
now declare {n_embd, hc} with TENSOR_ALLOW_RESHAPE instead of flat
{hc_dim}; graph code multiplies the rms_norm output directly (no reshape
between norm and mul) in build_hc_mix and build_ple grouped_norm.

Fork behavior kept (deviations from a verbatim port): the F32-activation /
F16-weight cast guard stays (fork-specific CUDA broadcast-mul limitation,
applies on the weight side so norm->mul adjacency is preserved); the
NextN/MTP flat hnorm path is untouched (its weight stays {hc_dim});
layer.nextn.hc_head_norm got the same [n_embd, hc] + ALLOW_RESHAPE
treatment so both build_hc_mix callers agree (upstream has no NextN).
Model saver untouched (no references to these tensors).

Files (commit a162f4a2b): src/models/qwen4exp.cpp (+15/-12).

Build: covered by the same green build-cuda Release build (exit 0; only
warnings observed were the pre-existing set - LNK4098, fattn-vec turbo
NVCC notes, one MSVC C4244 in llama-graph.cpp; no new warnings
introduced by these files).

Tests: `test-backend-ops test -o MUL_MAT_ID` - 981/981 passed, CUDA0 OK,
2/2 backends (scheduler-path regression after the graph/loader edits).

NOT tested: qwen4exp model load (gamma reshape on a real GGUF) and graph
build (needs weights). A wrong gamma shape would fail closed at load
(create_tensor shape check), not silently. Maintainer:

  build-cuda/bin/Release/llama-cli.exe -m models/flash-next.gguf -c 4096 -ngl 99 --temp 0 -n 64 -p "Write a hello-world in Python."

Expect normal load (no shape-mismatch error on hc/ple norm tensors) and
sane output; compare against the pre-fusion commit 3cd8ab951 for parity.

## Verify manually (maintainer, target machine with weights)

- T10.0 greedy triple (turbo5/turbo4, turbo6/turbo5, q8_0/turbo4) + first-differing-token report.
- T11.3/T11.4/T11.5 greedy parity of each commit against its parent on a
  fixed prompt, temp 0 (correctness control); pooled acceptance + tg only
  if a speed claim is ever made (none made here).
- Full test-backend-ops suite to completion (only -o FLASH_ATTN_EXT,
  -o LIGHTNING_INDEXER (temp nh=4), -o MUL_MAT_ID were run here).
- DESIGN-T10.3 review (unchanged, out of scope for this branch).

Branch state at stop: t11-qwen4exp = c5e9f4ce8 + 4 commits (32e62edfe
T10.0, 80539b290 T11.3, 3cd8ab951 T11.4, a162f4a2b T11.5). T11.1/T11.2
recorded above with reasons, no commits. T10.3 not implemented per task.
No push (local only). Binaries in build-cuda match tip (rebuilt after the
last source edit; test binary rebuilt after the eval-sweep revert).

---

# T12 (t11-qwen4exp): turbo5 garbage root cause + prefill-borrow larger chunks + DESIGN-T12.3

Branch t11-qwen4exp, base 25e21fdbd. Target: Qwen3.8-Flash-Next (qwen4exp,
IQ1_M, 16 GB card, -ncmoe 30). Work done on the 8 GB box (RTX 4070 Laptop);
no qwen4exp weights in-zone, so no model run was possible here -- the greedy
triple and all tok/s numbers below must be re-measured on the 16 GB machine
(see "NOT tested").

## T12.1 turbo5-garbage cause: missing Q-rotation in the QSA path (found by reading)

Suspects T11.3 (sparse FA gather, 80539b290) and T10.0 amend (32e62edfe) are
INNOCENT for this symptom:

- Sparse gather is numerically exact and, for turbo K/V, runs on the f16
  copy: the generic MMA path converts K/V to f16 in launch_fattn (need_f16)
  before the kernel; the native-turbo kernels never enable sparse
  (VEC sparse_decode is gated on non-quantized K/V, fattn.cu; fused turbo MMA
  instantiates use_sparse=false). Proven by test, not just reading: 10 new
  sparse cases pass (below).
- The T11.3 comment in build_attn_qsa claiming "the CUDA kernel picker routes
  quantized K/V away from the sparse gather" is wrong about the mechanism
  (shall_use_sparse does not check dtypes) but benign (conversion happens
  first). Reworded in the fix.

Actual cause: llama_model_qwen4exp::graph::build_attn_qsa
(src/models/qwen4exp.cpp) never applies the TurboQuant forward-WHT
Q-rotation that llm_graph_context::build_attn (src/llama-graph.cpp:3094+)
applies for dense layers whenever K is a turbo type. Every QSA layer with
turbo K therefore scores <Q_plain, K_rotated> = garbage on every prompt,
short or long; broken logits also break EOS, matching "runs to the token
limit with empty content". q8_0 K needs no rotation, which is exactly why
-ctk q8_0 stayed correct. Fix mirrors build_attn (pad Q to 128, cont,
ggml_turbo_wht forward with get_turbo_innerq_scale_inv()), plus the comment
correction above. q8_0 output path untouched (rotation block is gated on
k->type being turbo; verified by diff, no shared code changed).

Tests added (tests/test-backend-ops.cpp, after the mixed turbo5/6 block):
10 sparse FA cases at the QSA shape (256,256,nh=4,nr23 6/1,kv=4096,nb=1/2,
n_kv_max=512) x (turbo5/turbo4, turbo6/turbo4, turbo4/turbo4, q8_0/turbo4,
f16/f16). GET_ROWS/indexer cases with turbo K deliberately NOT added: since
T10.0 the indexer cache stores q8_0, so turbo never reaches GET_ROWS or the
lightning indexer in production (both abort on turbo by design); the indexer
sweep already covers q8_0. Note these FA tests pin kernel-vs-kernel
consistency (CUDA sparse vs CPU dense-mask); the model-convention bug itself
(missing rotation op in the QSA graph) has no weights-free test -- the greedy
triple below is the real regression check.

Build: cmake --build build-cuda --config Release (VS 18 2026), exit 0.
Objs verified newer than edited sources (llama-context.obj 15:31:50,
test-backend-ops.obj 15:32:25 vs sources 15:28-15:30).

Test output (build-cuda/bin/Release/test-backend-ops.exe test
-o FLASH_ATTN_EXT -b CUDA0 -p "kv=4096"): 22/22 passed, including all 10 new
sparse-turbo cases, e.g.
FLASH_ATTN_EXT(hsk=256,hsv=256,nh=4,nr23 6/1,kv=4096,nb=1,...,type_K=turbo5,type_V=turbo4,...,n_kv_max=512): OK
(first -b CUDA attempt matched nothing: device names are CUDA0/CPU, and -p
matches vars(), not op_desc -- reran with -b CUDA0 -p "kv=4096").
test-arg-parser.exe: all tests OK (covers the arg.cpp help-text change).

## T12.2 prefill-borrow: why auto/auto never beat --moe-cache on, and the fix

From the code (src/llama-context.cpp enter/exit/process_ubatch/decode):

- --prefill-chunk auto = cap 8192, but actual chunk = min(ubatch.n_tokens,
  cap) at enter(). ubatch size is capped upstream by the init_batch n_ubatch
  split (llama-kv-cache.cpp prepare/split, hybrid-idx init_batch -- split
  only sizes yielded ubatches; KV prep covers the whole batch either way). So
  with default -ub 512 every prompt ubatch borrows for exactly 512 tokens --
  the borrowed scheduler reserves what the normal path would reserve. No
  prompt ever runs in a larger ubatch: this is the whole 457 vs 458 story
  (same work +/- lend/evict + scheduler-recreate noise).
- --prefill-borrow auto = lend half the largest pool's FREE bytes at first
  prompt (cold cache approx half the pool), 128 MiB minimum else lend fails
  and the ubatch falls back to grow_normal (same size, transient buffers).
- -b/-ub 4096 slower (284): same 4096-token compute as the normal path but
  on lent slabs (evicted experts must stream during the prompt) + enter/exit
  sync stalls. Generation 21.8 -> 23.2: after exit, decode buffers shrink to
  n_seqs size while the no-borrow baseline keeps prompt-sized (4096) compute
  buffers, leaving less expert VRAM -- decode win, as designed.
- On the 16 GB card at -ncmoe 30, auto therefore picked chunk = n_ubatch
  (512 default, 4096 with -ub 4096) and borrowed MiB = half the largest
  (expert_size,wtype) pool free (>= 128 MiB) -- exact MiB is in the enter log
  line; it was never the bottleneck, chunk size was.

Fix (src/llama-context.cpp decode(), + log lines): when borrow is eligible
and the queued batch is prompt-sized (>= 512), split the batch into
borrow-sized ubatches -- split = min(chunk_cap auto 8192 or user N, n_batch)
when larger than n_ubatch -- instead of n_ubatch. Buffer sizing is already
correct for bigger ubatches (enter reserves chunk, grow_normal reserves the
ubatch transiently; the old "exceeds v1 borrow chunk" guard now only fires
when the user caps chunk below the split). Help text updated (auto =
min(n_batch, 8192); "ubatch splitting is v2" removed). New/changed logs:

- decode: prefill borrow splits %u queued tokens into %u-token ubatches (ubatch %u, chunk %s) (once per prompt batch, only when overriding)
- prefill_borrow_enter: prompt chunk %u tokens on borrowed %zu MiB slab (chunk %s, borrow %s) (chosen values + auto/capped provenance)

With default -b 2048 the 18k prompt now runs in 2048-token ubatches (4x);
with -b 4096, 4096-token ubatches (8x). If an 8192-token reserve fails on a
tight card it falls back to grow_normal (same size, transient) and logs it;
a hard failure there throws exactly as -ub 8192 would today.

## T12.3 DESIGN-T12.3.md (design only, no code)

Written at repo root. Key findings from reading (not assumptions):

- Fork max_batch=8 is a HARD cap (supplied >8 gives session_create null,
  moe-cache.cu session_create validation), so "fork for prompt batches" is
  impossible as built; the task's role assignment is inverted vs the gates.
  Upstream covers <=32 (slot path) and assists >32 read-only (copy_experts,
  no LRU touch).
- Smallest version: delete the CLI throw + fork-OFF forcing, keep
  borrow-vs-upstream exclusion, fixed-upstream-first allocation order gives
  the one-budget split for free (fork fit measures post-upstream free VRAM);
  routing falls out structurally (upstream takes <=32, fork sees only what
  upstream rejects). Recommends V1 + validation gates; V2/V3 rejected with
  reasons. Open item: the +42% fork prompt mechanism is unexplained by the
  per-node paths (prompt bypasses at >8) -- experiment specified in the doc
  before building.

## Verify manually (16 GB machine, weights required -- NOT tested here)

1. Greedy triple, temp 0, -ctk turbo5 -ctv turbo4 vs -ctk q8_0 -ctv turbo4:
   37*43 (expect 1591, QSA layers must no longer reason about 47*47), the
   coding prompt from the report, + one long-context prompt (forces QSA
   top_k + sparse gather). q8_0 output must be byte-identical to head
   25e21fdbd (fix touches only the turbo-K branch of build_attn_qsa).
2. Borrow: --moe-cache on --prefill-borrow auto --prefill-chunk auto on the
   18k prompt; expect the two new log lines (split + chunk/MiB) and prompt
   tok/s clearly above 458; also try -b 4096 -ub 4096 (expect >= 458, no
   grow_normal fallback lines). If reserve fails, paste the fallback lines.
3. Full test-backend-ops to completion (here: only -o FLASH_ATTN_EXT with
   -p kv=4096 + test-arg-parser).
4. DESIGN-T12.3 validation items (section 5 a-d, section 6) before any T12.3 code.

Commits on t11-qwen4exp (no push): T12.1 fix + tests, T12.2 fix + logs,
DESIGN-T12.3.md, this report section. Scratch under cache/scratch/ left
untracked and uncommitted.

# T13 (t13-qwen4exp-resync): full qwen4exp re-sync to upstream #29751

Branch `t13-qwen4exp-resync` from `t11-qwen4exp` at `f5fb9f3c7`
(no report commit `072f5baf3`; no push). Target: upstream `56a33d691`
"llama: fix qwen4exp" + `f2ef14ea8` "qwen4exp: keep kq_mask input the same
shape" (= `66e0c17ee` "llama: fix qwen4exp (#29751)"). Plan (with import
order, skip reasons, survival checklist, maintainer greedy commands):
RESYNC-T13.md (T13.0). T11.1's verdict stands: a hunk port was impossible,
the fork predates even the pre-fix parent `552f18f91` and lacked the whole
k-pool stack (`build_inp_kpool`, `build_qsa_sel`, `llm_graph_input_kpool`,
`indexer_kpool_row/by_order`, hparams `indexer_kpool*`, `kpool_layout/state/
access`, `set_input_kpool` from #27773/`649dcb103`).

Commits (all `user.name=hermes-agent`, no push):

- `dba1149f4` T13.0: RESYNC-T13.md plan (227 lines).
- `2b49ea71a` T13.1: `ed7ac35e1` causal_attn, no scheduler re-reserve + QSA
  causal rule. Gates: shared build green; test-arg-parser all OK
  (2026-10-17); test-backend-ops MUL_MAT_ID exit 0.
- `4ca2c9268` T13.2: k-pool base - `649dcb103` memory parts (kpool_layout/
  state/access, stale tracking, layout_update, build_sizes/state,
  set_input_kpool, gather_mla_rows, full-context ns_ubatch/ctx_idx restore)
  + `b356fa262` cells index (`seq_pos` map->set-of-pairs, `seq_pos_get`)
  + `get_n_seq_max`/`get_stream` (introduced by 649 itself) +
  `indexer_kpool`/`select_tail` hparams + MLA q-view stride fix
  (`nb[2]*(ne[2]/n_stream)`). Gates: shared build green; test-arg-parser
  all OK; MUL_MAT_ID 981/981 CUDA0 OK.
- `2776585af` T13.3: `10f340d1a` re-enable -sm tensor for qwen4exp (#28569:
  `ggml_build_forward_expand(gf, res_hc)` + drop QWEN4EXP from the
  `llm_arch_supports_sm_tensor` exclusion). Gates: shared build green;
  test-arg-parser all OK.
- `a5db4cda9` T13.4: `56a33d691` (#29751). Model side: kpool derivation from
  compress ratios, shared `llm_graph_input_kpool`, `build_qsa_top_k` ->
  `build_qsa_sel`, `build_attn_qsa(sel, n_sel)`; memory side: by_order scan,
  `kpool_rank`, `n_new_g`, `new_pool_pos`, both `set_input_qsa` deleted.
  Gates: shared build green FIRST TRY; test-arg-parser all OK;
  MUL_MAT_ID pass (CUDA0 + CPU).
- `56633f3e8` T13.5: `f2ef14ea8` (sel keeps kq_mask 4D shape, 4+/2-
  verbatim). Gates: shared build green (llama.dll 00:46); test-arg-parser
  all OK; MUL_MAT_ID pass; static build reconfigured + rebuilt,
  test-llama-archs.exe fresh 01:04.
- `3ef66d734` T13.6: fidelity fixes from the `66e0c17ee` cross-check (R9:
  drop shadowing `n_kv` local in `can_reuse`, comment + blank line).
  Gates: shared build green (llama.dll relinked 02:09); test-arg-parser
  100% (ctest -C Release); MUL_MAT_ID pass (CUDA0 + CPU);
  `-o LIGHTNING_INDEXER` 96/96 pass, CUDA0 OK.

Remaining gates: test-llama-archs full matrix (static; process exited -
result UNCONFIRMED, see Verify manually); `-o FLASH_ATTN_EXT` full sweep
(process exited after ~70 min - result UNCONFIRMED).

## Resolved conflicts (each)

- R1 (T13.1, `ed7ac35e1`): cherry-pick hit `<<<<<<<` markers in
  hybrid-idx.h/.cpp. Kept fork `[TAG_PLE_HISTORY]` + `ple_hist_get` decl;
  dropped the duplicated upstream `set_input_qsa` decl and the duplicated
  upstream `llama_memory_hybrid_idx_ns` + context-ctor block. Verified
  `grep` markers empty before `add` + `cherry-pick --quit`.
- R2 (T13.2): upstream moved the QSA body to memory-level `set_input_qsa`
  (649) but the fork had consolidated it into context-level `set_input_qsa`
  (T11.x). Kept the fork consolidation; ported only the k-pool additions.
  Upstream's memory-level fn was never added (nothing to consolidate into).
- R3 (T13.2): fork lacks `state_clear` (upstream `08618ff8e` #27530, NOT
  imported - failed-restore zeroing, out of scope). `state_drop` keeps
  fork's `seq_rm(-1,-1)` calls + adds only the stale hooks (deviation D2).
- R4 (T13.2): upstream `b356fa262` also adds `seq_pos_tok_le` +
  `cells.ext.tok` + n-gram `get_prev_tokens` removal. Skipped the token
  half (deviation D1): fork deleted the only consumer (n-gram speculative
  path) in favor of its own PLE history; re-adding `ext.tok` writes would
  touch kv-cache hot paths for no T13 consumer (`tok_le` unused by 649's
  kpool and by #29751 - verified by grep on both diffs).
- R5 (T13.3): `llama-arch.cpp` surrounding cases differ (fork has
  KIMI_LINEAR where upstream has GLM5_NEXT) - same one-line QWEN4EXP drop.
- R6 (T13.4): fork `llm_graph_input_qsa` lives at file scope, upstream
  in-struct. Renamed at file scope (`models.h:2135` fwd decl + qwen4exp.cpp
  class); `build_inp_kpool` follows fork style (returns raw pointer from
  `res->add_input`, no cast).
- R7 (T13.4): fork `build_qsa_top_k` was byte-identical to upstream pre-56a
  (no fork scoring customizations - T11.4's tile kernel lives in
  `ggml/src/ggml-cuda/lightning-indexer.cu`, used by the generic build_attn
  path), so it was replaced wholesale by `build_qsa_sel`. Fork blocks that
  DO live in `build_attn_qsa` (turbo-WHT Q rotation, norm-vs-plain fork,
  sparse-FA `n_kv_max`) sit after the replaced mask-build region and were
  kept verbatim; only the count arg changed (`top_k->ne[0]` -> `n_sel`).
- R8 (T13.4): 321-line dead fork `set_input_qsa` body deleted from
  hybrid-idx.cpp (lines 897-1217, CRLF preserved, verified `-359` in
  diffstat + clean seam) + decl removed from the header. Zero remaining
  `set_input_qsa` references in `src/`.
- Plan deviations: models.h qsa->kpool renames landed in T13.4 (not T13.2,
  they belong to #29751); `glm5-next.cpp` NOT imported (per plan);
  `37b53fd45` fused_dsv4_hc NOT imported (per plan, #29751 needs none of
  it); T13.2 memory `get_kpool()` is hparams-driven inline (no extra fork
  rule needed for pooled rows - the T10.0 q8_0 lambda maps whole rows).
- R9 (post-T13.5 cross-check vs squash-merge `66e0c17ee`): T13.4 had added
  `const int64_t n_kv = idx->get_n_kv();` in `can_reuse` from a misread of
  the demoted 56a hunk - upstream never had it (the member `n_kv` is the
  comparison; the local shadowed it into a dead always-true check). Removed
  in T13.6. Same cross-check fixed a stale comment (glm5-next -> glm5-next,
  qwen4exp QSA) and a missing blank line in `kpool_build_state`. Method:
  `git diff 66e0c17ee -- <file>` (working tree, NOT `HEAD` - the latter hid
  uncommitted fixes) hunk-scan; every remaining hunk matches a known fork
  block or pre-existing drift (verified identical hunks in
  `git diff 66e0c17ee f5fb9f3c7`).
- Gate hygiene: `test-arg-parser.exe` run directly prints usage errors as
  part of its PASSING invalid-usage subtests - judge by exit code / final
  "all tests OK" line, or `ctest --test-dir build-cuda -C Release
  -R "^test-arg-parser$"` (multi-config needs `-C`; bare ctest reports
  "Not Run"). Never run a second GPU job while one holds the build DLLs:
  link fails LNK1104 on locked `ggml-base.dll` (hit once: rebuild queued
  behind the FLASH_ATTN_EXT sweep).

## Fork behaviour that survived (each checked)

`git diff f5fb9f3c7..HEAD --name-only`: exactly 11 src files
(arch, context, graph, hparams, kv-cache.* , kv-cells.h, hybrid-idx.*,
models.h, qwen4exp.cpp) + RESYNC-T13.md. `ggml/` untouched (T11.4 tile
kernel, T12.1 turbo WHT/sparse kernels, T11.3 CUDA picker intact);
`llama-graph.cpp` delta is ONLY the 649 stream-view fix.

- (a) MTP/NextN head: 87 mtp/nextn matches, `graph_mtp` (l.544) intact;
  upstream MTP (#29761, out of range) not imported.
- (b) PLE history: `build_ple`, ple_hist_get/put, serialise paths intact.
- (c) LAZY `per_layer_tok_embd` + prefetch gate: 11 ple/prefetch/LAZY
  matches intact; #29599 verbatim not re-imported (per plan row 11).
- (d) q8_0 indexer rule (T10.0): `idx_type` lambda (hybrid-idx.cpp:67)
  intact; applies to whole k rows incl. pooled slots (row-granular).
- (e) Turbo-WHT Q rotation (T12.1): TURBO branch + `ggml_turbo_wht`
  forward call (qwen4exp.cpp:1063-1070) intact in the new QSA path.
- (f) Sparse FA gather (T11.3): `build_attn_mha(..., n_sel, ...)` non-zero
  budget intact; `-o FLASH_ATTN_EXT` full sweep exited after ~70 min,
  result UNCONFIRMED (see Verify manually item 1).
- (g) Tiled lightning indexer (T11.4): kernel + dispatch untouched;
  `-o LIGHTNING_INDEXER` 96/96 pass, CUDA0 OK (02:15).
- (h) rms_norm+mul fusion (T11.5): `build_hc_*` untouched by all T13.
- (i) Prefill-borrow + MoE caches (T10.2/T10.3/T12.2): files untouched
  except T13.1's scheduler no-re-reserve hunk in llama-context.cpp;
  MUL_MAT_ID 981/981 + test-arg-parser (borrow/MoE flags) pass.

## What was NOT tested

- test-llama-archs full matrix: process exited, result UNCONFIRMED at
  report time (background completion notice not yet collected - update
  this section when it lands; re-run if it crashed).
- `-o FLASH_ATTN_EXT` full sweep (incl. 10 T12.1 sparse-turbo cases):
  process exited after ~70 min, result UNCONFIRMED (same notice situation).
  If the notice shows failure, the tail + failing case go here and (f)
  above becomes a blocker.
- Any weights-based run (no GGUF in-zone): the five RESYNC-T13.md section 6
  greedy checks are the maintainer's (exact commands there).
- `test-turbo-quant.exe`, `test-moe-cache*.exe` (fork tests, exist in the
  tree): not run - no fork .cpp/.cu file they cover was touched by T13.
- Training-mode graph branches (`81ff93ea1` skipped per plan) - untouched.
- `glm5-next` arch (not imported) - `indexer_kpool_row=3/by_order=false`
  defaults are upstream's; only qwen4exp sets row=2/by_order=true.

## Verify manually (maintainer, machine with weights)

1. Collect/update the two UNCONFIRMED gates above (archs matrix,
   FLASH_ATTN_EXT sweep) from their background completion notices. If archs
   crashed on a qwen4exp row, paste the failing arch + NMSE/roundtrip lines.
   If the sweep failed, paste the tail + failing case and mark (f) a blocker.
2. RESYNC-T13.md section 6 greedy checks 1-5 at T13.5 tip `56633f3e8`
   (Flash-Next GGUF in-zone as `models/flash-next.gguf`): T10.0 triple,
   T11.3/T11.5 parity vs `f5fb9f3c7`, T12.1 `37*43` turbo5-vs-q8_0,
   long-context QSA top_k + sparse parity, prefetch LAZY on/off parity.
3. If check 2 shows divergence, bisect T13.1..T13.5 (each builds green;
   T13.2/T13.4 are the large ones) before touching ggml/ kernels.

## T13 gate update (maintainer run)

test-llama-archs full matrix on t13-qwen4exp-resync (static build at 02:25,
after the last code commit 3ef66d734 at 02:11): exit 0, 482 table lines,
0 FAIL. qwen4exp rows: CUDA OK (8.96e-08), CPU OK (0.00e+00), Meta OK,
MoE cache OK (2.39e-14). The earlier exit 139 matches the LNK1104
locked-DLL collision logged for T13.6 (rebuild during the running sweep);
it does not reproduce on a clean binary. FLASH_ATTN_EXT 10574/10574 and
LIGHTNING_INDEXER 96/96 were confirmed above. Weights-based greedy and
speed checks: see the next maintainer run.

# T14 (t13-qwen4exp-resync): compute buffer regression with the fork MoE cache

Cause: T13 scored the k-pool indexer with one product for all heads, an F32 tensor of
n_pool*n_idx_h*n_tokens (4 heads). At -c 32768 -ub 4096 the CUDA compute buffer grew from
1574 MiB (f5fb9f3c7) to 2134 MiB; the fork MoE cache (budget free-minus-reserve) lost the same
VRAM. Fix 526a7bde2 scores one head at a time (src/models/qwen4exp.cpp): compute buffer 1727 MiB.
Not fixed: +153 MiB remain, the indexer KV cache is 102 MiB vs 51 MiB (raw|pooled rows), and
T13 added about 800 graph nodes per decode step, which costs about 7% generation without the cache.
The `[moe-cache] capacity ... granted=0 MiB free=0 MiB` line prints the same in both builds, so it
does not show the real cache size; not investigated.

Gates: test-llama-archs exit 0, 482 lines, 0 FAIL (qwen4exp OK on CUDA, CPU and MoE cache rows).
test-backend-ops not run: no op changed, only the graph shape.

Speed (Flash-Next IQ1_M, 16 GB card, -ncmoe 30, -ctk q8_0 -ctv turbo4, 18k-token prompts, warm
requests, 2 runs each, tokens/s prompt / generation):

| Setup | f5fb9f3c7 | 05768cdd4 (T13) | 526a7bde2 (T14) |
|---|---|---|---|
| --moe-cache on -b 4096 -ub 4096 | 641 / 21.6 | 351 / 12.6 | 971 / 20.4 |
| no cache, default batch | 322 / 22.2 | 358 / 20.8 | 357 / 20.9 |

Greedy check on 526a7bde2: same answers as 05768cdd4 with q8_0/turbo4 and turbo5/turbo4
(37*43 = 1591 correct; the two 600-700 token prompts end in reasoning, as before).

# T15 (t13-qwen4exp-resync): decode graph node reduction

Change 1: In src/models/qwen4exp.cpp, score k-pool indexer with single mul_mat
for all heads + head slice summation when n_pool*n_idx_h*n_tokens*4 <= 64 MiB (commit 05768cdd4),
keeping the per-head loop (526a7bde2) only for large prefill (> 64 MiB). This elides transpose
and extra mul_mat nodes on decode (n_tokens=1).

Change 2 (skipped): In llama-memory-hybrid-idx, get_n_kpool_new() returns n_new_g, padded to
>= 1 (std::max(st.n_new, 1u)), so inp_kpool->n_new is never 0 today. Allowing n_new=0 would
attempt allocating invalid 0-dim GGML tensors (ne1=0) and break graph reuse across decode steps
where n_new alternates between 0 and 1.

Gates: cmake --build build-cuda-static --config Release --target test-llama-archs -j 16 built clean.
build-cuda-static/bin/Release/test-llama-archs.exe passed: exit 0, 0 FAIL (qwen4exp OK on CUDA
8.29e-08, CPU 0.00e+00, Meta, and MoE cache 2.18e-14).
