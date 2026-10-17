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
