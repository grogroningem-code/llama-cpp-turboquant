# RESYNC-T13: full qwen4exp re-sync to upstream #29751 + kq_mask followup

Branch: `t13-qwen4exp-resync` from `t11-qwen4exp` at `f5fb9f3c7`
(no report commit `072f5baf3`; no push).
Target: upstream `56a33d691` "llama: fix qwen4exp" + `f2ef14ea8`
"qwen4exp: keep kq_mask input the same shape"
(merged upstream as `66e0c17ee` "llama: fix qwen4exp (#29751)").

## 1. Why a hunk port is impossible (recap from REPORT-T10 T11.1)

The fork lacks the k-pool QSA machinery:

- `build_inp_kpool`, `build_qsa_sel` (qwen4exp.cpp)
- `llm_graph_input_kpool` (models.h / qwen4exp.cpp)
- `indexer_kpool_row`, `indexer_kpool_by_order`,
  `indexer_kpool`, `indexer_kpool_select_tail` (hparams)
- `kpool_layout`, `kpool_state`, `kpool_access`,
  `set_input_kpool`, `get_n_kpool*` (hybrid-idx memory)

Verified at base `f5fb9f3c7`:

- `grep indexer_kpool_row|indexer_kpool_by_order|llm_graph_input_kpool|build_qsa_sel|build_inp_kpool src/` empty.
- `src/llama-hparams.h` has no `indexer_kpool` at all
  (upstream pre-#29751 parent `552f18f91` already has
  `indexer_kpool` + `indexer_kpool_select_tail` from #27773).
- `src/models/glm5-next.cpp` absent (k-pool home model never imported).
- Fork `qwen4exp.cpp` 1620 lines vs upstream pre-fix 1310
  vs post-fix 1330; fork-vs-pre-fix diff 676 lines on
  qwen4exp.cpp alone (fork MTP/NextN + turbo + prefetch vs
  upstream k-pool + refactors).

So #29751 must come in as a full re-sync: first the missing
k-pool base + intermediate qwen4exp refactors, then #29751
itself and the followup.

## 2. Fork base and upstream range

- Fork qwen4exp base: upstream `36b101543`
  "qwen4exp: fix seq_cp, block position keying, mtmd input,
  cuda abort, add tests (#27941)", via fork catchup
  `925792c69` "qwen4exp: catch up to upstream through 36b101543"
  (2026-09-02).
- Upstream range examined:
  `36b101543..56a33d691` (591 commits total).
- Of those, commits touching the four areas
  (`src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.*`,
  `src/llama-graph.*`, `src/llama-hparams.h`, plus
  `src/models/models.h` for the arch-local QSA types):

| # | commit | date | files in scope | decision |
|---|--------|------|----------------|----------|
| 1 | `8e93a9773` CUDA + ggml: add sparse-fa for DSV4/GLM (#27970) | 2026-09-02 | qwen4exp.cpp (1-line `build_attn_mha` signature + `0` arg), llama-graph.* | SKIP (superseded). Fork already past it: T9 + T11.3 pass real `top_k->ne[0]` as `n_kv_max`, not `0`. |
| 2 | `c61b98b87` Nemotron-3-Puzzle support (#25444) | 2026-09-03 | qwen4exp.cpp (2-line `n_ff_exp` accessor rename), hparams, graph, models.h | SKIP (unrelated arch). Generic `n_ff_exp_arr` / `n_expert_used_arr` refactor for a new arch; no qwen4exp behavior change for Flash-Next. Revisit only if the build breaks on the renamed accessor. |
| 3 | `7bb0fc18f` metal: add sparse FA (#28098) | 2026-09-03 | qwen4exp.cpp (3-line TODO comment only) | SKIP (comment only, superseded by T11.3 which enables the sparse path). |
| 4 | `5fdfa6282` models: fix GDN normalization `max`->`rsqrt` (#28068) | 2026-09-06 | qwen4exp.cpp, models.h | SKIP (already in fork via T7 `c12ac3e43`). |
| 5 | `311d4211b` memory: avoid allocating V cache for indexer (#28330) | 2026-09-10 | hybrid-idx.cpp (+4: set `n_embd_head_*_mla_impl` to skip V) | SKIP (functionally present). Fork `ca9df6b90` already sets the same two MLA fields with the same effect, plus the fork `q8_0` indexer rule (T10.0) on top. |
| 6 | `41abbfd59` qwen4exp: enable rms_norm + mul fusion (#28896) | 2026-09-14 | qwen4exp.cpp | SKIP (already in fork via T11.5 `a162f4a2b`, with fork deviations documented there: F32/F16 cast guard, NextN flat path, nextn gamma shape). |
| 7 | `37b53fd45` qwen4exp: add hc ops (#28901) | 2026-09-16 | qwen4exp.cpp (`build_hc_mix` / `build_hc_combine` fused `ggml_dsv4_hc_pre/post` branches behind `cparams.fused_dsv4_hc_*`) | SKIP (deferred, not needed for #29751). Fork keeps its unfused hc path + T11.5 fusion. Importing would pull `ggml_dsv4_hc_*` ops + fused-node plumbing; #29751 does not touch `build_hc_*`, so it applies without this. Revisit as a separate perf port with weights. |
| 8 | `3cf03257f` CUDA: enable sparse fa for qwen4 (#28770) | 2026-09-20 | qwen4exp.cpp (1-line: pass `n_kv_max` instead of `0`) | SKIP (already in fork: T9 `9a6def4f3` CUDA side + T11.3 `80539b290` model one-liner). |
| 9 | `ed7ac35e1` context: do not re-reserve scheduler when toggling causal_attn (#28751) | 2026-09-28 | qwen4exp.cpp (`llm_graph_input_qsa` + `build_qsa_top_k`: new `causal_attn` param, `blk_bias` rule change, non-causal bias rule), hybrid-idx.* (`set_input_qsa` + `set_input_k_idxs` signature) | IMPORT first (T13.1). #29751's diff expects this 4-arg `set_input_qsa` as its base; without it #29751 does not apply. Small (28+/14-). |
| 10 | `649dcb103` add GLM-5.3-Flash GLM5-Next support (#27773) | 2026-09-30 | hybrid-idx.* (+604/-5: full k-pool layout/state/access), hparams.h (+2: `indexer_kpool`, `indexer_kpool_select_tail`), llama-graph.cpp (3-line stream-view + 2-line swiglu arch lists), models.h (+84: glm5-next arch + kpool types) | IMPORT second, PARTIAL (T13.2). Take ONLY: hparams 2-line hunk, hybrid-idx k-pool hunks, graph stream-view fix (needed: qwen4exp passes permuted q with streams), models.h kpool-adjacent types needed by hybrid-idx. Do NOT take: `src/models/glm5-next.cpp` (new arch, out of scope), conversion/gguf-py/mtmd/vocab/quant/saver/sweep parts, arch-list-only hunks for unrelated models. |
| 11 | `185103dcf` llama: llama_prefetch_rows (#29599) | 2026-09-30 | qwen4exp.cpp (PLE prefetch block rewrite) | SKIP (already in fork via T10.1 `068f62f08` adapted + T10.5.1 `abb0c6aed` LAZY flag). Fork gate (`lazy_has` + `can_prefetch`) differs deliberately; upstream verbatim would revert the fork adaptation. |
| 12 | `81ff93ea1` llama: properly handle KV on training (#28520) | 2026-09-30 | llama-graph.cpp only (38+/16-, training-mode KV handling) | SKIP (no qwen4exp hunk; shared graph training path, no Flash-Next behavior change; #29751 does not depend on it). Revisit only if the T13 build breaks in `build_attn_*` training branches. |
| 13 | `10f340d1a` model: re-enable -sm tensor for qwen4exp (#28569) | 2026-10-01 | qwen4exp.cpp (+2: `ggml_build_forward_expand(gf, res_hc)` after `hc_init`) | IMPORT third (T13.3). Fork lacks it (`res_hc` expand absent at `f5fb9f3c7:408`). Without it the PLE gather + `hc_init` REPEAT can split across CPU/meta devices (meta assert in the commit message). Trivial, no fork conflict. |
| 14 | `56a33d691` llama: fix qwen4exp (#29751 first commit) | 2026-09-30 | hparams.h (+4: `indexer_kpool_row=2`, `indexer_kpool_by_order=true` + validation), hybrid-idx.* (502-line rewrite: drop `set_input_qsa` block/bias path, add pooled-key path), models.h (+18/-), qwen4exp.cpp (370-line: `llm_graph_input_qsa`->`llm_graph_input_kpool`, `build_qsa_top_k`->`build_qsa_sel`, shared `build_inp_kpool`) | IMPORT fourth (T13.4). The goal commit. |
| 15 | `f2ef14ea8` qwen4exp: keep kq_mask input the same shape | 2026-10-01 | qwen4exp.cpp (4+/2-: `kq_mask` input shape fix) | IMPORT fifth (T13.5). Merged upstream as part of `66e0c17ee`; keep as a separate commit so the two upstream review units stay visible. |

Graph/hparams-only commits in range NOT listed above
(`48de2a1bc` classifier_pooling, `4da633776` RANK pooling,
`e9f824d8c` prec_policy/W4A4, `0f8a414b7` metal rescale gate,
`965506136` sched-reserve logging, plus ~15 arch-addition
commits touching only `models.h` enum lists) are all SKIP:
no qwen4exp/hybrid hunk, and #29751 applies without them.
If the T13 build fails on a moved graph/hparams symbol, the
failing commit will be pulled in and recorded in REPORT-T10.

Import order (chronological, dependency order):

1. T13.1 `ed7ac35e1` (causal_attn) - small, required base for #29751.
2. T13.2 `649dcb103` partial (k-pool base) - the big one.
3. T13.3 `10f340d1a` (-sm tensor) - trivial.
4. T13.4 `56a33d691` (#29751).
5. T13.5 `f2ef14ea8` (kq_mask shape).

## 3. Fork changes that must survive (at f5fb9f3c7)

Each checked after the re-sync (grep/build/test, plus
maintainer greedy checks where no weights-free test exists):

- (a) Own MTP/NextN head (`--spec-type draft-mtp`):
  `src/models/qwen4exp.cpp:27,143-154,185-191,264-274,286,500-708`
  (`layer.nextn.*`, `build_mtp_head`, draft-only checkpoint path,
  `t_h_nextn` export, shared `hc_head` mixer reuse).
  Upstream #29761 (Qwen4Exp MTP, `c061df198`, AFTER this range)
  is deliberately NOT imported; the fork head stays.
  Check: file still contains `build_mtp_head` + `draft-mtp`
  strings; `test-llama-archs` qwen4exp rows still pass.
- (b) PLE history (per-context n-gram history + serialise):
  `qwen4exp.cpp` PLE block (`build_ple`, `per_layer_tok_embd`,
  `ple_head_dim`/`ple_rows`), `llama-memory-hybrid` recurrent
  state handling. Upstream refactors must not drop the
  per-context history. Check: `build_ple` + `ple_n_heads`
  present; save/load-state qwen4exp tests pass if present.
- (c) TENSOR_READ_LAZY on `per_layer_tok_embd` + `llama_prefetch_rows`
  (T10.1 `068f62f08`, T10.5.1 `abb0c6aed`):
  `qwen4exp.cpp:182` flag, `:1445-1447` prefetch gate.
  Upstream #29599 verbatim is NOT re-imported (see row 11).
  Check: flag + `can_prefetch` gate present after re-sync;
  maintainer greedy check (REPORT-T10 T10.5 NOT-tested item 1).
- (d) Indexer cache `q8_0` when main cache is turbo (T10.0 `32e62edfe`):
  `src/llama-memory-hybrid-idx.cpp:63-68` `idx_type` lambda.
  Upstream has no equivalent (upstream indexer never sees turbo).
  The k-pool rewrite must keep this mapping (pooled keys are also
  indexer keys read with GET_ROWS: rotated turbo values would
  corrupt pooling scores the same way they corrupted block scores).
  Check: lambda present; no turbo GET_ROWS abort on load.
- (e) TurboQuant forward-WHT Q rotation in the QSA path for turbo K
  (T12.1 `1784e43ce`): `qwen4exp.cpp:1029-1036`
  (`k->type == TURBO*` -> pad Q, `ggml_turbo_wht` forward).
  #29751 rewrites `build_attn_qsa` into `build_qsa_sel` + pooled
  scoring; the rotation must be re-applied at the new scoring site
  (whatever Q meets pooled/raw indexer keys must be rotated when K
  is turbo, matching `llm_graph_context::build_attn` dense rule).
  Check: turbo branch present in the new QSA path; T12.1 sparse-turbo
  FA cases still pass; maintainer greedy triple (turbo5/turbo4 vs
  q8_0/turbo4) is the real regression check.
- (f) Sparse FA gather (T11.3 `80539b290`):
  `build_attn_qsa` passes `top_k->ne[0]` as `n_kv_max` (not `0`).
  #29751 keeps a sparse path (new `n_sel` budget); the non-zero
  budget must survive. Check: `n_kv_max`/`n_sel` non-zero in the
  new path; `-o FLASH_ATTN_EXT` incl. T12.1 sparse-turbo cases pass.
- (g) Tiled lightning indexer (T11.4 `3cd8ab951`):
  `ggml/src/ggml-cuda/lightning-indexer.cu` tile kernel +
  `n_head==4` dispatch + `LIGHTNING_INDEXER_TILE_TOKENS`.
  Untouched by #29751 (different file), but the new k-pool scoring
  uses plain ops for QSA ("CUDA lightning indexer takes 32/64 heads,
  QSA has a few" comment in #29751) - verify the tile kernel still
  builds and `-o LIGHTNING_INDEXER` (incl. temp nh=4 eval) still passes.
- (h) rms_norm+mul fusion (T11.5 `a162f4a2b`):
  loader gammas `[n_embd, hc]` + `TENSOR_ALLOW_RESHAPE`,
  `build_hc_mix`/`build_ple grouped_norm` direct mul (with fork
  F32/F16 cast guard + flat NextN hnorm path).
  #29751 does not touch `build_hc_*`; keep byte-identical.
  Check: `git diff` on the `build_hc_mix`/`grouped_norm` hunks empty
  (modulo #29751 context shifts); model-load shape check passes
  on weights (maintainer).
- (i) Prefill-borrow + both MoE caches (T10.2 `c5e9f4ce8`, T10.3
  `25e21fdbd`, T12.2 `1784e43ce` decode split):
  `src/llama-context.cpp` borrow state machine,
  `--prefill-borrow/--prefill-chunk`, `src/llama-moe-cache.*`,
  `ggml/src/ggml-cuda/moe-cache.cu` loan slabs.
  Untouched by #29751 (different files); re-verify with
  `-o MUL_MAT_ID` + `test-arg-parser` (borrow/MoE flags).
  Check: flags present in `--help`; MUL_MAT_ID 981/981.

## 4. Expected conflicts (to be resolved during import, recorded in REPORT-T10 T13)

- C1 (T13.2): fork `set_input_qsa` callers (fork 3-arg, T13.1 makes
  4-arg) vs upstream k-pool `set_input_kpool` replacement.
  Resolution: take upstream k-pool call, re-apply fork `idx_type q8_0`
  + turbo Q-rotation at the new scoring site (items d/e).
- C2 (T13.2/T13.4): fork MTP/NextN blocks + PLE-history + prefetch
  + turbo branches sit in hunks #29751 rewrites (graph ctor,
  `build_layer_attn` call, QSA class, `build_attn_qsa`).
  Resolution: keep fork blocks, move them to the new upstream
  structure (shared `inp_kpool`, `build_qsa_sel` signature).
  Never drop a fork block to make the upstream diff apply.
- C3 (T13.4): upstream sets
  `indexer_kpool_row=2 | pooled key` and `by_order=true`;
  fork indexer cache has no row concept.
  Resolution: take upstream hparams defaults; extend the fork
  `idx_type q8_0` rule to cover pooled-key rows (item d).
- C4 (T13.2): `src/models/models.h` upstream glm5-next arch entries
  vs fork arch list (fork MoE/arch additions).
  Resolution: keep both lists (additive merge); do NOT add
  `glm5-next.cpp` (out of scope, noted above).

## 5. Gates (per commit, then full)

- Build (VS 18 2026) after EACH commit:
  `cmake --build build-cuda --config Release` (shared, as in T10.5).
  Static `build-cuda-static` for `test-llama-archs` at the end
  (as in T10.5; that target does not exist in shared builds).
- `test-backend-ops` for every touched op after each commit;
  at minimum `-o MUL_MAT_ID` (scheduler path, repo precedent) +
  `-o FLASH_ATTN_EXT` (incl. the 10 T12.1 sparse-turbo QSA cases) +
  `-o LIGHTNING_INDEXER` after the k-pool commits.
- `test-llama-archs` on CUDA through all archs (static build) at
  T13.4/T13.5 (as in T10.5: expect exit 0, 0 FAILs).
- `test-arg-parser` after each commit (borrow/MoE flags).
- Never push. Scratch under `cache/scratch/t13/` stays untracked.
- Weights-based greedy checks are the maintainer's (no GGUF in-zone):
  exact commands go into REPORT-T10 T13 (see section 6).

## 6. Maintainer greedy checks (weights required, NOT run here)

All with a Flash-Next GGUF copy in-zone, e.g.
`models/flash-next.gguf` (any Flash-Next file; IQ1_M or Q4_0):

1. T10.0 triple (indexer q8_0 rule under the new k-pool layout),
   temp 0, fixed prompt, at T13.5 tip:
   `-m models/flash-next.gguf -c 4096 -ngl 99 -ctk turbo5 -ctv turbo4 --temp 0 -n 64 -p "Write a hello-world in Python."`
   same with `-ctk turbo6 -ctv turbo5`, same with `-ctk q8_0 -ctv turbo4`.
   Expect: all three load (no GET_ROWS abort); turbo pair identical
   to each other only if upstream guarantees it (else report first
   differing token); q8_0 run byte-identical to pre-T13 `f5fb9f3c7`.
2. T11.3/T11.5 parity: same fixed prompt, default cache types,
   at T13.5 vs `f5fb9f3c7`. Expect byte-identical (sparse + fusion
   are numerically equal by tbo).
3. T12.1 QSA rotation under k-pool: `37*43` prompt (expect 1591)
   with `-ctk turbo5 -ctv turbo4` vs `-ctk q8_0 -ctv turbo4` at T13.5.
   Expect: q8_0 byte-identical to `f5fb9f3c7`; turbo5 now sane
   (must no longer run to the token limit with empty content).
4. T11.4 indexer shape: any prompt long enough to force QSA top_k
   + sparse gather (long-context prompt), default types, T13.5 vs
   `f5fb9f3c7`. Expect byte-identical text.
5. T10.5.1 prefetch: `--tensor-read-lazy on -lv 4` on Flash-Next:
   expect `enabling prefetch for '...' (per_layer_tok_embd)` INFO
   line; greedy parity vs `--tensor-read-lazy off` byte-identical.

## 7. Done criteria / blockers

Done when #29751 (`56a33d691`) + followup (`f2ef14ea8`) are
committed on `t13-qwen4exp-resync` with all gates passing, or a
blocker is written into REPORT-T10 T13 (what failed, exact error,
which conflict could not be resolved, what was NOT tested).
