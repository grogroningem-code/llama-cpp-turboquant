# TASKS: Draft-vocab MTP head (tracer slices) - v4 after Opus R3

Phase 0 - gates (unattended, Opus-sub medium, no OpenRouter):
- T0.1 Opus re-review SPEC v4 + TASKS v4 -> record verdict in `docs/REVIEW-SPEC-OPUS.md` (append R4, keep R1-R3). BLOCKER = stop.
- T0.2 DoD sign-off recorded by reviewer (code gates + AC map in SPEC). No silent merge.

Phase 1 - loader and flags (no model needed):
- T1.1 DONE 2026-10-12 (CPU build, 4/4 gates green): `tools/vocab_freq/vocab_freq.cpp` + `common/spec-draft-vocab.h/.cpp` (magic/version/n_vocab/hash/size/id-range/no-dupe checks). Test `tests/test-spec-draft-vocab.cpp` RED then GREEN.
- T1.2 DONE 2026-10-12: struct fields in `common/common.h`, `--spec-draft-vocab` + reserved `--spec-draft-extra` in `common/arg.cpp`, default off; flag behavior covered in `test-spec-draft-vocab.cpp::test_flags`, `test-arg-parser` passes.
- BUILD LESSON 2026-10-12: after git stash push/pop cycles MSVC incremental left exe+DLL disagreeing on struct layout (moe-cache assert misfire); fix = `touch` touched sources and rebuild dependents before trusting a test verdict.
- BUILD LESSON 2026-10-12 (stale image): a CUDA build dir once killed mid-compile (LNK1136 corrupt obj, obj deleted) kept producing binaries that failed model load (`output.weight not found`, then instant-death) even after `touch` + full relink with zero errors; pristine-base binary loaded the same file. Fix = `rm -rf build-cuda`, reconfigure, clean rebuild (decisive: fresh binary loaded first try). When a load failure survives a clean relink, suspect the image before the code.
- T1.3 DONE 2026-10-12: `spec_draft_vocab_handle` in `common/spec-draft-vocab.h/.cpp` (per-sequence slots, accepted-only, position-stamped entries, FIFO overflow with no restore, `drop_deferred_from` rewind hook). RED (undeclared-type compile fail) then GREEN in `test-spec-draft-vocab.cpp::test_handle` (AC4).
- T1.4 DONE 2026-10-12 on live 4B (`Qwen3.5-4B-UD-Q4_K_XL.gguf`, user-staged
  outside zone, path taken on trust): our flag-off CUDA `llama-cli` vs
  pristine-HEAD `llama-cli` (worktree `llama-cpp-turboquant-base`, commit
  6e09de1). `parity PASS: temp 0 greedy` + `parity PASS: temp 0.7 fixed
  seed` (timing footer excluded — first run proved tokens identical and only
  t/s differed). AC1 holds with all T2.1a/b code present but off.

Phase 2 - draft-only restriction (model graph):
- T2.1a DONE 2026-10-12 (CPU `llama` target builds clean): plumbing slice, inert
  when off. `spec_draft_subset` hash in `src/llama-cparams.h` + default 0 +
  graph-key compare in `src/llama-graph.h`; once-per-model gathered-copy
  storage on `llama_model` (mutable, RAII ctx+buf); `set_spec_draft_vocab`
  + `spec_draft_vocab_ids` on `llama_context` (id validation, FNV-1a,
  idempotent reinstall, MTP head source = first nextn layer shared-head or
  model output, buft-matched gather via tensor_get/CPU row-pick/tensor_set);
  C API in `src/llama-ext.h`. Gating rule: subset nodes are built only when
  the explicit flag is set; flag-off graph untouched.
- T2.1b DONE 2026-10-12 (CPU `llama`+`llama-common`+`test-spec-draft-vocab`
  build clean; 4/4 gates green `spec-draft-vocab|arg-parser|turbo-quant|
  quantize-fns`): subset matmul branch in both non-chain heads; both extract
  sites handle `[n_sub, n]`; `get_logits_subset_ith` + C API (stride defined
  in `src/`); `output_reorder` subset row width; `common_sampler_sample_subset`
  via shared impl (backend early-out full-vocab only); MTP ctor loads DV32 +
  installs subset + owns handle (loud abort on bad file/chain combo);
  backend chains detached under subset; draft callsite switch; rewind hook.
- T2.2 DONE 2026-10-12 on live 4B (clean-rebuild CUDA `llama-cli`,
  `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-vocab
  models/draft_vocab_4b_32k.bin`, sha256 `9b616bd8...e0ffc1d`, 131096 bytes):
  flag-on smoke EXIT 0, 16 sane tokens; AC5 `AC5_TEMP0_MATCH` (temp 0) +
  `AC5_SEED42_MATCH` (temp 0.7 seed 42), flag-on final tokens identical to
  flag-off (timing footers excluded). Smoke DV32 is synthetic (`models/
  corpus-smoke.txt`); regenerate from real prompts before the AC6 bench.
- T2.2 Exactness test per SPEC AC5 (no KL without named distributions).
- T2.3 Build: `cmake -B build -DGGML_CUDA=ON`, affected targets. Gates: `test-turbo-quant`, `test-quantize-fns` pass; `test-backend-ops -o MUL_MAT -o GET_ROWS` (plus turbo SET_ROWS names if touched) with zero skips on touched cases. Opus review diff -> append `docs/REVIEW-SPEC-OPUS.md`. Fix loop until no BLOCKER.
- T2.3 R1 verdict 2026-10-12: FAIL, 1 BLOCKER (B1 repack-buffer gather segfault) + S1-S6 + N1-N5, saved in `docs/REVIEW-T2.3-OPUS.md`. Fix loop DONE same day: B1 extra-buft refusal, S1 pre-alloc checks + `test_bad_counts`, S2 refuse-reswap/disarm-only clear/graph asserts/API note, S3 WEIGHTS usage, S4 SPEC single-head amendment, S6 parity MTP + subset-vs-full cases, N1 empty-list reject, N3 centralized hash, N4 comment trims, N5 row_w. Explained: S5 (no weights in unit tests; covered by live AC5 + harness), N2 (optional per reviewer). Re-verify: CPU+CUDA clean, `test-spec-draft-vocab: all tests OK`, parity 5/5 PASS (temp0, seed42, mtp temp0, mtp seed42, subset-vs-full). R2 re-review pending.
- T2.3 R2 verdict 2026-10-12: PASS (saved by worker to `docs/REVIEW-T2.3-OPUS.md` successor note; reviewer sandbox denied the write, verdict taken from stdout). B1/S1-S4 fixed, S5 accepted with the SPEC AC3 deviation recorded above, N2 accepted. New NITs N6-N10 fixed same day: review labels removed, parity usage + accept-count compare for flag-off MTP cases, repack refusal logs `--no-repack` hint + clearer abort, getter docs describe real behavior, unneeded const_cast + dead clear() removed. T2.3 DONE.

Phase 3 - Lenovo bench (needs model inside workspace):
- T3.2 bench proxy DONE 2026-10-12 on 4B (matched public corpora, interleaved
  on/off): code 1.19x, prose 1.06x, both above the AC6 bars; see
  `docs/BENCH-PRELIM-DRAFT-VOCAB.md`. Full AC6 on the 27B target needs 24GB
  VRAM (not available on this box) plus a user-prompt DV32; merge stays manual.
- MoE validation DONE 2026-10-12 on live `Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS`
  (user path string, `--fit on`, 95GB RAM absorbs experts): HumanEval DV32
  (`models/draft_vocab_35bmoe_32k.bin`, n_vocab 248320), flag-on smoke EXIT 0
  with sane tokens, AC5 `MOE_AC5_MATCH` (temp 0 flag on/off identical).
  Vulkan leg: `build-vulkan` (CUDA+Vulkan, SDK 1.4.350) compiling for 780M
  validation.
- 32k matrix DONE 2026-10-12 (quiet machine, `-c 32768`, n=1024, 16 runs,
  `bench-32k.log`): 4B tg 1.04-1.12x on CUDA and Vulkan; MoE CUDA-offload
  neutral 0.95-0.97x; MoE Vulkan/780M +1.06x both prompts; pp flat
  everywhere. See `docs/BENCH-PRELIM-DRAFT-VOCAB.md`. Single run per cell.
- BUILD LESSON (Vulkan): nested `vulkan-shaders-gen` ExternalProject
  configure fails under this shell (no `cl` on PATH at the custom step) even
  with vcvarsall in the invoking .bat. Workaround (build dirs only, no
  source edits): configure the tool manually with the outer's 7 detected
  `*_GLSLC_SUPPORT=ON` flags (read them from
  `.../vulkan-shaders-gen-prefix/tmp/vulkan-shaders-gen-cfgcmd.txt`),
  install exe to `build-vulkan/Release/`, fake the 3 stamps, delete stale
  generated hpp/comp.cpp, resume outer build. Also: tool path uses
  top-level `CMAKE_BINARY_DIR` (`build-vulkan/Release/`), NOT the
  ggml-vulkan subdir.
- T3.1 Stage model GGUF under `llama-cpp-turboquant/models/` (user provides; never read outside zone). Record sha/size.
- T3.2 Baseline vs flag-ON: `-c 8192/16384 -fa on -ctk q8_0 -ctv turbo3 --spec-type draft-mtp --spec-draft-n-max 2 --fit on`. Record pp/tg, acceptance, splits, n_ctx, gathered-head VRAM delta, `p_min` behavior.
- T3.3 Threshold verdict over 3+ runs: code tg >= 1.03x baseline, prose tg >= 0.99x baseline. AC5 exactness per SPEC (temp 0 equality, temp > 0 fixed-seed match).

Exit gates:
- Code merge only after T2.3 clean + human manual merge.
- AC6 bench may stay BLOCKED if no model in workspace; report explicitly, do not fabricate numbers.
- T4 MoE chain fix DONE 2026-10-13 (SPEC-CHAIN-MOE.md, guarded loop, no
  checkpoints): ported the dense chain branch to `qwen35moe.cpp::graph_mtp`
  (+206 lines, MoE FFN + shared expert the only difference), one-line gate
  re-applied for QWEN35MOE, two loud-failure guards (pair-shape assert at
  extract, id range check in speculative). Compile fixes: `inp_attn->mctx`
  instead of casting base `mctx`, added `llama-kv-cache.h` include, used
  `llama_vocab_n_tokens(llama_model_get_vocab(...))` in common. Validation:
  smoke GREEN (60 candidates, 32 accepted / 64), AC5 chain-2 vs seq-2 MATCH
  (timestamp-only diff), 32k p1 chain-2 LONG32K:0 coherent tokens, 4B
  chain-2 unchanged 111.7 t/s, ctest 4/4. Opus review PASS, no BLOCKERs,
  5 NITs (docs/REVIEW-CHAIN-MOE-OPUS.md). Payoff bench (MoE short prompt,
  n=256, `bench-moe-chain2.log`): S2V 5.0/5.0, C2 4.9/4.9, C3 5.1/5.1,
  C4 4.4/4.4 - chain correct but not faster; MoE best stays S2V, 4B best
  stays C4. Caveat: absolute tg (5 t/s) below earlier MoE runs - placement
  differs (no --fit here); A/B internally consistent.
- T5 chain+subset remap DONE 2026-10-13 (SPEC-CHAIN-MOE.md T5, Opus R1 found
  1 BLOCKER wrong get_rows shape, fixed as prescribed, R2 PASS
  `docs/REVIEW-T5-OPUS.md` + `REVIEW-T5R2-OPUS.md`): persistent I32 idmap
  tensor in setter, dense+MoE remap with position/id split, abort lifted
  for chain_graph only. 4B C2S clean, 36/64 accepted, AC5 C2S vs S2V MATCH
  30/30. Regression check: pristine base S2 4.8 vs ours 5.3 (same regime)
  - our diff does NOT slow the fork. Placement root-cause found instead:
  ngl99 oversubscribes 8GB VRAM (16.75GB buffer, fit fails, sysmem
  fallback crawl); ngl15 fits (20.5); Opus-advised ncmoe28 -> 26.6 tg
  (`docs/ADVICE-MOE-SPEED-OPUS.md`). Final MoE best: plain S2
  ngl99+ncmoe28; subset/chain/stacked at/below noise. 4B best stays C4.
- BACKLOG (Opus NITs 1-3,5): n_last update on guard fire; loud assert for
  future micro-batch split; NaN-argmax watch (only if -1 reappears);
  shared-helper extraction (post-gate wide work, owner decision).
- OPS PERMISSION 2026-10-12: user allows the agent to kill AND relaunch the
  workspace server (`build-cuda/bin/Release/llama-server.exe` via
  `scripts/run-server-4b.bat`) itself when builds need the DLLs unlocked
  (Windows file locking) or testing needs a fresh server. Scope is ONLY that
  binary/bat; never touch other servers. Target it by path via wmic, not bare
  taskkill. Model path inside the bat is a user-supplied string, never
  verified from inside the zone.
