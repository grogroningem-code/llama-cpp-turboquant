# HANDOFF — draft-vocab restricted MTP head (turboquant-kv-cache slice)

Status 2026-10-12: slice implemented, reviewed (R2 PASS), exact (AC5 on 4B +
35B MoE), measured (28 runs, 2 models, 2 backends). Merge is manual by repo
owner. Next agent: start at "Open work, ranked".

## 1. Original task

Port the impactful ideas of the OpenJet megakernel fork (single-model CUDA
engine for Qwen3.8-27B: persistent cooperative decode kernel, on-device MTP
verification, bundled draft vocab; 51->78 t/s reasoning, 63->107 greedy on
RTX 3090) into `TheTom/llama-cpp-turboquant`, branch
`feature/turboquant-kv-cache`, for Qwen3.8-27B and Qwen3.6-35B-A3B on CUDA
(Vulkan if feasible). Full port was estimated at: dense clone ~2 mo, MoE
2-3 mo, generic framework 6-12 mo, Vulkan x2-3. The fork's own numbers show
the fused kernel is ~+6% no-spec (40.4->42.9) while speculation acceptance
spans 1.53x-1.93x — so the agreed smallest slice was: **draft-vocab
restricted MTP draft head** (shrink the 248320-row head to a 32768-row
gathered copy used only for MTP drafting; verify path stays full-vocab).

Standing constraints (do not renegotiate silently):
- Work ONLY inside `hermes_public_projects/llama-cpp-turboquant/` (+ its
  `../llama-cpp-turboquant-base` pristine worktree). NEVER read/list/copy
  anything outside: no `C:\Users\Elex\llama-cpp-turboquant\`, no models dir,
  no browser profiles, no other Hermes profiles. Model GGUF paths below are
  user-supplied strings used read-only at runtime; never verify by listing.
- Everything in session is PUBLIC (training data). Never echo secrets; if a
  token appears in tool output, stop and report.
- No external side effects: never push, post, upload, or publish. Local
  builds/bench only.
- Feature is default-off with kill-switch semantics: flag-off binary must
  behave as unmodified (AC1). Fail-closed everywhere: bad DV32, chained
  MTP, second spec list, repack-required head -> abort or refuse, never
  silently fall back to full head.
- Review gates via `claude -p ... --model opus --effort medium` (Claude
  subscription; NEVER OpenRouter). BLOCKER = fix loop, no silent merge.
  Opus is reviewer/advisor only.
- ASCII-only in repo files. Merge stays manual by the owner.
- OPS PERMISSION (2026-10-12, still valid): agent may kill AND relaunch ONLY
  the workspace server (`build-cuda/bin/Release/llama-server.exe` via
  `scripts/run-server-4b.bat`). Target by path via wmic, never bare
  `taskkill`. Outside PIDs (e.g. 8456/4020-class servers) are untouchable.
- Never bench while building (`-j 8` contaminates timings; one full MoE
  dataset was withdrawn for exactly this). Verify idle (`tasklist` shows no
  cl/nvcc/link/cmake) before every measurement.

## 2. Environment (verified 2026-10-12)

- Box: RTX 4070 Laptop 8GB (sm_89, CUDA 13.3, 7068 MiB free typical) +
  AMD Radeon 780M iGPU (Vulkan1, ~49GB addressable shared) + ~95GB RAM.
- Workspace: `C:\Users\Elex\AppData\Local\hermes\profiles\contributor-trained-on-my-data\hermes_public_projects\llama-cpp-turboquant`,
  shallow clone, branch `feature/turboquant-kv-cache`.
- Pristine control: `../llama-cpp-turboquant-base` @ `6e09de1` (used once to
  prove a load failure was a stale image, not code).
- Toolchain: NO VS2022/MSBuild on bash PATH; CMake generator MUST be
  `Visual Studio 18 2026` (`-G "Visual Studio 18 2026" -A x64`). Vulkan SDK
  1.4.350 at `%VULKAN_SDK%` (glslc present; all 7 glslc feature tests ON).
- Builds: `build-cpu`, `build-cuda` (`-DGGML_CUDA=ON`), `build-vulkan`
  (`-DGGML_CUDA=ON -DGGML_VULKAN=ON`). Full CUDA+Vulkan build ~40+ min.
- Models (user strings, IQ4_XS 4B / IQ4_XS 35B-MoE; n_vocab 248320 both):
  `C:\Users\Elex\llama-cpp-turboquant\build\bin\Release\Qwen3.5-4B-UD-Q4_K_XL.gguf`
  `C:\Users\Elex\llama-cpp-turboquant\build\bin\Release\Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf`
- No GGUF inside workspace. Server port for workspace runs: 9931
  (`scripts/run-server-4b.bat` forwards extra args).
- Build gotchas (all hit at least once): kill mid-compile corrupts `.obj`
  (LNK1136 -> delete file, rebuild); linking while workspace server runs
  locks DLLs (LNK1104 -> stop it first); `git stash push/pop` + MSVC
  incremental = stale exe+DLL struct mismatch (`touch` sources, rebuild, or
  clean `rm -rf build-cuda` before bisecting); after ANY suspicious load
  failure (`output.weight not found`), clean-rebuild before blaming code.
- Vulkan build gotcha (full saga in TASKS "BUILD LESSON (Vulkan)"): the
  nested `vulkan-shaders-gen` ExternalProject configure cannot see `cl.exe`
  from this shell even under vcvarsall. Workaround, build-dirs only, no
  source edits: (1) read the outer's detected flags from
  `build-vulkan/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/tmp/vulkan-shaders-gen-cfgcmd.txt`
  (all 7 `*_GLSLC_SUPPORT=ON`); (2) configure
  `ggml/src/ggml-vulkan/vulkan-shaders` manually with those `-D`s into
  scratch; (3) build+install, copy `vulkan-shaders-gen.exe` to
  `build-vulkan/Release/` (top-level `CMAKE_BINARY_DIR`, NOT the
  ggml-vulkan subdir); (4) create empty configure/build/install stamp files
  in `.../vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-stamp/Release/`;
  (5) delete stale generated `ggml-vulkan-shaders.hpp` + `*.comp.cpp`;
  (6) resume outer build. Helper bats live in profile scratch
  (`cache/scratch/vk*.bat`), NOT in the repo.

## 3. Design (see SPEC-DRAFT-VOCAB.md v4 + AC3-deviation note)

- DV32 file: magic `DV32` (`0x32335644`), version, n_vocab, count, FNV-1a
  tokenizer hash (init `1469598103934665603ull`), 32768 u32 ids. Loader
  pre-alloc checks: count 0 / count>n_vocab / size>1MB / short file rejected;
  empty/dupe/out-of-range rejected. Single-head only: chained flag-on MTP
  aborts (`GGML_ABORT`); SPEC amended, AC3 deviation recorded (no committed
  graph test for the abort path).
- Install sites: `src/models/qwen35.cpp` + `qwen35moe.cpp` (subset
  `build_lora_mm` branch emitting `result_output_subset`,
  `GGML_ASSERT(subset==0||active)`); extract in `src/llama-context.cpp`
  (`ggml_nbytes`, stride `ne[0]`); plumbing via `llama-cparams.h`
  (`spec_draft_subset`), `llama-graph.h` (cache key), `llama-model.h`
  (gathered-copy storage), `llama-context.h` (`set_spec_draft_vocab`,
  `spec_draft_vocab_ids`, `get_logits_subset_ith`), `llama-ext.h` C API
  (`llama_get_logits_subset_ith`).
- Gathered copy follows the LoRA precedent: same-device buft, explicit
  refusal when repack buffers would be needed (with `--no-repack` hint),
  `ggml_dup_tensor` + `alloc_ctx_tensors_from_buft`, WEIGHTS usage, sync
  get/set, once-per-model, disarm-only clear (no reswap).
- Sampler: `common/sampling.cpp` `set_logits_subset` +
  `common_sampler_sample_subset`; `common/speculative.cpp` uses centralized
  `spec_draft_vocab_tokenizer_hash`, loads/installs DV32 in ctor, detaches
  backend chains, rewind hook on `drop_deferred_from`. Handle
  (`spec_draft_vocab_handle`): slots, accepted-only binary_search, FIFO.
- Flags: `--spec-draft-vocab <file>` (default off),
  `--spec-draft-extra` nonzero rejected (ring/extras deferred, no v1).
  `LLAMA_SPEC_CHAIN_SUB` 32768 baseline. Explicit flag wins when absent.

## 4. What was done (with proof)

- SPEC/TASKS v1->v4 through Opus gates R1 (3 BLOCKERs) -> R2 (2) -> R3 (1)
  -> R4 PASS (`docs/REVIEW-SPEC-OPUS.md`).
- T1 loader/tool/flags/tests; T1.4 flag-off CUDA parity (temp-0 + seed-42,
  `scripts/parity-draft-vocab.sh`, footer excluded); T2.1a plumbing; T2.1b
  subset heads + sampler wiring; T2.2 vocab-freq + flag-on smoke + AC5;
  T2.3 Opus R1 FAIL (B1 repack-buffer + S1-S6/N1-N5) -> fix loop -> R2 PASS
  (`docs/REVIEW-T2.3-OPUS.md`, `REVIEW-T2.3R2-OPUS.md`); T3.2 bench proxy.
- Tests green: `test-spec-draft-vocab` (loader/range/dupe/hash/size +
  `test_bad_counts` + flags + handle) on CPU+CUDA; `ctest -R
  "spec-draft-vocab|arg-parser|turbo-quant|quantize-fns"` 4/4;
  `test-backend-ops -o MUL_MAT -o GET_ROWS` 119/119 zero skips; parity 5/5
  (temp0, seed42, mtp temp0, mtp seed42, subset-vs-full).
- Exactness: AC5 temp-0 MATCH on 4B (`AC5_TEMP0_MATCH`,
  `AC5_SEED42_MATCH`) and on live 35B MoE (`MOE_AC5_MATCH`, `--fit on`,
  quicksort prompt, n=16).
- New files: `common/spec-draft-vocab.h/.cpp`, `tools/vocab_freq/*`,
  `tests/test-spec-draft-vocab.cpp`, `scripts/parity-draft-vocab.sh`,
  `scripts/bench-interleaved.sh`, `scripts/run-server-4b.bat` (9931),
  `models/corpus-smoke.txt` (44 lines), `models/p1-prose.txt` (105KB Alice
  + analysis task), `models/p2-code.txt` (110KB repo sources + review
  task), `models/corpora/` (gitignored: HumanEval 164 tasks, Gutenberg
  alice+sherlock ~750KB), `models/draft_vocab_4b_32k.bin` (smoke, 131096B,
  sha256 `9b616bd8...e0ffc1d`), `models/draft_vocab_4b_code.bin`,
  `models/draft_vocab_4b_humaneval.bin`,
  `models/draft_vocab_4b_gutenberg.bin`,
  `models/draft_vocab_35bmoe_32k.bin` (HumanEval-derived, n_vocab 248320),
  `docs/BENCH-4B-BASELINE.md`, `docs/BENCH-PRELIM-DRAFT-VOCAB.md`,
  `docs/REVIEW-*.md`, `SPEC-DRAFT-VOCAB.md`, `TASKS-DRAFT-VOCAB.md`.
- Modified (16 files, +379/-28 at last count): `common/common.h`,
  `common/arg.cpp`, `common/sampling.cpp/.h`, `common/speculative.cpp`,
  `common/CMakeLists.txt`, `tests/CMakeLists.txt`,
  `tools/vocab_freq/CMakeLists.txt`, `src/llama-cparams.h`,
  `src/llama-context.h/.cpp`, `src/llama-ext.h`, `src/llama-model.h`,
  `src/llama-graph.h`, `src/models/qwen35.cpp`,
  `src/models/qwen35moe.cpp`.
- Corpus finding: smoke-vs-code 32k sets overlap 85% (Jaccard 0.74) — the
  head converges; per-domain sets still used for bench rigor.

## 5. Measurements (all in docs/BENCH-PRELIM-DRAFT-VOCAB.md + bench-*.log)

- 4B baselines (flag-off, `-c 8192 -ngl 99`): pp 74/35/187, tg 50/70/69
  across prose/code/long mixes; relaunch tg 49.1.
- 4B matched-corpora interleaved (n=128, temp 0): code tg 103.2->122.8
  (**1.19x**), prose 107.2->113.7 (**1.06x**). Above AC6 bars (>=1.03x code,
  >=0.99x prose).
- WITHDRAWN: `bench-moe.log` 0.93x (6 runs) — Vulkan build was compiling
  concurrently; struck from record, kept as a cautionary tale.
- 32k matrix (quiet machine, `-c 32768`, n=1024, off/on back-to-back;
  16 runs + 16 repeats = 28; full table in bench doc, log `bench-32k.log`):

  | leg | tg ratio (means) | runs |
  |---|---|---|
  | 4B CUDA p1/p2 | 1.04x / 1.11x | 1 each |
  | 4B Vulkan p1/p2 | 1.04x / 1.09x | 3 each |
  | MoE CUDA-offload p1/p2 | 0.95x / 0.97x | 1 each |
  | MoE Vulkan p1/p2 | 1.11x / 1.08x | 3 each |

  pp flat everywhere (decode-only slice, as designed). Repeat detail: iGPU
  spread ~=8% (one 4B on-run dipped below off, 18.0 vs 18.9); MoE-Vulkan on
  won 5/6 runs with tighter series than off. Working hypothesis: on Vulkan
  the head matmul is relatively pricier so the 8x shrink shows; on
  CUDA-offload MoE, expert PCIe traffic dominates (Amdahl). Caveats: prompt
  length by file size (~30k tokens vs 32768, no truncation errors); MoE legs
  `--fit` spill-bound (placement variance); pp-only outlier noted (B-p1-R2
  pp 176 vs ~153, tg unaffected).

## 6. Open work, ranked (start here)

1. **Draft acceptance study (days, this box).** Highest leverage: fork's
   1.53x->1.93x spread is acceptance, and acceptance is prompt-dependent.
   Log MTP accept counts per prompt class (emitted already; parity script
   greps them), build per-domain DV32s (`llama-vocab-freq -m <model> -o
   <out> <corpus...>`), sweep `--spec-draft-n-max` / `p_min`. Needs: real
   user prompts (ask owner) or more public corpora into `models/corpora/`.
2. **MoE-cache x subset matrix (weeks, this box).** The branch has
   moe-cache/turbokv work never benched against the subset on the 35B.
   Same 32k protocol: moe-cache on/off x subset on/off, 3 runs/cell.
   Needs: live 35B path (ask owner to confirm current filename).
3. **Head requant study (weeks).** Slice shrinks rows 8x; int4 head shrinks
   bytes; they stack. PPL-vs-speed on 4B here first.
4. **Fused persistent kernel (months, BLOCKED).** Needs Linux + 24GB VRAM;
   unrunnable/unbuildable here. Do not start without owner confirming
   hardware. Note: fork requires Linux, single model+conversation.
5. **Vulkan kernels (bad trade).** Measured +1.06-1.12x with zero kernel
   work; custom kernels cost x2-3 for single digits. Skip until 1-3 are
   exhausted.
6. **Owner-side only:** full AC6 on 27B (24GB box + user-prompt DV32),
   manual merge of this slice.

## 7. Reproduce / continue commands (run from repo root)

- Configure+build CUDA: `cmake -B build-cuda -S . -G "Visual Studio 18 2026" -A x64 -DGGML_CUDA=ON && cmake --build build-cuda --config Release --target llama-cli llama-server llama-vocab-freq -j 8` (stop workspace server first if relinking).
- Unit gates: `./build-cuda/bin/Release/test-spec-draft-vocab.exe` (-> `all tests OK`); `ctest --test-dir build-cuda -R "spec-draft-vocab|arg-parser|turbo-quant|quantize-fns"`.
- Parity: `bash scripts/parity-draft-vocab.sh <4B-gguf> <unused> <unused> <unused>` (reads script header first; 5/5 PASS incl. MTP + subset-vs-full; 5th arg = DV32 for the subset case).
- Build a draft set: `./build-cuda/bin/Release/llama-vocab-freq.exe -m "<model>" -o models/draft_vocab_X_32k.bin <corpus files...>` (-> `wrote 32768 ids (n_vocab 248320)`).
- Flag-on smoke: `llama-cli -m "<model>" -c 8192 -n 16 --temp 0 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-vocab <dv32> -p "..."` (add `--fit on` for 35B, `-dev Vulkan1` + build-vulkan binary for 780M).
- 32k bench cell: same with `-c 32768 -f models/p1-prose.txt -n 1024`; record `[ Prompt: ... ]` footer; machine MUST be idle (no cl/nvcc/link/cmake in tasklist), one job at a time.
- Devices: `llama-cli --list-devices` (expect CUDA0 + Vulkan0 + Vulkan1 from build-vulkan binary).

## 9. CORRECTION (Opus review 2026-10-12 — supersedes parts of Sec. 5-6)

The "acceptance is everything, kernel is 6%" framing in earlier sections is
WRONG and must not be handed off as fact:
- The fork's llama.cpp baseline ALREADY speculates (`--spec-type draft-mtp
  --spec-draft-n-max 2`). Both sides speculate with the same head, so
  acceptance is roughly equal on both sides. The author's alpha math
  (0.55->0.70) inverted the ratios against a non-spec baseline and is
  meaningless — discard it.
- The fork's own credit: "verifying 4-5 drafted tokens per pass at almost
  no extra cost" — depth 3-4 vs llama.cpp's 2, possible because the whole
  cycle stays on-GPU with no CPU round trip. The 1.5-1.9x is CHEAP DEEP
  VERIFY, i.e. exactly the "months, blocked" item. The 6% row is no-spec
  (memory-bound both sides) and says nothing about the speculative case.
- Step C hope ("+12% from a better 131KB file") is impossible at temp 0:
  subset drafts are a subset of full drafts under identical verify, so
  acceptance can only stay equal or drop. Per-domain sets can at best
  preserve alpha while cutting draft cost.
- Steps B/C measure only; they speed up nothing by themselves. The real
  lever in that list was Step D (depth sweep), but stock llama.cpp pays a
  verify pass + CPU round trip per extra draft token, so depth likely stays
  2-3 with modest gain.
- Target gap: nothing was ever run on the Lenovo (4090 16GB) + 27B. All
  numbers here are 4B-proxy or spill-bound 35B-MoE on the 8GB box. The 27B
  baseline is the first unmeasured thing that matters.
- pp gap: fork claims 1.2-1.5x prefill too (e.g. 1595 vs 1050 t/s at 1.8k
  ctx). This slice leaves pp flat "by design" — prefill was dropped from
  the question without justification. It is a separate lever (batched
  tensor-core prefill), still open.
- What survives review: the slice itself (1.04-1.19x matches the fork's own
  "+3-8%" credit for the draft-vocab trick — keep it), item 2 (moe-cache
  matrix), item 5 (skip Vulkan kernels).

## 11. SPEC-CHAIN (Opus findings 2026-10-12, all three verified in-branch)

- `--spec-chain N` exists (`common/arg.cpp:4158`: "all draft tokens in one
  GPU decode", default off). Round-trip removal is largely GENERIC:
  per-model gate is one line —
  `llama_model_supports_mtp_chain` (`src/llama-context.cpp:4880`) returns
  true ONLY for `LLM_ARCH_QWEN35`; `qwen35moe` excluded. On-GPU draft
  sampling already default (`--spec-draft-backend-sampling`). What still
  hits CPU is the VERIFY accept/reject (after full-logits readback).
- CONFLICT with this slice (verified, exact lines): subset install ABORTs
  on `chain_heads || chain_graph` (`common/speculative.cpp:1531-1532`);
  subset also detaches backend-sampling chains (`:1556`
  `... && !chain_graph && !subset_on`). Root cause is one mechanism in both
  places: the subset head packs rows `[n_sub]`, so any in-graph/backend
  argmax emits subset POSITIONS as if they were vocab ids. CPU sampler
  avoids this via `set_logits_subset` mapping; graph/backend paths have no
  remap. Compat work = one in-graph gather-after-argmax through the id
  table (+ sampler-side mapping). Bounded, not yet spec'd.
- Consequence: today you choose slice XOR round-trip savings. All Sec. 5
  numbers are slice-with-sequential-MTP; `--spec-chain` was never benched
  here — cheapest unmeasured test on any hardware.
- qwen35moe chain enablement = gate line + validation (whether its MTP
  hookup supports the chain math is UNKNOWN; the 35B "MTP" file has a
  single nextn layer, so `chain_heads` stays false and sequential path
  applies there today).
- Whether the 27B reports arch `qwen35` is UNCONFIRMED. The Lenovo run
  diagnoses it for free: "chained MTP is not supported for this model"
  warning = gate needs extension.
- DFlash 2: third, separate thing. Block-diffusion drafter with per-model
  TRAINED weights (27B drafter exists, PyPI `dflash`); no llama.cpp support
  (`dflash-llama` planning stage only; runs on vLLM/SGLang/Transformers/MLX
  today). Watch-item, not work — no action in this stack until a llama.cpp
  port exists.

## 12. ARCH COVERAGE (owner: slice must work for qwen3.5, MoE, qwen4/flash-next)

Measured 2026-10-12 in-branch. Rule: install (`set_spec_draft_vocab`,
`src/llama-context.cpp:2050`) is ARCH-GENERIC (uses `model.output` /
`nextn.shared_head_head` for any arch). The ONLY per-arch code is the graph
branch emitting `result_output_subset` at each `build_lora_mm` head site,
plus the one-line chain gate. Sampler mapping, DV32, verify path, backend
sampling are generic.

| arch | file | subset branch | chain gate | MTP? |
|---|---|---|---|---|
| qwen35 (3.5 dense) | qwen35.cpp:835-838 | DONE (AC5 green) | OPEN today | yes |
| qwen35moe (35B-A3B) | qwen35moe.cpp:739-742 | DONE (AC5 green) | CLOSED: gate line + validation | yes, 1 nextn layer |
| qwen3next (flash-next line) | qwen3next.cpp head sites ~223 (trunk), ~819 (MTP) | TODO: replicate assert+branch at both (both use `build_lora_mm`, same pattern) | CLOSED | yes (nextn; `mtp_only` mode :42) |
| qwen4exp | qwen4exp.cpp head site ~490 | TODO: verify site, replicate | CLOSED | likely (70 nextn/mtp hits; confirm) |
| dflash drafters | dflash.cpp (1023 lines) | N/A (separate-model draft path) | N/A | n/a (block diffusion) |

Replication recipe per arch (est. 1-2 h + AC5 run each): copy the
`GGML_ASSERT(...spec_draft_subset...); if (output_subset_active(...))`
branch from qwen35.cpp:835 to each head site; rebuild; AC5 temp-0
flag-on/off diff on that arch's model; parity script gains a case.
Graph-key (`llama-graph.h`) and extract (`llama-context.cpp`) are already
generic — no changes there.
DFlash status: branch ALREADY implements draft-dflash v1 + v2 autodetect
(`is_dflash2 = selector_top_k > 0`, :993; `--spec-type draft-dflash`,
:40). "No llama.cpp DFlash support" is stale for THIS fork. DFlash drafts
come from a separately trained drafter GGUF per target model (27B drafter
on PyPI `dflash`); subset slice does not apply to that path. Watch-item:
validate a 27B+drafter pair if owner stages both files.
OPEN QUESTION for the Lenovo leg: 27B's `general.architecture` value.
Checked 2026-10-12 via HF API (no weights pulled): `Qwen/Qwen3.8-27B`
config.json has `model_type: qwen3_5`, `architectures:
["Qwen3_5ForConditionalGeneration"]`, hidden 5120 / FF 17408 (matches the
megakernel dims). In-branch map (`llama-arch.cpp:41`) binds gguf string
"qwen35" -> LLM_ARCH_QWEN35, so a proper 27B GGUF SHOULD report qwen35 and
the slice + chain gate apply as-is. NOT confirmed from a real GGUF file
(none in zone) — runtime check stands: absence of the "chained MTP is not
supported" warning + passing tokenizer-hash load = confirmation.

## 13. CHAIN STUDY (2026-10-12/13 — answers AC1 asks)

Best settings found (flag diffs vs previous incumbent seq-2+subset):
- 4B dense: `--spec-draft-n-max 4 --spec-chain 4`, NO subset. tg 102.7 ->
  107.7 on code (**+4.9%**, 3 tight runs, wins every round); prose flat.
  Full table in `docs/BENCH-PRELIM-DRAFT-VOCAB.md`, log `bench-chain.log`.
- 35B MoE: NO CHANGE (S2V incumbent stands). Chain engages after a
  one-line gate flip (AC5 MATCH) but fails at 32k prompts with
  `invalid token[2] = -1` -> decode failure -> 0 tokens (repro
  `dbg-moe-c2.log`; short prompts unaffected). Gate REVERTED same night;
  tree verified clean (`QWEN35MOE` count in diff = 0), binaries rebuilt
  CUBUILD:0, unit tests OK. This is the next bug to root-cause (chain
  hookup vs MoE MTP mask at length), not a setting.
Fork state: 22 files, +1078/-28 over base; slice default-off, fail-closed;
chain gate qwen35-only (as upstream); no other behavior changes.
Build (current, verified this night): CUDA
`cmake -B build-cuda -S . -G "Visual Studio 18 2026" -A x64 -DGGML_CUDA=ON`
then `cmake --build build-cuda --config Release --target llama-cli
llama-server -j 8` (stop workspace server before relink; clean `rm -rf
build-cuda` after any kill/stash incident). Vulkan: same plus
`-DGGML_VULKAN=ON` into `build-vulkan`, then the shader-gen workaround in
TASKS "BUILD LESSON (Vulkan)" (manual tool build with 7 detected flags,
exe to `build-vulkan/Release/`, fake 3 stamps, delete stale hpp/comp.cpp).
Unit gates: `test-spec-draft-vocab` (`all tests OK`), `ctest -R
"spec-draft-vocab|arg-parser|turbo-quant|quantize-fns"` 4/4.
Current binaries in both build dirs are post-revert and green.

## 14. OPUS REVIEW OF MOE CHAIN FAILURE (2026-10-13, `docs/REVIEW-MOE-CHAIN-OPUS.md`)

Root cause (Opus 90%, evidence-chained, accepted): `qwen35moe.cpp` has NO
chain branch. Gate flip turns on `chain_graph` (`speculative.cpp:1519`),
chain batch rows 1+ get token 0 + zero hidden state (`:2018-2025`), and
`llama-context.cpp:2979-2981` trusts the graph to emit `[2, n_chain]`
(id, prob) pairs — but the MoE builder always returns full
`[n_vocab, n_outputs]` logits (`qwen35moe.cpp:744-748`; dense checks
`mtp_chain` at `qwen35.cpp:551`, MoE never reads it). `:2071-2072` then
reads raw logits as ids/probs; a logit in [-2,-1) truncates to id -1 ->
batch poison -> `llama_decode` failure. NOT length-dependent in mechanism:
prompt changes hidden state -> different arbitrary logits; short prompts
got lucky (or fake prob went negative and stopped the loop). Chain length
irrelevant (fails 2/3/4 alike). CORRECTION to our AC5 read: temp-0 MATCH
proves nothing (target replaces wrong drafts); 14.2 vs 24-28 t/s fits
near-always-rejected garbage drafts. All MoE chain runs were garbage
drafts, harmless at temp 0, fatal when id -1 lands in the batch.
Alternatives ranked: batch-split micro-batch hazard 7% (real hazard for
all models, doesn't fit dense-OK evidence); NaN-logit CUDA argmax -1 at 3%
(possible NEXT failure after the fix — argmax starts at -1, returns -1 on
all-NaN rows).
Fix (generic graph code, NO cuda kernel — owner was right): port chain
branch to qwen35moe (~190 lines copy, ~60 net with a shared helper taking
a feed-forward callback; per-model part ~40 lines: routed experts + gated
shared expert replace `build_ffn`; all ops exist in ggml). Loud-failure
guards (~10 lines generic): assert `t_logits->ne[0] == 2` in chain mode,
reject ids outside `[0, n_vocab)` at `:2071`. Better: gate should query
whether the graph implements chain mode, not a hardcoded arch list.
Chain+subset compat needs +15 lines in the helper (subset head -> argmax
-> subset-position-to-vocab-id lookup before `get_rows(tok_embd)`), NOT
unlocked by the MoE branch alone. NOTE for bench interpretation: dense
chain argmax defaults to first 32768 ids BY ID ORDER
(`LLAMA_SPEC_CHAIN_SUB`, `qwen35.cpp:694-709`), not by frequency — chain
vs subset compares different implicit vocabs.
Verdict: FAIL (code fix required, no setting avoids it). Confirm-before-fix
(open, no GPU work): acceptance counts from a verbose short MoE chain run
should show near-zero `draft_n_accepted` — predicted, unmeasured.

## 15. MOE CHAIN FIX LANDED (2026-10-13, guarded loop slice T4)

Spec: `SPEC-CHAIN-MOE.md` (AC1-AC4 all GREEN). Diff (3 files):
`qwen35moe.cpp::graph_mtp` +206-line chain branch mirroring dense
(`qwen35.cpp:549-739`); only the FFN differs (MoE experts + gated shared
expert). Gate re-applied for QWEN35MOE. Guards: pair-shape assert at
extract, id range check in speculative. Compile fixes: `inp_attn->mctx`
(no cast), `llama-kv-cache.h` include, public vocab-size accessors.
Validation: smoke GREEN (60 candidates, 32 accepted/64 — the confirm
Opus asked for, inverted: real acceptance, fix works); AC5 chain-2 vs
seq-2 MATCH (timestamp-only diff, now meaningful); 32k p1 LONG32K:0
coherent tokens; 4B chain-2 unchanged 111.7 t/s; ctest 4/4.
Review: `docs/REVIEW-CHAIN-MOE-OPUS.md` — PASS, no BLOCKERs, 5 NITs
(backlog in TASKS; notable: chain argmax limited to first 32768 ids by
id order on MoE too, NaN-argmax still a watch-item, micro-batch split
rechecked as impossible in current code).
Payoff (`bench-moe-chain2.log`, MoE short prompt n=256): S2V 5.0/5.0,
C2 4.9/4.9, C3 5.1/5.1, C4 4.4/4.4 — chain CORRECT, not faster; MoE best
stays S2V, 4B best stays C4. Caveat: absolute 5 t/s below earlier MoE
runs (placement differs, no --fit); A/B internally consistent.
Next: T5 chain+subset remap spec'd in SPEC-CHAIN-MOE.md (approach A:
fixed-data idmap graph input, ~60-80 lines; approach B rejected with
evidence). Needs its own slice + review + 4B/MoE validation.

## 10. Lenovo package (runs on the 16GB box, no code, ~1-2 h)

Prereqs: 27B IQ4_XS GGUF (~14GB, fits 16GB with room for ctx; confirm exact
filename on that box — placeholder `Qwen3.8-27B-IQ4_XS.gguf` below MUST be
substituted), feature-branch `llama-cli` CUDA build, one DV32 built there:
`llama-vocab-freq -m <27B> -o draft_vocab_27b_32k.bin <representative corpus>`
(all 248320-vocab Qwen3 files share the tokenizer hash check; the tool
aborts loudly on mismatch, so a wrong guess fails safe).

A. True baseline matrix (6 runs, `-c 32768 -n 1024 --temp 0`, quiet box,
   back-to-back, record `[ Prompt: ]` footer + MTP accept counts):

   for d in 2 3 4; do
     llama-cli -m <27B> -ngl 99 -c 32768 -f <long-prompt> -n 1024 --temp 0 \
       --spec-type draft-mtp --spec-draft-n-max $d
     llama-cli -m <27B> -ngl 99 -c 32768 -f <long-prompt> -n 1024 --temp 0 \
       --spec-type draft-mtp --spec-draft-n-max $d \
       --spec-draft-vocab draft_vocab_27b_32k.bin
   done

   This answers depth (2 vs 3 vs 4) AND vocab on/off on the hardware that
   matters. Expect: depth 3 helps iff verify stays cheap; vocab on ≈ same
   tokens, slightly faster tg (the +3-8% credit).
B. Profile one speculation cycle (Nsight Systems, 4B is fine — overhead
   SHARE is what transfers, not absolute times):
   `nsys profile -o mtp_cycle --trace=cuda,nvtx llama-cli -m <4B> -ngl 99 -c 8192 -p <short> -n 32 --temp 0 --spec-type draft-mtp --spec-draft-n-max 2`
   Read: gaps between kernels (CPU sync), launch counts per step, the
   full-vocab logits readback (248320 x 4B x (1+d) per step) vs matmul
   time. Large overhead share => case for C; small share => fork approach
   won't pay off on this stack, stop there.
C. Scoped proposal IF B shows overhead dominance (spec, not code yet):
   capture the verify batch in a CUDA graph, accept/reject on-GPU, read
   back only accepted token ids instead of full logits. Windows-buildable,
   16GB-friendly, no persistent kernel, no Linux. This is what makes depth
   3-4 affordable in stock llama.cpp and is the portable half of the
   fork's idea. Needs its own SPEC + Opus review before implementation.

## 8. Verify manually (could not confirm)

 - Exact prompt token counts in 32k runs (footer t/s only; files sized
  ~105-110KB as ~30k-token proxy; confirm via verbose `n_p` if challenged).
- B1 repack-refusal path judged fixed from code reading only — needs a
  CPU-resident Q4 head (`--no-repack`) to fire for real.
- AC3 abort path (chained flag-on) has no committed graph test (SPEC
  deviation recorded as accepted).
- MoE CUDA-offload neutral cells are single runs (1 each); Vulkan cells
  have 3. Promote with repeats if citing.
- Model paths/filenames are owner-supplied strings; re-confirm before each
  bench (filenames changed once already: `models/` -> `build/bin/Release/`).
- `build-vulkan` contains manual stamp/tool hacks (Sec. 2); a clean
  reconfigure will re-hit the shader-gen issue — re-apply workaround.
