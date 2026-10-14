# Rebase A/B results (new base: snapshot bbca6c5 + control-live + candidate d94a5e644)

Protocol (owner): `-c 131072 -fa on -ctk turbo5 -ctkd turbo5 -ctv turbo4 -ctvd turbo4
--fit on -fitt 100 -b 4096 -ub 2048 --no-mmap -f models/p1-prose.txt -n 512
--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.01 --seed 42
--spec-type draft-mtp --spec-draft-n-max 2`. MoE adds `-ngl 99 -ncmoe 28`.
Vulkan rows add `--device Vulkan1`. 3 reps per cell, tg = Generation t/s.

## Gate 1: snapshot vs control (TheTom picks no-slowdown)

- CUDA units: test-turbo-quant Done, test-quantize-fns EXIT 0, kv-stream 4/4 PASS (control).
- CUDA 4B tg: SNAP 80.9 vs CTL 80.8 -> parity, PASS.
- Vulkan backend-ops MUL_MAT+GET_ROWS: SNAP 115/115, CTL 115/115, zero fails -> PASS.

## Gate 4: flag-off parity control vs candidate -> IDENTICAL (4B build-string only diff; MoE identical).

## Gate 5: AC5 temp-0 chain/subset vs seq -> IDENTICAL all 6 (4B S2V/C2/C2S, MoE S2V/C2/C2S).

## CUDA 4B owner-flags matrix (tg means, n=3)

SNAP 80.9 / CTL 80.8 / CAND 80.0 / C2 70.3 / C3 49.7 / C4 34.1 / S2V 82.3 / C2S 69.6.
- CAND vs CTL: -0.7/-0.3/-1.3 in all 3 paired reps (consistent sign, Opus-flagged).
  Parity within 1% but NOT clean: needs alternating-order CTL/CAND x6 rerun to
  separate a real ~1% deficit from run-order drift (CTL always ran first).
  -> DONE 2026-10-14: alternating A/B x6 (4B-CUDA, owner flags): CTL
  83.5/81.1/79.3/79.3/63.6*/73.4, CAND 82.9/81.4/75.2/78.8/76.3/80.3
  (*CTL thermal/burst outlier). Means: CAND 79.2 vs CTL 76.7 (with outlier) /
  79.3 (without) -> dead even. Earlier -1% sign was noise. Gate2: PASS.
  (Also visible: pp drift 2700->2300 across the session = warmup/throttle.)
- Chain hurts at temp 0.6: C2 -13%, C3 -39%, C4 -58% (Opus: PASS; mechanism
  "acceptance collapse" unproven, no acceptance numbers logged).
- S2V vs S2 same-binary: +2.8% (82.27 vs 80.03, higher all 3 reps) — same ~3% as
  Vulkan (Opus correction: cross-binary CTL baseline was wrong). Provisional:
  single seed/prompt; needs multi-seed rerun.
  -> DONE 2026-10-14: CAND S2 vs S2V x5 seeds. CUDA: S2V wins 5/5
  (S2 79.6/83.1/81.9/82.7/79.9 vs S2V 82.7/87.6/86.0/87.1/83.5,
  means 81.4 vs 85.4 = +4.8%). CONFIRMED, stronger than provisional +2.8%.

## Vulkan 4B owner-flags matrix (tg means, n=3)

CTL-S2 13.50 / CAND-S2 13.80 / C2 11.17 / S2V 14.17 / C2S 11.37.
- S2V vs S2 same-binary: +2.7% (14.17 vs 13.80). Same ~3% as CUDA (Opus correction:
  the +5%-vs-CTL framing used a cold-GPU R1 baseline; corrected). Provisional.
  -> DONE 2026-10-14: x5 seeds: S2V wins 4/5 (S2 13.7/14.0/13.4/13.1/13.9 vs
  S2V 14.2/13.2/14.1/14.5/14.1, means 13.6 vs 14.0 = +2.9%) but noisy incl. one
  -5.7% loss: stays provisional.
- Chain hurts here too (C2 -17% vs CAND-S2).

## Vulkan MoE owner-flags matrix (tg means, n=3)

CTL-S2 11.57 / CAND-S2 11.37 / C2 10.20 / S2V 11.17 / C2S 10.13.
- Parity across the board with one correction (Opus): the R3 dip is CAND-specific
  (CTL-S2 R3 11.7 was its BEST run; CAND cells dipped). Cause unknown — R3 CAND
  numbers flagged, not contention-proven.
- Chain hurts (C2 -12% vs cleanest S2 reads).

## CUDA MoE matrix (-c 32768 valid regime; tg, n=3, heavy env noise, see note)

SNAP 25.4/26.4/11.4 / CTL 24.8/26.4/12.9 / CAND 19.6/9.2/29.5 /
C2 23.3/21.7/24.4 / C3 8.6/15.9/20.9 / C4 17.0/4.8/17.1 /
S2V 20.1/20.0/21.9 / C2S 19.3/5.2/19.9.
- Gate 1 MoE: pairwise SNAP/CTL parity every round (both dip together in R3) -> PASS.
- CAND-S2 vs CTL-S2 anti-correlated across rounds (19.6/9.2/29.5 vs 24.8/26.4/12.9):
  environment-dominated, no binary signal separable. Best quiet-window numbers
  agree ~25-29 for all three binaries.
- C2 ~23 stable-ish, below S2. S2V ~20.6 stable (below S2 ~20% on this MoE).
- Chain C3/C4 collapse at temp 0.6.
- NOTE: -c 131072 on this 8GB box crawls for all binaries (SNAP 9.6 / CTL 9.4-10.5 /
  CAND 9.7, gate1 CTL 38.2/9.9): owner 131k protocol is Lenovo-16GB-only for MoE.
  Placement verified healthy (CPU 9968 MiB / CUDA0 6415 MiB + compute 1004 MiB):
  the crawl is 131k KV/compute pressure, not flags. -c 32768 probe: 95.9/26.1.
- KV-arena attempt (--kv-stream-arena-mib, user's steer): 512/256/128 all fail
  context creation ("arena leaves insufficient free VRAM for the streamed
  attention kernel"); -ncmoe 34 + -b 2048 still fails ("no KV capacity after
  reserving compute"). 131k MoE needs more VRAM than this box has, arena or not.
  Stays Lenovo-only.
- R1 131k cells VOID (regime-invalid, kept only as no-crash evidence).

## CUDA MoE matrix: 131k VOID on 8GB, 32k table below (scripts/bench-moe-cuda-ab.sh).

## Opus final review (2026-10-14, opus-review-final.log): (a) PASS (b) PASS (c) PASS.
- Merge-blocker raised + RESOLVED: S2V text change at temp 0.6. Root cause is the
  harmless one: speculative.cpp:2208-2213 uses the subset sampler ONLY for draft
  proposals on ctx_dft (gated by draft_vocab); verify path stays subset=false;
  temp-0 AC5 MATCH proves target argmax identical; divergences are late
  (1300-2200 chars identical prefix, synonym-level); S2V repeat byte-identical
  (3356/3356, seed 42). Different drafts -> different accepted samples. No leak.
- Accepted gaps (not blocking): no acceptance-rate logging (can't separate
  cheaper-draft vs more-accepts); S2V ~neutral on MoE (opt-in flag, fine).

## GPU+iGPU MIX (--device CUDA0,Vulkan1, MoE -c 32768, candidate Vulkan build)

CTL-S2 7.2 / CAND-S2 8.3 / CAND-C2S 6.2 (tg, n=1).
- Runs clean, no errors, but SLOWER than CUDA-only (~25) and Vulkan-only (~11.5).
  Split transfer overhead dominates on this box. Works, not beneficial.

## Build lessons (new)

- Snapshot worktree renamed `llama-cpp-turboquant-snap` -> `lct-snap` (shorter).
- Nested vulkan-shaders-gen configure fails under shell MSBuild (no cl on PATH).
  Recipe: build tool via scratch `vkgen-manual.bat` (vcvarsall x64, all seven
  `*_GLSLC_SUPPORT=ON` from `.../vulkan-shaders-gen-prefix/tmp/*-cfgcmd.txt`,
  `TrackFileAccess=false` for FTK1011, short BLD `C:/vkg` for C1041 PDB length),
  install exe to `build-vulkan/Release/`, touch the 8 per-config stamps under
  `.../vulkan-shaders-gen-stamp/Release/`, resume outer build.
- NEVER reuse a build-vulkan dir across a rebase: old-base objects + new headers =
  ABI mismatch, cli segfaults on --version. `rm -rf build-vulkan` after any rebase.
- Background shells die silently when -j8 MSBuild exits (no EXIT echo); verify by
  binaries + idle procs, not by the echo line.
- Header-only struct changes (new field in common.h) do NOT trigger rebuild of
  TUs with only transitive includes (cli-context.obj stayed 2026-10-13): result
  is silent OOB write + garbage reads (probabilistic=109/61/112 varying
  run-to-run). After any shared-header layout change, delete stale .objs of
  direct includers (map via grep '#include' + find -name basename.obj) and
  rebuild, then re-probe the value.

## T6: upstream PR 27694 probabilistic draft + rejection verify (2026-10-14)

- Port: `--spec-draft-sampling {greedy,probabilistic}`, default greedy
  (flag-off identical). Commits `318d04928` (port) + `978960ad0` (Opus-review
  fixes: MTP begin reset, result_q cleared with cleared drafts, ckpt-replay
  gap recorded in SPEC-PROB-DRAFT.md).
- AC2 (temp 0): G vs P byte-identical 3205/3205 chars PASS (flag wiring only;
  rejection never runs at temp 0 by design).
- AC3 (temp 0.6, 4B owner flags, n=512): single-text 4 rounds G 80.8/80.5/
  82.5/82.0 vs P 89.7/89.5/92.8/90.4 (+11.2%, 4/4, both orders).
- M1 multi-seed (seeds 42-46, 10 distinct texts): G 83.1/85.3/81.4/83.4/84.6
  vs P 92.1/85.4/87.2/89.5/86.3, means 83.6 vs 88.1 = +5.4%, P wins 5/5
  (range +0.1%..+10.8%, gain scales with text predictability). HONEST NUMBER.
- Opus impl review: core math faithful, greedy not slowed (81.5 vs CTL 80.8),
  chain fallback + subset interaction correct. Known gap: ckpt-restore drops
  accepted tokens to exact-match recheck (speed lost on that path only).
- Verdict: real, consistent gain at temp 0.6; keep flag (default-off).
  Flag for this box: add `--spec-draft-sampling probabilistic` to the S2V set.

## T7: upstream batch 27978/29638/28068/29393 (2026-10-14, commits c12ac3e43 + fa0f5e530)

- #29638 EOG stop (match path exact port + same guard in T6 rejection path);
  #27978 mmid any-count fast path (byte-identical to upstream); #28068 GDN
  fix via build_gdn_l2_norm (5 models present; bailingmoe3/kimi-k3 absent);
  #29393 RMS_NORM+SCALE CUDA fusion (fuses the GDN pairs).
- AC1/AC2: build green, arg-parser + spec-draft-vocab green.
- AC3: 4B coherent, MoE no crash/no -1. MoE fit-refused identically pre+post
  (box condition, also fails pre-port; 95.9/26.1 era numbers were old-base).
- AC4: 4B pre vs post neutral (PRE 1490/42.0, POST-G 1489/43.6, POST-P
  1506/45.2; whole box in a depressed regime that day, pre+post equally).
- AC5: temp-0 pre vs post differs (1525 chars, content differs) = GDN live.
- Opus review: no blockers, all faithful, rejection-EOG deviation correct.
  NITs closed: comment style; fused-vs-CPU-ref temp-0 text identical (fused
  kernel exact; GGML_CUDA_DISABLE_FUSION does NOT exist in this tree, CPU
  -ngl 0 is the reference); n_used=10 MUL_MAT_ID added, 2/2 OK on CUDA0;
  alice ppl pre 1.6299 vs post 1.6219 (within +-0.015, no degradation).
- Value of batch is correctness (EOG, GDN), not speed. No new flag.