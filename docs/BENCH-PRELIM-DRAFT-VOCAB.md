# Preliminary draft-vocab bench (4B proxy, smoke DV32) - 2026-10-12

NOT the AC6 gate bench. Preliminary mechanics run with the synthetic smoke
draft set (`models/draft_vocab_4b_32k.bin` from `models/corpus-smoke.txt`).
Same binary, temp 0, n=128, -c 2048, -ngl 99, full-head MTP (off) vs subset
MTP (on). Raw log: `bench-prelim.log` (workspace root, gitignored scratch).

| cond | prompt | run1 | run2 | run3 | mean |
|------|--------|------|------|------|------|
| off | prose | 102.9 | 103.0 | 104.2 | 103.4 |
| on | prose | 99.9 | 111.2 | 112.1 | 107.7 |
| off | code | 101.0 | 97.2 | 85.8 | 94.7 |
| on | code | 123.1 | 123.3 | 116.4 | 120.9 |

Ratios (on/off): prose 1.04x (gate >= 0.99x), code 1.28x (gate >= 1.03x).

Confounds (do not cite these numbers as the AC6 result):
- Ordering: all off runs preceded all on runs, so GPU warmup favors on.
- Corpus: smoke DV32 is synthetic; acceptance rate (hence speedup) depends
  on the real prompt distribution.
- Variance: off-code spread 85.8-101.0 across 3 runs is unexplained noise;
  n=3 is thin, and run1-on-prose (99.9) undercuts the trend.
- Proxy: 4B mechanics only; thresholds were sized for the 27B target.

Next: regenerate the DV32 from real prompts, interleave on/off runs, raise
n, then judge AC6.

## Matched-domain run (same day, public corpora)

HumanEval prompts -> `draft_vocab_4b_humaneval.bin` (code), Gutenberg
(alice+sherlock) -> `draft_vocab_4b_gutenberg.bin` (prose); each prompt type
benched against its own set, on/off interleaved, n=128 temp 0 (`bench-matched.log`):

| cond | code (HumanEval set) | prose (Gutenberg set) |
|------|----------------------|-----------------------|
| off | 104.1 / 104.0 / 104.1 / 100.7 (mean 103.2) | 108.5 / 103.6 / 109.3 / 107.5 (mean 107.2) |
| on | 123.7 / 120.5 / 123.7 / 123.2 (mean 122.8) | 113.6 / 113.1 / 115.0 / 113.1 (mean 113.7) |

## 32k-context matrix (same day, quiet machine, 16 runs, `-c 32768`, n=1024)

Prompts `models/p1-prose.txt` (105KB Alice + style analysis) and
`models/p2-code.txt` (110KB repo sources + code review), off/on back to back
per cell, full log `bench-32k.log`. Machine verified idle (0 build procs)
before leg D; no builds ran during the matrix.

| leg | pp off -> on | tg off -> on | tg ratio |
|---|---|---|---|
| 4B CUDA p1 (1 run) | 2834 -> 2626 | 82.1 -> 85.0 | 1.04x |
| 4B CUDA p2 (1 run) | 2473 -> 2417 | 91.3 -> 101.6 | 1.11x |
| 4B Vulkan/780M p1 (3 runs) | 320 -> 318 | 18.9 -> 19.7 | 1.04x |
| 4B Vulkan/780M p2 (3 runs) | 307 -> 318 | 19.9 -> 21.7 | 1.09x |
| MoE CUDA-offload p1 (1 run) | 218 -> 220 | 25.8 -> 24.4 | 0.95x |
| MoE CUDA-offload p2 (1 run) | 202 -> 202 | 29.5 -> 28.7 | 0.97x |
| MoE Vulkan/780M p1 (3 runs) | 161 -> 153 | 13.4 -> 14.9 | 1.11x |
| MoE Vulkan/780M p2 (3 runs) | 148 -> 151 | 16.0 -> 17.2 | 1.08x |

Repeat detail (Vulkan legs, tg): 4B p1 off 18.8/19.0/18.9, on
21.1/20.0/18.0 (R3-on dips below off: iGPU variance is real, ~=8%); 4B p2
off 20.3/20.6/18.9, on 22.2/20.7/22.2; MoE p1 off 14.2/14.2/11.7 (off dips,
on stays 15.0/15.0/14.8); MoE p2 off 15.8/16.0/16.2, on 16.7/17.3/17.7
(on wins all 3). One pp-only outlier (B-p1-R2-off pp 176 vs ~153, tg
unaffected).

Read: prefill is flat everywhere (subset changes decode only, as designed).
Decode: 4B gains on both backends (~1.1x); MoE over CUDA-offload is neutral
(~0.96x, single runs); MoE over Vulkan gains mildly (~1.06x both prompts).
Working hypothesis for the split: on Vulkan the head matmul is relatively
pricier (less-tuned kernels), so shrinking it 8x shows up more; on
CUDA-offload MoE, expert PCIe traffic dominates and the head saving is
Amdahl-irrelevant. Caveats: ONE run per cell (no repeats), prompt length by
file size (~30k tokens vs `-c 32768`, no truncation errors), MoE legs are
`--fit` spill-bound so placement variance applies.

## Chain study (2026-10-12/13, `--spec-chain` vs sequential MTP, log `bench-chain.log`)

Cells: S2V = seq-2 + subset (incumbent), S2 = seq-2, C2/C3/C4 = chain-2/3/4
(no subset — the abort at `speculative.cpp:1531` forces it). `-c 32768`,
n=1024, temp 0, quiet machine throughout.

4B CUDA (3 runs on p2-code, 1 run on p1-prose):

| cell | p1 tg | p2 tg (3 runs) | p2 mean |
|---|---|---|---|
| S2V | 84.7 | 102.1 / 103.4 / 102.7 | 102.7 |
| S2 | 81.0 | 91.7 (1 run) | — |
| C2 | 82.9 | 101.5 / 102.0 / 101.3 | 101.6 |
| C3 | 83.6 | 103.8 / 104.1 / 102.1 | 103.3 |
| C4 | 77.5 | 107.7 / 108.4 / 107.1 | 107.7 |

Read: on code, round-trip removal (C2, 101.6) ≈ subset saving (S2V, 102.7)
— the two optimizations do not stack, and the conflict is now priced.
Depth pays iff acceptance is high: C4 best on code (**107.7, +4.9% over
incumbent, wins all 3 rounds**), C4 worst on prose (77.5 vs ~83). No
fallback warnings: chain engaged on qwen35. pp roughly flat (~2.5k).
**New best 4B setting: `--spec-draft-n-max 4 --spec-chain 4`, no subset.**

MoE CUDA-offload (1 run/cell): S2V 25.1/27.8, S2 24.2/28.2 (subset
neutral, as before); pre-flip C3 (fallback seq-3) 20.5/26.8 (depth costs
without chain). Gate flip (`+QWEN35MOE`, then REVERTED) engaged real chain
on MoE: AC5 chain-2 vs seq-2 MATCH (`MOE_CHAIN_AC5_MATCH`), no warnings.
But 32k-prompt chain runs failed: `invalid token[2] = -1` -> `llama_decode:
failed to decode` -> 0 tokens (`Generation: 1000000.0 t/s`), repro
`dbg-moe-c2.log`. Short-prompt/32k-ctx chain works and matches sequential
exactly — the -1 is LENGTH-dependent (long ~30k prompt), not ctx-dependent.
Root cause unknown (in-graph argmax/sampling edge on MoE at length).
Gate reverted, binaries rebuilt (CUBUILD:0, unit tests OK). MoE chain is a
found bug with repro, not a setting. **MoE best stays S2V (incumbent).**
Next: root-cause the -1 (chain hookup vs MoE MTP mask at length), then
re-attempt enablement; or ship the in-graph remap (Sec. 11) which fixes
chain+subset compat for all arches at once.

## MoE run (same day, live 35B-A3B, HumanEval set, interleaved, n=64)

`bench-moe.log`: on 14.7 / 15.3 / 17.1 (mean 15.7), off 14.4 / 18.9 / 17.4
(mean 16.9). Ratio 0.93x. WITHDRAWN as a measurement: the Vulkan build was
compiling (`-j 8`) on the same machine through all 6 runs, so CPU contention
contaminates every number. The Amdahl read below is hypothesis only until a
quiet-machine rerun. Lesson recorded twice: never bench while building.

## ngl placement cliff + fast regime (2026-10-13, `bench-ngl-sweep.log`)

MoE short-prompt runs sat at tg ~5 t/s (both our binary AND the pristine
base: 4.8 vs 5.3, so NOT a regression from our diff). Verbose load shows
`offloaded 42/42 layers` but `CUDA0 model buffer = 16754.70 MiB` on an 8GB
card, plus `fit could not prove a viable placement; restoring the pre-fit
parameters` (with and without `--fit`); smi during decode: 569MB VRAM,
GPU 46% at full clocks. Diagnosis: `-ngl 99` oversubscribes VRAM, Windows
WDDM pages weights to RAM, everything crawls.
Sweep (S2 short, n=96): ngl15 pp 97.3/tg 22.7, ngl25 34.2/8.5,
ngl35 38.2/6.1, ngl99 48/5. Cliff between 15 and 25. Rule: size -ngl to
real VRAM (here ~15 layers); re-probe if VRAM availability changes.
In-regime A/B at ngl15 (`bench-moe-fastregime.log`, n=128): S2 20.5,
S2V 21.1, C2 20.6, C2S 20.5, C3S 16.3. No errors anywhere. Read: all
equal within noise; deeper chain (C3S 0.80x) hurts on short prompts.
**MoE best stays S2V; 4B best stays C4.** 32k confirmation of C2S/C3S
pending (killed one slow-regime attempt at pp 50: same paging trap).
Opus advice on next speed levers requested (`docs/ADVICE-MOE-SPEED-OPUS.md`,
pending at time of writing).

## Fast-regime verdict (2026-10-13, Opus-advised, `bench-ncmoe*.log`)

Opus refined the mechanism (sysmem fallback, not paging; smi untrusted
under WDDM) and predicted lever #1: `-ngl 99 --n-cpu-moe N` (attention +
shared + router + KV on GPU, only routed experts of first N layers on
CPU) at ~1.3x. Measured (S2 short, n=96-128): ncmoe34 23.3, ncmoe28 27.1,
ncmoe22 7.9 (overshoot crawl). Prediction exact.
Final 3x-interleaved at ncmoe28 (`bench-ncmoe-final.log`), tg means:
S2 26.6 (25.1/27.4/27.4), S2V 24.0 (20.7/22.2/29.1),
C2S 22.9 (23.0/20.5/25.3). Variance high (knife-edge VRAM); gaps 10-14%,
marginally real. **MoE best on this box: plain S2 at ngl99+ncmoe28.**
Subset/chain/stacked do not pay for MoE decode here (their value may
appear at 32k or on the 27B target). T5 stands as correct-but-not-faster.
User-side: NVIDIA Control Panel `CUDA - Sysmem Fallback Policy` to
`Prefer No Sysmem Fallback` turns overshoot into a fast crash instead of
a silent crawl. Untested next (Opus ranked): thread sweep, expert-only
requant, prompt-lookup drafting, turbo KV at 32k.

## Theirs-vs-mine A/B (2026-10-13, user-ordered, Tiel MoE, user cache flags)
Theirs = C:\Users\Elex\llama-cpp-turboquant\build\bin\Release\llama-cli.exe
(10871 af0acacd7, HAS --spec-chain, NO --spec-draft-vocab: tree behind this branch).
Mine = build-cuda build. Flags: -c 32768 -fa on -ctk q8_0 -ctv turbo3 --fit on -fitt 100.
CUDA paging regime (16.75GB model vs 8GB VRAM, fit refuses on BOTH binaries):
 NOSPEC theirs 3.4 = mine 3.4. S2 theirs 3.8 / mine 5.2 (both crawl, noise).
 C2 theirs 5.2 = mine 5.2, zero errors both. C2S mine-only (theirs rejects the flag).
CPU (-ngl 0, n=32): S2 theirs 14.5 / mine 14.1 (parity); C2S mine 16.1 (+11-14%), clean.
Vulkan1 780M (mine, build-vulkan): S2 21.4 / C2 21.5 / C2S 22.0, all clean.
Vulkan-iGPU beats CUDA-dGPU here (model fits 46GB shared, no spill).
VERDICT: no fork slowdown anywhere measurable; mine-only stacked C2S is the
fastest CPU config and the fastest Vulkan config on this box.
Harness lesson: ${var%%:*} parsing breaks on C:\ paths (bin became "C",
exit 127); CPU offload is -ngl 0, "--device CPU" is a usage error.

## Proper-config CUDA A/B (2026-10-13, spill via -ncmoe 28, -fitt 200, Tiel, pp+tg)
Flags: -c 4096 -ngl 99 -ncmoe 28 -fa on -ctk q8_0 -ctv turbo3 --fit on -fitt 200.
 fit refuses placement on BOTH binaries (keeps explicit split) — environmental,
 16.75GB model vs 8GB VRAM, not a fork difference.
 THEIRS-S2F pp111.0/tg29.4 | MINE-S2F pp112.0/tg28.6 (parity).
 THEIRS-C2F pp108.5/tg28.9 | MINE-C2F pp106.4/tg26.5 (parity-ish, single runs).
 MINE-C2SF pp109.7/tg25.4, clean (mine-only feature).
 Earlier CUDA 3-5 t/s numbers RETRACTED: -ngl 99 with no spill path, invalid config.
Mixed CUDA0+Vulkan1 spill (mine, build-vulkan, same flags, --device CUDA0,Vulkan1):
 MIX-S2 pp126.0/tg14.2 | MIX-C2S pp121.4/tg15.3, clean. Syntax works, but slower
 than CPU spill (28-29) and pure-Vulkan1 (21-22). Hypothesis: row-split across
 devices adds transfer overhead; worst of both here. Not pursued further.
Standing: proper CUDA ~28-29 both binaries (parity); Vulkan1 ~22 mine-only;
 CPU ~14-16 (C2S best). No fork slowdown in any valid config.

## Turbo5/4 round (2026-10-13, user KV flags, sampling recipe, pp+tg)
Recipe: temp 0.6 top-p 0.95 top-k 20 min-p 0 seed 42, -fa on, -b 4096 -ub 2048.
DIVERGENCE FOUND: theirs supports KV turbo5/turbo6 + ctkd/ctvd; this branch
 stops at turbo4 ("Unsupported cache type: turbo5"). Their tree is NEWER on
 turbo-KV, this branch is NEWER on spec/subset/chain-MoE. Manual merge must
 unite both (user owns merge).
4B pure-fit (-c 131072): theirs turbo5/4 pp2673/tg87.7 | mine turbo4/4
 pp2701/tg85.8. PARITY (pp +1%, tg -2%).
MoE spill (-ncmoe 28, -fitt 200): theirs pp125.4/tg30.6 | mine pp126.2/tg29.8.
 PARITY. Mine-C2S-stacked pp129.9/tg23.4 clean (mine-only).
Turbo KV itself is huge: 4B tg 50-57 (q8_0/turbo3) -> 86-88 (turbo4/5).
Same-binary run-to-run swing (theirs tg 57.7->41.9, pp 1555->1320) exceeds the
 inter-binary gap: co-tenancy with live servers on shared DDR5; keep aux idle.
FINAL A/B VERDICT: fork parity in every valid config (4B pp+tg, MoE pp+tg,
 CPU, chain); mine-only C2S fastest on CPU/Vulkan; no slowdown anywhere.
