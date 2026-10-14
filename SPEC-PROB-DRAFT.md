# SPEC-T6: probabilistic draft + rejection verify (port of ggml-org/llama.cpp#27694)

## What (upstream, merged 2026-10-02, NOT in our tree)
- Drafters today offer their argmax; target accepts only on exact match.
  At temp > 0 acceptance collapses.
- PR adds: draft samples probabilistically (q), target verifies by rejection
  sampling accept w.p. min(1, p/q), residual from norm(max(0, p-q)).
  Output distribution preserved exactly (Leviathan et al. construction).
- Applies to draft-simple and MTP only (Eagle3 unevaluated, DFlash structural
  no-benefit). New flag `--spec-draft-sampling {greedy,probabilistic}`,
  default greedy (vLLM convention). Upstream: temp 0 identical tokens;
  gains rise with temperature.
- Full upstream diff: cache/scratch/pr27694.diff (+304/-11, 7 files).

## Port map (upstream file -> ours, anchors verified 2026-10-14)
1. common/arg.cpp: add `--spec-draft-sampling` after `--spec-draft-backend-sampling`
   (same builder pattern). Env LLAMA_ARG_SPEC_DRAFT_SAMPLING.
2. common/common.h: `common_params_speculative_draft::probabilistic = false`.
3. common/sampling.cpp: `#include <random>`; `common_sampler::rng` (mt19937);
   init seeded `llama_sampler_get_seed(chain) ^ 0x9e3779b9u` (NOT the chain seed:
   draft/target streams must differ); clone + copy carry rng.
   `common_sampler_get_seed` already exists (line 762) - reuse.
4. common/sampling.cpp + .h: new `common_sampler_sample_and_accept_n_rejection`
   (verbatim port of PR hunk; grammar-mask + residual logic included).
5. common/speculative.h: `common_speculative_draft_params` += `result_q`
   (ptr to per-token candidate dists), `temp` (1.0), `seed` (DEFAULT_SEED).
6. common/speculative.cpp draft_simple (198-408, structure matches PR base):
   `smpls_cfg` vec + `spec_retune` helper (top_k 10 + temperature chain,
   seed mixed ^ 0x85ebca6bu); begin() resets; process(): if !probabilistic,
   result_q=nullptr (greedy path byte-identical to today); if result_q set,
   retune + skip reset + capture cur_p per token + use sampled id.
7. Same for draft_mtp (1397+, incl. our chain-graph region: port applies to the
   MTP sampling site; subset path (line 2208-2213): when draft_vocab is on AND
   probabilistic, result_q captures the SUBSET candidates - rejection math stays
   valid (q supported on subset). Chain path: leave greedy-only (user: no chain).
8. common_speculative_draft (3271+): truncate result_q with result (PR hunk).
9. tools/server/server-context.cpp (CLI runs through server ctx in this tree):
   slot.spec_draft_q; use_spec_rejection() = temp > 0; wire result_q/temp/seed
   into draft params; verify switch (synth > replay > rejection > match).
   Port replay helper only if our tree has the ckpt-restore path it serves;
   else skip with note.

## Acceptance
- AC1: builds CUDA + Vulkan, all existing tests green.
- AC2 (temp 0): probabilistic run byte-identical to greedy (upstream control).
- AC3 (temp 0.6, owner flags, 4B): probabilistic tg > greedy tg (upstream: gains
  rise with temp; our regime temp 0.6 must show uplift or we report no-gain).
- AC4: flag absent = byte-identical to control binary (default greedy).
- AC5: MoE smoke, no `-1`/crash; chain untouched.
- Out of scope: Eagle3/DFlash, acceptance-rate logging (accepted gap), server
  ckpt-replay helper unless trivially portable.

## Test plan (4B, owner flags, -c 131072, p1, n=512, temp 0.6)
- greedy (flag off) x2 vs probabilistic x2, pp+tg; then temp-0 AC2 pair.
- If AC3 fails (no uplift): report numbers, keep flag (upstream feature,
  default-off, no regression) - do NOT tune further without user steer.
