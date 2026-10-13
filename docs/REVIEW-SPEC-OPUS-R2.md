**Verdict: 2 blockers remain, both left over from R1 blocker 1. v2 adds no new blockers.** Line numbers below are for the v2 files, checked against the current code.

## Remaining BLOCKERs

**1. Nothing in the plan passes the id list from `common/` into `src/`.** This was part of the fix R1 asked for and is still missing (SPEC:10-16, TASKS:14).
- The new loader and set handle live in `common/` (SPEC:15-16). But the subset projection, the id map, and the copy of the head gathered at load time (SPEC:22) are all built in `src/`. No step in the plan connects the two.
- The draft head runs in a separate MTP context, and today the only way to control it from outside is `llama_set_mtp_chain()` in `src/llama-ext.h:131`.
- The plan needs a new API listed in `src/llama-ext.h` and `llama-context.cpp`, plus a new field in `llama-cparams.h`. That API should take the id list and say when the gathered head is created. It also has to say which device buffer the gathered head goes into when the model is only partly on the GPU (AC2).
- The id map becomes a graph input, so `src/llama-graph.cpp` (where inputs are filled) also belongs in the file list.

**2. Smaller draft logits would still be read as full-vocabulary logits.** This is the memory-safety problem from R1, moved to a new place (SPEC:13, SPEC:17).
- SPEC:13 only protects the copy into the context's logits buffer. On the non-chained path, the draft is sampled with `common_sampler_sample()` (`common/speculative.cpp:2147`). That function reads `n_vocab` floats from `llama_get_logits_ith()` (`common/sampling.cpp:138-156`).
- With logits sized to the subset, it reads past the end of the row and assigns the wrong token ids. "Draft path consumes subset positions via the map" (SPEC:17) names no code that would do this.
- This affects every `qwen35moe` draft. `llama_model_supports_mtp_chain()` returns true only for `LLM_ARCH_QWEN35` (`llama-context.cpp:4718`), so the MoE model always takes the non-chained path.
- Fix: list `common/sampling.cpp` and a getter API for the subset logits and the map. Or choose the other option R1 offered: a full-size buffer filled with -inf outside the subset, which leaves the sampler unchanged.

## R1 items confirmed fixed in the text
- **Blocker 2 (existing restriction):** fixed at SPEC:11 and SPEC:18. The chained path already emits `[token id, prob]` pairs from an in-graph argmax (`qwen35.cpp:711-723`), so the in-graph map from subset position to token id is the right place for it.
- **Blocker 3 (file format):** fixed at SPEC:8.
- **Items 4-11:** fixed at SPEC:7, SPEC:22, SPEC:21, SPEC:8 with SPEC:30, SPEC:40, SPEC:44-51, SPEC:41, and TASKS:8.
- **NITs:** the model name is explained at SPEC:4, `--spec-draft-vocab-off` is gone, the review path is fixed at TASKS:4, and the test filters are named at TASKS:16.

Two points are not blockers but still need a fix:
- AC5 (SPEC:40) compares "flag-on drafts vs flag-off drafts". The drafts are meant to differ, so that comparison will fail. Only the tokens the target model outputs should match.
- SPEC:11 does not say which wins on the chained path when both `LLAMA_SPEC_CHAIN_SUB` and `--spec-draft-vocab` are set.

Verify manually:
- AC3 (SPEC:38) and the temperature-0 check in AC5 require identical results with the flag on and off. When drafts differ, the target model checks batches of a different size, and CUDA kernels can give slightly different numbers for different batch sizes. I could not run a model here, so I have not confirmed whether exact equality holds.
- `docs/REVIEW-SPEC-OPUS.md` lines 77-86 are a leftover duplicate of the R1 NIT and "Verify manually" sections. TASKS:4 says to append R2 to that file. I left it unchanged, as you asked.
