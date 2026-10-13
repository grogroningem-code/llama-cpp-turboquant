**Verdict: 1 BLOCKER remains, and it is narrow.** The R2 blocker about passing the id list from `common/` into `src/` is fixed. The R2 blocker about the sampler is only half fixed: the new getter is placed in the wrong layer.

All the referenced files exist: `src/llama-ext.h`, `src/llama-context.cpp`, `src/llama-cparams.h`, `src/llama-graph.cpp`, `src/llama-graph.h` and `common/sampling.cpp`.

## R2 blocker 1 (no API from common/ into src/): FIXED
- SPEC:16 lists the full chain: a setter in `src/llama-ext.h` and `src/llama-context.cpp`, a new field in `src/llama-cparams.h`, and the id map filled as a graph input in `src/llama-graph.cpp`.
- It also says when the gathered head copy is created and which device holds it under partial offload: the same device as the source head tensor, with overflow following the `--fit` spill.
- TASKS T2.1 (line 14) lists the same files.

## R2 blocker 2 (sampler reads subset logits as full vocab): STILL OPEN, narrower
- **Fixed part:** SPEC:17 names `common/sampling.cpp`, adds a draft-sample call that understands the subset, and leaves the verify sampler untouched. It also records why the full-size -inf buffer option was rejected.
- **Getter in the wrong layer:** SPEC:17 places "a getter for subset logits" under `common/sampling.cpp`. The logits live in `llama_context`, and today `common/` reaches them only through `llama_get_logits_ith` (`common/sampling.cpp:153`), which reads `n_vocab` floats per row. `common/` cannot read the subset row without a new public function in `src/llama-ext.h` and `src/llama-context.cpp`. That function must also define the subset row's length and stride. Neither SPEC:16 nor T2.1 lists it. The id map does not need a getter, because `common/` already holds the id list.
- **Call site not listed:** the non-chained draft is sampled at `common/speculative.cpp:2147` with `common_sampler_sample`. SPEC:15 lists `speculative.cpp` only for the set handle, so nothing says this call switches to the new subset call. Until it does, every `qwen35moe` draft still reads past the end of the subset row.
- **Fix (text only):** add "subset logits getter (row count and stride)" to the SPEC:16 API list and to T2.1. Add "switch the draft call at `common/speculative.cpp:2147` to the subset sample call" to SPEC:15 and T2.1.

## Other R2 items: FIXED
- AC5 (SPEC:43) now compares only the target model's final tokens and expects the drafts to differ. T3.3 matches.
- Precedence is stated at SPEC:18: the explicit flag wins over the `LLAMA_SPEC_CHAIN_SUB` environment variable, and the bench records both. T2.1 repeats it.

## Minor issues (not blockers)
- SPEC:16 says the setter takes "the id list (or its hash handle)". A hash alone cannot rebuild the ids, so it should take the id list. The setter should also name its target context: the draft context `ctx_dft`, like the existing `llama_set_mtp_chain` (`src/llama-ext.h:131`).
- SPEC:16 says the gathered head copy is created "at model load", but the ids only arrive through the context setter, which runs after the model has loaded. It should say "on the first setter call, once per model".
- Both file titles (SPEC:1, TASKS:1) still say "v2 after Opus R1", and T0.1 still says "SPEC v2 ... append R2". These should say v3 and R3.

I have not written anything to `docs/REVIEW-SPEC-OPUS.md`.

Verify manually: the earlier doubt is still open. Exact temperature-0 equality in AC3 and AC5 may break on CUDA, because the target model checks batches of different sizes when the drafts differ. Confirming it needs a model run, which I could not do here.
