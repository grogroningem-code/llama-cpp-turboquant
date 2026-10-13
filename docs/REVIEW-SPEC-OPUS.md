**Verdict: not ready to implement. There are 3 blockers.** The main one is that the spec puts the change in `common/speculative.cpp`, but the matrix multiply it wants to shrink is built inside the model graph in `src/`. I checked each finding below against the code.

There is already an untracked review in `docs/REVIEW-SPEC-OPUS.md`. I re-checked its claims instead of trusting them. Its main claims hold, but its note that the file "is empty" is out of date.

## BLOCKER

1. **SPEC:9, SPEC:11, TASKS:13: the file list is wrong, and the spec hides work in the model graph.**
   - The draft head's output projection over the vocabulary (the "lm_head" matmul) is built in `src/models/qwen35.cpp:832-835` and `src/models/qwen35moe.cpp:736-738`. `common/` only calls decode and reads the logits back.
   - On the non-chained path, the logits copy assumes the output is `n_tokens x n_vocab` (`src/llama-context.cpp:2360`, `2822`, `3022`). If the graph returns a 32k-row output, that copy reads past the end of the tensor. This is a memory-safety bug, not only a speed concern.
   - The graph-reuse comparison (`src/llama-graph.h:843-853`) has no field for the token subset. A changed subset would silently reuse a stale graph.
   - Fix: list `src/models/qwen35*.cpp`, `src/llama-context.cpp`, `src/llama-graph.h`, and the public API call that passes the id list in. Also state how the logits are returned: either subset-sized logits with a mapping back to token ids, or a full-size buffer filled with -inf outside the subset.

2. **SPEC:9-10: the spec ignores the vocabulary restriction that already exists.**
   - `qwen35.cpp:694-709` already limits the chained draft path to the first `LLAMA_SPEC_CHAIN_SUB` rows. It defaults to 32768 and is on by default.
   - The in-graph argmax (`:711`) and the next-token embedding lookup (`:723`) work on positions inside the subset. With an arbitrary id list, both need a mapping from subset position back to the real token id.
   - The softmax (`:712`) runs over the subset only, so draft confidence rises and the `p_min` cutoff behaves differently.
   - The MoE file has no such restriction, so the two architectures start from different baselines.
   - AC1 ("flag off equals baseline") must say which baseline it means.

3. **SPEC:7, SPEC:10: "refuse on `n_vocab` mismatch" cannot work with the planned file format.** A raw int32 array contains no vocabulary size and no tokenizer identity. A file built with a different tokenizer passes the `< n_vocab` check. Fix: add a header with magic, version, `n_vocab`, count and a tokenizer hash, and require all of them to match.

## SHOULD-FIX

4. **SPEC:8, SPEC:10: the extras ring makes the graph shape change at every step.** The set changes each step, which forces a graph rebuild and breaks CUDA graph reuse. A simpler v1 drops the ring and uses a fixed set, gathered once at load time into a quantized copy of the head. If the ring stays, use a fixed-size padded slot table passed in as a graph input.

5. **SPEC:9, SPEC:11: the gather method and its memory cost are not specified.** Calling `ggml_get_rows` on a quantized head every step dequantizes it to f32. That works against AC2 on an 8GB card. Put the VRAM cost of the gathered head into AC2.

6. **SPEC:10 (rewind): the semantics are incomplete.** The spec must state:
   - one ring per sequence, because the server can run parallel slots (`-np`);
   - only tokens the target accepted enter the ring, never unverified drafts;
   - each entry stores the position it was added at;
   - ids evicted by overflow are not restored after a rewind.

   There is no rewind API to hook. Hook the same place the existing trim runs: `drop_deferred_from` (`common/speculative.cpp:1653`, called at `:1777`, `:1945`, `:2067`).

   The spec should also say that a rewind bug costs only acceptance rate, not output correctness.

7. **SPEC:19: the loader bounds are incomplete.** Add these checks:
   - every id is at least 0 (the ids are signed);
   - file size equals header plus count x 4 bytes;
   - no duplicate ids;
   - count is no larger than `n_vocab`.

   The 1MB cap is fine: 32768 ids take 128KB.

8. **SPEC:29: AC5 tests a step that does not exist.** Acceptance is a plain "draft token equals the token the target sampled" check (`common/sampling.cpp:665-674`). There is no reject-and-resample step. The output is already exact by construction, so test it directly:
   - temperature 0: identical tokens with the flag on and off;
   - temperature above 0: a fixed-seed comparison of token distributions.

   "KL < 0.001" (KL measures how far apart two distributions are) does not say which two distributions are compared.

9. **TASKS: the test plan does not prove AC1-AC7.**
   - AC1: no task compares flag off against the unmodified binary.
   - AC2: no task starts the server with `--fit on` or records `n_ctx`.
   - AC3: "verify logits unchanged" needs a model test through the graph. The fake-vocabulary test in T1.3 cannot cover it.
   - AC5: inherits the problem in item 8.
   - Neither file has a table mapping each AC to a task.

10. **SPEC:30: AC6 has no pass/fail threshold.** "+3-8 pct" is a target, not a criterion. Use something like "token generation on code prompts at least 1.03 times baseline, prose at least 0.99 times baseline, over 3 or more runs".

11. **TASKS:8: build wiring is missing.** The plan needs CMake entries for `tools/vocab_freq`, for `common/spec-draft-vocab.cpp` in `common/CMakeLists.txt`, and for the test in `tests/CMakeLists.txt`.

## NIT

- **SPEC:4:** "Qwen3.8-27B" looks wrong. The `qwen35` architecture is Qwen3.5/3.6.
- **SPEC:10:** `--spec-draft-vocab-off` does nothing when the default is already off. Drop it, or say what it overrides.
- **TASKS:4:** the review path `docs/REVIEW-SPEC.md` does not match the file in the tree, `docs/REVIEW-SPEC-OPUS.md`.
- **TASKS:14:** name the `test-backend-ops` filters to run, for example `-o MUL_MAT` and `-o GET_ROWS`. Per AGENTS.md, a green run without filters does not prove the touched cases ran.

## Scope creep

There is no hidden Vulkan or full-model work: SPEC:11 and SPEC:22 rule both out. The only hidden scope is the `src/` graph and context work in blocker 1.

Verify manually:
- I did not check whether the MoE chained path has another vocabulary limit somewhere other than `qwen35moe.cpp`.
- I did not read the model metadata, so the size of the per-step dequantize cost in item 5 is unconfirmed.
e `qwen35` architecture is Qwen3.5/3.6 (`qwen35moe.cpp:18`). Please check the model name.
- **SPEC:10:** `--spec-draft-vocab-off` does nothing if the default is already off. Either drop it or say what it overrides (presets or environment variables).
- **TASKS:4:** the review verdict path is `docs/REVIEW-SPEC.md`, but the file in the tree is `docs/REVIEW-SPEC-OPUS.md`, and it is empty.
- **TASKS:14:** "touched ops only" is fine, but name the test filter (for example `-o MUL_MAT`, `GET_ROWS`), per the AGENTS.md coverage notes.

**Scope creep check:** I found no hidden Vulkan or full-model change; SPEC:11 and SPEC:22 rule those out. The only hidden scope is the model-graph work in blocker 1.

Verify manually:
- I did not check whether the `qwen35moe` chained path (`llama_model_supports_mtp_chain`) has its own vocabulary restriction. I only found `LLAMA_SPEC_CHAIN_SUB` in `qwen35.cpp`.
- The 670 MB figure assumes n_embd 5120 for the 27B model. I did not read the model's metadata to confirm it.

---
R2 verdict (docs/REVIEW-SPEC-OPUS-R2.md): 2 blockers remain, both from R1 blocker 1 (no common/-to-src/ API; sampler misread). Fixed in SPEC v3/TASKS v3 except getter placement and call site.
R3 verdict (docs/REVIEW-SPEC-OPUS-R3.md): 1 narrow blocker remains (getter in wrong layer + call site unlisted). Fixed in SPEC v4/TASKS v4.
R4 verdict (docs/REVIEW-SPEC-OPUS-R4.md): PASS, no blockers. DoD sign-off: SPEC v4 + TASKS v4 frozen; flag-off parity (AC1) required; code checks are test-turbo-quant, test-quantize-fns, test-backend-ops -o MUL_MAT -o GET_ROWS with zero skips on touched cases; AC6 may be BLOCKED without model, never fabricated; merge stays manual by user. Recorded 2026-10-12, reviewer opus/medium via Claude CLI.
