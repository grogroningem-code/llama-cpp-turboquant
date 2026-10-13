**Verdict: PASS.** No blockers remain. R3's one blocker is fixed in both files, and every code location the spec cites exists in the source.

**Each v4 claim, checked:**

1. **The getter is in `src/` and defines row count and stride.** SPEC line 16 adds a setter in `src/llama-ext.h` and `src/llama-context.cpp`, "plus a subset-logits getter there defining row count and stride." TASKS line 14 (T2.1) lists "subset-logits getter (row count, stride) in `src/llama-context.cpp`." Confirmed.
2. **It targets the draft context `ctx_dft`, using an existing pattern.** SPEC line 16 says the setter targets `ctx_dft` "same pattern as `llama_set_mtp_chain`", and TASKS line 14 says the same. That function really exists: it is declared at `src/llama-ext.h:131` and defined at `src/llama-context.cpp:4725`. Confirmed.
3. **The gathered head copy is created on the first setter call, on the same device as the source head.** SPEC line 16: "created on the first setter call, once per model, after the tokenizer check; it lives on the same device as the source head tensor, overflow following the `--fit` spill." Confirmed.
4. **The draft call at `common/speculative.cpp:2147` switches to the subset call.** SPEC line 15 names line 2147 (`common_sampler_sample`) and says it switches to the new subset-aware call. TASKS line 14 lists the switch at `common/speculative.cpp:2147`. In the code, line 2147 is `common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);`, inside the non-chained draft loop. Confirmed.
5. **The `common/sampling.cpp` call reads through the `src/` getter.** SPEC line 17 says the call "reads through the `src/` subset getter (row count and stride defined there)" and that the verify sampler is untouched. TASKS line 14 lists it. Confirmed.
6. **Both titles say v4.** SPEC line 1 and TASKS line 1 both read "v4 after Opus R3". Confirmed.

**Non-blocking notes for implementation:**

- **Mapping the top candidates back to token ids.** Right after the call, `common/speculative.cpp:2150-2152` reads the top 3 candidates through `common_sampler_get_candidates`. Once drafting uses the subset, those candidates must hold real token ids, not positions within the subset. SPEC line 20 covers this in general terms. T2.1's membership test should check it at this exact spot.
- **Where the getter is declared.** TASKS line 14 names only `src/llama-context.cpp` for the getter. SPEC line 16 implies it is also declared in `src/llama-ext.h`. Declare it in both places.
- **Where the R4 verdict goes.** T0.1 says to append it to `docs/REVIEW-SPEC-OPUS.md`, but there is also an empty `docs/REVIEW-SPEC-OPUS-R4.md`. Pick one so the record is not split. I did not write to either file.

**Sign-off before implementation starts (T0.2):** This R4 PASS is recorded in the review file T0.1 names. SPEC v4 and TASKS v4 are frozen; any change to the files the spec says to change means another review. The draft vocab file defaults to off, so with the flag off the binary must behave exactly like the unmodified one (AC1). The T2.3 test commands are agreed as the code checks: `test-turbo-quant`, `test-quantize-fns`, and `test-backend-ops -o MUL_MAT -o GET_ROWS` with zero skips on the cases the change touches. The criteria-to-task map at SPEC lines 48-54 is accepted as the definition of done. The benchmark (AC6) may be reported as blocked with a reason if no model is in the workspace, never with made-up numbers. Merging stays a manual step by the user.
