# T2.3 re-review (R2): draft-vocab restricted MTP head (Opus)

Saved by the worker: the reviewer's sandbox denied writing this file, so the
verdict below is taken verbatim from the reviewer's stdout.

**Gate: PASS.**

**B1 (the blocker) is fixed.**
- **Code:** `src/llama-context.cpp:2111-2128` now looks up the CPU's extra buffer types (repack, AMX, KleidiAI) the same way `llama-adapter.cpp:307-316` does. If the head sits in one of them, the setter returns false. This happens before any allocation and before the `ggml_backend_tensor_get` that used to crash, and the caller then aborts.
- **Spec:** SPEC line 16 now describes this refusal, so it matches the code.
- **Limit:** the check only covers the CPU's extra types. The only other backend that has them is Hexagon, which is outside the CPU+CUDA scope.

**S1-S4, S6, N1, N3-N5:**
- **S1, fixed:** the loader now checks count 0, count above the vocabulary size, size above 1 MB and file too short, all before it allocates (`common/spec-draft-vocab.cpp:72-98`). `test_bad_counts` reaches each of these checks, and its asserts run in Release builds because of `#undef NDEBUG`.
- **S2, fixed:** clearing now only turns the subset off and never frees the shared copy. Installing a different list is refused, and both `graph_mtp` builds assert that the subset is active whenever the context says it is on. The API comment notes that the setter is not thread-safe.
- **S3, fixed:** the gathered copy is marked as weights (`:2142`), so the scheduler keeps the matmul next to it.
- **S4, fixed:** SPEC lines 11, 18 and 21 now say flag-on is single-head only, which matches the abort.
- **N1, N3, N5, fixed:** an empty list is rejected, the tokenizer hash has a single definition, and the row stride is named `row_w`.
- **S6, partly fixed:** the harness now runs the MTP draft path and the subset-vs-full case. It still does not compare draft and accept counts in the flag-off cases (see N7 below).
- **N4, mostly fixed:** comments are shorter, but the fix added new review labels (N6).

**The two explained items:**
- **S5, accepted on one condition.** Draft candidates can only come from the installed list (`common/sampling.cpp:168-183`). The target context never sets the subset field, so verify logits cannot change, and the repo has no small MTP model to test through the graph. The harness case only shows identical final tokens, so write this deviation down where AC3 is tracked (the SPEC or the T2.3 line in TASKS). Otherwise a later gate check will think a graph test exists.
- **N2, accepted.** It was optional and matches the spec.

**New issues in the fix hunks (all NITs):**
- **N6:** the fix added review labels to comments: "(B1)" at `src/llama-context.cpp:2113`, "(S1)" at `common/spec-draft-vocab.cpp:72`, and T1.4, S6, AC1 and AC5 in the parity script.
- **N7:** `scripts/parity-draft-vocab.sh:61-62` says accept counts "are printed for info only", but `run_case` sends stderr to `/dev/null`, so nothing is printed. In the flag-off MTP cases the counts should be identical between the two binaries, and comparing them is the check R1 asked for. The usage line also does not mention the optional fifth argument (the draft-vocab file).
- **N8:** when the repack refusal triggers, the abort says "missing head or bad ids" (`common/speculative.cpp:1544`). That names the wrong cause and does not point the user to `--no-repack`.
- **N9:** the docs at `src/llama-ext.h:140,145` say both getters return null or 0 "when off". In fact `spec_draft_vocab_ids` returns the model's list even when this context has turned the subset off. `get_logits_subset_ith` sets the count before its check, and in debug builds it aborts instead of returning null. Nothing breaks today, because the only caller asserts.
- **N10:** the `const_cast` at `src/llama-context.cpp:2052` is unneeded because the fields are already `mutable`. The `clear()` at `:2152` is overwritten seven lines later.

Verify manually: I did not run the repack refusal on a real repacked head, because that needs a Q4_0 or Q4_K output head kept on the CPU on an AVX2 machine. B1 is judged fixed from reading the code.

Worker note: N6-N10 were all fixed the same day (see TASKS T2.3 R2 line); S5's condition was recorded in SPEC (AC3 map) and TASKS.
