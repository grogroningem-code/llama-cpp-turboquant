# T5 review: chain drafting plus subset head (Opus)

**Verdict: FAIL, with 1 BLOCKER.** The design is sound, but the remap call has the wrong shape. Once the subset head is installed and the build gets past model load, every chain graph build will abort. The fix is one line, applied in two files. Everything else checks out apart from one SHOULD-FIX and some NITs. I checked all of this by reading the source. Nothing was built or run.

## Findings

**1. BLOCKER: the idmap lookup returns n_sub values instead of one id, so the graph build asserts.**
At `src/models/qwen35.cpp:723` and `src/models/qwen35moe.cpp:798` the code calls `ggml_get_rows(ctx0, model.output_subset_ids_w, id_j)`.
- The idmap is 1-D `[n_sub]`, so `ne0 = n_sub` and `ne1 = 1`.
- `ggml_get_rows` (`ggml/src/ggml.c:3974`) builds its result as `[a->ne[0], b->ne[0]]`, which here is `[n_sub, 1]`, not a single id. It picks row `id_j` from a tensor that has only one row, so any position above 0 reads out of bounds.
- The next line, `ggml_reshape_2d(id_real_j, 1, 1)`, asserts `nelements == 1` (`ggml.c:3736`). With n_sub = 32k that is false, so `GGML_ASSERT` fires on the first chain graph build.

The `p_j` line already does this correctly by reshaping the softmax to `[1, n]` before `get_rows`. The idmap needs the same treatment:
`ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, model.output_subset_ids_w, 1, model.output_subset_ids_w->ne[0]), id_j)`.
That returns a `[1, 1]` I32 tensor. It then passes both `reshape_2d(...,1,1)` and `get_rows(tok_embd_w, ...)` (that call needs `a->ne[2] == b->ne[1]`, which is 1 == 1). Acceptance check 1 (the 4B chain-plus-subset run) would have caught this immediately, so it has not been reached yet.

**2. Checked and correct: the idmap's lifetime, buffer and device (item a).**
- `output_subset_ids_w` sits in the same `output_subset_ctx` context and buffer as `output_subset_w`. It lives as long as the model, and the setter never frees it.
- The setter's metadata pool is `3*overhead` for 2 tensors, which is enough.
- The buffer is the head's buffer type, flagged as weights, so the scheduler runs the lookup on the head's backend.
- I32 `get_rows` is supported on CUDA (`ggml-cuda.cu:6677`, `getrows.cu:311/434`) and on CPU (`ops.cpp:5093`).
- `ggml_backend_tensor_set` fills it once. The repack guard covers the buffer type it shares with the head.

**3. Checked and correct: types and shapes (item b), apart from finding 1.**
`argmax` over `[n_sub, 1]` gives a `[1]` I32 tensor, which is a valid index. The I32 source gives an I32 result (`ggml.c:3971`). Casting it to F32 for the pair is exact while ids stay below 2^24 (Qwen's vocabulary is about 152k, so this holds).

**4. Checked and correct: subset position versus real vocab id at all three uses (item c).**
In both files:
- `p_j` uses the subset position `id_j`, which matches the `[n_sub]` softmax.
- The emitted pair and the `tok_embd` lookup that feeds the next chain row both use the real id `id_real_j`.

No use is left on the wrong one. The host reader needs no change. Its new range check (`speculative.cpp:2080`) would catch any leaked subset position only if that position were 152k or more, which it never is, so it would **not** catch this mix-up. The in-graph code is the only protection.

**5. Checked and correct: the guard (item d).**
The assert `spec_draft_subset == 0 || output_subset_active(hash, head_w2)` fails closed if the head source differs. The chain branch's head choice (`shared_head_head`, else `output`) matches the setter's choice from the first MTP layer. Graph reuse compares `cparams.spec_draft_subset` (`llama-graph.h:848`), so a cached graph cannot skip the remap.
- NIT: a model with several MTP layers that have different shared heads would hit this assert at a layer offset above 0. That is acceptable fail-closed behavior, but it is undocumented.

**6. Checked and correct: other ways subset and chain drafting interact (item e).**
- Backend sampling stays detached whenever the subset is on (`speculative.cpp:1106`).
- The chain path returns before any host-side sampling, and the sequential fallback uses `common_sampler_sample_subset`.
- `draft_vocab` only does rewind bookkeeping.
- The verify step runs on the target context, where `spec_draft_subset` is 0. The flag is stored per context, so the shared model's target graph is untouched.
- The `LLAMA_SPEC_CHAIN_SUB` slice of the head is correctly skipped.
- NIT: when both chain and subset are on, `output_reorder` swaps logit rows n_sub values wide (`llama-context.cpp:3370`), but chain rows are 2 values wide. Before this change it used a width of n_vocab, which was just as wrong. It is harmless only because a chain batch has a single sequence and so no swaps. This is not new; it belongs with the known batch-split item in the backlog.

**7. Checked: this diff cannot cause the load-time error "check_tensor_dims: tensor output.weight not found" (item f).**
- The diff does not touch `llama-model-loader.cpp` or `llama-model.cpp`. In `qwen35.cpp` and `qwen35moe.cpp` only graph-building code changed, not the tensor-loading code.
- Both architectures load `output.weight` as `TENSOR_NOT_REQUIRED` (`qwen35moe.cpp:52`, `qwen35.cpp:51`). `check_tensor_dims` throws "not found" only when `required` is true (`llama-model-loader.cpp:892-895`), so current source cannot throw it for these architectures.
- The new tensors are named in a private context after load and are never registered with the loader.

So the error comes from binaries built from different sources, or from loading a model of a different architecture. The stale-build theory is consistent with this. `llama-model.h` is included only inside `src/`, plus `examples/turboanchorkv` and `tests/test-quantize-stats`. If `llama_model` objects with different layouts are mixed inside one `llama.dll`, a partial build, or an older `llama.dll`/`ggml*.dll` found first on PATH, can corrupt `hparams` and load flags. Run `where llama.dll` after the clean rebuild.

**8. SHOULD-FIX: `p` is a renormalized probability, which affects how acceptance should be measured (item g).**
`p_j` is the softmax over 32k subset rows, so it is always at least the full-vocabulary probability. The host compares this `p` against `p_min` (`speculative.cpp:2088`), so subset chains stop later and draft deeper. That raises the number of tokens drafted and lowers the acceptance rate per drafted token, while greedy argmax tokens stay the same.
- Sequential subset drafting also samples over the subset and renormalizes. Plain chain drafting already renormalizes over the default first-32768-id slice. The three modes are therefore roughly comparable, but none of them is the true probability.
- For acceptance check 3 (the MoE speed comparison), set `p_min` to 0 or report it. Compare accepted tokens per second, not the acceptance rate alone.
- Acceptance check 2 (temperature-0 token match) is unaffected, because the argmax order is the same.

## Fix list
1. BLOCKER: reshape the idmap to `[1, n_sub]` before `get_rows` in both files (finding 1).
2. SHOULD-FIX: record `p_min` and the renormalized `p` in the benchmark note (finding 8).

Verify manually:
- After the fix, do a real chain-plus-subset run on the 4B model with `GGML_SCHED`/debug assertions on. I proved the BLOCKER from the ggml source only; nothing was built or run because this review was read-only.
- Confirm the clean rebuild makes the `output.weight` error go away. Static reading rules out this diff as the cause but cannot show which stale binary is being loaded.
