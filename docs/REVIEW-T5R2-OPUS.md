# T5 review, round 2: chain drafting plus subset head (Opus)

**Verdict: PASS.** The BLOCKER from round 1 is fixed exactly as prescribed, in both files, and no other file changed after round 1. The round-1 SHOULD-FIX and NITs are not blockers and stay open.

## 1. The fix matches the prescription, and the shapes now fit together
`src/models/qwen35.cpp:723` and `src/models/qwen35moe.cpp:798` both read, character for character:
`ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, model.output_subset_ids_w, 1, model.output_subset_ids_w->ne[0]), id_j)`.
Shapes, step by step:
- The idmap is created with `ggml_new_tensor_1d(I32, n_ids)` (`src/llama-context.cpp:2140`). It is contiguous, so it can be reshaped to `[1, n_sub]`.
- `id_j` is the argmax over `[n_sub, 1]`, which gives a `[1]` I32 tensor. `get_rows` needs `a->ne[2] == b->ne[1]`, which is 1 == 1, and returns `[1, 1]` I32 holding the real vocabulary id.
- `reshape_2d(id_real_j, 1, 1)` now has exactly 1 element, so the assert passes. The F32 cast and the `concat` with `p_j` (`[1,1]` each, joined on dimension 0) give the `[2,1]` pair.
- `get_rows(tok_embd_w, id_real_j)`: the index tensor is `[1,1]` I32, and the embedding table has `ne[2] == 1 == b->ne[1]`. The result is `[n_embd, 1]` and goes into `build_proj` as before.
- `p_j` still uses the subset position `id_j`, and the pair and the next chain row still use the real id `id_real_j`. Round 1 finding 4 still holds.

## 2. Nothing else changed after round 1
I compared each changed or untracked file's modification time with `docs/REVIEW-T5-OPUS.md` (13:12:56). Only three are newer:
- `src/models/qwen35.cpp` and `src/models/qwen35moe.cpp` (13:18:18), which hold the fix;
- `docs/REVIEW-T5R2-OPUS.md`, which is this output file.

`git diff HEAD --stat` lists the same 22 files as the snapshot taken at session start. In the diff, `output_subset_ids_w` appears only on the two fixed lines. Nothing unexpected.

## 3. Verdict
PASS for T5. Still open from round 1, none of them blockers:
- SHOULD-FIX: the benchmark note should record `p_min` and state that `p` is renormalized over the subset (round 1 finding 8).
- NIT: models with several MTP layers that use different shared heads hit an undocumented assert.
- NIT: `output_reorder` swaps logit rows at the wrong width when chain drafting is on (an older problem).

Verify manually: I did not re-run the 4B chain-plus-subset run (36 of 64 drafted tokens accepted) or the temperature-0 token match (30 of 30 lines), so both results are taken as reported.
