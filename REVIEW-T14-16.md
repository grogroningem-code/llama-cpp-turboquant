# Review: T14-T16 (`src/models/qwen4exp.cpp::build_qsa_sel`)

Commits reviewed on `t13-qwen4exp-resync`:
- `526a7bde2` (T14: per-head indexer scoring loop)
- `bf5b8218d` (T15: single `mul_mat` when `n_pool*n_idx_h*n_tokens*4 <= 64 MiB`, per-head loop above)
- `4d54111dd` (T16: decode-only `n_tokens == 1` selection-mask build via `ggml_fill` + `ggml_set_rows`)

## Findings

None (0 blocker, 0 major, 0 minor).

## Verification Summary

1. **Per-head loop vs single product (`src/models/qwen4exp.cpp:955-974`)**:
   - `q` enters at `[idx_dim, n_idx_h, n_tokens, 1]`. `qh = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3))` produces a contiguous `[idx_dim, n_tokens, n_idx_h, 1]` tensor with `qh->nb[1] = idx_dim*4` and `qh->nb[2] = idx_dim*n_tokens*4`.
   - Each head view `q_h = ggml_view_2d(ctx0, qh, idx_dim, n_tokens, qh->nb[1], h*qh->nb[2])` is contiguous and selects `[idx_dim, h, t]` for all `t`, matching the `(h, t)` column order of `ggml_reshape_2d(ctx0, q, idx_dim, n_idx_h*n_tokens)` in the `else` branch.
   - `ggml_mul_mat(ctx0, k2, q_h)` produces `[n_pool, n_tokens]` F32 equal to the `slice` view (`stride = kq->nb[2]`, `offset = h*kq->nb[1]`) of the 3D-reshaped single `mul_mat`. Elementwise `ggml_relu` commutes with slicing, and both branches accumulate `h = 0 .. n_idx_h - 1` in identical left-to-right `ggml_add` order before `ggml_scale(..., 1.0f/sqrtf(idx_dim))`.

2. **64 MiB threshold switch (`src/models/qwen4exp.cpp:955`)**:
   - `n_pool`, `n_idx_h`, and `n_tokens` are `int64_t` (`lines 891, 893`; `llm_graph_context::n_tokens` is `int64_t`), so `n_pool * n_idx_h * n_tokens * 4` is evaluated in 64-bit signed arithmetic and cannot overflow under valid context/batch bounds (`n_pool <= UINT32_MAX - 64` asserted in `llama-memory-hybrid-idx.cpp:804`).
   - Both branches are reachable (e.g. `n_tokens = 1` decode takes `else`; `-c 32768 -ub 4096` with `n_pool = 8256, n_idx_h = 4, n_tokens = 4096` -> `541,065,216 > 67,108,864` takes `if`).

3. **T16 selection mask (`src/models/qwen4exp.cpp:996-1022`)**:
   - At `n_tokens == 1`, `mask_all` has shape `[1, n_kv + 1, 1]` and type `kq_mask->type` (`F16` when `flash_attn` is enabled, `F32` otherwise) filled with `-INFINITY`; `zeros` has shape `[1, n_sel, 1]` and type `GGML_TYPE_F32` filled with `0.0f`; `sel_idx` is reshaped to `[n_sel, 1, 1]` (`I32`).
   - This matches the shapes, types, and values of the `n_tokens > 1` `repeat_4d` + `reshape_3d` path (which stays untouched in `else`). `ggml_set_rows` supports `dst = F16/F32`, `src0 = F32`, `src1 = I32` across backends, scatters `0.0f` to selected rows in `[0, n_kv)` and sentinel row `n_kv`, and `ggml_view_4d` trims row `n_kv` to `[n_kv, kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]]` before `ggml_add(ctx0, sel, kq_mask)`.

4. **Edge cases & allocator lifetime (`src/models/qwen4exp.cpp:66-85, 861, 996-1001`)**:
   - `n_idx_h >= 1`, `kpool >= 2`, `n_pool >= 64` (`kpool_pad` in `llama-memory-hybrid-idx.cpp:660`), and `n_tokens >= 1` prevent zero-dim or null-score paths. `n_sel > n_kv` is valid because `set_rows` overwrites duplicate sentinel indices (`n_kv`) with identical `0.0f` values.
   - `ggml_fill` on leaf tensors `ggml_new_tensor_3d(ctx0, ...)` is safe with `ggml-alloc`: `ggml_gallocr_alloc_graph_impl` (`ggml/src/ggml-alloc.c:768-777`) allocates `node->src[0]` on demand and immediately reuses its buffer in-place for `GGML_OP_FILL` (`n_children == 1, n_views == 0`), freeing it right after `ggml_add(ctx0, sel, kq_mask)` consumes the `sel` view.
   - Existing `build-cuda-static/bin/Release/test-llama-archs.exe -a qwen4exp` passes on CUDA, iGPU, Meta, and MoE cache.

## Doubts

- **Leaf allocation order across layers (`src/models/qwen4exp.cpp:999-1000`, `ggml/src/ggml-alloc.c:724-727`)**: In the `else` branch, `mask_seed` and `zero_seed` are ops derived from `sel_idx`, tying their allocation strictly to layer execution. In the `n_tokens == 1` branch, `ggml_new_tensor_3d` creates leaf tensors (`op == GGML_OP_NONE`). Today `ggml_build_forward_expand` only adds non-`GGML_TENSOR_FLAG_INPUT` leafs with `n_views > 0` to `graph->leafs`, so `mask_all`/`zeros`'s `src[0]` tensors are not pre-allocated in the leaf loop and their buffers are reused per layer. If `ggml` graph construction ever changed to register all `OP_NONE` inputs in `graph->leafs`, all 48 QSA layers' `mask_all` and `zeros` buffers would stay live simultaneously (though at `n_tokens == 1` that would only be ~3.5 MiB total at `n_kv = 32768`).

VERDICT: CLEAN
