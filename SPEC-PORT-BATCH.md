# Spec: upstream batch port (T7)

Port 4 merged upstream PRs onto `candidate`, in cheap-first order.

## PRs
1. #29638 `common: stop accepting draft tokens at EOG` (1 hunk, sampling.cpp).
   Correctness: EOG draft tokens accepted mid-draft stayed in context.
   Tree anchor verified: `common/sampling.cpp:744` pre-image matches.
   API `llama_vocab_is_eog` present (`include/llama.h:1147`).
   NOTE: also apply the analogous guard in T6 `sample_and_accept_n_rejection`
   (same bug shape; upstream rejection path postdates the EOG PR, deviation
   documented here).
2. #27978 `CUDA: let any expert count use the fast mm_ids_helper path`.
   `ggml/src/ggml-cuda/mmid.cu`: pow2-padded slots, `case 10`, generalized
   static_assert. Anchor verified (`:63`, `:156`, `:159` match pre-image).
   Effect on our models ~0 (top-8 already fast), enables other counts.
3. #28068 `models: fix GDN normalization from max to rsqrt`.
   New `build_gdn_l2_norm` helper in `src/models/models.h` (anchor `:11`
   matches); swap `ggml_l2_norm` at qwen35.cpp:435-436, qwen35moe.cpp:459-460
   (+ other models in diff if present in tree: bailingmoe3, kimi-k3,
   kimi-linear, qwen3next, qwen4exp — port only files that exist here).
   NUMERICS CHANGE: temp-0 outputs WILL differ post-port (bugfix, expected).
4. #29393 `CUDA: fuse RMS_NORM + SCALE into one kernel`.
   `ggml-cuda.cu` can_fuse + try_fuse cases, `norm.cu` do_scale template
   param + `ggml_cuda_op_rms_norm_scale_fused`, `norm.cuh` decl.
   This is why #28068 rides along: GDN emits RMS_NORM+SCALE pairs.

## Acceptance
- AC1: full CUDA build green (llama-cli, llama-server), no new warnings.
- AC2: unit tests green (arg-parser, spec-draft-vocab).
- AC3: 4B smoke coherent; MoE smoke no `-1`/crash.
- AC4: aggregate A/B vs pre-port binaries (saved aside): report pp/tg delta.
- AC5: GDN temp-0 output differs (proves the fix is live, not dead code).

## Test plan
- Save pre-port binaries, build all 4, run AC2/AC3, then 4B A/B pre vs post
  (owner flags, 1 round G + 1 round P) + MoE spot check.
- Opus medium review of the 4 diffs post-port.
