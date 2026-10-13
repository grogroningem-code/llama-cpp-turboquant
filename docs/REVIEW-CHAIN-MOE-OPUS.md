**Verdict: PASS. There are no BLOCKERs and no SHOULD-FIX items, only 5 NITs.** I could not save the review: the write to `docs/REVIEW-CHAIN-MOE-OPUS.md` was refused because permission wasn't granted. That file is still empty. The findings are below; grant write access and I'll save them there.

**(a) The MoE chain branch matches the dense one.** The new branch (`qwen35moe.cpp:613-814`) follows `qwen35.cpp:549-739` line for line: the per-row KV stores (including the V-stride case), the mask row views, M-RoPE positions, the catch-up rows, the argmax plus probability packing into pairs, and the hidden-state concat. There are two small differences, both harmless:
- The MoE version gets the KV cache from `inp_attn->mctx`, which is already the right type, instead of casting. That is at least as safe.
- It declares `kq_scale` inside the branch, with no shadowing.

**(b) The MoE feed-forward step inside the chain is correct.** Its `build_moe_ffn` arguments are identical to both the main MoE layer (`:505-536`) and the sequential draft path (`:889-925`): gating, weight normalization and scale, expert tensors and scales. The gated shared expert and the residual add also match. `build_moe_ffn` takes its token count from its input tensor (`llama-graph.cpp:2004`) and reads no batch-wide state, so the 1-row slices are fine.

**(c) The guards are in the right place.**
- The new width check (`ne[0] == 2`) runs at logits extraction in `decode`. It will stop any future model that is added to the chain list without a chain branch.
- It cannot fire wrongly on dense. The only caller always sends tokens for a single sequence, flushes other sequences' rows first, and turns chain mode off right after the decode.

**(d) The micro-batch split cannot happen with the current code.** `defer_capacity()` (`speculative.cpp:1461-1465`) caps the catch-up rows at the micro-batch size minus `n_max`. The adaptive chain depth never goes above `n_max`. So catch-up rows plus chain rows always fit in one micro-batch. Only the caller guarantees this, though; `llama-context` itself does not check it.

**(e) The non-chain MoE and dense paths are untouched.** The new branch runs only while chain mode is on. The chain-plus-subset combination is still rejected at setup.

**NITs:**
1. When the new id check in `speculative.cpp:2078` fires, it returns without updating `n_last`. The adaptive depth controller then gets the previous round's draft count. Also, `n_vocab_dft` could be computed once, outside the loop.
2. A one-line `GGML_ASSERT(!cparams.mtp_chain || n_outputs_prev == 0)` in `decode` would make any future micro-batch split fail loudly.
3. If a row of logits is all NaN, the in-graph argmax returns -1, and that -1 is used to look up a token embedding on the GPU before the host check sees it. Dense already had this, and the spec parks it.
4. `LLAMA_SPEC_CHAIN_SUB` (default 32768) now also limits the MoE argmax to the first 32,768 vocab ids, so chain drafts never contain higher ids. Keep this in mind when comparing chain and sequential acceptance rates.
5. About 190 lines are copied from the dense model, so the two copies can drift apart; the spec accepts this. The new comments are hard-wrapped, which AGENTS.md asks to avoid.

On the acceptance criteria, AC3 (no BLOCKERs) is met. For AC1, AC2 and AC4 I relied on the runs you reported and did not re-run them.

Verify manually:
- Save this review to `docs/REVIEW-CHAIN-MOE-OPUS.md` once write access is granted.
- AC1, AC2 and AC4 rest on your measurements, not on anything I ran in this review.
