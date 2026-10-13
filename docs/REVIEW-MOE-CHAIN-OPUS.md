## Verdict

**The bug is in generic plumbing, not a numeric or long-context problem. It can be fixed in generic graph code without any CUDA kernel. With the current tree, though, it fails: making the gate return true for qwen35moe cannot work until the MoE draft graph gets a chain branch.**

## 1. Most likely root cause (high confidence, about 90%)

`qwen35moe.cpp` has no chain code. With the gate flipped, the MoE draft returns ordinary logits, and `speculative.cpp` reads those raw logit values as if they were (token id, probability) pairs.

Here is the chain of evidence, each step read from source:

1. **Fact.** Flipping `llama_model_supports_mtp_chain` (`src/llama-context.cpp:4880-4882`) turns on `chain_graph` (`common/speculative.cpp:1519`).
2. **Fact.** `speculative.cpp:2018-2025` builds the chain batch:
   - Row 0 gets the real last token and its hidden state.
   - Rows 1 to n-1 get token 0 and an all-zero hidden state.
   - Every chain row is marked as an output.

   This only makes sense if the graph replaces rows 1 and later with its own in-graph argmax.
3. **Fact.** `speculative.cpp:2055` sets chain mode on the draft context. In that mode `llama-context.cpp:2979-2981` copies `ggml_nbytes(t_logits)` straight into the logits buffer and trusts the graph to have produced `[2, n_chain]` pairs.
4. **Fact.** The dense builder checks `params.cparams.mtp_chain` at `qwen35.cpp:551` and produces those pairs (`:711-719`). The MoE MTP builder (`qwen35moe.cpp:556-750`) never reads `mtp_chain`. It always returns full logits of shape `[n_vocab, n_outputs]` (`:744-748`).
5. **Fact.** `speculative.cpp:2071-2072` then sets:
   - draft token j = `(int) lp[2j]`, which is the raw logit of row 0 at vocab id 2j;
   - probability j = `lp[2j+1]`, another raw logit.

   `p_min` defaults to 0.0 (`common/common.h:331`), so the loop keeps going while that "probability" is 0 or more.
6. **Fact (log).** The error comes from the target model's batch (`srv decode`, `dbg-moe-c2.log:54-57`), not from the draft decode. `token[2]` is draft 1. **Inference:** the logit at row 0, vocab id 2 fell between -2 and -1 and was truncated to -1.

**Why it looks length-dependent (inference).** Prompt length has no direct role. The prompt changes the hidden state, and that changes four arbitrary logits (vocab ids 0 to 3). For short prompts those values happened to truncate to valid ids, or the fake "probability" came out negative and stopped the loop. Any prompt can hit it. The chain-length setting does not matter either, which matches the failure at chain lengths 2, 3 and 4.

**Why the "working" runs prove nothing.** At temperature 0 the target model checks every draft token and replaces any wrong one with its own choice. So output that is token-identical to sequential decoding only shows the drafts were harmless, not that they were right. The 14.2 tokens/s for chain against 24-28 for sequential fits drafts that are almost always rejected. That is an inference, and the two runs used different prompts.

**Why the 4B dense model works:** its builder has the chain branch.

## 2. Alternative hypotheses, ranked

1. **The batch gets split across micro-batches (about 7%).** If the catch-up rows plus the chain rows are more than `n_ubatch`, the batch is split. The second piece would see `n_catchup = 0` and a shorter chain. Its pairs would be written at offset `n_outputs_prev * n_vocab` (`llama-context.cpp:2973-2974`), while `speculative.cpp` reads packed pairs from offset 0. This can depend on length. It does not fit the evidence well, because the dense model at the same 32k prompt is fine. It is still a real hazard in the chain code for every model.
2. **NaN or -inf logits inside a real chain graph (about 3% now, relevant after the fix).** The CUDA argmax starts from `argmax = -1` (`ggml/src/ggml-cuda/argmax.cu:12`). It returns -1 if every value in a row is NaN or -inf, for example after a long-context overflow in the MoE block. The MoE graph never runs argmax today, so this cannot be the current cause. It is a plausible next failure once the chain branch is ported.

## 3. Fix direction and scope

- **Main fix: give the MoE builder a chain branch.** Add a `cparams.mtp_chain` branch to `qwen35moe.cpp`, modeled on `qwen35.cpp:551-739`. Only the feed-forward step inside the block differs: the routed experts plus the gated shared expert (`qwen35moe.cpp:681-718`) replace `build_ffn`.
  - Size: about 190 lines as a copy, or about 60 net lines if the projection, attention and per-step argmax loop move into a shared helper that takes a feed-forward callback. That helper is generic; the per-model part is about 40 lines.
  - Every operation it needs already exists in ggml, so no new kernel is required: argmax, softmax, get_rows, concat, and the expert matrix multiply.
- **Make the failure loud (about 10 lines).** These are generic changes:
  - In chain mode, `llama-context` should check that `t_logits->ne[0] == 2`.
  - `speculative.cpp:2071` should reject ids outside `[0, n_vocab)`.

  With these, a missing chain branch stops with a clear error instead of producing garbage drafts. Better still, have the gate ask whether the model's graph actually implements chain mode, instead of keeping a hard-coded list of architectures.
- **Does it unlock chain plus the draft-vocab subset? No, not by itself.**
  - Subset mode produces packed rows `n_sub` wide.
  - The argmax over those rows returns a position inside the subset, not a vocab id.
  - The chain feeds that id into `get_rows(tok_embd_w, id_j)` (`qwen35.cpp:723`), so the next step would embed the wrong token.
  - `speculative.cpp:1531-1532` blocks this combination on purpose, and `:1556` keeps backend sampling off for the same reason.

  Unlocking it needs one extra lookup in the chain loop: the subset head, then argmax, then a lookup from subset position to vocab id. That is about 15 lines in the shared helper. If the helper is built first, both models get the change at once.
- **Side note on the dense chain.** By default its argmax only looks at the first 32,768 vocab ids (`LLAMA_SPEC_CHAIN_SUB`, `qwen35.cpp:694-709`). That cuts off the vocab by id order, not by token frequency. It is worth knowing when comparing chain and subset results.

## 4. Can `--spec-chain` work on qwen35moe at 32k prompts without a per-model CUDA kernel?

**Not with a setting. It needs a code fix, and the fix is generic graph code, so the model owner is right that it is fixable without a CUDA kernel.** The cause is a missing graph branch, not a numeric limit, so no setting avoids it: chain length, context size and `p_min` all leave the misread in place.

Verify manually (I could not run anything in this read-only review):
- Run the short-prompt chain on the 35B with acceptance counts logged. Near-zero acceptance would confirm that the drafts were misread logits all along.
- After porting the chain branch, rerun the 30k prompt. If -1 ids still appear, look at hypothesis 2 (NaN logits in the MoE block). If drafts land at the wrong positions, look at hypothesis 1 (the batch split).
