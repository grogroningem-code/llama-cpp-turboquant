# SPEC: MoE chain branch (qwen35moe) + loud-failure guards

## Problem
`--spec-chain` on qwen35moe misreads full logits as (id, prob) pairs
(Opus root cause, `docs/REVIEW-MOE-CHAIN-OPUS.md`): the MoE MTP builder has
no `mtp_chain` branch, so chain mode drafts garbage; a logit in [-2,-1)
truncates to id -1 and poisons the target batch at long prompts
(repro `dbg-moe-c2.log`). Gate is qwen35-only (reverted experiment).

## What ships in this slice
1. Chain branch in `qwen35moe.cpp::graph_mtp`, modeled line-for-line on
   `qwen35.cpp:551-739`: same guard (`mtp_chain && n_seqs_unq == 1 &&
   ubatch.token`), same `build_proj`, same per-row block (norms, QKV,
   rope_multi, per-row KV store, mask views, MHA, gate, wo, residual),
   same catchup/chain loop with in-graph argmax + [id, p] pairs. ONLY
   difference: the FFN section uses `build_moe_ffn` + gated shared expert
   (copy of the sequential path 681-718, no new pattern). Direct port, no
   shared-helper refactor (that is post-gate wide work).
2. Gate: `llama_model_supports_mtp_chain` also returns true for
   `LLM_ARCH_QWEN35MOE` (re-apply of the validated experiment).
3. Loud-failure guards (generic, ~10 lines): chain mode asserts
   `t_logits->ne[0] == 2` at extract (`llama-context.cpp` ~2979);
   `speculative.cpp` ~2071 rejects ids outside `[0, n_vocab)`. A missing
   branch must abort loudly, never draft garbage.
4. TDD note (adapted): the RED already exists as a repro (`dbg-moe-c2.log`
   fails with invalid token -1 on current tree). GREEN = new smoke script
   `scripts/chain-moe-smoke.sh` asserting: no "invalid token" in output,
   `draft_n_accepted > 0` in verbose timings, AC5 temp-0 chain-2 vs seq-2
   match on MoE short prompt, 32k p1 completes with >= 100 real tokens.

## Acceptance criteria
- AC1: `scripts/chain-moe-smoke.sh` passes end to end on the 35B MoE.
- AC2: unit gates green (`test-spec-draft-vocab` OK; ctest 4/4).
- AC3: Opus review of the diff: no BLOCKERs (fix loop on any).
- AC4: 4B chain behavior unchanged (C4 still best on code; no fallback
  warnings on qwen35).

## Non-goals (parked, not in this slice)
- No shared-helper extraction (post-gate question).
- No chain+subset remap (+15 lines) — NEXT slice after this lands.
- No batch-split micro-batch fix (Opus 7% hazard) — TASKS backlog.
- No NaN-argmax hardening — watch-item, only if -1 reappears post-fix.
- No qwen3next/qwen4exp chain work.
- No OpenRouter, no merge (owner merges manually).

## DoD sign-off
Approver defaulted to owner at manual merge; none was recorded (user
waived checkpoints). Tests prove AC; owner judges DoD. Reviewer: Opus via
Claude subscription (standing preference, recorded deviation from the
cloud-review chain).

## T5 follow-up: chain+subset remap (IMPLEMENTING 2026-10-13)

Goal: allow `--spec-draft-vocab` + `--spec-chain` together (today
`speculative.cpp:1531` aborts). Value: stack cheap subset drafts with
no-CPU-round-trip depth. Evidence it may matter: 4B C2~=S2V, so stacking
is the only untested combo; MoE chain now correct but 0.88-1.02x, so
cheaper drafts are its remaining lever.

Design: subset install keeps `model.output` intact and stashes the
gathered copy in `model.output_subset_w/_s` plus `output_subset_ids`
(vocab id per subset row); graphs branch on `cparams.spec_draft_subset`
(see `qwen35.cpp:835-842`). The chain branch must (a) use the subset head
when `model.output_subset_active(hash, head_w2)`, skipping the
`LLAMA_SPEC_CHAIN_SUB` sub-view (already n_sub wide), (b) translate the
in-graph argmax position through a persistent I32 idmap tensor (new
`model.output_subset_ids_w`, created + filled in the setter beside the
gathered copy — same proven leaf mechanism as `output_subset_w`, no new
graph-input plumbing) to a real vocab id BEFORE pair emission AND before
`get_rows(tok_embd)` recurrence, (c) keep `p` from the subset softmax
(renormalized; noted, accepted). Host-side pair reader needs no change.
Fail-closed: subset requested but `output_subset_active` false in chain
-> GGML_ASSERT (mirrors `:835`); `chain_heads` (multi-head path) keeps
the abort, only `chain_graph` is lifted.

AC: (1) 4B C2S runs clean, acceptance > 0, no guard fires; (2) AC5 temp-0
C2S vs S2V token MATCH; (3) MoE A/B (`--fit on`): C2S/C3S vs S2V vs C2,
pp+tg, faster-or-parity verdict with numbers; (4) Opus review, no
BLOCKERs. Rejected approach B (build-time memcpy into fresh graph
tensor): sched allocator does not promise to preserve leaf data.
