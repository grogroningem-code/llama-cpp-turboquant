# T2.3 review: draft-vocab restricted MTP head (Opus)

Scope: working-tree diff against HEAD 6e09de1 (22 files, +936/-28), rebuilt with `git diff HEAD` because /tmp was not readable. Checked against SPEC-DRAFT-VOCAB.md v4, AC1-AC7.
Taken as already verified (not re-run): CPU and CUDA builds, ctest 4/4, test-backend-ops 119/119 with zero skips, live AC5 exactness at temp 0 and seed 42.

## Findings

### B1 BLOCKER - gathering the head from a CPU repack buffer calls a null function pointer
`src/llama-context.cpp:2111-2130`: the setter takes `buft` from the head buffer without the extra-buffer-type check the spec requires (precedent `src/llama-adapter.cpp:337-350`). It then calls `ggml_backend_tensor_get(head, ...)`. The CPU repack buffers set `get_tensor = nullptr` (`ggml/src/ggml-cpu/repack.cpp:4761`, same in amx/kleidiai), and `ggml_backend_tensor_get` calls it without a check (`ggml/src/ggml-backend.cpp:491`). Extra buffer types are on by default.
Reach: the output head stays on the CPU (partial offload, which is the AC2 setup, or `-ngl` below n_layer+1), and its type is repackable on that CPU: Q4_0, Q4_K or IQ4_NL with AVX2 on x86; Q6_K or Q8_0 on ARM with NEON (for example GB10). Result: a segfault at startup instead of a clean error.
The spec's fallback is also not enough on its own: falling back to a CPU buffer fixes where the copy goes, but repacked source rows cannot be read back at all. Fix: detect an extra buffer type on `head->buffer` and return false (the caller then aborts with a clear message), or read the rows from the GGUF file. Add a line to the spec.

### S1 SHOULD-FIX - loader allocates before the size and count checks the spec requires
`common/spec-draft-vocab.cpp:55` runs `std::vector<int32_t> ids(count)` with `count` taken straight from the file header. None of the spec's checks (`count <= n_vocab`, exact file size, file size <= 1MB) runs before this allocation. A corrupt header with count 0xFFFFFFFF asks for a 16 GB zero-filled vector: either `bad_alloc` (caught, clean abort) or an OOM kill, depending on the OS. The trailing-byte check only catches files that are too long. Fix: check the file size and count first. Add tests for an oversized file and a huge count.

### S2 SHOULD-FIX - shared model state is changed through const_cast; freeing it can leave other contexts with dangling graphs
`src/llama-context.cpp:2052-2064`: the subset copy lives on `llama_model`, which all contexts share. A second install with a different list, or a clear (`ids == null`, line 2065), frees `output_subset_w` while another context's cached graph may still point to it. The graph-reuse key compares that context's own unchanged hash, so the stale graph gets reused: use-after-free. If that context rebuilds instead, `output_subset_active` fails and the graph silently uses the full head. Its `spec_draft_subset` is still nonzero, though, so `output_reorder` (line 3341) and `get_logits_subset_ith` (line 1741) read full-vocab rows at subset width and produce garbage drafts. Drafts only, so verify stays correct, but this is not fail-closed. No caller in this diff does either today, but `llama_set_spec_draft_vocab` is a public API in `llama-ext.h`. Fix: refuse a different list once one is installed, never free in clear (only zero `cparams`), and `GGML_ASSERT` in both `graph_mtp` builds that a nonzero `spec_draft_subset` means the subset is active. The setter is also not thread-safe; note it in the API comment.

### S3 SHOULD-FIX - the gathered copy is not marked as weights, so the scheduler may place the matmul away from it
`src/llama-context.cpp:2123`: the buffer keeps the default usage. The scheduler pins an op to its weight's backend only when the buffer usage is `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` (`ggml/src/ggml-backend.cpp:1112`). With a CPU-resident head under partial offload, the subset matmul can run on CUDA and copy the ~130 MB matrix every draft step. That would skew the AC6 bench. Fix: `ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS)`.

### S4 SHOULD-FIX - the chained path aborts instead of getting the spec'd position-to-id mapping and flag precedence
The spec says the qwen35 chained path gains the position-to-id mapping and that `--spec-draft-vocab` wins over `LLAMA_SPEC_CHAIN_SUB`. The chain graph is untouched (`src/models/qwen35.cpp:691-709`). Instead the flag aborts with `chain_heads || chain_graph` (`common/speculative.cpp:1545`). Chaining is opt-in (`common/speculative.cpp:61-64`) and the abort is loud, so nothing is silently wrong. Either implement the mapping or amend the spec (and the AC2 and AC6 notes) to say v1 is single-head only.

### S5 SHOULD-FIX - AC3 has no committed test through the graph
AC3 asks for a model test through the graph. `tests/test-spec-draft-vocab.cpp` covers only the loader, the flags and the rewind handle; no test calls the setter, the subset getter, `output_reorder` or `common_sampler_sample_subset`. From the code, AC3 holds by construction: candidates come only from the id list (`common/sampling.cpp:171-186`), and the target context keeps `spec_draft_subset == 0`. The live T2.2 run shows identical final tokens, which is weaker than identical verify logits. Add a small test that installs a subset on a tiny model and checks that the subset logits equal the full logits at the mapped ids, and that candidate ids are members of the list.

### S6 SHOULD-FIX - the AC1 parity harness never runs the MTP path
`scripts/parity-draft-vocab.sh:59-60` runs plain generation with no `--spec-type draft-mtp`. The changed draft code (constructor, sampler call site, rewind hook) never runs there. Also, at temp 0 the final tokens cannot reveal draft-path changes. Add draft-mtp cases and compare draft and accept counts with the base binary, not only stdout.

### N1 NIT - an empty id list passes the loader and fails only at the first draft
With `count == 0`, the setter takes the clear branch and returns true (`src/llama-context.cpp:2065`). The log says "subset head on (0 ids)", and the first draft hits `GGML_ASSERT` (`common/sampling.cpp:174`). Reject `count == 0` in the loader.

### N2 NIT - the gather stages the whole head in host RAM
`src/llama-context.cpp:2129-2130` copies the full head to the host (about 1 GB for a 27B Q6_K head) to keep 32k rows. That is a large spike on 8-16 GB laptops. Reading one row at a time with `ggml_backend_tensor_get(head, dst, ids[i]*nb[1], row_bytes)` avoids it. This matches the spec, so it is optional.

### N3 NIT - the tokenizer hash is duplicated and skips some tokens
The same loop appears in `common/speculative.cpp:68` and `tools/vocab_freq/vocab_freq.cpp:82`, with a comment saying they must match. Move it into `common/spec-draft-vocab.cpp`. Pieces that are empty or longer than 256 bytes are skipped, so vocabularies that differ only in those tokens hash the same.

### N4 NIT - comment style
Several new comments are 3-5 hard-wrapped lines (`common/sampling.cpp:166-170`, `common/speculative.cpp` constructor, `src/llama-ext.h`). Some cite task labels ("(AC3)", "v1"), which AGENTS.md asks to avoid. All added lines are ASCII-only (checked).

### N5 NIT - readability and scope
The nested ternary for the row stride at `src/llama-context.cpp:2957` would read better as a named `row_w`, like `output_reorder` uses. The `encode()` change (line 2499) is outside this feature's path.

## What holds
- Flag-off: every new branch is gated on `spec_draft_subset != 0` or a non-null `draft_vocab`. The sampler refactor runs the same statements in the same order when `subset == false`, and the graph-key field compares equal. Flag-off output is bit-identical by construction.
- Fail-closed: a bad file, failed install, chain combo, or missing subset row aborts loudly. The exceptions are B1 (segfault, not a clean abort) and S2 (silent garbage drafts).
- Lifetime: the copy is owned by `llama_model` through RAII (context and buffer) and outlives contexts. The scale is a shared {1} tensor, so sharing it is correct.
- Tests: no tautologies. The loader tests write real files and run the real loader; the flag tests run the real parser; the handle test checks eviction, rewind and slot isolation with real expected values.

Gate: FAIL - BLOCKER B1 (repack-buffer gather crash).
```

Your already-verified items are taken as given. AC2 (the 8 GB partial-offload run) and AC6 (the speed bench) are still open and are not judged here.

Verify manually:
- I did not reproduce the B1 crash; it comes from reading the code. To confirm it, run with the output head on the CPU (for example `-ngl` set to the layer count, without the output layer) using a Q4_0 or Q4_K head on an AVX2 machine.
- I did not check whether `--fit on` keeps the output head on the GPU on your 8 GB laptop. If it does, B1 won't show up in your AC2 run, but it is still reachable in other setups.
