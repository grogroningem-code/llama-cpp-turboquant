# REVIEW-T10.3: Review of Strata-Style Prompt Buffers Borrowed from MoE Slab

**Status**: REVISE DESIGN BEFORE IMPLEMENTATION (Major Blockers Found)
**Target Document**: `llama-cpp-turboquant/DESIGN-T10.3.md`
**Specification**: `llama-cpp-turboquant/SPEC-PORT-STRATA.md` (Task T10.3)
**Reviewer**: Headless Inference Model (Code Review Gate)
**Date**: 2026-10-17

---

## 1. Executive Summary

The core premise of T10.3—borrowing VRAM from the fork's MoE slot cache during prompt processing to allow large compute ubatches without permanently stranding VRAM from expert offloading—is sound and addresses a real throughput bottleneck in MoE inference (as verified in `src/llama-context.cpp:1140-1165`).

However, `DESIGN-T10.3.md` contains **four critical architectural blockers** regarding GGML scheduler internals, slab memory layout contiguity, CUDA buffer deallocation semantics, and CUDA graph invalidation. If implemented as currently drafted, the changes will either fail to compile, trigger immediate `cudaErrorInvalidValue` / `cudaErrorIllegalAddress` crashes at runtime, corrupt memory on warm caches, or fail to allocate compute memory.

The design requires revision on these points before any code is written.

---

## 2. Findings by Severity

### 2.1 Blockers (Must fix before coding)

#### B1: Fundamental Misunderstanding of `ggml_backend_sched_reserve_size` and Buffer Binding
- **Citation**: `DESIGN-T10.3.md:49-52`, `DESIGN-T10.3.md:117-118` vs `ggml/include/ggml-backend.h:330-331`, `ggml/src/ggml-backend.cpp:2130-2158`, `ggml/src/ggml-alloc.c:938`.
- **Issue**: `DESIGN-T10.3.md` Section 2 (Step 2) states:
  > "Wrap: `ptr` becomes a `ggml_backend_buffer` via the new device-ptr wrapper. Build the prompt-sized reserve graph and `ggml_backend_sched_reserve_size` into it. If reserve fails: return the loan immediately and continue on the normal path."
  And in Section 7 (Manual check):
  > "Whether `ggml_backend_sched_reserve_size` can target an externally wrapped buffer without scheduler changes (assumed yes - must prove with a spike per the spike skill)."
- **Analysis & Evidence**:
  1. `ggml_backend_sched_reserve_size(sched, measure_graph, sizes)` (`ggml-backend.h:330`, `ggml-backend.cpp:2130-2142`) does **not** allocate or bind memory into an external buffer. It is strictly a dry-run sizing query (`no_alloc = true` in `ggml_gallocr_reserve_n_size`, `ggml-alloc.c:950-959`) that populates the `size_t * sizes` array with the byte requirement of each backend.
  2. `ggml_backend_sched` (`ggml-backend.cpp:2032`) owns a `ggml_gallocr` instance initialized with `sched->bufts`. During `ggml_backend_sched_reserve` (`ggml-backend.cpp:2144-2159`), the scheduler delegates allocation directly to `ggml_gallocr_reserve_n`, which internally invokes `ggml_vbuffer_alloc(galloc->bufts[i], ...)` (`ggml-alloc.c:938`).
  3. There is **no mechanism** in `ggml_backend_sched` to attach an existing arbitrary `ggml_backend_buffer_t` as its compute buffer. Compute buffers are always allocated by the scheduler via its configured `ggml_backend_buffer_type_t`.
- **Fix Shape**: Do not attempt to pass a wrapped `ggml_backend_buffer` to `ggml_backend_sched_reserve_size`. Instead, follow the verified pattern established in `kv_stream_phase_arena` (`ggml-cuda.cu:1195-1229`, `src/llama-context.cpp:668-699`):
  - Implement a dedicated `ggml_backend_buffer_type_t` (e.g. `ggml_backend_cuda_borrowed_moe_buft`) whose `alloc_buffer` callback wraps the lent slab pointer into a `ggml_backend_buffer` without allocating fresh VRAM.
  - Recreate or rebind the scheduler using this buffer type for the prompt phase.

---

#### B2: Slab Contiguity and Fragmentation in `moe_cache_lend`
- **Citation**: `DESIGN-T10.3.md:46-48` vs `ggml/src/ggml-cuda/moe-cache.cu:118-130`, `moe-cache.cu:1151-1152`, `moe-cache.cu:1307-1347`.
- **Issue**: Section 2 (Step 1) specifies:
  > "`moe_cache_lend(device, bytes, &ptr, &n_slots)` carves whole FREE slots from the largest pool with enough free slots (never partial slots, never resident ones). Lent slots get a `loaned` state: LRU/admission/dispatch skip them."
- **Analysis & Evidence**:
  1. In `moe-cache.cu:1307-1347`, each pool allocates a single contiguous `cudaMalloc` memory block: `char * slab` of size `slot_count * shape.expert_size`. Individual slot addresses are computed by linear index offset: `pool->slab + (size_t)job.slot * pool->expert_size` (`moe-cache.cu:1152`).
  2. Free slots are maintained in `std::vector<int> free_slots` (`moe-cache.cu:126`). Slots are added, popped, and recycled during inference. After any decode steps have executed, free slots become non-contiguous across the slab index range.
  3. A scheduler compute buffer requires a single, contiguous virtual address span `[ptr, ptr + bytes)`.
  4. `moe_cache_lend` cannot return a single `void * ptr` for `n_slots * expert_size` simply because `free_slots.size() >= n_slots`. If free slots are disjoint (e.g., slots 2, 5, 11 are free, but 0, 1, 3, 4, 6... are resident), they cannot be used as a contiguous buffer.
  5. Restricting the loan to "never resident ones" guarantees that on a warm cache (turn 2+ of any server or chat session where decode has populated the cache with resident experts), `free_slots` will be empty or fragmented, causing `moe_cache_lend` to fail consistently.
- **Fix Shape**:
  - `moe_cache_lend` must search for a **contiguous run** of slot indices `[slot_start, slot_start + n_slots - 1]`.
  - If contiguous free slots are unavailable, `moe_cache_lend` must be permitted to evict resident experts in a contiguous window (invalidating their entries in `pool->map` and popping them from LRU). Because prompt processing does not run decode concurrently, evicting resident experts is safe; decode will naturally reload needed experts upon resumption.

---

#### B3: Illegal `cudaFree` on Sub-Allocated Pointer in Buffer Wrapper Destructor
- **Citation**: `DESIGN-T10.3.md:28-31`, `DESIGN-T10.3.md:87` vs `ggml/src/ggml-cuda/ggml-cuda.cu:808-831`, `ggml-cuda.cu:1175-1178`.
- **Issue**: Section 1 and Section 5 propose introducing `ggml_backend_cuda_buffer_from_device_ptr(ptr, size)` as a general device-pointer wrapper.
- **Analysis & Evidence**:
  1. In `ggml-cuda.cu:1175-1178`, the destructor for CUDA buffer contexts is implemented as:
     ```cpp
     ggml_backend_cuda_buffer_context::~ggml_backend_cuda_buffer_context() {
         if (arena == nullptr) {
             CUDA_CHECK(cudaFree(dev_ptr));
             return;
         }
         // ... arena release ...
     }
     ```
  2. If `ptr` is an offset within an existing `moe_cache_pool` slab (`slab + slot_offset`) and is wrapped into a standard `ggml_backend_cuda_buffer_context` without an arena owner, destroying the buffer will execute `cudaFree(dev_ptr)`.
  3. Passing an interior pointer to `cudaFree` is an illegal operation that returns `cudaErrorInvalidValue` and crashes CUDA runtime context.
- **Fix Shape**: The wrapper must use a specialized buffer interface or set an explicit flag (e.g. `owns_memory = false`) in its context so `~ggml_backend_cuda_buffer_context()` never calls `cudaFree` on borrowed slab ranges.

---

#### B4: Stale CUDA Graphs on Return to Decode
- **Citation**: `DESIGN-T10.3.md:71-73` vs `ggml/src/ggml-cuda/ggml-cuda.cu:3952-3971`, `ggml-cuda.cu:4159-4184`, `src/llama-context.cpp:1044-1051`.
- **Issue**: Section 4 states:
  > "While a loan is active, skip capture (force `graph_reset_fn`, same as the phase-arena path). Decode captures are unaffected (loan is returned first)."
- **Analysis & Evidence**:
  1. During decode, CUDA graphs capture kernel executions with fixed buffer virtual addresses baked into the node parameters (`ggml-cuda.cu:4159-4184`, `5968-5977`).
  2. According to Section 2 (Step 4) and Section 3, when the prompt finishes, `moe_cache_return` returns the loan, and the scheduler "re-reserves decode-sized buffers".
  3. Re-reserving decode buffers reallocates or shifts the compute buffer addresses. The decode graph captured prior to the prompt now holds dangling or mismatched memory addresses.
  4. Claiming "Decode captures are unaffected" is false. If decode executes a previously captured graph whose buffers were re-reserved, it will access stale memory, resulting in silent corruption or GPU kernel panic.
- **Fix Shape**: Invalidate all captured CUDA graphs on the decode transition. A mandatory call to `ggml_backend_cuda_graph_reset(backend)` (`ggml-cuda.cu:3952`) must be executed inside `moe_cache_return` or before the next decode token step whenever compute buffers have been re-reserved.

---

### 2.2 Should-Fix (Architectural & Operational Improvements)

#### S1: Infeasibility of Partial Trim with `cudaMalloc`
- **Citation**: `DESIGN-T10.3.md:67-70`, `DESIGN-T10.3.md:105` vs `ggml/src/ggml-cuda/moe-cache.cu:1716-1723`, `moe-cache.cu:3324-3328`, `ggml-cuda.cu:520`.
- **Issue**: The proposal suggests that under allocator pressure, `ggml_moe_cache_trim` "must skip loaned ranges (else use-after-free). Proposed: trim returns 0 for loaned bytes; the chunk reserve then fails and step 2 fallback shrinks/retries".
- **Analysis**: In `moe-cache.cu:1718`, each pool slab is a single allocation. `cudaFree` cannot release partial ranges of a `cudaMalloc` allocation. If a pool contains any active loan, `cudaFree(pool->slab)` cannot be called at all. Therefore, `trim` cannot free "unloaned slots" in that pool—it can only skip the entire pool slab (freeing 0 bytes for that pool), or free other unrelated pools that have no active loans.

#### S2: Excessive Scheduler Re-creation Overhead on Small Prompts
- **Citation**: `DESIGN-T10.3.md:43-44`, `DESIGN-T10.3.md:54-55` vs `src/llama-context.cpp:1108-1138`.
- **Issue**: Re-creating the scheduler and measuring graphs via `ggml_backend_sched_new` and `graph_reserve` carries non-trivial CPU latency (~5-15 ms per invocation). Doing this twice for every short prompt in an interactive chat session introduces a noticeable regression.
- **Fix Shape**: Gate the borrowing mechanism behind a minimum token threshold: only borrow when `prompt_tokens >= min_borrow_tokens` (e.g. >= 512 or 1024 tokens). Small prompts should continue using the default decode-sized reserve without triggering a phase switch.

#### S3: Refined Scope vs `kv_stream_phase_arena`
- **Citation**: `DESIGN-T10.3.md:32-34` vs `src/llama-context.h:312`, `src/llama-context.cpp:1275-1354`.
- **Issue**: The design claims it "reuses that phase-switch skeleton, not a new one". However, `kv_stream_phase_arena` is strictly coupled to `memory->get_kv_stream_targets()` and streaming KV ring layouts. A standard model configuration (non-streaming KV) does not instantiate `kv_stream_phase_arena`.
- **Fix Shape**: Clarify that T10.3 implements an independent phase-switch state machine within `llama_context` for MoE borrowing, or explicitly forbid enabling `--prefill-borrow-mib` concurrently with `--kv-stream-arena-mib`.

---

## 3. Answers to Section 7 Open Questions

### Q1: Trim-during-loan: skip loaned ranges (proposed) or abort the loan?
- **Answer**: **Skip the entire pool slab that hosts the loan; never abort the loan synchronously.**
- **Technical Rationale**:
  1. `ggml_moe_cache_trim` (`moe-cache.cu:3342`) is invoked synchronously from deep inside `ggml_cuda_device_malloc` / `cuMemCreate` (`ggml-cuda.cu:520, 622`) when an allocation fails.
  2. If `trim` attempts to "abort the loan", it has no safe way to notify or unwind the active GGML compute graph executing on another thread or higher up the call stack. Freeing the memory under an active execution causes immediate segfaults.
  3. Because `pool->slab` is allocated as a single `cudaMalloc` chunk, CUDA cannot free partial allocations. Any pool with an active loan must be completely skipped by `trim`. `trim` can still free auxiliary buffers (`d_input`, `d_out`, `d_act_q8`) and any other pools without active loans.
  4. If allocator pressure cannot be relieved, `cudaMalloc` will fail, triggering the normal GGML out-of-memory error handling which cleans up the graph and returns the loan on the error unwind path.

### Q2: Flag shape: `--prefill-borrow-mib` + `--prefill-chunk auto|<n>` (proposed), or a single `--prefill auto`?
- **Answer**: **Use `--prefill-borrow auto|<MiB>` (default `off`) paired with `--prefill-chunk auto|<n>` (default `auto`).**
- **Technical Rationale**:
  - Requiring users to specify an exact MiB number is fragile: users do not know the exact slot size of the model's expert pools (e.g. Q4_K vs IQ1_M) or how many slots fit the hardware.
  - With `--prefill-borrow auto`, the runtime computes the borrow size as `floor(available_slab_bytes * 0.5)` or targets a compute buffer adequate for the selected chunk size (capped at 8192 tokens).
  - Specifying an explicit MiB should remain available as an advanced tuning override.

### Q3: v1 scope: fork slot cache only (proposed), or also lend from the T10.2 upstream cache?
- **Answer**: **Fork slot cache (`--moe-cache`) only in v1.**
- **Technical Rationale**:
  - The upstream T10.2 MoE cache (`--moe-cache-mib`, PR #29887) manages memory completely differently via scheduler-owned host tensors and an LRU upload queue.
  - The fork slot cache (`moe-cache.cu`) owns dedicated, pre-allocated VRAM slabs (`moe_cache_pool.slab`) that are immediately amenable to device buffer slicing.
  - Coupling v1 to T10.2 would delay T10.3 and complicate the review surface.

### Q4: Lend granularity: whole slots only (proposed) - and a minimum slot count?
- **Answer**: **Whole slots only, with a minimum threshold of at least 128 MiB (or 8-16 contiguous slots).**
- **Technical Rationale**:
  - Whole slots are mandatory to maintain pool slot accounting and prevent alignment faults in CUDA GEMM/GEMV kernels.
  - A minimum threshold is necessary because the overhead of scheduler re-reservation and CUDA graph reset (~10-20 ms) negates any throughput advantage if only a tiny buffer (e.g. 16-32 MiB) is borrowed.

### Q5: Strata attribution: which exact mechanism is taken from Strata source?
- **Answer**: **Attribute the architectural concept of dynamically borrowing prompt compute staging buffers from an offloaded MoE expert cache slab, and returning them prior to token generation.**
- **Technical Rationale**:
  - Credit Strata (MIT License, Niko1221/Strata) in the commit message and PR description for the concept of phase-borrowed prefill buffers.
  - State clearly that the implementation is a clean-room re-architecture tailored to GGML's backend scheduler, `ggml_gallocr`, and the fork's slot cache.

---

## 4. Manual Verification Items Status

1. **Strata docs/DETAILS.md lines ~160-170 and ~326-340 against this design**:
   - *Status*: File `docs/DETAILS.md` does not exist in the repository root or docs directory. Strata concepts cited in `SPEC-PORT-STRATA.md` are taken from the spec's design notes.
2. **Chunk-size practice (8192 tokens) on target machine**:
   - *Status*: 8192 token chunks are standard for large prompt batching on high-end hardware, but on the 8GB local test environment (RTX 4070 Laptop), an 8192-token MoE compute buffer exceeds total free VRAM. The auto-chunk logic must clamp to available borrowed bytes.
3. **Whether `ggml_backend_sched_reserve_size` can target an externally wrapped buffer without scheduler changes**:
   - *Status*: **VERIFIED NO** (see Blocker B1). `ggml_backend_sched_reserve_size` is an inspection function that writes to `size_t * sizes`; it cannot bind an external buffer. A custom `ggml_backend_buffer_type_t` or scheduler extension is required.

---

## 5. Verified Facts vs. Inferences

| Item / Claim | Category | Confidence | Source / Justification |
| :--- | :--- | :--- | :--- |
| `sched_reserve()` reserves for worst-case `n_tokens` | Verified Fact | 100% | `src/llama-context.cpp:1140-1165` |
| `ggml_moe_cache_trim()` surrenders full pool slabs | Verified Fact | 100% | `ggml/src/ggml-cuda/moe-cache.cu:1718, 3324-3342` |
| `ggml_backend_sched_reserve_size` only writes sizes | Verified Fact | 100% | `ggml/src/ggml-backend.cpp:2130-2142`, `ggml-alloc.c:950` |
| `~ggml_backend_cuda_buffer_context` calls `cudaFree` | Verified Fact | 100% | `ggml/src/ggml-cuda/ggml-cuda.cu:1175-1178` |
| Free slots are fragmented after decode steps | Verified Fact | 100% | `ggml/src/ggml-cuda/moe-cache.cu:118-130, 1152` |
| CUDA graphs only support decode (`ne[1] == 1`) | Verified Fact | 100% | `ggml/src/ggml-cuda/ggml-cuda.cu:4015` |
| Re-reserving decode buffers invalidates CUDA graphs | Verified Fact | 100% | `ggml/src/ggml-cuda/ggml-cuda.cu:3952-3971, 4159` |
| Recreating sched adds ~5-15ms latency per prompt | Inference | High (~90%) | Based on graph traversal and alloc structures overhead |
| Contiguous eviction avoids borrow failures on warm cache | Inference | High (~95%) | Slot layout in slab is linear; contiguous run needed |

---

## 6. Required Next Steps

1. **Update `DESIGN-T10.3.md`** to eliminate reliance on `ggml_backend_sched_reserve_size` for buffer injection and replace it with a dedicated `ggml_backend_buffer_type_t` that safely wraps borrowed slab slices.
2. **Specify Contiguous Slot Allocation & Eviction** in `moe_cache_lend` to handle warm-cache fragmentation.
3. **Add `ggml_backend_cuda_graph_reset`** to the decode transition path to prevent stale graph execution.
4. **Update `moe_cache_trim` specification** to clarify that pools hosting active loans are skipped in their entirety.
5. Resubmit revised `DESIGN-T10.3.md` for maintainer review prior to coding.
