#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Draft-vocab file (DV32): top-freq token ids used to shrink the MTP draft head.
// Layout: magic "DV32", version u32, n_vocab u32, count u32, tokenizer hash u64,
// then count LE int32 ids. Loader throws std::runtime_error on any bad check.
std::vector<int32_t> spec_draft_vocab_load(const std::string & path, int32_t n_vocab, uint64_t tokenizer_hash);

// FNV-1a over bytes. Tool and loader use it for the tokenizer hash.
uint64_t spec_draft_vocab_fnv1a(const void * data, size_t n);

struct llama_vocab; // from llama.h; defined here to keep this header light

// FNV-1a tokenizer hash over vocab pieces in id order. Single definition
// shared by the tool (writes it into DV32) and the MTP ctor (checks it).
// Pieces that are empty or do not fit the 256-byte buffer are skipped, so
// vocabularies differing only in those tokens hash the same by design.
uint64_t spec_draft_vocab_tokenizer_hash(const struct llama_vocab * vocab);

// Per-sequence rewind handle over one immutable fixed draft set (v2-ready
// semantics; v1 uses the fixed set only, no per-step mutation of the set):
// - one slot per sequence: parallel slots never share entries
// - only target-accepted tokens enter: accept() rejects ids outside the set
// - entries carry positions: (pos, id) pairs in acceptance order
// - overflow FIFO-evicts the oldest entry; evicted entries are never restored
// - drop_deferred_from() is the hook the speculative rewind path calls
// A handle bug costs acceptance rate only, never correctness.
struct spec_draft_vocab_handle {
    spec_draft_vocab_handle(std::vector<int32_t> ids, int32_t n_seqs = 1, size_t seq_capacity = 128);

    bool contains(int32_t id) const;

    // record a target-accepted token; returns false (no entry) if id is not in the set
    bool accept(int32_t seq, int32_t pos, int32_t id);

    // drop entries with pos >= from_pos; returns the dropped count
    size_t drop_deferred_from(int32_t seq, int32_t from_pos);

    const std::vector<std::pair<int32_t, int32_t>> & entries(int32_t seq) const;

  private:
    std::vector<int32_t> m_sorted; // fixed set, sorted, for membership
    std::vector<std::vector<std::pair<int32_t, int32_t>>> m_slots;
    size_t m_capacity;
};
