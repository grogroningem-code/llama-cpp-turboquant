#include "spec-draft-vocab.h"

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <unordered_set>

uint64_t spec_draft_vocab_fnv1a(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t spec_draft_vocab_tokenizer_hash(const struct llama_vocab * vocab) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    uint64_t hash = 1469598103934665603ull;
    char piece[256];
    for (int32_t id = 0; id < n_vocab; id++) {
        const int32_t len = llama_token_to_piece(vocab, id, piece, sizeof(piece), 0, true);
        if (len > 0) {
            const uint64_t h = spec_draft_vocab_fnv1a(piece, (size_t) len);
            hash ^= h;
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

std::vector<int32_t> spec_draft_vocab_load(const std::string & path, int32_t n_vocab, uint64_t tokenizer_hash) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("spec-draft-vocab: cannot open " + path);
    }
    auto read_u32 = [&](uint32_t & v) {
        if (fread(&v, 4, 1, f) != 1) {
            fclose(f);
            throw std::runtime_error("spec-draft-vocab: short read in " + path);
        }
    };
    uint32_t magic, version, file_vocab, count;
    uint64_t file_hash;
    read_u32(magic);
    read_u32(version);
    read_u32(file_vocab);
    read_u32(count);
    if (fread(&file_hash, 8, 1, f) != 1) {
        fclose(f);
        throw std::runtime_error("spec-draft-vocab: short read in " + path);
    }
    if (magic != 0x32335644u) {
        fclose(f);
        throw std::runtime_error("spec-draft-vocab: bad magic");
    }
    if (version != 1u) {
        fclose(f);
        throw std::runtime_error("spec-draft-vocab: bad version");
    }
    if ((int64_t) file_vocab != (int64_t) n_vocab) {
        fclose(f);
        throw std::runtime_error("spec-draft-vocab: n_vocab mismatch");
    }
    if (file_hash != tokenizer_hash) {
        fclose(f);
        throw std::runtime_error("spec-draft-vocab: tokenizer hash mismatch");
    }
    // validate header against the real file size BEFORE allocating:
    // a corrupt count must fail here, not inside a multi-GB vector alloc
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        throw std::runtime_error("spec-draft-vocab: cannot seek in " + path);
    }
    const long fsize = ftell(f);
    if (fsize < 0 || fseek(f, 24, SEEK_SET) != 0) {
        fclose(f);
        throw std::runtime_error("spec-draft-vocab: cannot stat " + path);
    }
    if (count == 0) {
        fclose(f);
        throw std::runtime_error("spec-draft-vocab: empty id list");
    }
    if ((int64_t) count > (int64_t) n_vocab) {
        fclose(f);
        throw std::runtime_error("spec-draft-vocab: count exceeds n_vocab");
    }
    if ((uint64_t) fsize > 1048576u) {
        fclose(f);
        throw std::runtime_error("spec-draft-vocab: file too large");
    }
    if (24ull + (uint64_t) count * 4ull > (uint64_t) fsize) {
        fclose(f);
        throw std::runtime_error("spec-draft-vocab: file too short for count");
    }
    std::vector<int32_t> ids(count);
    if (count > 0 && fread(ids.data(), 4, count, f) != count) {
        fclose(f);
        throw std::runtime_error("spec-draft-vocab: short id read");
    }
    // no trailing bytes allowed
    uint8_t tail;
    if (fread(&tail, 1, 1, f) != 0) {
        fclose(f);
        throw std::runtime_error("spec-draft-vocab: trailing bytes");
    }
    fclose(f);
    // id checks
    std::unordered_set<int32_t> seen;
    seen.reserve(ids.size() * 2 + 1);
    for (int32_t id : ids) {
        if (id < 0 || (int64_t) id >= (int64_t) n_vocab) {
            throw std::runtime_error("spec-draft-vocab: id out of range");
        }
        if (!seen.insert(id).second) {
            throw std::runtime_error("spec-draft-vocab: duplicate id");
        }
    }
    return ids;
}

spec_draft_vocab_handle::spec_draft_vocab_handle(std::vector<int32_t> ids, int32_t n_seqs, size_t seq_capacity)
    : m_sorted(std::move(ids)),
      m_slots(n_seqs < 0 ? 0 : (size_t) n_seqs),
      m_capacity(seq_capacity) {
    std::sort(m_sorted.begin(), m_sorted.end());
}

bool spec_draft_vocab_handle::contains(int32_t id) const {
    return std::binary_search(m_sorted.begin(), m_sorted.end(), id);
}

bool spec_draft_vocab_handle::accept(int32_t seq, int32_t pos, int32_t id) {
    auto & slot = m_slots.at(seq); // throws std::out_of_range on bad slot
    if (!contains(id)) {
        return false;
    }
    while (m_capacity > 0 && slot.size() >= m_capacity) {
        slot.erase(slot.begin()); // FIFO evict; evicted entries are never restored
    }
    slot.emplace_back(pos, id);
    return true;
}

size_t spec_draft_vocab_handle::drop_deferred_from(int32_t seq, int32_t from_pos) {
    auto & slot = m_slots.at(seq); // throws std::out_of_range on bad slot
    const size_t before = slot.size();
    slot.erase(std::remove_if(slot.begin(), slot.end(), [&](const std::pair<int32_t, int32_t> & e) {
        return e.first >= from_pos;
    }), slot.end());
    return before - slot.size();
}

const std::vector<std::pair<int32_t, int32_t>> & spec_draft_vocab_handle::entries(int32_t seq) const {
    return m_slots.at(seq); // throws std::out_of_range on bad slot
}
