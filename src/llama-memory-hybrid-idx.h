#pragma once

#include "llama-memory-hybrid.h"

#include <array>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache that holds one indexer key per token, for hybrid
// architectures with block-sparse attention layers (qwen4exp QSA).
// The indexer cache is a side buffer addressed by the cells of the attention cache: same
// size, padding, stream count and slots, so cell j is the same token in both.

// TODO: this memory module is pending complete reimplementation - do not use for model other than Qwen4

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   gdn_replay_req,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    // Defined out of line because kpool_layout is incomplete here.
    ~llama_memory_hybrid_idx();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // [TAG_PLE_HISTORY]
    // The qwen4exp PLE hash of a token mixes in the ple_ngram_size - 1 tokens before it, which
    // a decode ubatch does not carry. It lives here because it is per-context per-sequence
    // state: it must follow the seq_* operations and the state blob, like the caches next to it.
    struct ple_history {
        // position the next token of this sequence must have; -1 means the window is not trusted
        llama_pos next_pos = -1;

        // the tokens at [next_pos - toks.size(), next_pos), oldest first, at most ple_ngram_size - 1
        // it can be shorter near a sequence start or after a rewind; the caller pads the front with EOS
        std::vector<llama_token> toks;
    };

    // history for seq_id, default-constructed (and so untrusted) on first use
    // const because set_input updates it through a const memory context
    ple_history & ple_hist_get(llama_seq_id seq_id) const;

    // The model's indexer pool size.
    uint32_t get_kpool() const { return hparams_idx.indexer_kpool; }

    // Whether pools are kpool consecutive cells in sequence order (qwen4exp) instead of kpool consecutive positions.
    bool get_kpool_by_order() const { return hparams_idx.indexer_kpool_by_order; }

    // Which cells of a sequence make up which pool of kpool consecutive positions (or cells, in order mode).
    // It is kept here because it outlives the batch: pools are fixed by the positions relative to the
    // sequence's first one, so a ubatch only ever appends to it. Sequence edits drop it, see mem_idx_stale.
    struct kpool_layout;

    const kpool_layout & kpool_layout_update();
    const kpool_layout & kpool_layout_get() const;

    // The pooled keys persist in the idx cache across batches. A sequence edit can regroup the pools
    // from some position on, which stales every pooled key at or after it. POS_CLEAN means none.
    using stale_pos_t = std::array<llama_pos, LLAMA_MAX_SEQ>;

    static constexpr llama_pos POS_CLEAN = std::numeric_limits<llama_pos>::max();

    static stale_pos_t stale_pos_clean() {
        stale_pos_t res;
        res.fill(POS_CLEAN);
        return res;
    }

    const stale_pos_t & mem_idx_stale_get() const { return mem_idx_stale; }
    void mem_idx_stale_clear() { mem_idx_stale.fill(POS_CLEAN); }

private:
    // forget seq_id (or, for seq_id < 0, everything) in every cache at once, so that a restore
    // that failed partway cannot leave the indexer cache holding cells the attention cache does
    // not. seq_id < 0 drops the whole context, as the caches themselves do on a failed restore.
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // unique_ptr because kpool_layout is incomplete here
    std::unique_ptr<kpool_layout> kpool_lay;

    // whether the current layout has cells shared between sequences (kpool_layout is incomplete here, so out of line)
    bool kpool_layout_shared() const;

    // seq_id < 0 stales every sequence, p0 < 0 stales the sequence from its first position
    void mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0);

    // the position an edit at p0 stales the sequence from
    llama_pos mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const;

    stale_pos_t mem_idx_stale = stale_pos_clean();

    // [TAG_PLE_HISTORY] empty for every architecture but qwen4exp, the only one that asks for a history
    mutable std::unordered_map<llama_seq_id, ple_history> ple_hist;

    // the seq_* halves of the history bookkeeping, one per llama_memory_i operation
    void ple_hist_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1);
    void ple_hist_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
    void ple_hist_keep(llama_seq_id seq_id);
    void ple_hist_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift);
    void ple_hist_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1);

    void ple_hist_state_write(llama_io_write_i & io, llama_seq_id seq_id) const;
    void ple_hist_state_read (llama_io_read_i  & io, llama_seq_id seq_id);
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    class kpool_access {
    public:
        ggml_tensor * gather_key_gate(ggml_tensor * idxs) const;
        ggml_tensor * scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const;
        ggml_tensor * gather_pooled(ggml_tensor * idxs) const;

    private:
        friend class llama_memory_hybrid_idx_context;

        kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd);

        ggml_context * ctx;
        ggml_tensor  * key_gate;
        ggml_tensor  * pooled;
    };

    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context(); // Defined out of line because kpool_state is incomplete here.

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer, and for the full and update contexts, which build no sparse graph
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // [TAG_PLE_HISTORY] the per-sequence n-gram history of the owning memory, for set_input
    llama_memory_hybrid_idx::ple_history & get_ple_hist(llama_seq_id seq_id) const;

    // glm5-next and qwen4exp, complete pools of kpool cells per sequence, scored as whole pools.
    uint32_t get_n_kpool    () const; // Padded pool count, where the last pool is always unused.
    uint32_t get_n_kpool_new() const; // Pools to re-pool this ubatch, padded to a stable bound, never below 1.
    bool get_kpool_cache_safe() const;
    kpool_access get_kpool_access(ggml_context * ctx, int32_t il, int64_t n_embd) const;
    ggml_tensor * gather_mla_rows(ggml_context * ctx, ggml_tensor * idxs, int64_t n_rows, int64_t n_embd, int32_t il) const;
    // new_pool_pos (I32 [4*n_new]): M-RoPE position of each new pool's first member, for pooled keys rotated at pooling time
    void set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
                         ggml_tensor * gather_mask, bool gather, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
                         const llama_ubatch * ubatch, ggml_tensor * new_pool_pos = nullptr) const;

private:
    llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // the indexer cells of each ubatch, kept for pools in cache order (qwen4exp): token s*n + i of ubatch u
    // sits in cell idxs[s][i] of stream strm[s] of sinfos_kpool[u], and several cells can share a position
    const slot_info_vec_t sinfos_kpool;

    // null unless the model has an indexer and this is a batch context
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;

    // Which pools of the layout this ubatch must re-pool. The layout itself belongs to the memory.
    struct kpool_state;
    kpool_state kpool_build_sizes() const;
    void kpool_build_state(const llama_ubatch & ubatch);
    const kpool_state & kpool_cur() const;

    // unique_ptr because kpool_state is incomplete here.
    std::unique_ptr<kpool_state> kpool_st;

    // The ubatch kpool_st was built for, guards against reads before apply.
    size_t i_kpool = SIZE_MAX;

    // Whether this context tracks k-pool states.
    bool kpool_track() const;

    // Positions each sequence must re-pool from, cleared only after the first ubatch succeeds
    llama_memory_hybrid_idx::stale_pos_t mem_idx_stale_batch = llama_memory_hybrid_idx::stale_pos_clean();
};
