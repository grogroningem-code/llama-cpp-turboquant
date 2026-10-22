#pragma once
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

// Hidden states of the first n tokens of a text file: trunk states (input of
// the MTP layer) and MTP states (input of the MTP LM head). The contexts stay
// alive so the full-head logits of each row can be read back.
struct lowrank_states {
    llama_context * ctx     = nullptr;
    llama_context * ctx_dft = nullptr;
    const float   * h_trunk = nullptr;
    const float   * h_mtp   = nullptr;
};

// Returns false (after printing an error) if the file is unreadable, too short
// or a decode fails. Call lowrank_states_free afterwards in every case.
bool lowrank_states_collect(llama_model * model, const std::string & path, int n, int64_t H, lowrank_states & st);

void lowrank_states_free(lowrank_states & st);

// Prints recall@k for each k: the share of rows whose full-head argmax is in
// the top k of the low-rank scores (scores * down * h).
void lowrank_print_recall(const lowrank_states & st, bool mtp, int n, int64_t H, int64_t V, int rank,
        const std::vector<float> & down_f32, const std::vector<float> & scores_f32,
        const std::vector<int> & ks, const std::string & path);
