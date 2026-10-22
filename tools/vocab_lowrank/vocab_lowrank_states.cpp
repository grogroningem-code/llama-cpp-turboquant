#include "vocab_lowrank_states.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

static llama_context * make_ctx(llama_model * model, int n, bool mtp) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = std::max(512, n + 64);
    cparams.n_batch  = n;
    cparams.n_ubatch = n;
    if (mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    } else {
        cparams.embeddings = true;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx) {
        llama_set_embeddings_nextn(ctx, true, false);
    }
    return ctx;
}

bool lowrank_states_collect(llama_model * model, const std::string & path, int n, int64_t H, lowrank_states & st) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fprintf(stderr, "error: cannot open '%s'\n", path.c_str());
        return false;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(text.size() + 8);
    int32_t n_tok = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, false);
    if (n_tok < 0) {
        toks.resize(-n_tok);
        n_tok = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, false);
    }
    if (n_tok < n + 1) {
        fprintf(stderr, "error: '%s' has only %d tokens, need %d\n", path.c_str(), n_tok, n + 1);
        return false;
    }

    st.ctx     = make_ctx(model, n, false);
    st.ctx_dft = make_ctx(model, n, true);
    if (!st.ctx || !st.ctx_dft) {
        fprintf(stderr, "error: failed to create trunk or MTP context\n");
        return false;
    }

    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i]    = toks[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0]= 0;
        batch.logits[i]   = 1;
    }
    const bool ok_trunk = llama_decode(st.ctx, batch) == 0;
    llama_batch_free(batch);
    if (!ok_trunk) {
        fprintf(stderr, "error: llama_decode failed on trunk tokens of '%s'\n", path.c_str());
        return false;
    }
    st.h_trunk = llama_get_embeddings_nextn(st.ctx);

    llama_batch batch_mtp = llama_batch_init(n, (int32_t) H, 1);
    batch_mtp.token = (llama_token *) malloc(sizeof(llama_token) * n);
    batch_mtp.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch_mtp.token[i]    = toks[i + 1];
        batch_mtp.pos[i]      = i + 1;
        batch_mtp.n_seq_id[i] = 1;
        batch_mtp.seq_id[i][0]= 0;
        batch_mtp.logits[i]   = 1;
        std::memcpy(batch_mtp.embd + (size_t) i * H, st.h_trunk + (size_t) i * H, (size_t) H * sizeof(float));
    }
    const bool ok_mtp = llama_decode(st.ctx_dft, batch_mtp) == 0;
    free(batch_mtp.token);
    batch_mtp.token = nullptr;
    llama_batch_free(batch_mtp);
    if (!ok_mtp) {
        fprintf(stderr, "error: llama_decode failed on MTP tokens of '%s'\n", path.c_str());
        return false;
    }
    st.h_mtp = llama_get_embeddings_nextn(st.ctx_dft);
    return true;
}

void lowrank_states_free(lowrank_states & st) {
    if (st.ctx_dft) {
        llama_free(st.ctx_dft);
    }
    if (st.ctx) {
        llama_free(st.ctx);
    }
    st = lowrank_states();
}

void lowrank_print_recall(const lowrank_states & st, bool mtp, int n, int64_t H, int64_t V, int rank,
        const std::vector<float> & down_f32, const std::vector<float> & scores_f32,
        const std::vector<int> & ks, const std::string & path) {
    const float * h_all = mtp ? st.h_mtp : st.h_trunk;
    std::vector<int> hits(ks.size(), 0);
    std::vector<float> z(rank);
    std::vector<float> lr(V);
    for (int t = 0; t < n; ++t) {
        const float * h = h_all + (size_t) t * H;
        const float * full_logits = llama_get_logits_ith(mtp ? st.ctx_dft : st.ctx, t);
        const int64_t full_argmax = std::max_element(full_logits, full_logits + V) - full_logits;

        for (int k = 0; k < rank; ++k) {
            const float * d_row = down_f32.data() + (size_t) k * H;
            float sum = 0.0f;
            for (int64_t d = 0; d < H; ++d) {
                sum += d_row[d] * h[d];
            }
            z[k] = sum;
        }
        for (int64_t v = 0; v < V; ++v) {
            const float * s_row = scores_f32.data() + (size_t) v * rank;
            float sum = 0.0f;
            for (int k = 0; k < rank; ++k) {
                sum += s_row[k] * z[k];
            }
            lr[v] = sum;
        }

        const float target_lr = lr[full_argmax];
        int strictly_greater = 0;
        for (int64_t v = 0; v < V; ++v) {
            if (lr[v] > target_lr) {
                strictly_greater++;
            }
        }
        for (size_t ki = 0; ki < ks.size(); ++ki) {
            if (strictly_greater < ks[ki]) {
                hits[ki]++;
            }
        }
    }
    for (size_t ki = 0; ki < ks.size(); ++ki) {
        printf("lowrank: recall@%d on %d %s hidden states from '%s': %d / %d = %.4f (%.2f%%)\n",
                ks[ki], n, mtp ? "mtp" : "trunk", path.c_str(), hits[ki], n,
                (double) hits[ki] / n, 100.0 * hits[ki] / n);
    }
    fflush(stdout);
}
