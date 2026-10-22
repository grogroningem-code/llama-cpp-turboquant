// Build a low-rank SVD sidecar GGUF for the MTP draft head.
// usage: llama-vocab-lowrank -m model.gguf -o lowrank.gguf [-r 128] [-q 2] [--eval file.txt [-n 200] [-k 128]]
#include "vocab_lowrank_svd.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void usage(const char * prog) {
    printf("usage: %s -m model.gguf -o lowrank.gguf [-r 128] [-q 2] [--eval file.txt [-n 200] [-k 128]]\n", prog);
}

int main(int argc, char ** argv) {
    std::string model_path, out_path, eval_path;
    int rank = 128;
    int n_iter = 2;
    int eval_n = 200;
    int eval_k = 128;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "-o" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "-r" && i + 1 < argc) {
            rank = std::atoi(argv[++i]);
        } else if (a == "-q" && i + 1 < argc) {
            n_iter = std::atoi(argv[++i]);
        } else if (a == "--eval" && i + 1 < argc) {
            eval_path = argv[++i];
        } else if (a == "-n" && i + 1 < argc) {
            eval_n = std::atoi(argv[++i]);
        } else if (a == "-k" && i + 1 < argc) {
            eval_k = std::atoi(argv[++i]);
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || out_path.empty() || rank <= 0) {
        usage(argv[0]);
        return 1;
    }

    // 1. Open GGUF and find the MTP/LM head tensor
    ggml_context * ctx_meta = nullptr;
    gguf_init_params gparams = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx_meta };
    gguf_context * ctx_gguf = gguf_init_from_file(model_path.c_str(), gparams);
    if (!ctx_gguf) {
        fprintf(stderr, "error: failed to open GGUF '%s'\n", model_path.c_str());
        return 1;
    }

    ggml_tensor * head_meta = nullptr;
    for (ggml_tensor * cur = ggml_get_first_tensor(ctx_meta); cur; cur = ggml_get_next_tensor(ctx_meta, cur)) {
        std::string tname(cur->name);
        if (tname.find(".nextn.shared_head_head.weight") != std::string::npos) {
            head_meta = cur;
            break;
        }
    }
    if (!head_meta) {
        head_meta = ggml_get_tensor(ctx_meta, "output.weight");
    }
    if (!head_meta) {
        head_meta = ggml_get_tensor(ctx_meta, "token_embd.weight");
    }
    if (!head_meta) {
        fprintf(stderr, "error: no output.weight or token_embd.weight in '%s'\n", model_path.c_str());
        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
        return 1;
    }

    const int64_t H = head_meta->ne[0];
    const int64_t V = head_meta->ne[1];
    if (rank > H || rank > V) {
        fprintf(stderr, "error: rank %d exceeds head dims [%lld, %lld]\n", rank, (long long) H, (long long) V);
        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
        return 1;
    }

    printf("lowrank: dequantizing '%s' [%lld, %lld] (%s)...\n",
            head_meta->name, (long long) H, (long long) V, ggml_type_name(head_meta->type));
    fflush(stdout);

    const int tid = gguf_find_tensor(ctx_gguf, head_meta->name);
    const size_t offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, tid);
    const size_t nbytes = ggml_nbytes(head_meta);

    std::vector<uint8_t> raw(nbytes);
    {
        FILE * fp = fopen(model_path.c_str(), "rb");
        if (!fp) {
            fprintf(stderr, "error: cannot open '%s'\n", model_path.c_str());
            return 1;
        }
        if (_fseeki64(fp, (int64_t) offset, SEEK_SET) != 0 || fread(raw.data(), 1, nbytes, fp) != nbytes) {
            fprintf(stderr, "error: short read on '%s'\n", head_meta->name);
            fclose(fp);
            return 1;
        }
        fclose(fp);
    }

    // W [V, H] in float32 (row v has length H)
    std::vector<float> W((size_t) V * H);
    if (head_meta->type == GGML_TYPE_F32) {
        std::memcpy(W.data(), raw.data(), nbytes);
    } else {
        const auto * traits = ggml_get_type_traits(head_meta->type);
        if (!traits || !traits->to_float) {
            fprintf(stderr, "error: no to_float for type %s\n", ggml_type_name(head_meta->type));
            return 1;
        }
        traits->to_float(raw.data(), W.data(), V * H);
    }
    raw.clear();
    raw.shrink_to_fit();
    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);

    std::vector<float> down_f32, scores_f32;
    double energy_share = 0.0;
    compute_randomized_svd(W, H, V, rank, n_iter, down_f32, scores_f32, energy_share);
    W.clear();
    W.shrink_to_fit();

    std::vector<ggml_fp16_t> down_f16(down_f32.size());
    for (size_t i = 0; i < down_f32.size(); ++i) {
        ggml_fp16_t hval = ggml_fp32_to_fp16(down_f32[i]);
        down_f16[i] = hval;
        down_f32[i] = ggml_fp16_to_fp32(hval);
    }

    std::vector<ggml_fp16_t> scores_f16(scores_f32.size());
    for (size_t i = 0; i < scores_f32.size(); ++i) {
        ggml_fp16_t hval = ggml_fp32_to_fp16(scores_f32[i]);
        scores_f16[i] = hval;
        scores_f32[i] = ggml_fp16_to_fp32(hval);
    }

    // 4. Write sidecar GGUF with down [n_embd, r] and scores [r, n_vocab] in F16
    {
        const size_t meta_size = 4 * ggml_tensor_overhead() + 1024;
        ggml_init_params iparams = { meta_size, nullptr, true };
        ggml_context * ctx_out = ggml_init(iparams);

        ggml_tensor * t_down   = ggml_new_tensor_2d(ctx_out, GGML_TYPE_F16, H, rank);
        ggml_tensor * t_scores = ggml_new_tensor_2d(ctx_out, GGML_TYPE_F16, rank, V);
        ggml_set_name(t_down,   "down");
        ggml_set_name(t_scores, "scores");
        t_down->data   = down_f16.data();
        t_scores->data = scores_f16.data();

        gguf_context * g_out = gguf_init_empty();
        gguf_set_val_u32(g_out, "lowrank.rank",         (uint32_t) rank);
        gguf_set_val_u32(g_out, "lowrank.n_embd",       (uint32_t) H);
        gguf_set_val_u32(g_out, "lowrank.n_vocab",      (uint32_t) V);
        gguf_set_val_f32(g_out, "lowrank.energy_share", (float) energy_share);

        gguf_add_tensor(g_out, t_down);
        gguf_add_tensor(g_out, t_scores);

        if (!gguf_write_to_file(g_out, out_path.c_str(), false)) {
            fprintf(stderr, "error: failed to write GGUF to '%s'\n", out_path.c_str());
            gguf_free(g_out);
            ggml_free(ctx_out);
            return 1;
        }
        gguf_free(g_out);
        ggml_free(ctx_out);
        printf("lowrank: wrote '%s' (down [%lld, %d] F16, scores [%d, %lld] F16)\n",
                out_path.c_str(), (long long) H, rank, rank, (long long) V);
    }

    // 5. Optional recall check on eval_n hidden states from eval_path
    if (!eval_path.empty() && eval_n > 0) {
        W.clear();
        W.shrink_to_fit();
        llama_backend_init();
        std::ifstream in(eval_path, std::ios::binary);
        if (!in) {
            fprintf(stderr, "error: cannot open eval file '%s'\n", eval_path.c_str());
            return 1;
        }
        std::stringstream ss;
        ss << in.rdbuf();
        std::string text = ss.str();

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 99;
        mparams.load_mtp = true;
        llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
        if (!model) {
            fprintf(stderr, "error: failed to load model '%s' for eval\n", model_path.c_str());
            return 1;
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        std::vector<llama_token> toks(text.size() + 8);
        int32_t n_tok = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, false);
        if (n_tok < 0) {
            toks.resize(-n_tok);
            n_tok = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, false);
        }
        if (n_tok < eval_n) {
            fprintf(stderr, "error: eval file has only %d tokens, need %d\n", n_tok, eval_n);
            return 1;
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = std::max(512, eval_n + 64);
        cparams.n_batch = eval_n;
        cparams.n_ubatch = eval_n;
        cparams.embeddings = true;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "error: failed to create context for eval\n");
            return 1;
        }
        llama_set_embeddings_nextn(ctx, true, false);

        llama_context_params cparams_dft = llama_context_default_params();
        cparams_dft.n_ctx = std::max(512, eval_n + 64);
        cparams_dft.n_batch = eval_n;
        cparams_dft.n_ubatch = eval_n;
        cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        llama_context * ctx_dft = llama_init_from_model(model, cparams_dft);
        if (ctx_dft) {
            llama_set_embeddings_nextn(ctx_dft, true, false);
        }

        llama_batch batch = llama_batch_init(eval_n, 0, 1);
        batch.n_tokens = eval_n;
        for (int i = 0; i < eval_n; ++i) {
            batch.token[i]    = toks[i];
            batch.pos[i]      = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0]= 0;
            batch.logits[i]   = 1;
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "error: llama_decode failed on eval tokens\n");
            return 1;
        }

        auto eval_recall = [&](const float * h_all, int n_rows, const char * label) {
            int hits = 0;
            for (int t = 0; t < n_rows; ++t) {
                const float * h = h_all + (size_t) t * H;
                // Compute full head argmax directly on W [V, H] or from llama_get_logits_ith
                // Wait: for trunk, llama_get_logits_ith(ctx, t) is exact; for MTP, llama_get_logits_ith(ctx_dft, t) is exact!
                const float * full_logits = (std::strcmp(label, "trunk") == 0)
                    ? llama_get_logits_ith(ctx, t)
                    : llama_get_logits_ith(ctx_dft, t);
                int full_argmax = (int) (std::max_element(full_logits, full_logits + V) - full_logits);

                std::vector<float> z(rank, 0.0f);
                for (int k = 0; k < rank; ++k) {
                    const float * d_row = down_f32.data() + (size_t) k * H;
                    float sum = 0.0f;
                    for (int64_t d = 0; d < H; ++d) {
                        sum += d_row[d] * h[d];
                    }
                    z[k] = sum;
                }

                std::vector<float> lr(V);
                for (int64_t v = 0; v < V; ++v) {
                    const float * s_row = scores_f32.data() + (size_t) v * rank;
                    float sum = 0.0f;
                    for (int k = 0; k < rank; ++k) {
                        sum += s_row[k] * z[k];
                    }
                    lr[v] = sum;
                }

                float target_lr = lr[full_argmax];
                int strictly_greater = 0;
                for (int64_t v = 0; v < V; ++v) {
                    if (lr[v] > target_lr) {
                        strictly_greater++;
                    }
                }
                if (strictly_greater < eval_k) {
                    hits++;
                }
            }
            printf("lowrank: recall@%d on %d %s hidden states from '%s': %d / %d = %.4f (%.2f%%)\n",
                    eval_k, n_rows, label, eval_path.c_str(), hits, n_rows,
                    (double) hits / n_rows, 100.0 * hits / n_rows);
            fflush(stdout);
        };

        const float * h_trunk = llama_get_embeddings_nextn(ctx);
        eval_recall(h_trunk, eval_n, "trunk");

        if (ctx_dft && n_tok >= eval_n + 1) {
            llama_batch batch_mtp = llama_batch_init(eval_n, (int32_t) H, 1);
            batch_mtp.token = (llama_token *) malloc(sizeof(llama_token) * eval_n);
            batch_mtp.n_tokens = eval_n;
            for (int i = 0; i < eval_n; ++i) {
                batch_mtp.token[i]    = toks[i + 1];
                batch_mtp.pos[i]      = i + 1;
                batch_mtp.n_seq_id[i] = 1;
                batch_mtp.seq_id[i][0]= 0;
                batch_mtp.logits[i]   = 1;
                std::memcpy(batch_mtp.embd + (size_t) i * H, h_trunk + (size_t) i * H, (size_t) H * sizeof(float));
            }
            if (llama_decode(ctx_dft, batch_mtp) == 0) {
                const float * h_mtp = llama_get_embeddings_nextn(ctx_dft);
                eval_recall(h_mtp, eval_n, "mtp");
            }
            free(batch_mtp.token);
            batch_mtp.token = nullptr;
            llama_batch_free(batch_mtp);
        }

        if (ctx_dft) {
            llama_free(ctx_dft);
        }
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
    }

    return 0;
}
