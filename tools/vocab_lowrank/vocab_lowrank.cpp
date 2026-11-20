// Build a low-rank SVD sidecar GGUF for the MTP draft head.
// usage: llama-vocab-lowrank -m model.gguf -o lowrank.gguf [-r 128] [-q 2] [--eval file.txt [-n 200] [-k 128]]
#include "vocab_lowrank_svd.h"
#include "vocab_lowrank_states.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// _fseeki64/_ftelli64 are MSVC names; elsewhere off_t is 64-bit and fseeko/ftello do the same
#ifndef _WIN32
#define _fseeki64 fseeko
#define _ftelli64 ftello
#endif

static void usage(const char * prog) {
    printf("usage: %s -m model.gguf -o lowrank.gguf [-r 128] [-q 2] [--weight-act calib.txt] [--weight-freq freq.bin|file.txt] [--eval file.txt [-n 200] [-k 128]]\n", prog);
}

int main(int argc, char ** argv) {
    std::string model_path, out_path, eval_path, weight_act_path, weight_freq_path;
    int rank = 128;
    int n_iter = 2;
    int eval_n = 200;
    std::vector<int> eval_ks;

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
        } else if (a == "--weight-act" && i + 1 < argc) {
            weight_act_path = argv[++i];
        } else if (a == "--weight-freq" && i + 1 < argc) {
            weight_freq_path = argv[++i];
        } else if (a == "--eval" && i + 1 < argc) {
            eval_path = argv[++i];
        } else if (a == "-n" && i + 1 < argc) {
            eval_n = std::atoi(argv[++i]);
        } else if (a == "-k" && i + 1 < argc) {
            eval_ks.push_back(std::atoi(argv[++i]));
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (eval_ks.empty()) {
        eval_ks.push_back(128);
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

    // Optional frequency weighting (--weight-freq): scale row v of W by sqrt(freq_v / mean(freq) + 10)
    std::vector<float> sqrt_freq;
    if (!weight_freq_path.empty()) {
        std::vector<double> freq((size_t) V, 0.0);
        FILE * ff = fopen(weight_freq_path.c_str(), "rb");
        if (!ff) {
            fprintf(stderr, "error: cannot open freq file '%s'\n", weight_freq_path.c_str());
            return 1;
        }
        _fseeki64(ff, 0, SEEK_END);
        int64_t fsize = _ftelli64(ff);
        _fseeki64(ff, 0, SEEK_SET);
        uint32_t magic = 0;
        if (fsize >= 24 && fread(&magic, 4, 1, ff) == 1 && (magic == 0x54435644u || magic == 0x32335644u)) {
            uint32_t version = 0, file_vocab = 0, count = 0;
            uint64_t hash = 0;
            if (fread(&version, 4, 1, ff) != 1 || fread(&file_vocab, 4, 1, ff) != 1 ||
                fread(&count, 4, 1, ff) != 1 || fread(&hash, 8, 1, ff) != 1) {
                fprintf(stderr, "error: short header read in '%s'\n", weight_freq_path.c_str());
                fclose(ff);
                return 1;
            }
            if ((int64_t) file_vocab != V) {
                fprintf(stderr, "error: freq file n_vocab %u != %lld\n", file_vocab, (long long) V);
                fclose(ff);
                return 1;
            }
            if (magic == 0x54435644u && fsize == 24 + (int64_t) V * 8) {
                std::vector<int64_t> counts((size_t) V);
                if (fread(counts.data(), 8, (size_t) V, ff) != (size_t) V) {
                    fprintf(stderr, "error: short counts read in '%s'\n", weight_freq_path.c_str());
                    fclose(ff);
                    return 1;
                }
                for (int64_t v = 0; v < V; ++v) {
                    freq[v] = (double) std::max<int64_t>(0, counts[v]);
                }
            } else if (magic == 0x32335644u && fsize == 24 + (int64_t) count * 4 + (int64_t) V * 8) {
                _fseeki64(ff, 24 + (int64_t) count * 4, SEEK_SET);
                std::vector<int64_t> counts((size_t) V);
                if (fread(counts.data(), 8, (size_t) V, ff) != (size_t) V) {
                    fprintf(stderr, "error: short counts tail read in '%s'\n", weight_freq_path.c_str());
                    fclose(ff);
                    return 1;
                }
                for (int64_t v = 0; v < V; ++v) {
                    freq[v] = (double) std::max<int64_t>(0, counts[v]);
                }
            } else if (magic == 0x32335644u && fsize == 24 + (int64_t) count * 4) {
                std::vector<int32_t> ids(count);
                if (count > 0 && fread(ids.data(), 4, count, ff) != count) {
                    fprintf(stderr, "error: short id read in '%s'\n", weight_freq_path.c_str());
                    fclose(ff);
                    return 1;
                }
                for (uint32_t r = 0; r < count; ++r) {
                    if (ids[r] >= 0 && ids[r] < V) {
                        freq[ids[r]] = 1.0 / (double) (r + 1);
                    }
                }
            } else {
                fprintf(stderr, "error: unrecognized DV header/size in '%s'\n", weight_freq_path.c_str());
                fclose(ff);
                return 1;
            }
            fclose(ff);
        } else if (fsize == (int64_t) V * 8) {
            _fseeki64(ff, 0, SEEK_SET);
            std::vector<int64_t> counts((size_t) V);
            if (fread(counts.data(), 8, (size_t) V, ff) != (size_t) V) {
                fprintf(stderr, "error: short raw int64 read in '%s'\n", weight_freq_path.c_str());
                fclose(ff);
                return 1;
            }
            fclose(ff);
            for (int64_t v = 0; v < V; ++v) {
                freq[v] = (double) std::max<int64_t>(0, counts[v]);
            }
        } else {
            fclose(ff);
            // Treat as text corpus and tokenize with vocab_only model
            llama_backend_init();
            llama_model_params mp = llama_model_default_params();
            mp.vocab_only = true;
            llama_model * vmodel = llama_model_load_from_file(model_path.c_str(), mp);
            if (!vmodel) {
                fprintf(stderr, "error: failed to load vocab from '%s'\n", model_path.c_str());
                return 1;
            }
            const llama_vocab * vocab = llama_model_get_vocab(vmodel);
            std::ifstream in(weight_freq_path, std::ios::binary);
            std::stringstream ss;
            ss << in.rdbuf();
            std::string text = ss.str();
            std::vector<llama_token> buf(text.size() + 4);
            int32_t n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), buf.data(), (int32_t) buf.size(), false, false);
            if (n < 0) {
                buf.resize(-n);
                n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), buf.data(), (int32_t) buf.size(), false, false);
            }
            for (int32_t i = 0; i < n; ++i) {
                if (buf[i] >= 0 && buf[i] < V) {
                    freq[buf[i]] += 1.0;
                }
            }
            llama_model_free(vmodel);
        }

        double sum_freq = 0.0;
        int64_t n_nonzero = 0;
        for (int64_t v = 0; v < V; ++v) {
            sum_freq += freq[v];
            if (freq[v] > 0.0) {
                n_nonzero++;
            }
        }
        double mean_freq = sum_freq / (double) V;
        if (mean_freq <= 0.0) {
            fprintf(stderr, "error: mean(freq) <= 0 in '%s'\n", weight_freq_path.c_str());
            return 1;
        }
        printf("lowrank: applying frequency weighting from '%s' (nonzero=%lld/%lld, mean=%.6f, smooth=10.0)...\n",
                weight_freq_path.c_str(), (long long) n_nonzero, (long long) V, mean_freq);
        fflush(stdout);

        sqrt_freq.resize((size_t) V);
        for (int64_t v = 0; v < V; ++v) {
            float sf = (float) std::sqrt(freq[v] / mean_freq + 10.0);
            sqrt_freq[v] = sf;
            float * w_row = W.data() + (size_t) v * H;
            for (int64_t d = 0; d < H; ++d) {
                w_row[d] *= sf;
            }
        }
    }

    // Load model once if needed for --weight-act or --eval
    llama_model * model = nullptr;
    if (!weight_act_path.empty() || (!eval_path.empty() && eval_n > 0)) {
        llama_backend_init();
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 99;
        mparams.load_mtp = true;
        model = llama_model_load_from_file(model_path.c_str(), mparams);
        if (!model) {
            fprintf(stderr, "error: failed to load model '%s'\n", model_path.c_str());
            return 1;
        }
    }

    // Optional activation weighting (--weight-act): C = X^T X / n + eps*I, Cholesky C = L L^T
    std::vector<double> L_act;
    if (!weight_act_path.empty()) {
        lowrank_states cal;
        if (!lowrank_states_collect(model, weight_act_path, eval_n, H, cal)) {
            lowrank_states_free(cal);
            return 1;
        }
        const float * h_mtp = cal.h_mtp;

        std::vector<double> C((size_t) H * H, 0.0);
        const double inv_n = 1.0 / (double) eval_n;
        for (int t = 0; t < eval_n; ++t) {
            const float * x = h_mtp + (size_t) t * H;
            for (int64_t i = 0; i < H; ++i) {
                const double xi = (double) x[i] * inv_n;
                double * c_row = C.data() + (size_t) i * H;
                for (int64_t j = 0; j <= i; ++j) {
                    c_row[j] += xi * (double) x[j];
                }
            }
        }
        lowrank_states_free(cal);

        double trace_C = 0.0;
        for (int64_t i = 0; i < H; ++i) {
            trace_C += C[(size_t) i * H + i];
        }
        const double eps = 1e-4 * (trace_C / (double) H);
        for (int64_t i = 0; i < H; ++i) {
            C[(size_t) i * H + i] += eps;
        }
        printf("lowrank: calib '%s' (%d MTP states): mean_diag=%.6f, eps=%.6e, computing Cholesky...\n",
                weight_act_path.c_str(), eval_n, trace_C / (double) H, eps);
        fflush(stdout);

        L_act.assign((size_t) H * H, 0.0);
        for (int64_t i = 0; i < H; ++i) {
            const double * c_row = C.data() + (size_t) i * H;
            double * l_row_i = L_act.data() + (size_t) i * H;
            for (int64_t j = 0; j <= i; ++j) {
                const double * l_row_j = L_act.data() + (size_t) j * H;
                double sum = c_row[j];
                for (int64_t k = 0; k < j; ++k) {
                    sum -= l_row_i[k] * l_row_j[k];
                }
                if (i == j) {
                    if (sum <= 0.0) {
                        fprintf(stderr, "error: Cholesky failed at i=%lld (sum=%e)\n", (long long) i, sum);
                        return 1;
                    }
                    l_row_i[j] = std::sqrt(sum);
                } else {
                    l_row_i[j] = sum / l_row_j[j];
                }
            }
        }
    }

    std::vector<float> down_f32, scores_f32;
    double energy_share = 0.0;
    compute_randomized_svd(W, H, V, rank, n_iter, down_f32, scores_f32, energy_share, L_act.empty() ? nullptr : &L_act);
    W.clear();
    W.shrink_to_fit();
    L_act.clear();
    L_act.shrink_to_fit();

    if (!sqrt_freq.empty()) {
        for (int64_t v = 0; v < V; ++v) {
            float inv_sf = 1.0f / sqrt_freq[v];
            float * s_row = scores_f32.data() + (size_t) v * rank;
            for (int k = 0; k < rank; ++k) {
                s_row[k] *= inv_sf;
            }
        }
    }

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
        lowrank_states ev;
        const bool ok = lowrank_states_collect(model, eval_path, eval_n, H, ev);
        if (ok) {
            lowrank_print_recall(ev, false, eval_n, H, V, rank, down_f32, scores_f32, eval_ks, eval_path);
            lowrank_print_recall(ev, true,  eval_n, H, V, rank, down_f32, scores_f32, eval_ks, eval_path);
        }
        lowrank_states_free(ev);
        if (!ok) {
            return 1;
        }
    }

    if (model) {
        llama_model_free(model);
    }

    return 0;
}
