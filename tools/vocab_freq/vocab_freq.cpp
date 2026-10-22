// Build a DV32 draft-vocab file: top-N frequent token ids from text files.
// usage: vocab_freq -m model.gguf -o draft_vocab_32k.bin [-n 32768] file [file ...]
#include "llama.h"
#include "spec-draft-vocab.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void usage(const char * prog) {
    printf("usage: %s -m model.gguf -o out.bin [-n 32768] [--counts] file [file ...]\n", prog);
}

int main(int argc, char ** argv) {
    std::string model_path, out_path;
    int32_t top_n = 32768;
    bool write_counts = false;
    std::vector<std::string> files;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "-o" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "-n" && i + 1 < argc) {
            top_n = atoi(argv[++i]);
        } else if (a == "-c" || a == "--counts") {
            write_counts = true;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            files.push_back(a);
        }
    }
    if (model_path.empty() || out_path.empty() || files.empty() || top_n <= 0) {
        usage(argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        printf("error: cannot load model '%s'\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<int64_t> freq(n_vocab, 0);
    std::vector<llama_token> buf;
    for (const auto & path : files) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            printf("error: cannot open '%s'\n", path.c_str());
            return 1;
        }
        std::stringstream ss;
        ss << in.rdbuf();
        const std::string text = ss.str();
        buf.resize(text.size() + 4);
        int32_t n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), buf.data(), (int32_t) buf.size(), false, false);
        if (n < 0) {
            buf.resize(-n);
            n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), buf.data(), (int32_t) buf.size(), false, false);
        }
        if (n < 0) {
            printf("error: tokenize failed for '%s'\n", path.c_str());
            return 1;
        }
        for (int32_t i = 0; i < n; i++) {
            if (buf[i] >= 0 && buf[i] < n_vocab) {
                freq[buf[i]]++;
            }
        }
    }

    // tokenizer hash over pieces in id order (shared with the MTP ctor check)
    const uint64_t hash = spec_draft_vocab_tokenizer_hash(vocab);

    // Counts mode (DVCT v1): when -c / --counts is passed, write all n_vocab uint64 counts
    // after the 24-byte header (magic "DVCT" = 0x54435644u).
    if (write_counts) {
        FILE * fc = fopen(out_path.c_str(), "wb");
        if (!fc) {
            printf("error: cannot open '%s' for writing\n", out_path.c_str());
            return 1;
        }
        uint32_t magic = 0x54435644u, version = 1u, uvocab = (uint32_t) n_vocab, count = (uint32_t) n_vocab;
        fwrite(&magic, 4, 1, fc);
        fwrite(&version, 4, 1, fc);
        fwrite(&uvocab, 4, 1, fc);
        fwrite(&count, 4, 1, fc);
        fwrite(&hash, 8, 1, fc);
        fwrite(freq.data(), 8, freq.size(), fc);
        fclose(fc);
        printf("wrote %d token counts (n_vocab %d) to '%s'\n", n_vocab, n_vocab, out_path.c_str());
        llama_model_free(model);
        return 0;
    }

    std::vector<int32_t> order(n_vocab);
    for (int32_t i = 0; i < n_vocab; i++) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return freq[a] == freq[b] ? a < b : freq[a] > freq[b];
    });
    if (top_n > n_vocab) {
        top_n = n_vocab;
    }
    std::vector<int32_t> ids(order.begin(), order.begin() + top_n);
    // file order: rank order (most frequent first)
    FILE * f = fopen(out_path.c_str(), "wb");
    if (!f) {
        printf("error: cannot open '%s' for writing\n", out_path.c_str());
        return 1;
    }
    uint32_t magic = 0x32335644u, version = 1u, uvocab = (uint32_t) n_vocab, count = (uint32_t) ids.size();
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&uvocab, 4, 1, f);
    fwrite(&count, 4, 1, f);
    fwrite(&hash, 8, 1, f);
    fwrite(ids.data(), 4, ids.size(), f);
    fclose(f);

    // self-check through the loader
    try {
        std::vector<int32_t> back = spec_draft_vocab_load(out_path, n_vocab, hash);
        if (back != ids) {
            printf("error: loader roundtrip mismatch\n");
            return 1;
        }
    } catch (const std::exception & e) {
        printf("error: loader roundtrip failed: %s\n", e.what());
        return 1;
    }
    printf("wrote %d ids (n_vocab %d) to '%s'\n", top_n, n_vocab, out_path.c_str());
    llama_model_free(model);
    return 0;
}
