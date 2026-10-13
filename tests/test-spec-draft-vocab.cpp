#include "spec-draft-vocab.h"
#include "arg.h"
#include "common.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#undef NDEBUG
#include <cassert>

static const int32_t N_VOCAB = 64000;
static const uint64_t TOK_HASH = 0x123456789abcdef0ull;

// write one DV32 file, return path
static std::string write_file(const std::vector<int32_t> & ids, uint32_t n_vocab, uint64_t hash, uint32_t magic = 0x32335644u, uint32_t version = 1u) {
    static int n = 0;
    std::string path = "test-spec-draft-vocab-" + std::to_string(n++) + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    assert(f);
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&n_vocab, 4, 1, f);
    uint32_t count = (uint32_t) ids.size();
    fwrite(&count, 4, 1, f);
    fwrite(&hash, 8, 1, f);
    if (!ids.empty()) {
        fwrite(ids.data(), 4, ids.size(), f);
    }
    fclose(f);
    return path;
}

static void must_fail(const std::string & path, int32_t n_vocab, uint64_t hash) {
    bool threw = false;
    try {
        spec_draft_vocab_load(path, n_vocab, hash);
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw);
    remove(path.c_str());
}

static void test_ok(void) {
    std::vector<int32_t> ids;
    for (int32_t i = 0; i < 100; i++) {
        ids.push_back(i * 7);
    }
    std::string path = write_file(ids, N_VOCAB, TOK_HASH);
    std::vector<int32_t> got = spec_draft_vocab_load(path, N_VOCAB, TOK_HASH);
    assert(got == ids);
    remove(path.c_str());
}

static void test_bad_header(void) {
    std::vector<int32_t> ids = {1, 2, 3};
    must_fail(write_file(ids, N_VOCAB, TOK_HASH, 0xdeadbeefu), N_VOCAB, TOK_HASH); // bad magic
    must_fail(write_file(ids, N_VOCAB, TOK_HASH, 0x32335644u, 99u), N_VOCAB, TOK_HASH); // bad version
    must_fail(write_file(ids, 32000, TOK_HASH), N_VOCAB, TOK_HASH); // n_vocab mismatch
    must_fail(write_file(ids, N_VOCAB, 0x1111111111111111ull), N_VOCAB, TOK_HASH); // hash mismatch
}

static void test_bad_ids(void) {
    must_fail(write_file({1, -5, 3}, N_VOCAB, TOK_HASH), N_VOCAB, TOK_HASH); // negative id
    must_fail(write_file({1, N_VOCAB, 3}, N_VOCAB, TOK_HASH), N_VOCAB, TOK_HASH); // id past end
    must_fail(write_file({4, 4, 5}, N_VOCAB, TOK_HASH), N_VOCAB, TOK_HASH); // duplicate ids
}

static void test_bad_size(void) {
    // count says 3 but file holds 2
    {
        std::string path = "test-spec-draft-vocab-trunc.bin";
        FILE * f = fopen(path.c_str(), "wb");
        assert(f);
        uint32_t magic = 0x32335644u, version = 1u, n_vocab = N_VOCAB, count = 3;
        fwrite(&magic, 4, 1, f);
        fwrite(&version, 4, 1, f);
        fwrite(&n_vocab, 4, 1, f);
        fwrite(&count, 4, 1, f);
        fwrite(&TOK_HASH, 8, 1, f);
        int32_t two[2] = {1, 2};
        fwrite(two, 4, 2, f);
        fclose(f);
        must_fail(path, N_VOCAB, TOK_HASH);
    }
    // trailing bytes after ids
    {
        std::string path = write_file({1, 2}, N_VOCAB, TOK_HASH);
        FILE * f = fopen(path.c_str(), "ab");
        assert(f);
        uint32_t extra = 0;
        fwrite(&extra, 4, 1, f);
        fclose(f);
        must_fail(path, N_VOCAB, TOK_HASH);
    }
    // missing file
    must_fail("test-spec-draft-vocab-nope.bin", N_VOCAB, TOK_HASH);
}

static void test_bad_counts(void) {
    must_fail(write_file({}, N_VOCAB, TOK_HASH), N_VOCAB, TOK_HASH); // empty list
    // huge count, tiny file: must fail on the header check, not in alloc
    {
        std::string path = "test-spec-draft-vocab-huge.bin";
        FILE * f = fopen(path.c_str(), "wb");
        assert(f);
        uint32_t magic = 0x32335644u, version = 1u, n_vocab = N_VOCAB, count = 0xffffffffu;
        fwrite(&magic, 4, 1, f);
        fwrite(&version, 4, 1, f);
        fwrite(&n_vocab, 4, 1, f);
        fwrite(&count, 4, 1, f);
        fwrite(&TOK_HASH, 8, 1, f);
        fclose(f);
        must_fail(path, N_VOCAB, TOK_HASH);
    }
    // count past n_vocab
    {
        std::vector<int32_t> ids;
        for (int32_t i = 0; i < 10; i++) {
            ids.push_back(i);
        }
        std::string path = "test-spec-draft-vocab-bigcount.bin";
        FILE * f = fopen(path.c_str(), "wb");
        assert(f);
        uint32_t magic = 0x32335644u, version = 1u, n_vocab = 8u, count = 10u;
        fwrite(&magic, 4, 1, f);
        fwrite(&version, 4, 1, f);
        fwrite(&n_vocab, 4, 1, f);
        fwrite(&count, 4, 1, f);
        fwrite(&TOK_HASH, 8, 1, f);
        fwrite(ids.data(), 4, ids.size(), f);
        fclose(f);
        must_fail(path, 8, TOK_HASH);
    }
    // file over the 1MB cap
    {
        std::string path = write_file({1, 2}, N_VOCAB, TOK_HASH);
        FILE * f = fopen(path.c_str(), "ab");
        assert(f);
        std::vector<char> pad(2 * 1024 * 1024, 0);
        fwrite(pad.data(), 1, pad.size(), f);
        fclose(f);
        must_fail(path, N_VOCAB, TOK_HASH);
    }
}

static void test_flags(void) {
    auto to_argv = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    // defaults: off
    {
        common_params params;
        assert(params.speculative.draft.draft_vocab_path.empty());
        assert(params.speculative.draft.draft_vocab_extra == 0);
    }

    // set path
    {
        common_params params;
        std::vector<std::string> argv = {"binary_name", "-m", "model.gguf", "--spec-draft-vocab", "draft.bin"};
        assert(true == common_params_parse(argv.size(), to_argv(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
        assert(params.speculative.draft.draft_vocab_path == "draft.bin");
    }

    // explicit zero extra is accepted
    {
        common_params params;
        std::vector<std::string> argv = {"binary_name", "-m", "model.gguf", "--spec-draft-extra", "0"};
        assert(true == common_params_parse(argv.size(), to_argv(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
        assert(params.speculative.draft.draft_vocab_extra == 0);
    }

    // nonzero extra is rejected in v1
    {
        common_params params;
        std::vector<std::string> argv = {"binary_name", "-m", "model.gguf", "--spec-draft-extra", "1"};
        assert(false == common_params_parse(argv.size(), to_argv(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    }
}

static void test_handle(void) {
    // fixed set {10, 20, 30}, 2 parallel slots, capacity 2 per slot
    spec_draft_vocab_handle h({10, 20, 30}, 2, 2);

    // accepted-only: non-member never enters
    assert(h.accept(0, 0, 10));
    assert(!h.accept(0, 1, 99));
    assert(h.entries(0).size() == 1);

    // entries carry positions; overflow FIFO-evicts the oldest
    assert(h.accept(0, 1, 20));
    assert(h.accept(0, 2, 30));
    assert(h.entries(0).size() == 2);
    assert(h.entries(0)[0].first == 1);
    assert(h.entries(0)[0].second == 20);

    // rewind hook drops pos >= from_pos ...
    assert(h.drop_deferred_from(0, 2) == 1);
    // ... but the evicted entry is NOT restored
    assert(h.drop_deferred_from(0, 0) == 1);
    assert(h.entries(0).empty());

    // parallel slots: no cross-slot leak
    assert(h.accept(1, 0, 30));
    assert(h.entries(1).size() == 1);
    assert(h.entries(0).empty());
    assert(h.drop_deferred_from(0, 0) == 0);
    assert(h.entries(1).size() == 1);

    // bad slot index throws
    bool threw = false;
    try {
        h.accept(7, 0, 10);
    } catch (const std::out_of_range &) {
        threw = true;
    }
    assert(threw);
}

int main(void) {
    try {
        test_ok();
        test_bad_header();
        test_bad_ids();
        test_bad_size();
        test_bad_counts();
        test_flags();
        test_handle();
    } catch (const std::exception & e) {
        fprintf(stderr, "test-spec-draft-vocab: exception: %s\n", e.what());
        return 1;
    }
    printf("test-spec-draft-vocab: all tests OK\n");
    return 0;
}
