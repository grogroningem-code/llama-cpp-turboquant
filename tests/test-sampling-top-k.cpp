// common_sampler_top_k_logits (the top-k fast path of common_sampler) against llama_sampler_top_k.
// Built into test-sampling, which calls test_top_k_logits().

#include "ggml.h"
#include "llama.h"
#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>

// llama_sampler_top_k on all logits, which common_sampler_top_k_logits must reproduce
static std::vector<llama_token_data> top_k_ref(const std::vector<float> & logits, int32_t k) {
    std::vector<llama_token_data> cur(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        cur[i] = { (llama_token) i, logits[i], 0.0f };
    }

    llama_token_data_array arr = { cur.data(), cur.size(), -1, false };

    llama_sampler * smpl = llama_sampler_init_top_k(k);
    llama_sampler_apply(smpl, &arr);
    llama_sampler_free(smpl);

    return std::vector<llama_token_data>(arr.data, arr.data + arr.size);
}

static bool same_token_data(const std::vector<llama_token_data> & a, const std::vector<llama_token_data> & b) {
    return a.size() == b.size() && (a.empty() || memcmp(a.data(), b.data(), a.size()*sizeof(llama_token_data)) == 0);
}

// whether the k + 1 largest logits are pairwise distinct: the only case the helper may answer
static bool top_k_unique(const std::vector<float> & logits, int32_t k) {
    std::vector<float> v(logits);
    std::partial_sort(v.begin(), v.begin() + k + 1, v.end(), std::greater<float>());
    for (int32_t i = 0; i < k; ++i) {
        if (!(v[i] > v[i + 1])) {
            return false;
        }
    }
    return true;
}

static void test_top_k_logits(const std::vector<float> & logits, int32_t k, bool expect_ok) {
    // shared across calls, so every call starts from a dirty vector, as set_logits does with its cur
    static std::vector<llama_token_data> out;
    const bool ok = common_sampler_top_k_logits(logits.data(), (int32_t) logits.size(), k, out);
    if (ok != expect_ok) {
        fprintf(stderr, "top-k logits n=%zu k=%d: got %d, want %d\n", logits.size(), k, ok, expect_ok);
        GGML_ASSERT(false);
    }
    if (ok) {
        GGML_ASSERT(same_token_data(out, top_k_ref(logits, k)));
    }
}

void test_top_k_logits() {
    std::mt19937 rng(42);

    const std::vector<int32_t> ks = { 1, 2, 20, 40, 127, 128 };

    for (int32_t n : { 200, 4096 }) {
        std::normal_distribution<float> nd(0.0f, 3.0f);
        std::vector<float> logits(n);

        for (int rep = 0; rep < 4; ++rep) {
            for (auto & l : logits) {
                l = nd(rng);
            }
            for (int32_t k : ks) {
                if (k < n) {
                    test_top_k_logits(logits, k, top_k_unique(logits, k));
                }
            }
        }

        // ascending ids: every logit enters the list
        for (int32_t i = 0; i < n; ++i) {
            logits[i] = 1e-3f*i;
        }
        test_top_k_logits(logits, 20, true);

        // descending ids: no logit after the first k + 1 enters
        for (int32_t i = 0; i < n; ++i) {
            logits[i] = -1e-3f*i;
        }
        test_top_k_logits(logits, 20, true);

        std::fill(logits.begin(), logits.end(), 0.5f);
        test_top_k_logits(logits, 20, false);

        // all -inf but k + 1 of them
        std::fill(logits.begin(), logits.end(), -INFINITY);
        for (int32_t i = 0; i <= 20; ++i) {
            logits[(i*7919) % n] = (float) i;
        }
        test_top_k_logits(logits, 20, true);

        // all -inf but k: the k + 1-th largest is -inf, still distinct from the k-th
        logits[0] = -INFINITY;
        test_top_k_logits(logits, 20, true);

        // all -inf but k - 1: -inf at the k-th and the k + 1-th
        logits[7919 % n] = -INFINITY;
        test_top_k_logits(logits, 20, false);
    }

    // ties at the k-th and inside the top k fall back, ties below it do not matter
    {
        const int32_t n = 4096;
        const int32_t k = 20;
        std::vector<float> logits(n);
        for (int32_t i = 0; i < n; ++i) {
            logits[i] = (float) ((i*35761u) % 100000u); // distinct, scrambled
        }
        test_top_k_logits(logits, k, true);

        std::vector<float> v(logits);
        std::sort(v.begin(), v.end(), std::greater<float>());

        const auto with_value_at = [&](float value, int32_t rank) {
            std::vector<float> l(logits);
            *std::find(l.begin(), l.end(), v[rank]) = value;
            return l;
        };

        test_top_k_logits(with_value_at(v[k - 1], k),     k, false);
        test_top_k_logits(with_value_at(v[3],     4),     k, false);
        test_top_k_logits(with_value_at(v[k],     k + 1), k, true);

        // +0 and -0 compare equal
        std::vector<float> l(n);
        for (int32_t i = 0; i < n; ++i) {
            l[i] = -1.0f - (float) i;
        }
        l[5] = 0.0f;
        test_top_k_logits(l, k, true);
        l[9] = -0.0f;
        test_top_k_logits(l, k, false);
    }

    // a logit that ties the k-th arrives last, before and after a new maximum moved the threshold
    {
        const int32_t n = 4096;
        std::vector<float> logits(n);
        for (int32_t k : ks) {
            for (int32_t i = 0; i < n; ++i) {
                logits[i] = -(float) i;
            }
            logits[n - 1] = logits[k - 1];
            test_top_k_logits(logits, k, false);

            for (int32_t i = 0; i < n; ++i) {
                logits[i] = -(float) i;
            }
            logits[k + 1] = 1.0f;
            logits[n - 1] = k >= 2 ? logits[k - 2] : 1.0f;
            test_top_k_logits(logits, k, false);
        }
    }

    // NaN in the first k + 1 logits or after them, in blocks with no other logit above the threshold
    {
        const int32_t n = 4096;
        std::vector<float> logits(n);
        for (int32_t i = 0; i < n; ++i) {
            logits[i] = -(float) i;
        }
        for (int32_t at : { 0, 5, 20, 21, 1000, n - 1 }) {
            std::vector<float> l(logits);
            l[at] = std::numeric_limits<float>::quiet_NaN();
            test_top_k_logits(l, 20, false);
        }
    }

    // arguments the helper does not take
    {
        const std::vector<float> logits = { 1.0f, 2.0f, 3.0f };
        test_top_k_logits(logits, 0, false);
        test_top_k_logits(logits, 3, false);
        test_top_k_logits(logits, 4, false);

        std::vector<float> big(1000);
        for (int32_t i = 0; i < 1000; ++i) {
            big[i] = (float) i;
        }
        test_top_k_logits(big, 128, true);
        test_top_k_logits(big, 129, false);
    }

    // k = -1, +inf at the top and a tie of two +inf, NaN in a tail block shorter than 64
    {
        std::vector<float> l(200);
        for (int32_t i = 0; i < 200; ++i) {
            l[i] = -(float) i;
        }
        test_top_k_logits(l, -1, false);
        l[150] = INFINITY;
        test_top_k_logits(l, 20, true);
        l[170] = INFINITY;
        test_top_k_logits(l, 20, false);
        l[170] = -170.0f;
        l[199] = std::numeric_limits<float>::quiet_NaN();
        test_top_k_logits(l, 20, false);
    }
}
