#include "reasoning-budget.h"
#include "unicode.h"
#include "arg.h"

#include "llama.h"
#include "ggml.h"

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

// Reasoning budget sampler test helper
// These tests use nullptr vocab which safely falls back to treating all tokens as complete
// (The UTF-8 boundary detection logic is tested separately in test_utf8_boundary_detection)
static void test_reasoning_budget(
    const char * test_name,
    const std::vector<llama_token> & sequence,
    const std::vector<llama_tokens> & start_seqs,
    const std::vector<llama_tokens> & end_seqs,
    const std::vector<llama_token> & forced_tokens,
    int32_t budget,
    common_reasoning_budget_state initial_state,
    size_t expected_force_start,   // token index where forcing should start (SIZE_MAX = never)
    size_t expected_force_end      // token index where forcing should end (after this, no more forcing)
) {
    // Find the maximum token ID to ensure our vocab covers all tokens
    llama_token max_token = 0;
    for (auto t : sequence) max_token = std::max(max_token, t);
    for (const auto & seq : start_seqs) {
        for (auto t : seq) max_token = std::max(max_token, t);
    }
    for (const auto & seq : end_seqs) {
        for (auto t : seq) max_token = std::max(max_token, t);
    }
    for (auto t : forced_tokens) max_token = std::max(max_token, t);

    // Create a minimal sampler with mock vocabulary
    // For this test, we use nullptr as vocab since we're testing state transitions
    // The UTF-8 boundary check will treat all tokens as complete (safe fallback)
    auto * sampler = common_reasoning_budget_init(
        nullptr,  // vocab - not used for basic state machine tests
        start_seqs,
        end_seqs,
        forced_tokens,
        budget,
        initial_state
    );

    // Create a test token data array for checking forcing behavior
    // Vocab size must be large enough to include all tokens (start, end, forced, sequence)
    std::vector<llama_token_data> cur;
    const size_t n_vocab = (size_t)max_token + 1;
    for (size_t i = 0; i < n_vocab; i++) {
        cur.emplace_back(llama_token_data{(llama_token)i, logf((float)(i+1)), 0.0f});
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };

    size_t actual_force_start = SIZE_MAX;
    size_t actual_force_end = SIZE_MAX;

    // Feed the sequence and track when forcing occurs
    for (size_t i = 0; i < sequence.size(); i++) {
        // Check if we're in forcing state by applying and seeing if logits are modified
        cur_p.selected = -1;
        for (size_t j = 0; j < cur.size(); j++) {
            cur[j].logit = logf((float)(j+1));  // reset logits
        }

        llama_sampler_apply(sampler, &cur_p);

        // Check if forcing is active (all logits except one should be -INFINITY)
        size_t finite_count = 0;
        llama_token finite_token = -1;
        for (size_t j = 0; j < cur.size(); j++) {
            if (std::isfinite(cur[j].logit)) {
                finite_count++;
                finite_token = cur[j].id;
            }
        }

        llama_sampler_accept(sampler, sequence[i]);

        fprintf(stderr, "    i=%zu: token=%d, finite_count=%zu, finite_token=%d\n", i, (int)sequence[i], finite_count, (int)finite_token);

        if (finite_count == 1) {
            if (actual_force_start == SIZE_MAX) {
                actual_force_start = i;
            }
            actual_force_end = i;
        } else if (actual_force_start != SIZE_MAX && actual_force_end != SIZE_MAX) {
            // Forcing stopped
            break;
        }
    }

    llama_sampler_free(sampler);

    // Verify forcing occurred at expected positions
    if (expected_force_start == SIZE_MAX) {
        if (actual_force_start != SIZE_MAX) {
            fprintf(stderr, "Test '%s' FAILED: Expected no forcing, but forcing occurred at %zu\n", test_name, actual_force_start);
            GGML_ASSERT(false && "Expected no forcing, but forcing occurred");
        }
    } else {
        if (actual_force_start == SIZE_MAX) {
            fprintf(stderr, "Test '%s' FAILED: Expected forcing but none occurred\n", test_name);
            GGML_ASSERT(false && "Expected forcing but none occurred");
        }
        if (actual_force_start != expected_force_start) {
            fprintf(stderr, "Test '%s' FAILED: Forcing started at %zu, expected %zu\n", test_name, actual_force_start, expected_force_start);
            GGML_ASSERT(false && "Forcing started at wrong position");
        }
    }

    if (expected_force_end != SIZE_MAX) {
        if (actual_force_end < expected_force_end) {
            fprintf(stderr, "Test '%s' FAILED: Forcing ended at %zu, expected >= %zu\n", test_name, actual_force_end, expected_force_end);
            GGML_ASSERT(false && "Forcing ended too early");
        }
    }

    fprintf(stderr, "  Test '%s' passed (force_start=%zu, force_end=%zu)\n", test_name, actual_force_start, actual_force_end);
    (void)sequence;
}

static llama_token get_forced_token(struct llama_sampler * sampler, llama_token max_token) {
    std::vector<llama_token_data> cur;
    const size_t n_vocab = (size_t) max_token + 1;
    for (size_t i = 0; i < n_vocab; i++) {
        cur.emplace_back(llama_token_data{(llama_token) i, logf((float) (i + 1)), 0.0f});
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(sampler, &cur_p);

    size_t finite_count = 0;
    llama_token finite_token = LLAMA_TOKEN_NULL;
    for (size_t i = 0; i < cur.size(); i++) {
        if (std::isfinite(cur[i].logit)) {
            finite_count++;
            finite_token = cur[i].id;
        }
    }

    GGML_ASSERT(finite_count == 1 && "sampler is not forcing exactly one token");
    return finite_token;
}

static void test_reasoning_budget_clone_mid_counting() {
    const std::vector<llama_token> start = {100};
    const std::vector<llama_token> end = {101};
    const std::vector<llama_token> forced = {102, 101};

    auto * sampler = common_reasoning_budget_init(nullptr, {start}, {end}, forced, 2, REASONING_BUDGET_IDLE);

    llama_sampler_accept(sampler, 100); // COUNTING, remaining=2
    llama_sampler_accept(sampler, 50);  // COUNTING, remaining=1

    auto * clone = llama_sampler_clone(sampler);
    llama_sampler_accept(clone, 51); // should exhaust the cloned remaining budget

    GGML_ASSERT(get_forced_token(clone, 102) == 102 && "cloned counting state lost remaining budget");

    llama_sampler_free(clone);
    llama_sampler_free(sampler);
}

static void test_reasoning_budget_clone_mid_forcing() {
    const std::vector<llama_token> start = {100};
    const std::vector<llama_token> end = {101};
    const std::vector<llama_token> forced = {102, 101};

    auto * sampler = common_reasoning_budget_init(nullptr, {start}, {end}, forced, 0, REASONING_BUDGET_FORCING);

    GGML_ASSERT(get_forced_token(sampler, 102) == 102);
    llama_sampler_accept(sampler, 102); // advance to the second forced token

    auto * clone = llama_sampler_clone(sampler);

    GGML_ASSERT(get_forced_token(clone, 102) == 101 && "cloned forcing state lost force position");

    llama_sampler_free(clone);
    llama_sampler_free(sampler);
}

static void test_reasoning_budget_force_manual() {
    const std::vector<llama_token> start  = {100};
    const std::vector<llama_token> end    = {101};
    const std::vector<llama_token> forced = {102, 101};

    // if COUNTING, force() succeeds and begins forcing the end sequence from the start
    {
        auto * sampler = common_reasoning_budget_init(nullptr, {start}, {end}, forced, 5, REASONING_BUDGET_IDLE);

        llama_sampler_accept(sampler, 100); // COUNTING, remaining=5
        llama_sampler_accept(sampler, 50);  // COUNTING, remaining=4
        GGML_ASSERT(common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_COUNTING);

        GGML_ASSERT(common_reasoning_budget_force(sampler) && "force() should succeed from COUNTING");
        GGML_ASSERT(common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_FORCING);

        // forces the configured sequence from force_pos=0, then transitions to DONE
        GGML_ASSERT(get_forced_token(sampler, 102) == 102);
        llama_sampler_accept(sampler, 102);
        GGML_ASSERT(get_forced_token(sampler, 102) == 101);
        llama_sampler_accept(sampler, 101);
        GGML_ASSERT(common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_DONE);

        llama_sampler_free(sampler);
    }

    // if IDLE, force() is a no-op
    {
        auto * sampler = common_reasoning_budget_init(nullptr, {start}, {end}, forced, 5, REASONING_BUDGET_IDLE);

        GGML_ASSERT(!common_reasoning_budget_force(sampler) && "force() must not transition from IDLE");
        GGML_ASSERT(common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_IDLE);

        llama_sampler_free(sampler);
    }

    // if DONE, force() is a no-op
    {
        auto * sampler = common_reasoning_budget_init(nullptr, {start}, {end}, forced, 5, REASONING_BUDGET_IDLE);

        llama_sampler_accept(sampler, 100); // COUNTING
        llama_sampler_accept(sampler, 101); // natural end -> DONE
        GGML_ASSERT(common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_DONE);

        GGML_ASSERT(!common_reasoning_budget_force(sampler) && "force() must not transition from DONE");
        GGML_ASSERT(common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_DONE);

        llama_sampler_free(sampler);
    }

    // if FORCING, force() is a no-op and must not rewind the force position
    {
        auto * sampler = common_reasoning_budget_init(nullptr, {start}, {end}, forced, 0, REASONING_BUDGET_FORCING);

        GGML_ASSERT(get_forced_token(sampler, 102) == 102);
        llama_sampler_accept(sampler, 102); // advance to the second forced token (force_pos=1)

        GGML_ASSERT(!common_reasoning_budget_force(sampler) && "force() must not transition from FORCING");
        GGML_ASSERT(common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_FORCING);
        GGML_ASSERT(get_forced_token(sampler, 102) == 101 && "force() must not rewind the force position");

        llama_sampler_free(sampler);
    }

    // a null sampler is safely ignored
    GGML_ASSERT(!common_reasoning_budget_force(nullptr));

    fprintf(stderr, "  Test 'manual force transition' passed\n");
}

static void test_reasoning_budget_end_match() {
    const std::vector<llama_tokens> start = {{100}};
    const std::vector<llama_tokens> end   = {{101}, {103, 104}};

    // natural end records the sequence that matched; re-arming clears it
    {
        auto * sampler = common_reasoning_budget_init(nullptr, start, end, {102, 101}, 5, REASONING_BUDGET_IDLE);

        GGML_ASSERT(common_reasoning_budget_get_end_match(sampler) == nullptr);

        llama_sampler_accept(sampler, 100); // COUNTING
        llama_sampler_accept(sampler, 50);
        llama_sampler_accept(sampler, 103);
        llama_sampler_accept(sampler, 104); // end matched via {103, 104}, DONE

        const llama_tokens * matched = common_reasoning_budget_get_end_match(sampler);
        GGML_ASSERT(matched != nullptr);
        GGML_ASSERT(*matched == llama_tokens({103, 104}));

        llama_sampler_accept(sampler, 100); // re-arm, COUNTING
        GGML_ASSERT(common_reasoning_budget_get_end_match(sampler) == nullptr);

        llama_sampler_free(sampler);
    }

    // overlapping end sequences: the longest one ending at the position wins
    {
        const std::vector<llama_tokens> end_overlap = {{104}, {103, 104}};

        auto * sampler = common_reasoning_budget_init(nullptr, start, end_overlap, {102, 104}, 5, REASONING_BUDGET_IDLE);

        llama_sampler_accept(sampler, 100); // COUNTING
        llama_sampler_accept(sampler, 103);
        llama_sampler_accept(sampler, 104); // both {104} and {103, 104} end here

        const llama_tokens * matched = common_reasoning_budget_get_end_match(sampler);
        GGML_ASSERT(matched != nullptr);
        GGML_ASSERT(*matched == llama_tokens({103, 104}));

        llama_sampler_free(sampler);
    }

    // forcing records the end sequence terminating forced_tokens
    {
        auto * sampler = common_reasoning_budget_init(nullptr, start, end, {102, 103, 104}, 0, REASONING_BUDGET_FORCING);

        llama_sampler_accept(sampler, 102);
        llama_sampler_accept(sampler, 103);
        GGML_ASSERT(common_reasoning_budget_get_end_match(sampler) == nullptr);
        llama_sampler_accept(sampler, 104); // forced sequence complete, DONE

        const llama_tokens * matched = common_reasoning_budget_get_end_match(sampler);
        GGML_ASSERT(matched != nullptr);
        GGML_ASSERT(*matched == llama_tokens({103, 104}));

        llama_sampler_free(sampler);
    }

    // forced_tokens not ending with a known end sequence records nothing
    {
        auto * sampler = common_reasoning_budget_init(nullptr, start, end, {102}, 0, REASONING_BUDGET_FORCING);

        llama_sampler_accept(sampler, 102); // forced sequence complete, DONE
        GGML_ASSERT(common_reasoning_budget_get_state(sampler) == REASONING_BUDGET_DONE);
        GGML_ASSERT(common_reasoning_budget_get_end_match(sampler) == nullptr);

        llama_sampler_free(sampler);
    }

    // a null sampler is safely ignored
    GGML_ASSERT(common_reasoning_budget_get_end_match(nullptr) == nullptr);

    fprintf(stderr, "  Test 'matched end sequence' passed\n");
}

// UTF-8 boundary detection unit test
// Tests common_utf8_is_complete() from reasoning-budget.h
static bool args_throw(std::initializer_list<const char*> args) {
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("prog"));
    for (auto a : args) argv.push_back(const_cast<char*>(a));
    common_params p;
    // the parser prints "error: ..." and returns false on invalid args
    // (it catches std::invalid_argument internally and restores defaults)
    return !common_params_parse((int)argv.size(), argv.data(), p, LLAMA_EXAMPLE_SERVER);
}

// Runs a token stream through a budget sampler, observing which tokens
// apply() forces during FORCING.

struct budget_run_result {
    common_reasoning_budget_state final_state;
    std::vector<llama_token> forced_seen;    // tokens forced by apply() during FORCING
};

static budget_run_result run_budget(
        const llama_tokens & tokens,
        const llama_tokens & start, const llama_tokens & end,
        const llama_tokens & budget_forced, int budget,
        int32_t end_nudge_budget,
        const llama_tokens & end_nudged) {
    budget_run_result r;

    struct llama_sampler * smpl = common_reasoning_budget_init(
        /* vocab = */ nullptr, {start}, {end}, budget_forced, budget,
        REASONING_BUDGET_IDLE, end_nudge_budget, end_nudged);
    if (!smpl) { throw std::runtime_error("sampler init failed"); }

    // candidate set for apply(): every token id in the stream plus one unused id
    std::vector<llama_token_data> cands;
    {
        std::vector<llama_token> ids;
        for (auto t : tokens) { if (std::find(ids.begin(), ids.end(), t) == ids.end()) ids.push_back(t); }
        ids.push_back(999);
        for (auto id : ids) { cands.push_back({ id, 0.0f, 0.0f }); }
    }

    for (auto t : tokens) {
        if (common_reasoning_budget_get_state(smpl) == REASONING_BUDGET_FORCING) {
            for (auto & c : cands) { c.logit = 0.0f; } // fresh logits each apply
            llama_token_data_array arr = { cands.data(), cands.size(), -1, false };
            llama_sampler_apply(smpl, &arr);
            for (size_t i = 0; i < arr.size; i++) {
                if (arr.data[i].logit > -INFINITY) { r.forced_seen.push_back(arr.data[i].id); break; }
            }
        }
        llama_sampler_accept(smpl, t);
    }

    r.final_state = common_reasoning_budget_get_state(smpl);
    llama_sampler_free(smpl);
    return r;
}

// reasoning end nudge: pre-exhaustion wrap-up message
static void test_reasoning_end_nudge() {
    const llama_tokens START = {1};
    const llama_tokens END = {2};
    const llama_tokens BFORCED = {7, 8, 2};       // budget-exhaustion close
    const llama_tokens NUDGED = {5, 6};           // end-nudge message tokens

    // E1: fires exactly once at the threshold; reasoning stays open; the
    // budget keeps draining and the exhaustion close still happens.
    // Stream layout (budget=10, threshold=3): t1-t7 drain to remaining=3
    // (nudge fires), t8-t9 accept the nudge bytes, t10-t12 drain to 0,
    // t13-t15 accept the exhaustion close.
    {
        llama_tokens tokens = START;
        for (int i = 0; i < 7; i++) { tokens.push_back(20 + i); }
        tokens.push_back(5); tokens.push_back(6);
        for (int i = 0; i < 3; i++) { tokens.push_back(29 + i); }
        tokens.push_back(7); tokens.push_back(8); tokens.push_back(2);
        tokens.push_back(36);
        auto r = run_budget(tokens, START, END, BFORCED, 10, 3, NUDGED);
        if (std::count(r.forced_seen.begin(), r.forced_seen.end(), 5) != 1 ||
            std::count(r.forced_seen.begin(), r.forced_seen.end(), 6) != 1) {
            throw std::runtime_error("end nudge did not fire exactly once at the threshold");
        }
        if (std::count(r.forced_seen.begin(), r.forced_seen.end(), 7) != 1 ||
            std::count(r.forced_seen.begin(), r.forced_seen.end(), 8) != 1 ||
            std::count(r.forced_seen.begin(), r.forced_seen.end(), 2) != 1) {
            throw std::runtime_error("budget exhaustion close did not happen after the end nudge");
        }
        if (r.final_state != REASONING_BUDGET_DONE) { throw std::runtime_error("block not done after end-nudge drain"); }
    }

    // E2: feature off (no message tokens) - nothing injected
    {
        llama_tokens tokens = START;
        for (int i = 0; i < 16; i++) { tokens.push_back(20 + i); }
        auto r = run_budget(tokens, START, END, BFORCED, 10, 3, {});
        if (std::count(r.forced_seen.begin(), r.forced_seen.end(), 5) != 0) {
            throw std::runtime_error("end nudge fired with empty nudge tokens");
        }
    }

    // E3: threshold >= budget fires on the first counted token
    // (budget=5, threshold=5): t1 fires, t2-t3 nudge bytes, t4-t7 drain,
    // t8-t10 exhaustion close
    {
        llama_tokens tokens = START;
        tokens.push_back(20);
        tokens.push_back(5); tokens.push_back(6);
        for (int i = 0; i < 4; i++) { tokens.push_back(21 + i); }
        tokens.push_back(7); tokens.push_back(8); tokens.push_back(2);
        auto r = run_budget(tokens, START, END, BFORCED, 5, 5, NUDGED);
        if (std::count(r.forced_seen.begin(), r.forced_seen.end(), 5) != 1) {
            throw std::runtime_error("end nudge did not fire immediately at threshold >= budget");
        }
        if (std::count(r.forced_seen.begin(), r.forced_seen.end(), 7) != 1) {
            throw std::runtime_error("exhaustion close missing in E3");
        }
    }

    // E4: budget exhaustion wins when they coincide (budget=1, threshold=1):
    // t1 exhausts, t2-t4 accept the close; no nudge bytes anywhere
    {
        llama_tokens tokens = START;
        tokens.push_back(20);
        tokens.push_back(7); tokens.push_back(8); tokens.push_back(2);
        tokens.push_back(21);
        auto r = run_budget(tokens, START, END, BFORCED, 1, 1, NUDGED);
        if (std::count(r.forced_seen.begin(), r.forced_seen.end(), 5) != 0 ||
            std::count(r.forced_seen.begin(), r.forced_seen.end(), 6) != 0) {
            throw std::runtime_error("end nudge fired on the exhaustion token");
        }
        if (std::count(r.forced_seen.begin(), r.forced_seen.end(), 7) != 1) {
            throw std::runtime_error("exhaustion close missing in E4");
        }
    }

    // E5: a re-armed block fires the nudge again (fired flag resets per
    // block). budget=6, threshold=2: each block drains t1-t4 to remaining=2
    // (nudge), t5-t6 nudge bytes, t7-t8 drain to 0, t9-t11 close, t12 re-arm.
    {
        llama_tokens tokens = START;
        for (int b = 0; b < 2; b++) {
            for (int i = 0; i < 4; i++) { tokens.push_back(20 + i); }
            tokens.push_back(5); tokens.push_back(6);
            tokens.push_back(24); tokens.push_back(25);
            tokens.push_back(7); tokens.push_back(8); tokens.push_back(2);
            if (b == 0) { tokens.push_back(1); } // re-arm between the blocks
        }
        auto r = run_budget(tokens, START, END, BFORCED, 6, 2, NUDGED);
        if (std::count(r.forced_seen.begin(), r.forced_seen.end(), 5) != 2 ||
            std::count(r.forced_seen.begin(), r.forced_seen.end(), 6) != 2) {
            throw std::runtime_error("end nudge did not re-arm per block");
        }
        if (std::count(r.forced_seen.begin(), r.forced_seen.end(), 7) != 2) {
            throw std::runtime_error("budget closes missing in E5");
        }
    }

    // E7: natural end before the threshold - no nudge
    {
        llama_tokens tokens = START;
        for (int i = 0; i < 4; i++) { tokens.push_back(20 + i); }
        tokens.push_back(2);
        auto r = run_budget(tokens, START, END, BFORCED, 100, 3, NUDGED);
        if (std::count(r.forced_seen.begin(), r.forced_seen.end(), 5) != 0) {
            throw std::runtime_error("end nudge fired on a naturally ended block");
        }
        if (r.final_state != REASONING_BUDGET_DONE) { throw std::runtime_error("natural end not DONE"); }
    }

    // E9: half-configured state - message set but the budget flag missing
    // (end_nudge_budget <= 0): the init disables the nudge (per-slot-once
    // warning) and nothing is ever injected.
    {
        llama_tokens tokens = START;
        for (int i = 0; i < 10; i++) { tokens.push_back(20 + i); }
        tokens.push_back(7); tokens.push_back(8); tokens.push_back(2);
        auto r = run_budget(tokens, START, END, BFORCED, 10, -1, NUDGED);
        if (std::count(r.forced_seen.begin(), r.forced_seen.end(), 5) != 0 ||
            std::count(r.forced_seen.begin(), r.forced_seen.end(), 6) != 0) {
            throw std::runtime_error("end nudge fired in the half-configured (message-only) state");
        }
        if (r.final_state != REASONING_BUDGET_DONE) { throw std::runtime_error("E9 exhaustion close not done"); }
    }
}

// review fix: a manual reasoning_end arriving during a transient nudge must be
// queued (return true) and chained right after the nudge text, not silently
// dropped while the sampler resumes counting
static void test_force_queued_during_nudge() {
    const llama_tokens START = {1};
    const llama_tokens END = {2};
    const llama_tokens BFORCED = {3, 2};   // budget-exhaustion close
    const llama_tokens NUDGED = {5, 6};    // end-nudge message tokens

    // stream: start, 7 counted tokens (drains budget 10 -> 3, end nudge fires
    // at the threshold), the two nudge bytes, then the close sequence.
    // The manual force is injected right after the first nudge byte.
    std::vector<llama_token> tokens = START;
    for (int i = 0; i < 7; i++) { tokens.push_back(10 + i); }
    tokens.push_back(5); tokens.push_back(6); // nudge bytes
    tokens.push_back(3); tokens.push_back(2); // chained close

    struct llama_sampler * smpl = common_reasoning_budget_init(
        /* vocab = */ nullptr, {START}, {END}, BFORCED, 10,
        REASONING_BUDGET_IDLE, /* end_nudge_budget */ 3, NUDGED);
    if (!smpl) { throw std::runtime_error("sampler init failed"); }

    bool force_queued = false;
    bool chained_forcing_seen = false;
    std::vector<llama_token> forced_seen;
    for (auto t : tokens) {
        // serve forced tokens while FORCING
        if (common_reasoning_budget_get_state(smpl) == REASONING_BUDGET_FORCING) {
            std::vector<llama_token_data> cands;
            for (int id = 1; id <= 30; id++) { cands.push_back({ (llama_token) id, 0.0f, 0.0f }); }
            llama_token_data_array arr = { cands.data(), cands.size(), -1, false };
            llama_sampler_apply(smpl, &arr);
            for (size_t i = 0; i < arr.size; i++) {
                if (arr.data[i].logit > -INFINITY) { forced_seen.push_back(arr.data[i].id); break; }
            }
        }

        llama_sampler_accept(smpl, t);

        // right after the first nudge byte is accepted we are mid transient
        // nudge: the manual force must be queued, not refused
        if (t == 5) {
            force_queued = common_reasoning_budget_force(smpl);
        }

        // after the nudge completes, the queued close must chain: still FORCING
        if (t == 6 && common_reasoning_budget_get_state(smpl) == REASONING_BUDGET_FORCING) {
            chained_forcing_seen = true;
        }
    }

    const auto st = common_reasoning_budget_get_state(smpl);
    llama_sampler_free(smpl);

    if (!force_queued) { throw std::runtime_error("manual force during nudge was refused instead of queued"); }
    if (!chained_forcing_seen) { throw std::runtime_error("queued close did not chain after the nudge"); }
    if (st != REASONING_BUDGET_DONE) { throw std::runtime_error("chained close did not complete"); }
    // exact stream: nudge bytes then close bytes, nothing extra
    const std::vector<llama_token> expect = {5, 6, 3, 2};
    if (forced_seen != expect) {
        throw std::runtime_error("chained close forced the wrong sequence");
    }
}

// reasoning revisit nudge: text tracker (counts sentence-start take-backs)
static int revisit_count_all(const std::vector<std::string> & pieces) {
    common_reasoning_revisit_tracker t(common_reasoning_revisit_default_retracts());
    int n = 0;
    for (const auto & p : pieces) { n += t.feed(p); }
    return n;
}

static void test_revisit_tracker() {
    // each sentence-start take-back counts once
    if (revisit_count_all({"Actually, no. Wait. Hmm, ok. done"}) != 3) { throw std::runtime_error("three take-backs not counted"); }
    // a longer phrase wins and counts once: "but wait" is not also "wait", "hmmm" is not "hmm"
    if (revisit_count_all({"Fine. But wait, x. Hmmm. ok"}) != 2) { throw std::runtime_error("overlapping phrases miscounted"); }
    // word boundaries: "waiting" is not "wait"
    if (revisit_count_all({"Waiting for the build. ok"}) != 0) { throw std::runtime_error("matched inside a word"); }
    // a take-back word in the middle of a sentence does not count
    if (revisit_count_all({"It is, which actually works. ok"}) != 0) { throw std::runtime_error("mid-sentence take-back counted"); }
    // phrases split across pieces, mixed case, markdown before the take-back
    if (revisit_count_all({"So X.\n", "**Hm", "m**, no. ok"}) != 1) { throw std::runtime_error("split phrase missed"); }
    // curly apostrophe normalizes to ASCII (custom list)
    {
        common_reasoning_revisit_tracker t({"Let's redo"});
        if (t.feed("Done. Let\xE2\x80\x99s redo it. ok") != 1) { throw std::runtime_error("curly apostrophe missed"); }
    }
    // nothing decidable until the character after the phrase arrives
    {
        common_reasoning_revisit_tracker t(common_reasoning_revisit_default_retracts());
        if (t.feed("Fine. Wait") != 0) { throw std::runtime_error("counted before the boundary was known"); }
        if (t.feed(",") != 1) { throw std::runtime_error("boundary arrival missed"); }
        t.reset();
        if (!t.buf.empty() || t.scanned != 0) { throw std::runtime_error("reset kept state"); }
    }
    // line end detection: a period is not a cut point (live 35B case: "in pip, `1.")
    {
        common_reasoning_revisit_tracker t({"y"});
        t.feed("one two");
        if (t.at_line_end()) { throw std::runtime_error("mid-line reported as end"); }
        t.feed(". in pip, `1.");
        if (t.at_line_end()) { throw std::runtime_error("period reported as line end"); }
        t.feed("0`.\n");
        if (!t.at_line_end()) { throw std::runtime_error("line end missed"); }
        t.feed("\n  ");
        if (!t.at_line_end()) { throw std::runtime_error("line end with trailing spaces missed"); }
    }
    // command line: repeated stages all survive, count and phrase list parse
    {
        const char * argv[] = { "prog",
            "--reasoning-revisit-nudge", "3500:light, with a colon: inside",
            "--reasoning-revisit-nudge", "5000:strict",
            "--reasoning-revisit-count", "2",
            "--reasoning-revisit-retract-phrases", "redo|scrap that" };
        common_params p;
        if (!common_params_parse(9, const_cast<char **>(argv), p, LLAMA_EXAMPLE_SERVER)) { throw std::runtime_error("revisit flags rejected"); }
        const auto & n = p.sampling.reasoning_revisit_nudges;
        if (n.size() != 2 || n[0].first != 3500 || n[0].second != "light, with a colon: inside" || n[1].first != 5000) {
            throw std::runtime_error("repeated --reasoning-revisit-nudge lost a stage");
        }
        if (p.sampling.reasoning_revisit_count != 2) { throw std::runtime_error("count not parsed"); }
        if (p.sampling.reasoning_revisit_retract_phrases.size() != 2) { throw std::runtime_error("phrases not split"); }
        if (!args_throw({"--reasoning-revisit-nudge", "no-colon"})) { throw std::runtime_error("missing N: accepted"); }
        if (!args_throw({"--reasoning-revisit-nudge", "100:"})) { throw std::runtime_error("empty message accepted"); }
        if (!args_throw({"--reasoning-revisit-count", "0"})) { throw std::runtime_error("count 0 accepted"); }
    }
}

// reasoning revisit nudge through the sampler with a real vocab: returns, per
// injected stage message, the index of the text token after which it fired
static std::vector<std::pair<int, size_t>> run_revisit(const llama_vocab * vocab, const std::string & text,
        const common_reasoning_revisit_cfg & cfg, const llama_tokens & start, const llama_tokens & end) {
    const llama_tokens toks = common_tokenize(vocab, text, false, false);
    struct llama_sampler * smpl = common_reasoning_budget_init(vocab, {start}, {end}, end, 100000,
        REASONING_BUDGET_IDLE, 0, {}, cfg);
    std::vector<llama_token_data> cands;
    {
        std::vector<llama_token> ids(toks);
        for (const auto & n : cfg.nudges) { ids.insert(ids.end(), n.tokens.begin(), n.tokens.end()); }
        ids.insert(ids.end(), start.begin(), start.end());
        ids.insert(ids.end(), end.begin(), end.end());
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        for (auto id : ids) { cands.push_back({ id, 0.0f, 0.0f }); }
    }
    for (auto t : start) { llama_sampler_accept(smpl, t); }
    std::vector<std::pair<int, size_t>> fired;
    for (size_t i = 0; i < toks.size(); i++) {
        llama_sampler_accept(smpl, toks[i]);
        llama_tokens forced;
        while (common_reasoning_budget_get_state(smpl) == REASONING_BUDGET_FORCING) {
            for (auto & c : cands) { c.logit = 0.0f; }
            llama_token_data_array arr = { cands.data(), cands.size(), -1, false };
            llama_sampler_apply(smpl, &arr);
            llama_token tok = -1;
            for (size_t k = 0; k < arr.size; k++) { if (arr.data[k].logit > -INFINITY) { tok = arr.data[k].id; break; } }
            if (tok < 0) { throw std::runtime_error("forcing with no surviving token"); }
            forced.push_back(tok);
            llama_sampler_accept(smpl, tok);
        }
        if (!forced.empty()) {
            int which = -1;
            for (size_t k = 0; k < cfg.nudges.size(); k++) { if (cfg.nudges[k].tokens == forced) { which = (int) k; } }
            if (which < 0) { throw std::runtime_error("forced sequence is not a stage message"); }
            if (common_reasoning_budget_get_state(smpl) != REASONING_BUDGET_COUNTING) { throw std::runtime_error("reasoning closed after a revisit nudge"); }
            // the message lands right after a line end
            std::string so_far;
            for (size_t k = 0; k <= i; k++) { so_far += common_token_to_piece(vocab, toks[k], false); }
            while (!so_far.empty() && so_far.back() == ' ') { so_far.pop_back(); }
            if (so_far.empty() || so_far.back() != '\n') {
                throw std::runtime_error("revisit nudge cut a sentence: ..." + so_far.substr(so_far.size() > 40 ? so_far.size() - 40 : 0));
            }
            fired.push_back({ which, i });
        }
    }
    llama_sampler_free(smpl);
    return fired;
}

static void test_reasoning_revisit_nudge() {
    std::string path = __FILE__;
    path = path.substr(0, path.find_last_of("/\\"));
    path = path.substr(0, path.find_last_of("/\\")) + "/models/ggml-vocab-qwen2.gguf";
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (!model) { throw std::runtime_error("cannot load " + path); }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const llama_tokens START = common_tokenize(vocab, "<<<BEGINREASON>>>", false, false);
    const llama_tokens END   = common_tokenize(vocab, "<<<ENDREASON>>>", false, false);
    common_reasoning_revisit_cfg cfg;
    cfg.nudges.push_back({ 60, common_tokenize(vocab, " [STRICT NUDGE] So: ", false, false) });
    cfg.nudges.push_back({ 20, common_tokenize(vocab, " [LIGHT NUDGE] ", false, false) }); // unsorted on purpose
    cfg.count = 2;

    std::string filler, half;
    for (int i = 0; i < 12; i++) { filler += "The value is computed here.\n"; }
    for (int i = 0; i < 6; i++)  { half   += "The value is computed here.\n"; }
    const std::string TB2 = "Actually, no.\nHmm, maybe not.\n"; // two take-backs
    const std::string TB1 = "Actually, no.\n";

    // take-backs before the first threshold do not count
    {
        auto f = run_revisit(vocab, TB2 + TB2 + filler + filler, cfg, START, END);
        if (!f.empty()) { throw std::runtime_error("A: counted take-backs before the threshold"); }
    }
    // two take-backs past the light threshold fire it; the strict one counts afresh
    {
        auto f = run_revisit(vocab, half + TB2 + filler, cfg, START, END);
        if (f.size() != 1 || f[0].first != 1) { throw std::runtime_error("B: expected only the light nudge"); }
        if (f[0].second + 1 < 20) { throw std::runtime_error("B: fired before its token count"); }
    }
    // one take-back is not enough
    {
        auto f = run_revisit(vocab, half + TB1 + filler + filler, cfg, START, END);
        if (!f.empty()) { throw std::runtime_error("C: fired on a single take-back"); }
    }
    // light, then two more take-backs past the strict threshold fire the strict nudge
    {
        auto f = run_revisit(vocab, half + TB2 + filler + TB1 + TB1 + half, cfg, START, END);
        if (f.size() != 2 || f[0].first != 1 || f[1].first != 0) { throw std::runtime_error("D: expected light then strict"); }
        if (f[1].second + 1 < 60) { throw std::runtime_error("D: strict fired before its token count"); }
    }
    // both thresholds passed before the take-backs: only the strict nudge fires
    {
        auto f = run_revisit(vocab, filler + half + TB2 + filler, cfg, START, END);
        if (f.size() != 1 || f[0].first != 0) { throw std::runtime_error("E: expected only the strict nudge"); }
    }
    // no vocab: the feature disables itself
    {
        struct llama_sampler * smpl = common_reasoning_budget_init(nullptr, {START}, {END}, END, 1000,
            REASONING_BUDGET_IDLE, 0, {}, cfg);
        for (auto t : START) { llama_sampler_accept(smpl, t); }
        for (auto t : common_tokenize(vocab, half + TB2 + TB2 + filler, false, false)) {
            llama_sampler_accept(smpl, t);
            if (common_reasoning_budget_get_state(smpl) == REASONING_BUDGET_FORCING) { throw std::runtime_error("F: fired without a vocab"); }
        }
        llama_sampler_free(smpl);
    }
    llama_model_free(model);
}

// budget close and end nudge with a real vocab: both wait for a line end.
// Returns, per forced injection, the index of the text token after which it
// started and the forced tokens.
struct text_run_event {
    size_t       after;
    llama_tokens forced;
    std::string  text_before; // reasoning text up to the injection
};

static std::vector<text_run_event> run_text(const llama_vocab * vocab, const std::string & text,
        const llama_tokens & start, const llama_tokens & end, const llama_tokens & budget_forced, int budget,
        int32_t end_nudge_budget, const llama_tokens & end_nudged, common_reasoning_budget_state * final_state) {
    const llama_tokens toks = common_tokenize(vocab, text, false, false);
    struct llama_sampler * smpl = common_reasoning_budget_init(vocab, {start}, {end}, budget_forced, budget,
        REASONING_BUDGET_IDLE, end_nudge_budget, end_nudged);
    std::vector<llama_token_data> cands;
    {
        std::vector<llama_token> ids(toks);
        ids.insert(ids.end(), budget_forced.begin(), budget_forced.end());
        ids.insert(ids.end(), end_nudged.begin(), end_nudged.end());
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        for (auto id : ids) { cands.push_back({ id, 0.0f, 0.0f }); }
    }
    for (auto t : start) { llama_sampler_accept(smpl, t); }
    std::vector<text_run_event> events;
    std::string so_far;
    for (size_t i = 0; i < toks.size(); i++) {
        if (common_reasoning_budget_get_state(smpl) == REASONING_BUDGET_DONE) { break; }
        llama_sampler_accept(smpl, toks[i]);
        so_far += common_token_to_piece(vocab, toks[i], false);
        llama_tokens forced;
        while (common_reasoning_budget_get_state(smpl) == REASONING_BUDGET_FORCING) {
            for (auto & c : cands) { c.logit = 0.0f; }
            llama_token_data_array arr = { cands.data(), cands.size(), -1, false };
            llama_sampler_apply(smpl, &arr);
            llama_token tok = -1;
            for (size_t k = 0; k < arr.size; k++) { if (arr.data[k].logit > -INFINITY) { tok = arr.data[k].id; break; } }
            if (tok < 0) { throw std::runtime_error("forcing with no surviving token"); }
            forced.push_back(tok);
            llama_sampler_accept(smpl, tok);
        }
        if (!forced.empty()) { events.push_back({ i, forced, so_far }); }
    }
    *final_state = common_reasoning_budget_get_state(smpl);
    llama_sampler_free(smpl);
    return events;
}

static bool ends_at_line(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) { s.pop_back(); }
    return !s.empty() && s.back() == '\n';
}

static void test_line_end_cut() {
    std::string path = __FILE__;
    path = path.substr(0, path.find_last_of("/\\"));
    path = path.substr(0, path.find_last_of("/\\")) + "/models/ggml-vocab-qwen2.gguf";
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (!model) { throw std::runtime_error("cannot load " + path); }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const llama_tokens START = common_tokenize(vocab, "<<<BEGINREASON>>>", false, false);
    const llama_tokens END   = common_tokenize(vocab, "<<<ENDREASON>>>", false, false);
    llama_tokens CLOSE = common_tokenize(vocab, "[OUT OF BUDGET]", false, false);
    CLOSE.insert(CLOSE.end(), END.begin(), END.end());
    const llama_tokens NUDGE = common_tokenize(vocab, "[WRAP UP] So:", false, false);

    const std::string LINE = "The value is computed here.\n";
    std::string lines, long_line;
    for (int i = 0; i < 20; i++) { lines += LINE; }
    for (int i = 0; i < 400; i++) { long_line += "word "; }
    const int line_toks = (int) common_tokenize(vocab, LINE, false, false).size();
    common_reasoning_budget_state st;

    // L1: the budget runs out mid-line; the close waits for the line end
    {
        const int budget = line_toks * 2 + 2; // two tokens into the third line
        auto ev = run_text(vocab, lines, START, END, CLOSE, budget, 0, {}, &st);
        if (ev.size() != 1 || ev[0].forced != CLOSE) { throw std::runtime_error("L1: expected one close"); }
        if (!ends_at_line(ev[0].text_before)) { throw std::runtime_error("L1: close cut a line: ..." + ev[0].text_before.substr(ev[0].text_before.size() - 30)); }
        if ((int) ev[0].after != line_toks * 3 - 1) { throw std::runtime_error("L1: close not at the first line end after the budget"); }
        if (st != REASONING_BUDGET_DONE) { throw std::runtime_error("L1: not done"); }
    }
    // L2: the budget runs out exactly at a line end; the close is immediate
    {
        auto ev = run_text(vocab, lines, START, END, CLOSE, line_toks * 2, 0, {}, &st);
        if (ev.size() != 1 || (int) ev[0].after != line_toks * 2 - 1) { throw std::runtime_error("L2: close not immediate at a line end"); }
    }
    // L3: a line with no end; the close comes after the 200-token grace
    {
        const int budget = line_toks + 3;
        auto ev = run_text(vocab, LINE + long_line + "\n" + lines, START, END, CLOSE, budget, 0, {}, &st);
        if (ev.size() != 1) { throw std::runtime_error("L3: expected one close"); }
        if ((int) ev[0].after != budget - 1 + 200) { throw std::runtime_error("L3: grace cap not 200 tokens (closed after " + std::to_string(ev[0].after) + ")"); }
    }
    // L4: the model closes the reasoning itself while the close waits
    {
        const std::string text = LINE + "The value is<<<ENDREASON>>> done.\n" + lines;
        auto ev = run_text(vocab, text, START, END, CLOSE, line_toks + 2, 0, {}, &st);
        if (!ev.empty()) { throw std::runtime_error("L4: forced a close after a natural end"); }
        if (st != REASONING_BUDGET_DONE) { throw std::runtime_error("L4: natural end not done"); }
    }
    // L5: the end nudge also waits for a line end, then the close follows
    {
        const int budget = line_toks * 10;
        auto ev = run_text(vocab, lines, START, END, CLOSE, budget, budget - line_toks - 2, NUDGE, &st);
        if (ev.size() != 2 || ev[0].forced != NUDGE || ev[1].forced != CLOSE) { throw std::runtime_error("L5: expected nudge then close"); }
        if (!ends_at_line(ev[0].text_before)) { throw std::runtime_error("L5: end nudge cut a line"); }
        if ((int) ev[0].after != line_toks * 2 - 1) { throw std::runtime_error("L5: end nudge not at the first line end past its threshold"); }
    }
    llama_model_free(model);
}

static void test_utf8_boundary_detection() {
    // Complete sequences
    GGML_ASSERT(common_utf8_is_complete("hello"));
    GGML_ASSERT(common_utf8_is_complete(""));
    GGML_ASSERT(common_utf8_is_complete("\xC2\xA0"));            // complete 2-byte UTF-8 (U+00A0)
    GGML_ASSERT(common_utf8_is_complete("\xE2\x80\x9C"));        // complete 3-byte UTF-8 (left double quote)
    GGML_ASSERT(common_utf8_is_complete("\xF0\x9F\x98\x80"));    // complete 4-byte UTF-8 (emoji)
    GGML_ASSERT(common_utf8_is_complete("abc\xC3\xA9"));         // ASCII + complete 2-byte

    // Incomplete sequences
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xC2", 1)));            // 2-byte start, missing continuation
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xE2\x80", 2)));        // 3-byte start + 1 cont, missing 1
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xE2", 1)));            // 3-byte start, missing 2
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xF0\x9F\x98", 3)));    // 4-byte start + 2 cont, missing 1
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xF0\x9F", 2)));        // 4-byte start + 1 cont, missing 2
    GGML_ASSERT(!common_utf8_is_complete(std::string("\xF0", 1)));            // 4-byte start, missing 3
    GGML_ASSERT(!common_utf8_is_complete(std::string("\x80", 1)));            // orphan continuation byte

    // Mixed: ASCII followed by start of multi-byte
    GGML_ASSERT(!common_utf8_is_complete(std::string("hello\xC3", 6)));       // ASCII + incomplete 2-byte
    GGML_ASSERT(common_utf8_is_complete(std::string("hello\xC3\xA9", 7)));    // ASCII + complete 2-byte
}

static int run_all_tests(void) {
    // unbuffered stdout so no test output is lost on a crash
    setvbuf(stdout, nullptr, _IONBF, 0);
#ifdef _MSC_VER
    // route CRT assert/heap-error reports to stderr instead of a modal dialog
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
#endif

    // Reasoning budget sampler tests
    printf("Testing reasoning budget sampler... ");

    // Test 1: Basic budget with start/end tokens - no forcing (natural end before budget exhausted)
    {
        const std::vector<llama_token> start = {100};  // start token
        const std::vector<llama_token> end = {101};    // end token
        const std::vector<llama_token> forced = {102}; // forced token (not used in this test)
        const std::vector<llama_token> sequence = {100, 50, 51, 101, 52}; // start, two tokens, end, one more

        test_reasoning_budget("natural end before budget exhausted", sequence, {start}, {end}, forced,
            5,      // budget of 5 tokens
            REASONING_BUDGET_IDLE,
            SIZE_MAX, SIZE_MAX); // no forcing expected (natural end)
    }

    // Test 2: Budget exhausted, forcing should occur
    // Flow: i=0 apply()->passthrough, accept(100)->COUNTING; i=1 accept(50)->remaining=1
    // i=2 accept(51)->remaining=0->FORCING; i=3 apply() forces token[0]; i=4 apply() forces token[1]
    // At i=4, accept() advances force_pos to 2 which equals forced_tokens.size(), so state becomes DONE
    {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101}; // forced message + end
        const std::vector<llama_token> sequence = {100, 50, 51, 52, 53}; // start + 4 tokens (budget=2)

        test_reasoning_budget("budget exhausted forcing", sequence, {start}, {end}, forced,
            2,      // budget of 2 tokens
            REASONING_BUDGET_IDLE,
            3,      // forcing starts at i=3 (accept at i=2 depletes budget, apply at i=3 forces)
            4);     // forcing continues through i=4 (accept at i=4 transitions to DONE)
    }

    // Test 3: Activate immediately with budget=0, forcing should start right away
    // Flow: init promotes COUNTING+budget=0 to FORCING, so apply() sees FORCING at i=0
    {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {100, 50, 51, 52}; // start token first, then 3 tokens

        test_reasoning_budget("activate immediately budget=0", sequence, {start}, {end}, forced,
            0,      // budget of 0 tokens
            REASONING_BUDGET_COUNTING, // starts counting, promoted to FORCING since budget=0
            0,      // forcing starts at i=0 (initialized in FORCING, apply forces immediately)
            1);     // forcing continues through i=1 (accept at i=1 transitions to DONE)
    }

    // Test 4: No start/end tokens configured - passthrough (no forcing)
    {
        const std::vector<llama_token> start = {};
        const std::vector<llama_token> end = {};
        const std::vector<llama_token> forced = {102};
        const std::vector<llama_token> sequence = {50, 51, 52, 53};

        test_reasoning_budget("no start/end configured", sequence, {start}, {end}, forced,
            2,      // budget
            REASONING_BUDGET_IDLE,
            SIZE_MAX, SIZE_MAX); // no forcing (no start/end configured)
    }

    // Test 5: Activate immediately with budget > 0, count down then force
    // Flow: i=0 accept(50)->remaining=1, i=1 accept(51)->remaining=0->FORCING
    // Forcing starts at i=2 (apply sees FORCING after accept at i=1 transitioned)
    {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {50, 51, 52, 53};

        test_reasoning_budget("activate immediately with budget", sequence, {start}, {end}, forced,
            2,      // budget of 2 tokens
            REASONING_BUDGET_COUNTING,
            2,      // forcing starts at i=2 (after 2 accepts deplete budget, apply at i=2 forces)
            3);     // forcing continues through i=3
    }

    // Test 6: Multi-block thinking. First block ends naturally at i=2, second
    // start tag at i=3 re-arms the budget, which then exhausts at i=5.
    // Regression: before this fix, DONE absorbed all subsequent tokens and a
    // second <think> block ran unbudgeted.
    // Flow: i=0 accept(100)->COUNTING rem=2; i=1 accept(50)->rem=1;
    //       i=2 accept(101)->end_matcher matches, DONE;
    //       i=3 accept(100)->re-arm, COUNTING rem=2;
    //       i=4 accept(60)->rem=1; i=5 accept(61)->rem=0->FORCING;
    //       i=6 apply()->forces token[0]=102, accept(62)->force_pos=1, stay FORCING;
    //       i=7 apply()->forces token[1]=101, accept(63)->force_pos=2->DONE.
    {
        const std::vector<llama_token> start = {100};
        const std::vector<llama_token> end = {101};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {100, 50, 101, 100, 60, 61, 62, 63};

        test_reasoning_budget("multi-block re-arms budget after DONE", sequence, {start}, {end}, forced,
            2,      // budget of 2 tokens (per block)
            REASONING_BUDGET_IDLE,
            6,      // forcing starts at i=6 (after second block exhausts at i=5)
            7);     // forcing continues through i=7
    }

    // Test 7: Multiple start sequences - the second sequence activates counting
    // Flow: i=0 accept(110), i=1 accept(111)->COUNTING rem=2; i=2 accept(50)->rem=1;
    //       i=3 accept(51)->rem=0->FORCING; i=4..5 apply() forces the end sequence
    {
        const std::vector<llama_tokens> start = {{100}, {110, 111}};
        const std::vector<llama_tokens> end = {{101}};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {110, 111, 50, 51, 52, 53};

        test_reasoning_budget("multiple start sequences", sequence, start, end, forced,
            2,      // budget of 2 tokens
            REASONING_BUDGET_IDLE,
            4,      // forcing starts at i=4 (accept at i=3 depletes budget)
            5);     // forcing continues through i=5
    }

    // Test 8: Multiple end sequences - natural end via the second sequence
    // Flow: i=0 accept(100)->COUNTING rem=5; i=1 accept(50)->rem=4;
    //       i=2 accept(103)->partial end, rem=3; i=3 accept(104)->end matched, DONE
    {
        const std::vector<llama_tokens> start = {{100}};
        const std::vector<llama_tokens> end = {{101}, {103, 104}};
        const std::vector<llama_token> forced = {102, 101};
        const std::vector<llama_token> sequence = {100, 50, 103, 104, 52};

        test_reasoning_budget("multiple end sequences", sequence, start, end, forced,
            5,      // budget of 5 tokens
            REASONING_BUDGET_IDLE,
            SIZE_MAX, SIZE_MAX); // no forcing expected (natural end)
    }

    test_reasoning_budget_clone_mid_counting();
    test_reasoning_budget_clone_mid_forcing();
    test_reasoning_budget_force_manual();
    test_reasoning_budget_end_match();

    // reasoning end nudge: pre-exhaustion wrap-up message (bounded feature)
    printf("Testing reasoning end nudge... ");
    test_reasoning_end_nudge();
    printf("OK\n");

    // review fix: manual force queued during a transient nudge
    printf("Testing queued force during nudge... ");
    test_force_queued_during_nudge();
    printf("OK\n");

    printf("Testing reasoning revisit tracker... ");
    test_revisit_tracker();
    printf("OK\n");

    printf("Testing reasoning revisit nudge... ");
    test_reasoning_revisit_nudge();
    printf("OK\n");

    printf("Testing line-end cut for the budget close and end nudge... ");
    test_line_end_cut();
    printf("OK\n");

    printf("OK (reasoning budget suites passed)\n");

    printf("Testing UTF-8 boundary detection... ");
    test_utf8_boundary_detection();
    printf("OK\n");

    return 0;
}

int main(void) {
    try {
        return run_all_tests();
    } catch (const std::exception & e) {
        fprintf(stderr, "TEST FAILED: %s\n", e.what());
        return 1;
    }
}