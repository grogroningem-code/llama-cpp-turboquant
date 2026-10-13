#include "reasoning-budget.h"
#include "common.h"
#include "trie.h"
#include "unicode.h"

#include "log.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<llama_tokens> seqs;
    common_aho_corasick ac;
    size_t state = 0;

    token_matcher(const std::vector<llama_tokens> & seqs) : seqs(collect(seqs)), ac(build_trie(this->seqs)) {}

    static std::vector<llama_tokens> collect(const std::vector<llama_tokens> & seqs) {
        std::vector<llama_tokens> res;
        for (const auto & seq : seqs) {
            if (!seq.empty() && std::find(res.begin(), res.end(), seq) == res.end()) {
                res.push_back(seq);
            }
        }
        return res;
    }

    static common_trie build_trie(const std::vector<llama_tokens> & seqs) {
        common_trie t;
        for (const auto & seq : seqs) {
            t.insert(std::vector<uint32_t>(seq.begin(), seq.end()));
        }
        return t;
    }

    // returns the index into seqs of the longest sequence ending at this token, or -1
    int32_t advance(llama_token token) {
        state = ac.next(state, (uint32_t) token);
        const int32_t p = ac.match_pattern(state);
        if (p >= 0) {
            state = 0;
        }
        return p;
    }

    void reset() { state = 0; }
};

// --- reasoning revisit tracker -------------------------------------------

const std::vector<std::string> & common_reasoning_revisit_default_retracts() {
    static const std::vector<std::string> v = {
        "actually", "but actually", "wait", "but wait", "no wait", "no,", "hmm", "hmmm", "hm,",
        "hold on", "but hold on", "oh", "on second thought", "on reflection", "scratch that",
        "let me reconsider", "let me rethink", "let me re-think", "let me re-read", "let me reread",
        "let me re-check", "let me recheck", "let me re-examine", "let me double-check",
        "let me double check", "let me verify", "let me look again", "let me step back",
        "or maybe", "or should i", "alternatively", "one more thing", "one concern",
        "but what if", "what if", "however",
    };
    return v;
}

static std::vector<std::string> revisit_lower_all(std::vector<std::string> v) {
    for (auto & s : v) {
        for (auto & c : s) {
            c = (char) std::tolower((unsigned char) c);
        }
    }
    return v;
}

common_reasoning_revisit_tracker::common_reasoning_revisit_tracker(std::vector<std::string> retract)
    : retract_phrases(revisit_lower_all(std::move(retract))) {}

void common_reasoning_revisit_tracker::reset() {
    buf.clear();
    scanned = 0;
}

static bool revisit_is_word(char c) {
    return std::isalnum((unsigned char) c) || c == '_' || (unsigned char) c >= 0x80;
}

// phrase at pos, on word boundaries; -1 = not decidable yet (text too short)
static int revisit_match_at(const std::string & buf, size_t pos, const std::string & ph) {
    if (pos + ph.size() >= buf.size()) {
        return buf.compare(pos, std::string::npos, ph, 0, buf.size() - pos) == 0 ? -1 : 0;
    }
    if (buf.compare(pos, ph.size(), ph) != 0) {
        return 0;
    }
    if (pos > 0 && revisit_is_word(ph.front()) && revisit_is_word(buf[pos - 1])) {
        return 0;
    }
    if (revisit_is_word(ph.back()) && revisit_is_word(buf[pos + ph.size()])) {
        return 0;
    }
    return 1;
}

// a sentence starts at pos: only spaces / markdown markers back to a
// sentence end, a newline or the start of the text
static bool revisit_sentence_start(const std::string & buf, size_t pos) {
    while (pos > 0) {
        const char c = buf[pos - 1];
        if (c == ' ' || c == '\t' || c == '*' || c == '-' || c == '#' || c == '>' || c == '"' || c == '(') {
            pos--;
            continue;
        }
        return c == '.' || c == '!' || c == '?' || c == ':' || c == '\n' || c == ')';
    }
    return true;
}

// a period is not a safe cut point ("version 1." / "e.g." / "x.py"), so only
// a newline counts
bool common_reasoning_revisit_tracker::at_line_end() const {
    size_t n = buf.size();
    while (n > 0 && (buf[n - 1] == ' ' || buf[n - 1] == '\t' || buf[n - 1] == '\r')) {
        n--;
    }
    return n > 0 && buf[n - 1] == '\n';
}

int common_reasoning_revisit_tracker::feed(const std::string & piece) {
    for (size_t i = 0; i < piece.size(); i++) {
        const unsigned char c = (unsigned char) piece[i];
        // right single quote U+2019 -> ASCII apostrophe
        if (c == 0xE2 && i + 2 < piece.size() && (unsigned char) piece[i + 1] == 0x80 && (unsigned char) piece[i + 2] == 0x99) {
            buf.push_back('\'');
            i += 2;
            continue;
        }
        buf.push_back((char) std::tolower(c));
    }
    int found = 0;
    while (scanned < buf.size()) {
        bool pending = false;
        size_t hit = 0;
        for (const auto & ph : retract_phrases) {
            const int m = revisit_match_at(buf, scanned, ph);
            if (m < 0) {
                pending = true;
            } else if (m > 0 && revisit_sentence_start(buf, scanned)) {
                hit = std::max(hit, ph.size());
            }
        }
        if (pending) {
            break; // a longer phrase may still complete here ("but" -> "but wait"): wait for more text
        }
        if (hit > 0) {
            found++;
            scanned += hit;
        } else {
            scanned++;
        }
    }
    // bound memory: keep a tail for sentence-start look-back
    if (scanned > 1024) {
        const size_t cut = scanned - 256;
        buf.erase(0, cut);
        scanned -= cut;
    }
    return found;
}

// After the budget runs out, the close waits for the model to end its line
// (at most this many extra tokens) so the budget message never cuts a word,
// number or path in half. Needs a vocab to see the text.
static const int32_t REASONING_BUDGET_LINE_GRACE = 200;

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    llama_tokens forced_tokens;

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force

    int32_t end_match;        // index into end_matcher.seqs of the sequence that transitioned to DONE, -1 if none

    // reasoning end nudge (bounded feature): fire once when the budget
    // drains to the threshold; transient FORCING, reasoning stays open
    llama_tokens end_nudge_tokens;     // precomputed message tokens (empty = disabled)
    int32_t end_nudge_budget = 0;      // threshold in remaining tokens
    bool end_nudge_fired = false;      // once per reasoning block
    bool end_nudging = false;          // active FORCING is an end-nudge injection
    // reasoning revisit nudge stages: fire after repeated take-backs
    std::vector<common_reasoning_revisit_nudge> revisit_nudges; // sorted by increasing `after`
    common_reasoning_revisit_tracker revisit_tracker { {} };
    int32_t revisit_need  = 3;         // take-backs needed per stage
    int32_t revisit_count = 0;         // take-backs seen past the next stage's count
    size_t revisit_next = 0;           // first stage not yet fired in this block
    size_t revisit_active = 0;         // stage being injected while revisit_nudging
    bool revisit_nudging = false;      // active FORCING is a revisit-nudge injection
    bool force_pending = false;        // manual force queued during a transient nudge
    bool line_end = true;              // reasoning text so far ends at a line start
    int32_t grace_left = 0;            // extra tokens left while waiting for a line end

    // the sequence apply() forces in the current FORCING state
    const llama_tokens & active_forced() const {
        if (end_nudging) {
            return end_nudge_tokens;
        }
        if (revisit_nudging) {
            return revisit_nudges[revisit_active].tokens;
        }
        return forced_tokens;
    }
};

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->end_nudge_fired = false;         // fresh block: end nudge re-arms
                ctx->revisit_next = 0;                // fresh block: revisit stages re-arm
                ctx->revisit_tracker.reset();
                ctx->revisit_count = 0;
                ctx->line_end = true;
                COM_TRC("activated, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            const int32_t match = ctx->end_matcher.advance(token);
            if (match >= 0) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            bool utf8_complete = true;
            if (ctx->vocab != nullptr) {
                const std::string piece = common_token_to_piece(ctx->vocab, token, false);
                utf8_complete = common_utf8_is_complete(piece);
                for (const char c : piece) {
                    if (c == '\n') {
                        ctx->line_end = true;
                    } else if (c != ' ' && c != '\t' && c != '\r') {
                        ctx->line_end = false;
                    }
                }
            }
            // safe point to inject text: the character is complete and, when
            // the text is known, the line has ended
            const bool safe_cut = utf8_complete && (ctx->vocab == nullptr || ctx->line_end);

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                ctx->grace_left--;
                if (safe_cut || (utf8_complete && ctx->grace_left <= 0)) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->end_matcher.reset();
                    LOG_INF("reasoning budget exhausted, closing %s after %d extra tokens\n",
                            safe_cut ? "at a line end" : "mid-line (grace used up)",
                            REASONING_BUDGET_LINE_GRACE - ctx->grace_left);
                }
            } else if (ctx->state == REASONING_BUDGET_COUNTING) {
                ctx->remaining--;
                if (ctx->remaining <= 0) {
                    if (safe_cut) {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, forcing end sequence\n");
                    } else {
                        // let the model finish its line (or character) first
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->grace_left = REASONING_BUDGET_LINE_GRACE;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, waiting for a line end\n");
                    }
                } else {
                    // reasoning end nudge: fire once per block when the budget
                    // first drains to the threshold. Exhaustion already won
                    // above when they coincide (remaining <= 0). The message
                    // is injected via a transient FORCING; the reasoning
                    // section stays open and the budget does not decrement
                    // while FORCING (the nudge effectively adds its own
                    // length to the usable budget headroom). Like the close,
                    // it waits for a line end; if none comes before the
                    // budget runs out, the close takes over.
                    if (ctx->state == REASONING_BUDGET_COUNTING && safe_cut &&
                        !ctx->end_nudge_tokens.empty() &&
                        !ctx->end_nudge_fired && ctx->remaining <= ctx->end_nudge_budget) {
                        ctx->end_nudge_fired = true;
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_matcher.reset();
                        ctx->end_nudging = true;
                        LOG_INF("reasoning end nudge fired (remaining=%d)\n", ctx->remaining);
                    }
                    // reasoning revisit nudge: scan the text; take-backs count
                    // only once the next stage's token count is reached, and a
                    // stage fires at the line end after `need` of them. Defers
                    // when the end nudge took this token.
                    if (ctx->revisit_next < ctx->revisit_nudges.size()) {
                        const int found = ctx->revisit_tracker.feed(common_token_to_piece(ctx->vocab, token, false));
                        const int32_t spent = ctx->budget - ctx->remaining;
                        if (spent >= ctx->revisit_nudges[ctx->revisit_next].after) {
                            ctx->revisit_count += found;
                        }
                        if (ctx->state == REASONING_BUDGET_COUNTING && utf8_complete &&
                            ctx->revisit_count >= ctx->revisit_need &&
                            ctx->revisit_tracker.at_line_end()) {
                            while (ctx->revisit_next + 1 < ctx->revisit_nudges.size() &&
                                   spent >= ctx->revisit_nudges[ctx->revisit_next + 1].after) {
                                ctx->revisit_next++;
                            }
                            ctx->revisit_active = ctx->revisit_next++;
                            ctx->revisit_count = 0; // the next stage counts afresh
                            ctx->state = REASONING_BUDGET_FORCING;
                            ctx->force_pos = 0;
                            ctx->end_matcher.reset();
                            ctx->revisit_nudging = true;
                            LOG_INF("reasoning revisit nudge %zu fired (spent=%d)\n", ctx->revisit_active + 1, spent);
                        }
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
        {
            // track the end sequence within the forced sequence so it is also reported on DONE
            const int32_t match = ctx->end_matcher.advance(token);
            ctx->force_pos++;
            if (ctx->force_pos >= ctx->active_forced().size()) {
                if (ctx->end_nudging || ctx->revisit_nudging) {
                    if (ctx->force_pending) {
                        // chained manual close: abandon the transient state and
                        // force the close sequence immediately (active_forced()
                        // now resolves to the plain forced sequence)
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->end_matcher.reset();
                        ctx->end_nudging = false;
                        ctx->revisit_nudging = false;
                        ctx->force_pending = false;
                        COM_TRC("%s", "queued manual force chained after the nudge\n");
                    } else {
                        // transient nudge complete (end or revisit nudge): the
                        // reasoning section stays open, the model keeps reasoning
                        // (any end-tag bytes inside the nudge text are discarded
                        // here - the matcher restarts)
                        ctx->state = REASONING_BUDGET_COUNTING;
                        ctx->end_match = -1;
                        ctx->end_matcher.reset();
                        ctx->end_nudging = false;
                        ctx->revisit_nudging = false;
                        COM_TRC("%s", "nudge injected, resuming counting\n");
                    }
                } else {
                    ctx->state = REASONING_BUDGET_DONE;
                    ctx->end_match = match;
                    COM_TRC("%s", "forced sequence complete, done\n");
                }
            }
            break;
        }
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->end_matcher.reset();
                ctx->end_match = -1;
                ctx->end_nudge_fired = false;         // fresh block: end nudge re-arms
                ctx->revisit_next = 0;                // fresh block: revisit stages re-arm
                ctx->revisit_tracker.reset();
                ctx->revisit_count = 0;
                ctx->line_end = true;
                COM_TRC("re-activated on new start tag, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    COM_TRC("%s", "budget=0, forcing immediately\n");
                }
            }
            break;
    }
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state != REASONING_BUDGET_FORCING) {
        // passthrough — don't modify logits
        return;
    }

    const llama_tokens & forced = ctx->active_forced();
    if (ctx->force_pos >= forced.size()) {
        return;
    }

    const llama_token forced_token = forced[ctx->force_pos];

    // set all logits to -inf except the forced token
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced_token) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
    ctx->end_match = -1;
    ctx->end_nudge_fired = false;
    ctx->end_nudging = false;
    ctx->revisit_next = 0;
    ctx->revisit_nudging = false;
    ctx->revisit_tracker.reset();
    ctx->revisit_count = 0;
    ctx->force_pending = false;
    ctx->line_end = true;
    ctx->grace_left = 0;
}

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl);

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx(*ctx)
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state,
        int32_t                           end_nudge_budget,
        const llama_tokens              & end_nudge_tokens,
        const common_reasoning_revisit_cfg & revisit) {
    // promote COUNTING with budget <= 0 to FORCING
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    common_reasoning_budget_ctx * ctx = new common_reasoning_budget_ctx {
        /* .vocab         = */ vocab,
        /* .start_matcher = */ token_matcher(start_seqs),
        /* .end_matcher   = */ token_matcher(end_seqs),
        /* .forced_tokens = */ forced_tokens,
        /* .budget        = */ budget,
        /* .remaining     = */ budget,
        /* .state         = */ initial_state,
        /* .force_pos     = */ 0,
        /* .end_match     = */ -1,
    };
    ctx->end_nudge_tokens = end_nudge_tokens;
    ctx->end_nudge_budget = end_nudge_budget;

    // reasoning end nudge: both halves must be configured to enable
    if (ctx->end_nudge_budget > 0 && ctx->end_nudge_tokens.empty()) {
        LOG_WRN("%s: reasoning end nudge budget=%d set but the message tokenized empty - end nudge disabled\n",
                __func__, ctx->end_nudge_budget);
        ctx->end_nudge_budget = 0;
    } else if (ctx->end_nudge_budget <= 0 && !ctx->end_nudge_tokens.empty()) {
        LOG_WRN("%s: reasoning end nudge message set but no --reasoning-end-nudge-budget - end nudge disabled\n", __func__);
        ctx->end_nudge_tokens.clear();
    }

    // reasoning revisit nudge: stages need a message and text matching needs a vocab
    for (const auto & stage : revisit.nudges) {
        if (stage.tokens.empty()) {
            LOG_WRN("%s: reasoning revisit nudge after=%d has an empty message - stage disabled\n", __func__, stage.after);
        } else {
            ctx->revisit_nudges.push_back(stage);
        }
    }
    if (!ctx->revisit_nudges.empty() && vocab == nullptr) {
        LOG_WRN("%s: reasoning revisit nudge needs a vocab - disabled\n", __func__);
        ctx->revisit_nudges.clear();
    }
    std::stable_sort(ctx->revisit_nudges.begin(), ctx->revisit_nudges.end(),
            [](const common_reasoning_revisit_nudge & a, const common_reasoning_revisit_nudge & b) { return a.after < b.after; });
    ctx->revisit_need = std::max(1, revisit.count);
    ctx->revisit_tracker = common_reasoning_revisit_tracker(
            revisit.retract_phrases.empty() ? common_reasoning_revisit_default_retracts() : revisit.retract_phrases);

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ ctx
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state,
        int32_t                           end_nudge_budget,
        const llama_tokens              & end_nudge_tokens,
        const common_reasoning_revisit_cfg & revisit) {
    return common_reasoning_budget_init_state(vocab, start_seqs, end_seqs, forced_tokens, budget, initial_state, end_nudge_budget, end_nudge_tokens, revisit);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl) {
    if (!smpl) {
        return nullptr;
    }

    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    if (ctx->end_match < 0) {
        return nullptr;
    }

    return &ctx->end_matcher.seqs[ctx->end_match];
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }

    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    // a manual reasoning_end arriving during a transient nudge (end or revisit
    // nudge) cannot act immediately; queue it so the nudge text completes
    // and the block closes right after, instead of the request being silently
    // lost while the sampler resumes counting (review fix)
    if (ctx->state == REASONING_BUDGET_FORCING &&
        (ctx->end_nudging || ctx->revisit_nudging)) {
        ctx->force_pending = true;
        return true;
    }

    // only a sampler that is actively counting down the budget may be forced;
    // any other state (idle, already forcing/waiting, or done) is left untouched
    if (ctx->state != REASONING_BUDGET_COUNTING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
