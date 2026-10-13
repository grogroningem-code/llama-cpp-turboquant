#pragma once

#include "llama.h"

#include "common.h"

#include <cstdint>
#include <vector>

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,         // waiting for start sequence
    REASONING_BUDGET_COUNTING,     // counting down tokens
    REASONING_BUDGET_FORCING,      // forcing budget message + end sequence
    REASONING_BUDGET_WAITING_UTF8, // budget exhausted, waiting for UTF-8 completion and a line end
    REASONING_BUDGET_DONE,         // passthrough forever
};

// Reasoning revisit nudge (see --reasoning-revisit-* flags): staged messages
// that fire once the model keeps taking itself back ("actually", "hmm", ...).
struct common_reasoning_revisit_nudge {
    int32_t      after = 0; // earliest reasoning token count (per block) to fire at
    llama_tokens tokens;    // precomputed message tokens
};

struct common_reasoning_revisit_cfg {
    std::vector<common_reasoning_revisit_nudge> nudges;
    int32_t                  count = 3;        // take-backs needed after a stage's token count
    std::vector<std::string> retract_phrases;  // empty = built-in defaults
};

// Text scanner behind the revisit nudge, exposed for tests. feed() takes the
// reasoning text piece by piece and returns how many new take-back phrases it
// found at a sentence start. Matching is ASCII case-insensitive on word
// boundaries; a match is decided once the character after it is known.
// reset() forgets the text so far.
struct common_reasoning_revisit_tracker {
    std::vector<std::string> retract_phrases;
    std::string buf;         // lowercased tail of the reasoning text
    size_t      scanned = 0; // match starts below this index were examined

    explicit common_reasoning_revisit_tracker(std::vector<std::string> retract);
    int  feed(const std::string & piece);
    void reset();
    bool at_line_end() const; // text so far ends with a newline
};

// Built-in take-back phrase list for the revisit tracker.
const std::vector<std::string> & common_reasoning_revisit_default_retracts();

// Creates a reasoning budget sampler that limits token generation inside a
// reasoning block (e.g. between <think> and </think>).
//
// State machine: IDLE -> COUNTING -> WAITING_UTF8 -> FORCING -> DONE
//   IDLE:         passthrough, watching for a start sequence
//   COUNTING:     counting down remaining tokens, watching for a natural end sequence
//   WAITING_UTF8: budget exhausted, allowing tokens to complete a UTF-8 sequence
//                 and (with a vocab) the current line, at most 200 extra tokens
//   FORCING:      forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:         passthrough forever
//
// Parameters:
//   vocab          - vocabulary (used for UTF-8 boundary detection; can be nullptr)
//   start_seqs     - token sequences, any of which activates counting
//   end_seqs       - token sequences, any of which naturally deactivates
//   forced_tokens  - token sequence forced when budget expires
//   budget         - max tokens allowed in the reasoning block
//   initial_state  - initial state
//   end_nudge_budget / end_nudge_tokens - reasoning end nudge: when the
//                  budget first drains to end_nudge_budget tokens remaining,
//                  end_nudge_tokens (a static wrap-up message) are injected
//                  via a transient FORCING and the reasoning section stays
//                  open; fires at most once per reasoning block, at a line
//                  end when a vocab is given. Disabled
//                  when end_nudge_budget <= 0 or end_nudge_tokens is empty.
//   revisit        - reasoning revisit nudge stages: stage k fires once per
//                  reasoning block after `count` take-backs were seen past
//                  its `after` token count (and past the previous firing).
//                  When several stages are due at once only the latest fires.
//                  Waits for a line end so the message never cuts a
//                  sentence, number or path in half.
//                  Needs a vocab (text matching); disabled without one.
//
struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state = REASONING_BUDGET_IDLE,
        int32_t                           end_nudge_budget = 0,
        const llama_tokens              & end_nudge_tokens = {},
        const common_reasoning_revisit_cfg & revisit = {});

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl);

// The end sequence that transitioned the sampler to DONE, or nullptr if none
// was recorded. Cleared when a new start sequence re-arms the sampler.
const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl);

// Manually transition the reasoning budget sampler into the FORCING state.
// Returns true if the transition occurred.
bool common_reasoning_budget_force(struct llama_sampler * smpl);
