import re
from utils import *

# A context checkpoint saved into a reused buffer (the buffer of an erased checkpoint) must restore the same state
# as one saved into a new buffer.
#
# tinygemma3's sliding window (4096) is longer than this context (1024), so the window never slides: the server
# restores a checkpoint on rollback only because its pruning threshold (pos_next - n_swa) clamps to 0. If the
# server learns to skip that restore, this test fails loudly with "restored no checkpoint". Because nothing is
# pruned, a checkpoint's size grows with the prompt, so the reused buffer here (from an erased, longer checkpoint)
# is larger than the data saved into it.
#
# The control is the same conversation on a fresh server, where the restored checkpoint sits in a new buffer.
# Both runs read every prompt in the same batches, so their outputs must match exactly on any backend. (Comparing
# with the prompt read from scratch would not work: after a restore the server re-reads the checkpoint's last
# token in a batch of its own, and that alone moves this model's logprobs by about 0.05, as much as a corrupted
# buffer does.)

NOTES = " ".join(f"Entry {i}: the guard at gate {i % 7} saw trader {i * 7 % 11} sell {i % 5} bottles on day {i}."
                 for i in range(24))
QUESTION = "\nQuestion: how many bottles did trader {} sell?\nAnswer:"


def complete(server, prompt, n_predict):
    # n_probs: tinygemma3 prints the same token whatever the state, so the probabilities carry the comparison
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": n_predict, "id_slot": 0, "cache_prompt": True, "n_probs": 5,
    })
    assert res.status_code == 200
    return res.body


def run(log_path, with_spares):
    server = ServerPreset.tinygemma3()
    server.temperature = 0.0
    server.n_slots = 1
    server.n_ctx = 1024
    server.n_batch = 32
    server.debug = True  # --verbose: the buffer reuse line is trace level
    server.log_path = str(log_path)  # kept by pytest's tmp_path for debugging a failure
    server.start()
    try:
        if with_spares:
            # a different chat: the next turn reads its prompt from the start and erases both of these
            # checkpoints, so their buffers become spares that the next checkpoints are saved into
            complete(server, "Ignore this. " + NOTES + QUESTION.format(1), 4)
        complete(server, NOTES + QUESTION.format(1), 4)
        # rolls back to the last checkpoint before the question's changed digit and restores it
        last = complete(server, NOTES + QUESTION.format(2), 16)
    finally:
        server.stop()
    with open(log_path, encoding="utf-8", errors="replace") as f:
        return last, f.read().splitlines()


def pos_max_of(line):
    m = re.search(r"pos_max = (\d+),", line)
    assert m is not None, f"no pos_max in log line: {line}"
    return m.group(1)


def restore(lines):
    """Index and pos_max of the run's only checkpoint restore (in the last turn)."""
    found = [i for i, l in enumerate(lines) if "restored context checkpoint" in l]
    assert len(found) == 1, f"expected one checkpoint restore, found {len(found)}"
    return found[0], pos_max_of(lines[found[0]])


def test_restore_from_reused_checkpoint_buffer(tmp_path):
    reused, lines = run(tmp_path / "reuse.log", with_spares=True)
    i_restore, pos_max = restore(lines)
    # the restored checkpoint must be one that was saved into a reused buffer: the "created" line right after a
    # "reusing" line, with the restored position
    reused_saves = []
    for i, l in enumerate(lines[:i_restore]):
        if "reusing a spare context checkpoint buffer" in l:
            created = next((m for m in lines[i + 1:i_restore] if "created context checkpoint" in m), None)
            if created is not None:
                reused_saves.append(pos_max_of(created))
    assert pos_max in reused_saves, f"the restored checkpoint (pos_max {pos_max}) was not saved into a reused buffer"
    assert reused["timings"]["prompt_n"] < 64, "the restored turn should read only the tail of its prompt"

    fresh, lines = run(tmp_path / "control.log", with_spares=False)
    i_restore, control_pos_max = restore(lines)
    assert not any("reusing a spare context checkpoint buffer" in l for l in lines[:i_restore]), \
        "the control's restored checkpoint should be in a new buffer"
    # the same restore and the same re-read: otherwise a difference below is not about the buffer
    assert control_pos_max == pos_max, "the control restored a different checkpoint"
    assert fresh["timings"]["prompt_n"] == reused["timings"]["prompt_n"], "the control re-read a different tail"

    assert reused["content"] == fresh["content"]
    assert reused["completion_probabilities"] == fresh["completion_probabilities"]
