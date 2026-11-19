import os
import re
import tempfile
import pytest
from utils import *

# A context checkpoint saved into a reused buffer (the buffer of an erased checkpoint) must restore the same state
# as one saved into a new buffer. tinygemma3 has sliding-window layers, so rolling back restores a checkpoint.
#
# The control is the same conversation on a fresh server, where the restored checkpoint sits in a new buffer.
# Both runs read every prompt in the same batches, so their outputs must match exactly on any backend. (Comparing
# with the prompt read from scratch would not work: after a restore the server re-reads the checkpoint's last
# token in a batch of its own, and that alone moves this model's logprobs by about 0.05, as much as a corrupted
# buffer does.)

NOTES = " ".join(f"Entry {i}: the guard at gate {i % 7} saw trader {i * 7 % 11} sell {i % 5} bottles on day {i}."
                 for i in range(24))
QUESTION = "\nQuestion: how many bottles did trader {} sell?\nAnswer:"


def start_server():
    server = ServerPreset.tinygemma3()
    server.temperature = 0.0
    server.n_slots = 1
    server.n_ctx = 1024
    server.n_batch = 32
    server.debug = True  # --verbose: the buffer reuse line is trace level
    fd, server.log_path = tempfile.mkstemp(prefix="test_ckpt_reuse_", suffix=".log")
    os.close(fd)
    server.start()
    return server


def complete(server, prompt, n_predict):
    # n_probs: tinygemma3 prints the same token whatever the state, so the probabilities carry the comparison
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": n_predict, "id_slot": 0, "cache_prompt": True, "n_probs": 5,
    })
    assert res.status_code == 200
    return res.body


def run(with_spares):
    server = start_server()
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
    with open(server.log_path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    os.remove(server.log_path)
    return last, lines


def restored_index(lines):
    i = next((i for i, l in enumerate(lines) if "restored context checkpoint" in l), None)
    assert i is not None, "the last turn restored no checkpoint"
    return i


def test_restore_from_reused_checkpoint_buffer():
    reused, lines = run(with_spares=True)
    i_restore = restored_index(lines)
    pos_max = re.search(r"restored context checkpoint \(pos_min = \d+, pos_max = (\d+),", lines[i_restore]).group(1)
    # the restored checkpoint must be one that was saved into a reused buffer: the "created" line right after a
    # "reusing" line, with the restored position
    reused_saves = []
    for i, l in enumerate(lines[:i_restore]):
        if "reusing a spare context checkpoint buffer" in l:
            created = next((m for m in lines[i + 1:i_restore] if "created context checkpoint" in m), None)
            if created is not None:
                reused_saves.append(re.search(r"pos_max = (\d+),", created).group(1))
    assert pos_max in reused_saves, f"the restored checkpoint (pos_max {pos_max}) was not saved into a reused buffer"

    fresh, lines = run(with_spares=False)
    i_restore = restored_index(lines)
    assert not any("reusing a spare context checkpoint buffer" in l for l in lines[:i_restore]), \
        "the control's restored checkpoint should be in a new buffer"

    assert reused["timings"]["prompt_n"] < 64  # the restored turn read only the tail
    assert reused["content"] == fresh["content"]
    assert reused["completion_probabilities"] == fresh["completion_probabilities"]
