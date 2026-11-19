import re
import pytest
from utils import *

# A rolled-back turn must answer the same when the context checkpoint it restores was saved into
# a reused buffer (the buffer of an erased checkpoint) as when the prompt is read from scratch.
# tinygemma3 has sliding-window layers, so each rollback here restores a checkpoint.

server = ServerPreset.tinygemma3()

NOTES = " ".join(f"Entry {i}: the guard at gate {i % 7} saw trader {i * 7 % 11} sell {i % 5} bottles on day {i}."
                 for i in range(24))


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinygemma3()
    server.temperature = 0.0
    server.n_slots = 1
    server.debug = True  # --verbose: the buffer reuse line is trace level
    server.log_path = "./tmp/test_ckpt_reuse.log"


def complete(prompt, n_predict, cache_prompt=True):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": n_predict, "id_slot": 0, "cache_prompt": cache_prompt, "n_probs": 5,
    })
    assert res.status_code == 200
    return res.body


def test_restore_from_reused_checkpoint_buffer():
    global server
    server.start()
    complete(NOTES + "\nQuestion: list every entry about gate 3.\nAnswer:", 200)
    # rolls back into the notes: restores a checkpoint, erases the newer one and saves the next into its buffer
    complete(NOTES + "\nQuestion: which trader came most often?\nAnswer:", 200)
    # rolls back again: restores the checkpoint saved into the reused buffer
    last = NOTES + "\nQuestion: how many bottles in all?\nAnswer:"
    restored = complete(last, 64)
    cold = complete(last, 64, cache_prompt=False)
    server.stop()

    lines = open(server.log_path).read().splitlines()
    reuse = [i for i, l in enumerate(lines) if "reusing a spare context checkpoint buffer" in l]
    assert reuse, "no checkpoint was saved into a reused buffer"
    # the checkpoint created right after the first reuse, and its restore by the last turn
    created = next(re.search(r"created context checkpoint .*pos_max = (\d+)", l)
                   for l in lines[reuse[0]:] if "created context checkpoint" in l)
    pos_max = created.group(1)
    assert any(re.search(rf"restored context checkpoint \(pos_min = \d+, pos_max = {pos_max},", l)
               for l in lines[reuse[0]:]), f"the checkpoint in the reused buffer (pos_max {pos_max}) was never restored"

    assert restored["timings"]["prompt_n"] < 64          # the restored turn read only the tail
    assert cold["timings"]["prompt_n"] > 500             # the control read the whole prompt
    assert restored["content"] == cold["content"]
    assert restored["completion_probabilities"] == cold["completion_probabilities"]
