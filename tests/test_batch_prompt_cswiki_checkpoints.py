import scripts.batch_prompt_cswiki_checkpoints as batch
import numpy as np
from types import SimpleNamespace


def test_generation_state_marks_limit_truncation_when_no_eos_was_generated():
    result = {"prompt_token_ids": [1, 10], "terminal_eos_id": 2,
              "max_new_tokens": 3, "continuation_token_ids": [1, 10, 7, 8, 9, 2]}
    state = batch.generation_state(result)
    assert state["stop_reason"] == "max_new_tokens"
    assert state["prematurely_ended"] is True
    assert state["generated_token_count"] == 3
    assert state["exact_prompt_preserved"] is True


def test_generation_state_marks_eos_when_model_emits_eos_in_generated_span():
    result = {"prompt_token_ids": [1, 10], "terminal_eos_id": 2,
              "max_new_tokens": 4, "continuation_token_ids": [1, 10, 7, 2, 9, 8, 2]}
    state = batch.generation_state(result)
    assert state["stop_reason"] == "eos"
    assert state["prematurely_ended"] is False
    assert state["generated_token_count"] == 1
    assert state["eos_generated_offsets"] == [1]


def test_exit_state_finds_earliest_matching_layer_and_saved_layers():
    class Model:
        cfg = SimpleNamespace(mask_token_id=lambda: 99, n_layers=4)
        def __call__(self, tokens, exit_layer):
            logits = np.zeros(tuple(tokens.shape) + (12,), dtype=np.float32)
            if exit_layer == 2:
                logits[:, 2, 7] = 10
                logits[:, 3, 1] = 10
            else:
                logits[:, 2, 7] = 10
                logits[:, 3, 8] = 10
            return batch.mx.array(logits)

    result = {"prompt_token_ids": [1, 10], "terminal_eos_id": 2,
              "max_new_tokens": 2, "continuation_token_ids": [1, 10, 7, 8, 2]}
    state = batch.exit_state(Model(), result, (2, 4))
    assert state["token_count"] == 2
    assert state["tokens"][0]["exit_layer"] == 2
    assert state["tokens"][0]["early_exited"] is True
    assert state["tokens"][1]["exit_layer"] == 4
    assert state["early_exit_token_ratio"] == 0.5
    assert state["mean_layers_saved"] == 1.0
