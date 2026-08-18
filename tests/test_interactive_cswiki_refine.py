from types import SimpleNamespace

import numpy as np

import scripts.interactive_cswiki_refine as refine


def test_full_sentence_refine_can_replace_input_content_but_keeps_boundaries():
    class Tokenizer:
        def encode(self, text):
            assert text == "Praha je hlavní."
            return SimpleNamespace(ids=[1, 3, 4, 5, 2])
        def decode(self, ids):
            return "/".join(map(str, ids))
        def token_to_id(self, token):
            return {"[BOS]": 1, "[EOS]": 2}[token]

    class Model:
        cfg = SimpleNamespace(n_layers=25, max_seq_len=256)
        def __call__(self, tokens, exit_layer):
            logits = np.zeros(tuple(tokens.shape) + (10,), dtype=np.float32)
            logits[..., 9] = 10
            logits[..., 0] = 1
            return refine.mx.array(logits)

    result = refine.full_sentence_refine(Model(), Tokenizer(), "Praha je hlavní.", passes=2)
    assert result["refinements"] == ["1/9/9/9/2", "1/9/9/9/2"]
    assert result["final_text"].startswith("1/") and result["final_text"].endswith("/2")


def test_parser_defaults_match_current_d64_checkpoint_shape():
    parser = refine.parser()
    args = parser.parse_args(["--cache-dir", "/tmp/cache", "--checkpoint", "/tmp/best.npz"])
    assert (args.d_model, args.d_ff, args.n_heads, args.n_layers, args.seq_len) == (64, 256, 4, 25, 256)
    assert args.route == "q2_q8_fp16"
