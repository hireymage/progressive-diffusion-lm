import json
import sys
from types import SimpleNamespace

import numpy as np

import src.data as data_module


class _Encoding:
    ids = [1, 2, 3, 4]


class _Tokenizer:
    def token_to_id(self, _token):
        return 2

    def encode(self, _text):
        return _Encoding()


def test_stream_forwards_dataset_identity_and_revision(monkeypatch):
    observed = {"closed": False}

    class ClosingDataset:
        def __iter__(self):
            def rows():
                try:
                    while True:
                        yield {"text": "example"}
                finally:
                    observed["closed"] = True

            self.iterator = rows()
            return self.iterator

    def fake_load_dataset(name, config, **kwargs):
        observed.update(name=name, config=config, kwargs=kwargs)
        return ClosingDataset()

    monkeypatch.setattr(data_module, "load_tokenizer", lambda _: _Tokenizer())
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    tokens = list(
        data_module.stream_wikipedia_tokens(
            tokenizer_path="unused",
            dataset_name="example/corpus",
            dataset_config="v1",
            dataset_revision="deadbeef",
            max_articles=1,
        )
    )

    assert tokens
    assert observed == {
        "closed": True,
        "name": "example/corpus",
        "config": "v1",
        "kwargs": {"split": "train", "streaming": True, "revision": "deadbeef"},
    }


def test_cache_identity_and_metadata_include_all_provenance(monkeypatch, tmp_path):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text('{"version":"test"}')
    builds = []

    def fake_stream(**kwargs):
        builds.append(kwargs)
        return iter([[1, 2, 3, 4, 5, 6, 7, 8]])

    monkeypatch.setattr(data_module, "stream_wikipedia_tokens", fake_stream)
    common = dict(
        tokenizer_path=str(tokenizer),
        cache_dir=str(tmp_path / "cache"),
        seq_len=4,
        max_articles=1,
        max_text_bytes=100,
        dataset_name="example/corpus",
        dataset_config="v1",
        dataset_revision="rev-a",
        train_split=0.5,
        seed=7,
    )

    first = data_module.build_and_cache_dataset(**common)
    second = data_module.build_and_cache_dataset(**common)
    assert len(builds) == 1
    assert np.array_equal(first[0], second[0])

    changed = dict(common, dataset_revision="rev-b")
    data_module.build_and_cache_dataset(**changed)
    assert len(builds) == 2

    meta_files = sorted((tmp_path / "cache").glob("meta_*.json"))
    assert len(meta_files) == 2
    metadata = [json.loads(path.read_text()) for path in meta_files]
    for meta in metadata:
        assert meta["dataset_name"] == "example/corpus"
        assert meta["dataset_config"] == "v1"
        assert meta["dataset_revision"] in {"rev-a", "rev-b"}
        assert meta["train_split"] == 0.5
        assert meta["seed"] == 7
        assert len(meta["tokenizer_sha256"]) == 64
        assert len(meta["train_sha256"]) == 64
        assert len(meta["val_sha256"]) == 64
