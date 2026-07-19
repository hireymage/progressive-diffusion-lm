import inspect

import pytest

from scripts import ablation_study, ptq_study, train_tokenizer


PINNED_WIKIPEDIA_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"


def test_ablation_generator_preserves_legacy_unregularized_protocol():
    for phase in ("screen", "full"):
        cfg = ablation_study.make_config("baseline", 42, phase)
        assert cfg["model"]["dropout"] == 0.0
        assert cfg["train"]["weight_decay"] == 0.0
        assert cfg["data"]["dataset_revision"] == PINNED_WIKIPEDIA_REVISION


def test_research_scripts_report_fp32_storage_as_32_bits():
    assert ablation_study.EFFECTIVE_BITS[16] == 32.0
    assert ptq_study.EFFECTIVE_BITS[16] == 32.0


def test_ptq_native_results_fail_closed_without_artifacts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing native ablation artifact"):
        ptq_study.load_ablation_natives()


def test_tokenizer_training_accepts_immutable_corpus_identity():
    signature = inspect.signature(train_tokenizer.train_tokenizer)
    assert "dataset_name" in signature.parameters
    assert "dataset_config" in signature.parameters
    revision = signature.parameters["dataset_revision"].default
    assert revision == PINNED_WIKIPEDIA_REVISION
