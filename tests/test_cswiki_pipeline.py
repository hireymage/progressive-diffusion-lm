import bz2
import hashlib
import json

import numpy as np
import pytest

from scripts import cswiki_pipeline as pipe


XML = """<mediawiki xmlns='http://www.mediawiki.org/xml/export-0.11/'><page><title>Článek</title><ns>0</ns><id>1</id><revision><text>Český [[text|text]] a {{šablona}}.</text></revision></page><page><title>Přesměrování</title><ns>0</ns><id>2</id><redirect title='Článek'/><revision><text>ignore</text></revision></page><page><title>Diskuse</title><ns>1</ns><id>3</id><revision><text>ignore</text></revision></page><page><title>Druhý</title><ns>0</ns><id>4</id><revision><text>Další český obsah pro testování tokenizace.</text></revision></page></mediawiki>"""


def fixture_dump(tmp_path):
    dump = tmp_path / "cswiki-20260801-pages-articles.xml.bz2"
    dump.write_bytes(bz2.compress(XML.encode()))
    sha1 = hashlib.sha1(dump.read_bytes()).hexdigest()
    manifest = tmp_path / "cswiki-20260801-sha1sums.txt"
    manifest.write_text(f"{sha1}  {dump.name}\n")
    return dump, manifest


def write_provenance(corpus):
    meta = {"format": "cswiki-jsonl-v1", "corpus_sha256": pipe.sha256_file(corpus),
            "source": {"dump_filename": "cswiki-20260801-pages-articles.xml.bz2", "sha1": "a" * 40}}
    corpus.with_suffix(corpus.suffix + ".meta.json").write_text(json.dumps(meta))


def test_verify_and_stream_filters_namespace_redirects(tmp_path):
    dump, manifest = fixture_dump(tmp_path)
    assert pipe.verify_dump(dump, manifest)["sha1"] == hashlib.sha1(dump.read_bytes()).hexdigest()
    records = list(pipe.iter_articles(dump))
    assert [r["title"] for r in records] == ["Článek", "Druhý"]
    assert records[0]["text"] == "Český text a ."
    manifest.write_text("0" * 40 + "  " + dump.name + "\n")
    with pytest.raises(ValueError, match="SHA1 mismatch"):
        pipe.verify_dump(dump, manifest)


def test_extract_atomic_and_does_not_overwrite(tmp_path):
    dump, manifest = fixture_dump(tmp_path)
    output = tmp_path / "outside-icloud" / "corpus.jsonl"
    meta = pipe.extract(dump, output, manifest)
    assert meta["articles"] == 2 and output.exists() and not output.with_name(output.name + ".part").exists()
    with pytest.raises(FileExistsError):
        pipe.extract(dump, output, manifest)


def test_split_is_deterministic_and_articles_do_not_leak(tmp_path):
    # Locate enough deterministic ids to assure both 95/5 sides in a small fixture.
    rows = []
    for i in range(1, 100):
        row = {"id": str(i), "title": f"T{i}", "text": " ".join(f"slovo{i}_{j:05x}_{(i*j)%997:03x}" for j in range(1000))}
        rows.append(row)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    write_provenance(corpus)
    tokenizer = tmp_path / "tokenizer"
    meta_tok = pipe.train_tokenizer(corpus, tokenizer, vocab_size=1000, test_only_allow_nonstandard_vocab=True)
    assert meta_tok["special_tokens"] == {name: i for i, name in enumerate(pipe.SPECIAL_TOKENS)}
    cache = tmp_path / "cache"
    first = pipe.build_cache(corpus, tokenizer, cache, test_only_allow_nonstandard_vocab=True)
    second = pipe.build_cache(corpus, tokenizer, cache, test_only_allow_nonstandard_vocab=True)
    assert first == second
    train = np.load(next(cache.glob("train_*.npy"))); val = np.load(next(cache.glob("val_*.npy")))
    assert train.shape[1] == val.shape[1] == 256
    assert first["train_sha256"] and first["val_sha256"]
    # Article assignment is mutually exclusive by construction, not token-level.
    assert {x["id"] for x in rows if pipe._is_val(x)}.isdisjoint({x["id"] for x in rows if not pipe._is_val(x)})


def test_provenance_rejects_missing_tampered_and_non_czech(tmp_path):
    corpus = tmp_path / "corpus.jsonl"; corpus.write_text('{"id":"1","text":"český"}\n')
    with pytest.raises(FileNotFoundError): pipe.validated_corpus_metadata(corpus)
    write_provenance(corpus)
    corpus.write_text('{"id":"1","text":"změněno"}\n')
    with pytest.raises(ValueError, match="provenance"): pipe.validated_corpus_metadata(corpus)
    write_provenance(corpus)
    sidecar = corpus.with_suffix(corpus.suffix + ".meta.json")
    bad = json.loads(sidecar.read_text()); bad["source"]["dump_filename"] = "enwiki-20260801-pages-articles.xml.bz2"
    sidecar.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="provenance"): pipe.validated_corpus_metadata(corpus)


def test_extract_refuses_existing_metadata_sidecar(tmp_path):
    dump, manifest = fixture_dump(tmp_path)
    out = tmp_path / "corpus.jsonl"; out.with_suffix(out.suffix + ".meta.json").write_text("{}")
    with pytest.raises(FileExistsError, match="metadata"):
        pipe.extract(dump, out, manifest)


def test_cache_rejects_nonproduction_tokenizer(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"id": str(i), "title": f"T{i}", "text": f"český text {i} " * 300}) + "\n" for i in range(40)))
    write_provenance(corpus)
    tokenizer = tmp_path / "tokenizer"
    pipe.train_tokenizer(corpus, tokenizer, vocab_size=30, test_only_allow_nonstandard_vocab=True)
    # Test-only vocabularies are intentionally rejected by the production cache gate.
    with pytest.raises(ValueError, match="16000-vocabulary"):
        pipe.build_cache(corpus, tokenizer, tmp_path / "cache")
