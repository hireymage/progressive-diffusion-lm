from scripts.check_bilingual_docs import check_bilingual_docs


def test_all_public_markdown_has_language_peer_status_and_valid_links():
    assert check_bilingual_docs() == []
