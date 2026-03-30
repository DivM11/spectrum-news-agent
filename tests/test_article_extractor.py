from __future__ import annotations

import pytest

from src.article_extractor import (
    ArticleExtractionResult,
    ArticleExtractor,
    EXTRACTION_STATUS_FAILED,
    EXTRACTION_STATUS_OK,
    EXTRACTION_STATUS_PAYWALLED,
    EXTRACTION_STATUS_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Stubs / helpers
# ---------------------------------------------------------------------------

class _StubConfig:
    """Minimal stub for newspaper.Config — accepts attribute assignment."""

    browser_user_agent: str = ""
    request_timeout: int = 10
    fetch_images: bool = True


class _StubArticle:
    def __init__(self, title: str, text: str) -> None:
        self.title = title
        self.text = text

    def download(self) -> None:
        pass

    def parse(self) -> None:
        pass


class _TimeoutStubArticle:
    title = ""
    text = ""

    def download(self) -> None:
        raise TimeoutError("timeout connecting to host")

    def parse(self) -> None:
        pass


class _ErrorStubArticle:
    title = ""
    text = ""

    def download(self) -> None:
        raise ValueError("connection refused")

    def parse(self) -> None:
        pass


class _EmptyTextArticle:
    title = "Paywalled Article"
    text = ""

    def download(self) -> None:
        pass

    def parse(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests for ArticleExtractor.extract
# ---------------------------------------------------------------------------

def test_extract_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = ArticleExtractor()
    stub = _StubArticle("Test Title", "This is a full text with many words here.")

    import newspaper
    monkeypatch.setattr(newspaper, "Config", _StubConfig)
    monkeypatch.setattr(newspaper, "Article", lambda url, config=None: stub)

    result = extractor.extract("https://example.com/article")

    assert result.status == EXTRACTION_STATUS_OK
    assert result.title == "Test Title"
    assert result.full_text is not None
    assert result.word_count > 0


def test_extract_empty_text_classified_as_paywalled(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = ArticleExtractor()
    stub = _EmptyTextArticle()

    import newspaper
    monkeypatch.setattr(newspaper, "Config", _StubConfig)
    monkeypatch.setattr(newspaper, "Article", lambda url, config=None: stub)

    result = extractor.extract("https://example.com/paywalled")

    assert result.status == EXTRACTION_STATUS_PAYWALLED
    assert result.full_text is None
    assert result.word_count == 0


def test_extract_timeout_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = ArticleExtractor()
    stub = _TimeoutStubArticle()

    import newspaper
    monkeypatch.setattr(newspaper, "Config", _StubConfig)
    monkeypatch.setattr(newspaper, "Article", lambda url, config=None: stub)

    result = extractor.extract("https://example.com/slow")

    assert result.status == EXTRACTION_STATUS_TIMEOUT
    assert result.full_text is None


def test_extract_generic_error(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = ArticleExtractor()
    stub = _ErrorStubArticle()

    import newspaper
    monkeypatch.setattr(newspaper, "Config", _StubConfig)
    monkeypatch.setattr(newspaper, "Article", lambda url, config=None: stub)

    result = extractor.extract("https://example.com/error")

    assert result.status == EXTRACTION_STATUS_FAILED
    assert result.full_text is None


def test_extract_url_preserved_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = ArticleExtractor()
    stub = _ErrorStubArticle()

    import newspaper
    monkeypatch.setattr(newspaper, "Config", _StubConfig)
    monkeypatch.setattr(newspaper, "Article", lambda url, config=None: stub)

    url = "https://example.com/article"
    result = extractor.extract(url)

    assert result.url == url


# ---------------------------------------------------------------------------
# Tests for ArticleExtractor.extract_batch
# ---------------------------------------------------------------------------

def test_extract_batch_returns_one_per_url(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = ArticleExtractor()
    stub = _StubArticle("Title", "Text text text.")

    import newspaper
    monkeypatch.setattr(newspaper, "Config", _StubConfig)
    monkeypatch.setattr(newspaper, "Article", lambda url, config=None: stub)

    urls = ["https://a.com/1", "https://b.com/2", "https://c.com/3"]
    results = extractor.extract_batch(urls)

    assert len(results) == 3


def test_extract_batch_mixed_results(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = ArticleExtractor()
    calls: list[str] = []

    def fake_article(url: str, config: object = None) -> object:
        calls.append(url)
        if "fail" in url:
            return _ErrorStubArticle()
        return _StubArticle("Title", "Some text content here.")

    import newspaper
    monkeypatch.setattr(newspaper, "Config", _StubConfig)
    monkeypatch.setattr(newspaper, "Article", fake_article)

    urls = ["https://ok.com", "https://fail.com"]
    results = extractor.extract_batch(urls)

    assert len(results) == 2
    statuses = {r.url: r.status for r in results}
    assert statuses["https://ok.com"] == EXTRACTION_STATUS_OK
    assert statuses["https://fail.com"] == EXTRACTION_STATUS_FAILED


def test_extract_batch_empty_list() -> None:
    extractor = ArticleExtractor()
    results = extractor.extract_batch([])
    assert results == []


# ---------------------------------------------------------------------------
# Tests for ArticleExtractionResult dataclass
# ---------------------------------------------------------------------------

def test_result_dataclass_fields() -> None:
    result = ArticleExtractionResult(
        url="https://x.com",
        title="Title",
        full_text="Body text",
        word_count=2,
        status=EXTRACTION_STATUS_OK,
    )
    assert result.url == "https://x.com"
    assert result.word_count == 2
    assert result.status == EXTRACTION_STATUS_OK


def test_result_full_text_can_be_none() -> None:
    result = ArticleExtractionResult(
        url="https://x.com",
        title="",
        full_text=None,
        word_count=0,
        status=EXTRACTION_STATUS_FAILED,
    )
    assert result.full_text is None
