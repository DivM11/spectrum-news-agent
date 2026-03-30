from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EXTRACTION_STATUS_OK = "ok"
EXTRACTION_STATUS_FAILED = "failed"
EXTRACTION_STATUS_PAYWALLED = "paywalled"
EXTRACTION_STATUS_TIMEOUT = "timeout"

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
_REQUEST_TIMEOUT = 15  # seconds


@dataclass
class ArticleExtractionResult:
    url: str
    title: str
    full_text: str | None
    word_count: int
    status: str


class ArticleExtractor:
    """Wraps newspaper4k for full-text article extraction."""

    def extract(self, url: str) -> ArticleExtractionResult:
        try:
            import newspaper  # type: ignore[import-untyped]

            config = newspaper.Config()
            config.browser_user_agent = _USER_AGENT
            config.request_timeout = _REQUEST_TIMEOUT
            config.fetch_images = False

            article = newspaper.Article(url, config=config)
            article.download()
            article.parse()

            text: str = article.text or ""
            title: str = article.title or url

            if not text:
                return ArticleExtractionResult(
                    url=url,
                    title=title,
                    full_text=None,
                    word_count=0,
                    status=EXTRACTION_STATUS_PAYWALLED,
                )

            return ArticleExtractionResult(
                url=url,
                title=title,
                full_text=text,
                word_count=len(text.split()),
                status=EXTRACTION_STATUS_OK,
            )
        except Exception as exc:
            error_str = str(exc).lower()
            if "timeout" in error_str:
                status = EXTRACTION_STATUS_TIMEOUT
            else:
                status = EXTRACTION_STATUS_FAILED
            logger.warning("Article extraction failed", extra={"url": url, "error": str(exc)})
            return ArticleExtractionResult(
                url=url,
                title=url,
                full_text=None,
                word_count=0,
                status=status,
            )

    def extract_batch(self, urls: list[str]) -> list[ArticleExtractionResult]:
        results = []
        for url in urls:
            results.append(self.extract(url))
        return results
