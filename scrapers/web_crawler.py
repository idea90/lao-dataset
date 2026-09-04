"""
Polite Web Crawler & Content Extractor for Lao language and grammar pages.
"""

import hashlib
import logging
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

from scrapers.base import BaseScraper
from core.cleaner import TextCleaner
from core.unicode_utils import contains_lao, lao_character_ratio
from config import CURATED_GRAMMAR_URLS

logger = logging.getLogger("WebCrawler")


class WebCrawler(BaseScraper):
    def __init__(self):
        super().__init__(source_name="web_crawl")
        self.cleaner = TextCleaner()
        self.visited_urls: Set[str] = set()

    def crawl_url(self, url: str) -> Optional[Dict[str, str]]:
        """Fetch a specific webpage, extract clean body text, and extract metadata."""
        if url in self.visited_urls:
            return None
        self.visited_urls.add(url)

        # Cache check using MD5 of URL
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
        cache_key = f"page_{url_hash}.json"
        cached = self.load_cache(cache_key)
        if cached:
            return cached

        html = self.get_url(url)
        if not html:
            return None

        # Extract clean text using Trafilatura & BS4
        text = self.cleaner.extract_text_from_html(html, url=url)
        cleaned_text = self.cleaner.clean_text(text)

        # Linguistic / Lao validation
        is_valid, reason = self.cleaner.is_valid_corpus_entry(cleaned_text, is_grammar_source=True)
        if not is_valid:
            logger.debug(f"Skipping {url}: {reason}")
            return None

        # Determine page title
        title = url
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
        except Exception:
            pass

        result = {
            "title": title,
            "url": url,
            "source": urlparse(url).netloc,
            "lao_ratio": round(lao_character_ratio(cleaned_text), 4),
            "text": cleaned_text
        }

        self.save_cache(cache_key, result)
        return result

    def crawl_curated_resources(self) -> List[Dict[str, str]]:
        """Crawl the curated linguistic and educational grammar URLs."""
        results = []
        logger.info(f"Crawling {len(CURATED_GRAMMAR_URLS)} curated grammar resources...")
        for url in CURATED_GRAMMAR_URLS:
            doc = self.crawl_url(url)
            if doc:
                results.append(doc)

        logger.info(f"Curated crawl finished: {len(results)} pages successfully processed.")
        return results
