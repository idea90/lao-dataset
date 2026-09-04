"""
Search Harvester: Uses search queries to find Lao grammar, linguistics,
and educational articles all around the internet.
"""

import logging
from typing import List, Set
from urllib.parse import urlparse

from scrapers.base import BaseScraper
from scrapers.web_crawler import WebCrawler
from config import SEARCH_QUERIES

logger = logging.getLogger("SearchHarvester")


class SearchHarvester:
    def __init__(self, crawler: WebCrawler):
        self.crawler = crawler
        self.discovered_urls: Set[str] = set()

    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[str]:
        """Search DuckDuckGo using duckduckgo_search library with fallback."""
        urls = []
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                if results:
                    for r in results:
                        href = r.get("href")
                        if href:
                            urls.append(href)
        except Exception as e:
            logger.warning(f"DuckDuckGo search query '{query}' encountered: {e}")

        return urls

    def harvest(self, max_queries: int = 5, results_per_query: int = 4) -> List[dict]:
        """Run search queries, collect top pages, crawl their contents."""
        collected_docs = []
        logger.info("Initiating internet search harvest for Lao language & grammar...")

        queries_to_run = SEARCH_QUERIES[:max_queries]
        for query in queries_to_run:
            logger.info(f"Searching web for: {query}")
            urls = self.search_duckduckgo(query, max_results=results_per_query)

            for url in urls:
                domain = urlparse(url).netloc.lower()
                # Exclude social media or video streaming sites
                if any(bad in domain for bad in ["youtube.com", "facebook.com", "tiktok.com", "twitter.com", "x.com"]):
                    continue

                if url not in self.discovered_urls:
                    self.discovered_urls.add(url)
                    doc = self.crawler.crawl_url(url)
                    if doc:
                        collected_docs.append(doc)

        logger.info(f"Search harvest finished: {len(collected_docs)} new articles extracted.")
        return collected_docs
