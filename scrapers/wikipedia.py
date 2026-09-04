"""
Wikipedia Scraper for Lao language & grammar articles.
Fetches articles from both Lao Wikipedia (lo.wikipedia.org)
and English linguistic articles (en.wikipedia.org).
"""

import logging
from typing import Dict, List, Optional
from scrapers.base import BaseScraper
from config import WIKIPEDIA_LAO_API, WIKIPEDIA_EN_API, WIKI_LAO_SEEDS, WIKI_EN_SEEDS

logger = logging.getLogger("WikipediaScraper")


class WikipediaScraper(BaseScraper):
    def __init__(self):
        super().__init__(source_name="wikipedia")

    def fetch_article(self, title: str, api_url: str) -> Optional[Dict[str, str]]:
        """Fetch article plaintext and metadata via MediaWiki API."""
        cache_key = f"{'lo' if 'lo.' in api_url else 'en'}_{title.replace(' ', '_')}.json"
        cached = self.load_cache(cache_key)
        if cached:
            logger.info(f"Loaded {title} from cache.")
            return cached

        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "inprop": "url",
            "explaintext": 1,
            "titles": title,
            "redirects": 1,
        }

        data = self.get_json(api_url, params=params)
        if not data or "query" not in data or "pages" not in data["query"]:
            return None

        pages = data["query"]["pages"]
        for page_id, page in pages.items():
            if page_id == "-1":
                logger.warning(f"Wikipedia page not found: {title}")
                return None

            extract = page.get("extract", "")
            if not extract or len(extract.strip()) < 50:
                return None

            result = {
                "title": page.get("title", title),
                "page_id": page.get("pageid", ""),
                "url": page.get("fullurl", f"https://{'lo' if 'lo.' in api_url else 'en'}.wikipedia.org/wiki/{title}"),
                "source": "Wikipedia (Lao)" if "lo." in api_url else "Wikipedia (English Linguistics)",
                "language": "lo" if "lo." in api_url else "en",
                "text": extract.strip()
            }

            self.save_cache(cache_key, result)
            return result

        return None

    def search_articles(self, query: str, api_url: str, limit: int = 10) -> List[str]:
        """Search for article titles matching a keyword."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json"
        }
        data = self.get_json(api_url, params=params)
        titles = []
        if data and "query" in data and "search" in data["query"]:
            for item in data["query"]["search"]:
                titles.append(item["title"])
        return titles

    def get_category_articles(self, category: str, api_url: str, limit: int = 50) -> List[str]:
        """Fetch article titles belonging to a specific category."""
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": limit,
            "format": "json"
        }
        data = self.get_json(api_url, params=params)
        titles = []
        if data and "query" in data and "categorymembers" in data["query"]:
            for item in data["query"]["categorymembers"]:
                # Only include standard articles (namespace 0)
                if item.get("ns") == 0:
                    titles.append(item["title"])
        return titles

    def harvest_all(self, max_search_per_seed: int = 5) -> List[Dict[str, str]]:
        """
        Execute comprehensive harvesting across Lao and English Wikipedia:
        1. Seed articles
        2. Category searches
        3. Linguistic keyword queries
        """
        results = []
        visited_titles = set()

        # 1. English Wikipedia Linguistic Seeds
        logger.info("Harvesting English Wikipedia Lao linguistics articles...")
        for seed in WIKI_EN_SEEDS:
            if seed not in visited_titles:
                doc = self.fetch_article(seed, WIKIPEDIA_EN_API)
                if doc:
                    results.append(doc)
                    visited_titles.add(seed)

        # 2. Lao Wikipedia Seeds
        logger.info("Harvesting Lao Wikipedia language & grammar articles...")
        for seed in WIKI_LAO_SEEDS:
            if seed not in visited_titles:
                doc = self.fetch_article(seed, WIKIPEDIA_LAO_API)
                if doc:
                    results.append(doc)
                    visited_titles.add(seed)

        # 3. Dynamic search on Lao Wikipedia for grammar & language terms
        for search_term in ["ໄວຍາກອນ", "ຫຼັກການຂຽນ", "ວັນນະຄະດີ", "ສຳນວນລາວ"]:
            found_titles = self.search_articles(search_term, WIKIPEDIA_LAO_API, limit=max_search_per_seed)
            for title in found_titles:
                if title not in visited_titles:
                    doc = self.fetch_article(title, WIKIPEDIA_LAO_API)
                    if doc:
                        results.append(doc)
                        visited_titles.add(title)

        logger.info(f"Wikipedia harvesting completed: {len(results)} articles collected.")
        return results
