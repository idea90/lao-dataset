"""
Wiktionary Scraper for Lao grammatical vocabulary, parts of speech, and usage.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from scrapers.base import BaseScraper
from config import WIKTIONARY_EN_API

logger = logging.getLogger("WiktionaryScraper")

WIKTIONARY_CATEGORIES = [
    "Lao classifiers",
    "Lao particles",
    "Lao pronouns",
    "Lao verbs",
    "Lao adjectives",
    "Lao nouns",
    "Lao adverbs"
]


class WiktionaryScraper(BaseScraper):
    def __init__(self):
        super().__init__(source_name="wiktionary")

    def get_category_members(self, category: str, limit: int = 40) -> List[str]:
        """Fetch member titles in a Wiktionary category."""
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": limit,
            "format": "json"
        }
        data = self.get_json(WIKTIONARY_EN_API, params=params)
        members = []
        if data and "query" in data and "categorymembers" in data["query"]:
            for item in data["query"]["categorymembers"]:
                if item.get("ns") == 0:
                    members.append(item["title"])
        return members

    def fetch_word_entry(self, word: str) -> Optional[Dict[str, Any]]:
        """Fetch raw wikitext or parsed section for a Lao word."""
        cache_key = f"word_{word}.json"
        cached = self.load_cache(cache_key)
        if cached:
            return cached

        params = {
            "action": "query",
            "prop": "extracts|revisions",
            "rvprop": "content",
            "titles": word,
            "explaintext": 1,
            "format": "json",
            "redirects": 1
        }
        data = self.get_json(WIKTIONARY_EN_API, params=params)
        if not data or "query" not in data or "pages" not in data["query"]:
            return None

        pages = data["query"]["pages"]
        for page_id, page in pages.items():
            if page_id == "-1":
                return None

            extract = page.get("extract", "")
            if not extract:
                continue

            entry = {
                "word": word,
                "url": f"https://en.wiktionary.org/wiki/{word}",
                "source": "Wiktionary",
                "text": extract.strip()
            }
            self.save_cache(cache_key, entry)
            return entry

        return None

    def harvest_vocabulary_and_grammar(self, limit_per_category: int = 25) -> List[Dict[str, Any]]:
        """Scrape structured entries across Lao grammatical categories."""
        results = []
        visited = set()

        for cat in WIKTIONARY_CATEGORIES:
            logger.info(f"Fetching Wiktionary category: {cat}")
            words = self.get_category_members(cat, limit=limit_per_category)
            for w in words:
                if w not in visited:
                    entry = self.fetch_word_entry(w)
                    if entry:
                        entry["category_tag"] = cat.replace("Lao ", "")
                        results.append(entry)
                        visited.add(w)

        logger.info(f"Wiktionary harvesting completed: {len(results)} lexical entries collected.")
        return results
