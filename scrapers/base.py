"""
Base scraper infrastructure with polite rate-limiting, error retries,
caching, and response normalization.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
import httpx

from config import RATE_LIMIT_DELAY, REQUEST_TIMEOUT, USER_AGENT, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LaoHarvester")


class BaseScraper:
    def __init__(self, source_name: str, rate_limit: float = RATE_LIMIT_DELAY):
        self.source_name = source_name
        self.rate_limit = rate_limit
        self.last_request_time = 0.0
        self.cache_dir = RAW_DIR / source_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": "lo,en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,*/*;q=0.8",
            },
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
        )

    def _respect_rate_limit(self):
        """Ensure polite delay between HTTP requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def get_url(self, url: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> Optional[str]:
        """Fetch URL content with retry logic and rate limiting."""
        for attempt in range(1, max_retries + 1):
            try:
                self._respect_rate_limit()
                logger.info(f"[{self.source_name}] GET {url} (attempt {attempt}/{max_retries})")
                response = self.client.get(url, params=params)
                if response.status_code == 200:
                    return response.text
                elif response.status_code in [404, 410]:
                    logger.warning(f"[{self.source_name}] HTTP {response.status_code} for {url}")
                    return None
                else:
                    logger.warning(f"[{self.source_name}] HTTP {response.status_code} for {url}. Retrying...")
            except Exception as e:
                logger.warning(f"[{self.source_name}] Request error on {url}: {e}")

            time.sleep(1.5 * attempt)

        return None

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch JSON data directly."""
        text = self.get_url(url, params=params)
        if text:
            try:
                return json.loads(text)
            except Exception as e:
                logger.error(f"[{self.source_name}] Failed to parse JSON from {url}: {e}")
        return None

    def save_cache(self, filename: str, content: Any):
        """Cache raw fetched content to disk."""
        path = self.cache_dir / filename
        try:
            if isinstance(content, (dict, list)):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(str(content))
        except Exception as e:
            logger.error(f"Failed to write cache {path}: {e}")

    def load_cache(self, filename: str) -> Optional[Any]:
        """Load content from raw cache if exists."""
        path = self.cache_dir / filename
        if path.exists():
            try:
                if filename.endswith(".json"):
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        return f.read()
            except Exception as e:
                logger.warning(f"Failed to read cache {path}: {e}")
        return None

    def close(self):
        """Close HTTP client session."""
        self.client.close()
