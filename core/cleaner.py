"""
Data cleaning, deduplication, and quality filtering pipeline for Lao language texts.
"""

import hashlib
import re
from typing import Dict, Optional, Set, Tuple
from bs4 import BeautifulSoup
import trafilatura

from core.unicode_utils import contains_lao, lao_character_ratio, normalize_lao_text

# Common web noise and boilerplate regex patterns
BOILERPLATE_PATTERNS = [
    re.compile(r"cookie(s)? policy|terms of service|privacy policy", re.IGNORECASE),
    re.compile(r"all rights reserved|copyright © \d{4}", re.IGNORECASE),
    re.compile(r"share this:|facebook|twitter|instagram|youtube|tiktok", re.IGNORECASE),
    re.compile(r"click here to read more|read more|advertisement|sponsored", re.IGNORECASE),
    re.compile(r"ຕິດຕາມພວກເຮົາ|ແບ່ງປັນ|ລິຂະສິດ|ນະໂຍບາຍຄວາມເປັນສ່ວນຕົວ", re.IGNORECASE),
]

# Keywords that indicate linguistic/grammar value even in English or bilingual texts
LINGUISTIC_KEYWORDS = [
    "grammar", "syntax", "phonology", "consonant", "vowel", "tone", "classifier",
    "morphology", "pronoun", "particle", "verb", "noun", "adjective", "sentence",
    "ໄວຍາກອນ", "ຫຼັກໄວຍາກອນ", "ພະຍັນຊະນະ", "ສະຫຼະ", "ວັນນະຍຸດ", "ລັກສະນະນາມ",
    "ຄຳນາມ", "ຄຳແທນນາມ", "ຄຳກິລິຍາ", "ຄຳຄຸນນາມ", "ປະໂຫຍກ", "ພາສາລາວ"
]


class TextCleaner:
    def __init__(self, min_chars: int = 50, min_lao_ratio: float = 0.15):
        self.min_chars = min_chars
        self.min_lao_ratio = min_lao_ratio
        self.seen_hashes: Set[str] = set()

    def extract_text_from_html(self, html_content: str, url: Optional[str] = None) -> str:
        """
        Extract primary article content using trafilatura with BeautifulSoup fallback.
        Trafilatura strips boilerplate, navigation, footers, and ads automatically.
        """
        if not html_content:
            return ""

        # Try trafilatura first
        extracted = trafilatura.extract(
            html_content,
            url=url,
            include_links=False,
            include_images=False,
            include_tables=True,
            favor_precision=True,
            deduplicate=True
        )

        if extracted and len(extracted.strip()) >= self.min_chars:
            return extracted

        # Fallback to BeautifulSoup clean text extraction
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        return text

    def clean_text(self, text: str) -> str:
        """Thoroughly clean, normalize, and strip noise from text."""
        if not text:
            return ""

        # Normalize unicode and whitespace
        text = normalize_lao_text(text)

        # Remove line-by-line boilerplate
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or len(line) < 3:
                continue

            # Check boilerplate
            is_noise = any(p.search(line) for p in BOILERPLATE_PATTERNS)
            if not is_noise:
                lines.append(line)

        cleaned = "\n".join(lines)
        return cleaned.strip()

    def is_valid_corpus_entry(self, text: str, is_grammar_source: bool = False) -> Tuple[bool, str]:
        """
        Determine if text is of sufficient quality and relevance to be included in the dataset.
        Returns: (is_valid, reason)
        """
        if not text or len(text) < self.min_chars:
            return False, f"Too short (< {self.min_chars} chars)"

        lao_ratio = lao_character_ratio(text)

        # If it's explicitly a linguistic or grammar source, allow bilingual text (e.g. English explaining Lao grammar)
        if is_grammar_source:
            if not contains_lao(text, min_chars=5):
                return False, "Grammar document lacks Lao script examples"
            return True, "Valid grammar reference"

        # General text must satisfy minimum Lao content ratio
        if lao_ratio < self.min_lao_ratio:
            # Check if it has strong linguistic relevance
            has_keywords = sum(1 for kw in LINGUISTIC_KEYWORDS if kw in text.lower())
            if has_keywords >= 2 and contains_lao(text, min_chars=10):
                return True, "Valid bilingual linguistic document"
            return False, f"Lao ratio too low ({lao_ratio:.2f} < {self.min_lao_ratio})"

        return True, "Valid Lao text"

    def is_duplicate(self, text: str) -> bool:
        """Check if document is duplicate based on SHA-256 fingerprint."""
        norm_key = re.sub(r"\s+", "", text)[:1000]
        hash_val = hashlib.sha256(norm_key.encode("utf-8")).hexdigest()
        if hash_val in self.seen_hashes:
            return True
        self.seen_hashes.add(hash_val)
        return False
