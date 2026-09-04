"""
Unicode and script utilities for the Lao language (ພາສາລາວ).
Handles Lao Unicode range detection, character classification,
normalization, and text quality scoring.
"""

import re
import unicodedata
from typing import Dict, List, Tuple

# Lao Unicode Block Range: U+0E80 - U+0EFF
LAO_UNICODE_START = 0x0E80
LAO_UNICODE_END = 0x0EFF

# Lao Consonant classifications for phonological & tone rules
LAO_HIGH_CONSONANTS = set("ຂສຖຜຝຫ")
LAO_MIDDLE_CONSONANTS = set("ກຈດຕບປຢອ")
LAO_LOW_CONSONANTS = set("ຄງຊຍທນພຟມຣລວຮ")
LAO_COMPOUND_CONSONANTS = {"ໝ", "ໜ", "ຫງ", "ຫຍ", "ຫນ", "ຫມ", "ຫລ", "ຫຼ", "ຫວ"}

# Tone marks (ວັນນະຍຸດ)
LAO_TONE_MARKS = {
    "\u0EC8": "ໄມ້ເອກ (Mai Ek - 1st tone mark)",
    "\u0EC9": "ໄມ້ໂທ (Mai Tho - 2nd tone mark)",
    "\u0ECA": "ໄມ້ຕີ (Mai Ti - 3rd tone mark)",
    "\u0ECB": "ໄມ້ຈັດຕະວາ (Mai Chattawa - 4th tone mark)",
}

# Lao vowels and special symbols
LAO_VOWELS = set("\u0EB0\u0EB2\u0EB3\u0EB4\u0EB5\u0EB6\u0EB7\u0EB8\u0EB9\u0EBB\u0EBC\u0EC0\u0EC1\u0EC2\u0EC3\u0EC4")
LAO_SIGNS = set("\u0EC6\u0ECD\u0EDC\u0EDD")  # Repetition sign (ໆ), Niggahit, ligature signs
LAO_NUMERALS = set("໐໑໒໓໔໕໖໗໘໙")


def is_lao_char(char: str) -> bool:
    """Check if a character falls within the Lao Unicode block."""
    if not char:
        return False
    return LAO_UNICODE_START <= ord(char) <= LAO_UNICODE_END


def lao_character_ratio(text: str) -> float:
    """
    Calculate the ratio of Lao script characters to total non-whitespace characters.
    Useful for filtering out irrelevant non-Lao noise.
    """
    if not text:
        return 0.0
    clean_chars = [c for c in text if not c.isspace()]
    if not clean_chars:
        return 0.0
    lao_count = sum(1 for c in clean_chars if is_lao_char(c))
    return lao_count / len(clean_chars)


def contains_lao(text: str, min_chars: int = 3) -> bool:
    """Check if the text contains at least `min_chars` Lao characters."""
    count = 0
    for c in text:
        if is_lao_char(c):
            count += 1
            if count >= min_chars:
                return True
    return False


def normalize_lao_text(text: str) -> str:
    """
    Canonical Unicode normalization (NFC) and cleanup for Lao script:
    - Normalizes decomposed characters into standard precomposed Lao glyphs.
    - Fixes duplicated tone marks or malformed diacritics.
    - Standardizes spaces around Lao punctuation.
    """
    if not text:
        return ""

    # NFC Normalization
    normalized = unicodedata.normalize("NFC", text)

    # Remove zero-width spaces, null bytes, invisible control marks
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "").replace("\x00", "")

    # Clean consecutive spaces (in Lao, spaces denote clause/sentence boundaries)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    # Deduplicate repeated diacritics / tone marks (e.g. ້້ -> ້)
    for tone in LAO_TONE_MARKS.keys():
        normalized = re.sub(f"{tone}+", tone, normalized)

    return normalized.strip()


def segment_lao_sentences(text: str) -> List[str]:
    """
    Segment Lao text into clause/sentence units.
    In Lao orthography, spaces typically separate clauses or sentences,
    along with full stops and newlines.
    """
    if not text:
        return []

    # Split by explicit punctuation or newlines
    raw_segments = re.split(r"[\n\r]+|[.!?।]+", text)
    sentences = []

    for seg in raw_segments:
        seg = seg.strip()
        if not seg:
            continue
        # If segment is long with multiple spaced clauses, break on multiple spaces or long clauses
        sub_clauses = re.split(r"\s{2,}", seg)
        for clause in sub_clauses:
            clause = clause.strip()
            if clause and len(clause) > 2:
                sentences.append(clause)

    return sentences


def analyze_lao_script(text: str) -> Dict[str, int]:
    """Provide detailed breakdown of Lao script composition in a text snippet."""
    consonants = 0
    vowels = 0
    tones = 0
    numerals = 0
    other_lao = 0

    for c in text:
        if not is_lao_char(c):
            continue
        if c in LAO_HIGH_CONSONANTS or c in LAO_MIDDLE_CONSONANTS or c in LAO_LOW_CONSONANTS:
            consonants += 1
        elif c in LAO_VOWELS:
            vowels += 1
        elif c in LAO_TONE_MARKS:
            tones += 1
        elif c in LAO_NUMERALS:
            numerals += 1
        else:
            other_lao += 1

    return {
        "consonants": consonants,
        "vowels": vowels,
        "tones": tones,
        "numerals": numerals,
        "other_lao": other_lao,
        "total_lao_chars": consonants + vowels + tones + numerals + other_lao,
    }
