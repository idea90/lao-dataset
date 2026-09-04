"""
Lao Grammar & Linguistic Information Extractor.
Parses crawled linguistic texts, extracts grammatical patterns,
syntactic rules, classifier pairs, and vocabulary definitions.
"""

import re
from typing import Any, Dict, List, Set
from core.knowledge_base import LAO_CLASSIFIERS_TABLE, LAO_CONSONANT_CLASSES
from core.unicode_utils import contains_lao


class GrammarExtractor:
    def __init__(self):
        self.known_classifiers = {c["classifier"]: c for c in LAO_CLASSIFIERS_TABLE}

    def extract_classifier_occurrences(self, text: str) -> List[Dict[str, str]]:
        """Identify instances of classifier usage in text."""
        occurrences = []
        pattern = re.compile(r"([໐-໙0-9]+|ໜຶ່ງ|ສອງ|ສາມ|ສີ່|ຫ້າ|ຫົກ|ເຈັດ|ແປດ|ເກົ້າ|ສິບ)\s*([ກ-ໝ]{1,5})")
        for match in pattern.finditer(text):
            num, candidate_clf = match.groups()
            if candidate_clf in self.known_classifiers:
                occurrences.append({
                    "number": num,
                    "classifier": candidate_clf,
                    "usage_category": self.known_classifiers[candidate_clf]["usage"],
                    "matched_text": match.group(0)
                })
        return occurrences

    def extract_consonant_class_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract discussions of Lao consonant classes and tone rules."""
        findings = []
        for class_name, chars in LAO_CONSONANT_CLASSES.items():
            for char in chars:
                if f"ພະຍັນຊະນະ {char}" in text or f"ອັກສອນ {char}" in text or f"'{char}'" in text or f'"{char}"' in text:
                    findings.append({
                        "class": class_name,
                        "character": char,
                        "context": "Consonant class classification reference"
                    })
        return findings

    def extract_bilingual_definitions(self, text: str) -> List[Dict[str, str]]:
        """
        Extract bilingual dictionary style definitions:
        e.g. 'ກິນ (kin) - to eat', 'ຫມາ (ma) - dog', etc.
        Must contain alphabetic English definition text (excludes numbers/tables).
        """
        definitions = []
        pattern = re.compile(
            r"([ກ-ໝ\u0EB0-\u0ECB]+)\s*(?:\(([^)]+)\))?\s*[-–—:]\s*([A-Za-z0-9\s,;'/()-]{3,80})",
            re.UNICODE
        )
        for match in pattern.finditer(text):
            lao_word, romanization, defn = match.groups()
            defn_clean = defn.strip()
            # Ensure defn contains actual English words (not numbers or statistics)
            if not re.search(r"[A-Za-z]{3,}", defn_clean):
                continue

            # Exclude boilerplate words
            if any(noise in defn_clean.lower() for noise in ["cookie", "wikipedia", "url", "http"]):
                continue

            if contains_lao(lao_word, min_chars=1) and len(defn_clean) > 2:
                definitions.append({
                    "lao_word": lao_word.strip(),
                    "romanization": (romanization or "").strip(),
                    "definition": defn_clean
                })
        return definitions

    def parse_wiktionary_grammar(self, wiktionary_text: str, word: str) -> Dict[str, Any]:
        """Extract parts of speech, pronunciation, and definitions from Wiktionary explaintext."""
        result = {
            "word": word,
            "parts_of_speech": [],
            "romanization": "",
            "pronunciation": "",
            "definitions": []
        }

        # Detect IPA / Pronunciation
        ipa_match = re.search(r"IPA(?:\([^)]*\))?:\s*\[([^\]]+)\]", wiktionary_text) or re.search(r"IPA(?:\([^)]*\))?:\s*\/([^/]+)\/", wiktionary_text)
        if ipa_match:
            result["pronunciation"] = ipa_match.group(1).strip()

        # Detect romanization: e.g. "• (khan)"
        rom_match = re.search(rf"{re.escape(word)}\s*•\s*\(([^)]+)\)", wiktionary_text)
        if rom_match:
            result["romanization"] = rom_match.group(1).strip()

        # POS sections to look for
        pos_list = ["Noun", "Verb", "Adjective", "Adverb", "Classifier", "Particle", "Pronoun", "Conjunction", "Preposition"]
        pos_pattern = re.compile(r"^={2,6}\s*(" + "|".join(pos_list) + r")\b.*$", re.IGNORECASE)

        lines = wiktionary_text.split("\n")
        current_pos = None

        for line in lines:
            line_s = line.strip()
            if not line_s:
                continue

            # Check if entering a new POS section
            match_pos = pos_pattern.match(line_s)
            if match_pos:
                matched_name = match_pos.group(1).capitalize()
                current_pos = matched_name
                if current_pos not in result["parts_of_speech"]:
                    result["parts_of_speech"].append(current_pos)
                continue

            # Check if entering non-POS section (e.g. References, Etymology, Pronunciation, See also)
            if re.match(r"^={2,6}\s*(Etymology|Pronunciation|Derived terms|See also|References|Descendants|Usage notes)\b", line_s, re.IGNORECASE):
                current_pos = None
                continue

            # If inside a POS block, look for definition lines
            if current_pos:
                # Ignore word line like "ຄັນ • (khan) (classifier ຄັນ)"
                if line_s.startswith(f"{word} •") or line_s.startswith(f"{word} "):
                    continue
                # Ignore hyphenation or rhymes
                if line_s.startswith("Hyphenation:") or line_s.startswith("Rhymes:"):
                    continue

                clean_def = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", line_s).strip()
                if clean_def and len(clean_def) >= 3 and not clean_def.startswith("="):
                    tagged_def = f"[{current_pos}] {clean_def}"
                    if tagged_def not in result["definitions"]:
                        result["definitions"].append(tagged_def)

        return result
