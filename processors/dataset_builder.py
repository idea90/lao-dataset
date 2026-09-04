"""
Dataset Builder: Transforms harvested texts, grammar definitions, and linguistic
rules into standardized AI training datasets (JSONL format for fine-tuning & pretraining).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from core.knowledge_base import LAO_GRAMMAR_RULES, LAO_CLASSIFIERS_TABLE, LAO_CONSONANT_CLASSES
from core.unicode_utils import lao_character_ratio
from processors.grammar_extractor import GrammarExtractor
from config import (
    DATASET_INSTRUCT_FILE,
    DATASET_PRETRAIN_FILE,
    DATASET_RULES_FILE,
    DATASET_CLASSIFIERS_FILE,
)

logger = logging.getLogger("DatasetBuilder")


class DatasetBuilder:
    def __init__(self):
        self.extractor = GrammarExtractor()
        self.instruct_records: List[Dict[str, Any]] = []
        self.pretrain_records: List[Dict[str, Any]] = []

    def build_rule_instruction_pairs(self) -> List[Dict[str, Any]]:
        """Synthesize instruction-tuning samples from the authoritative Lao grammar rules."""
        records = []

        # 1. Synthesize core grammar rule Q&A
        for idx, rule in enumerate(LAO_GRAMMAR_RULES, start=1):
            category = rule["category"]
            rule_name = rule["rule_name"]
            lao_name = rule["lao_name"]
            explanation = rule["explanation"]
            formula = rule["formula"]
            examples = rule["examples"]

            ex_text = "\n".join([
                f"- Lao: {ex['lao']}\n  Pronunciation: {ex.get('transcription', '')}\n  Breakdown: {ex.get('gloss', '')}\n  English: {ex.get('translation', '')}"
                for ex in examples
            ])

            # Q&A 1: Explain the rule
            inst_1 = f"Explain the rule for '{rule_name}' ({lao_name}) in Lao grammar."
            ans_1 = (
                f"### Lao Grammar: {rule_name} ({lao_name})\n\n"
                f"**Explanation:**\n{explanation}\n\n"
                f"**Pattern / Formula:**\n`{formula}`\n\n"
                f"**Examples:**\n{ex_text}"
            )
            records.append(self._format_instruct(f"rule_{idx}_explain", category, inst_1, "", ans_1))

            # Q&A 2: Lao language prompt
            inst_lao = f"ອະທິບາຍຫຼັກໄວຍາກອນກ່ຽວກັບ '{lao_name}' ພ້ອມຍົກຕົວຢ່າງປະກອບ."
            ans_lao = (
                f"### ຫຼັກໄວຍາກອນລາວ: {lao_name}\n\n"
                f"**ຄຳອະທິບາຍ:**\n{explanation}\n\n"
                f"**ໂຄງສ້າງ:**\n`{formula}`\n\n"
                f"**ຕົວຢ່າງ:**\n" + "\n".join([f"- {ex['lao']} ({ex.get('translation', '')})" for ex in examples])
            )
            records.append(self._format_instruct(f"rule_{idx}_lao", category, inst_lao, "", ans_lao))

        # 2. Synthesize Classifier Instruction Pairs
        for idx, clf in enumerate(LAO_CLASSIFIERS_TABLE, start=1):
            clf_word = clf["classifier"]
            usage = clf["usage"]
            roman = clf["transcription"]

            inst = f"In Lao grammar, what is the classifier '{clf_word}' ({roman}) used for?"
            ans = (
                f"In Lao, the classifier **'{clf_word}'** (pronounced *{roman}*) is used for: **{usage}**.\n\n"
                f"**Grammatical Formula:**\n"
                f"`[Noun] + [Number] + {clf_word}`  (e.g., to count items)\n"
                f"`[Noun] + {clf_word} + [ນີ້/ນັ້ນ]` (e.g., with demonstratives 'this/that')"
            )
            records.append(self._format_instruct(f"clf_{idx}_usage", "classifiers", inst, "", ans))

        # 3. Consonant Classes & Tone Rules
        for class_name, consonants in LAO_CONSONANT_CLASSES.items():
            cons_list = ", ".join(consonants)
            inst = f"Which consonants belong to the {class_name} class in Lao, and how do they affect tone?"
            ans = (
                f"The **{class_name}** consonants in Lao are:\n"
                f"> {cons_list}\n\n"
                f"In the Lao tone rule system, consonant classes determine the base tone of an unmarked syllable "
                f"(whether live or dead) and dictate how tone marks (ໄມ້ເອກ, ໄມ້ໂທ) modify pitch."
            )
            records.append(self._format_instruct(f"cons_{class_name[:3]}", "phonology", inst, "", ans))

        return records

    def process_wiktionary_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert scraped Wiktionary entries into vocabulary and grammatical Q&A."""
        records = []
        for idx, entry in enumerate(entries, start=1):
            word = entry.get("word", "")
            raw_text = entry.get("text", "")
            parsed = self.extractor.parse_wiktionary_grammar(raw_text, word)

            if not parsed["definitions"]:
                continue

            pos_str = ", ".join(parsed["parts_of_speech"]) if parsed["parts_of_speech"] else "Lexical item"
            pron_str = f" (IPA: /{parsed['pronunciation']}/)" if parsed["pronunciation"] else ""
            defns = "\n".join([f"- {d}" for d in parsed["definitions"][:4]])

            inst = f"Define the Lao word '{word}' and describe its grammatical category."
            ans = (
                f"**Word:** {word}{pron_str}\n"
                f"**Part of Speech:** {pos_str}\n\n"
                f"**Definitions & Usage:**\n{defns}"
            )
            records.append(self._format_instruct(f"wikt_{idx}", "lexicon", inst, "", ans))

        return records

    def process_crawled_documents(self, crawled_docs: List[Dict[str, Any]]) -> None:
        """Process harvested web & Wikipedia articles into pretraining corpus and instruction samples."""
        for idx, doc in enumerate(crawled_docs, start=1):
            text = doc.get("text", "").strip()
            if not text or len(text) < 80:
                continue

            # 1. Add to Pretraining Corpus
            title = doc.get("title", f"Doc {idx}")
            source = doc.get("source", "Web")
            url = doc.get("url", "")
            ratio = lao_character_ratio(text)

            pretrain_entry = {
                "id": f"lao_pretrain_{idx:05d}",
                "title": title,
                "source": source,
                "url": url,
                "lao_char_ratio": round(ratio, 4),
                "char_length": len(text),
                "text": text
            }
            self.pretrain_records.append(pretrain_entry)

            # 2. Extract bilingual definitions to create instruction pairs
            bilingual_pairs = self.extractor.extract_bilingual_definitions(text)
            for b_idx, bp in enumerate(bilingual_pairs[:10]):
                inst = f"What is the meaning and pronunciation of the Lao word '{bp['lao_word']}'?"
                roman = f" ({bp['romanization']})" if bp["romanization"] else ""
                ans = f"The Lao word **{bp['lao_word']}**{roman} translates to: **{bp['definition']}**."
                self.instruct_records.append(
                    self._format_instruct(f"doc_{idx}_vocab_{b_idx}", "vocabulary", inst, "", ans)
                )

    def _format_instruct(self, item_id: str, category: str, instruction: str, input_text: str, output_text: str) -> Dict[str, Any]:
        """Generate dual format: Alpaca format (instruction/input/output) + Chat format (messages)."""
        user_msg = f"{instruction}\n\n{input_text}".strip() if input_text else instruction
        return {
            "id": item_id,
            "category": category,
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output_text}
            ]
        }

    def save_all_datasets(self) -> Dict[str, int]:
        """Write all generated datasets to disk in standard AI formats."""
        # 1. Add rule-based instructions
        rule_instructions = self.build_rule_instruction_pairs()
        self.instruct_records.extend(rule_instructions)

        # Write Instruct Dataset (JSONL)
        DATASET_INSTRUCT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DATASET_INSTRUCT_FILE, "w", encoding="utf-8") as f:
            for rec in self.instruct_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Write Pretraining Corpus (JSONL)
        with open(DATASET_PRETRAIN_FILE, "w", encoding="utf-8") as f:
            for rec in self.pretrain_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Write Grammar Rules Reference (JSON)
        with open(DATASET_RULES_FILE, "w", encoding="utf-8") as f:
            json.dump(LAO_GRAMMAR_RULES, f, ensure_ascii=False, indent=2)

        # Write Classifiers Reference (JSON)
        with open(DATASET_CLASSIFIERS_FILE, "w", encoding="utf-8") as f:
            json.dump(LAO_CLASSIFIERS_TABLE, f, ensure_ascii=False, indent=2)

        stats = {
            "instruct_samples": len(self.instruct_records),
            "pretrain_documents": len(self.pretrain_records),
            "grammar_rules": len(LAO_GRAMMAR_RULES),
            "classifiers": len(LAO_CLASSIFIERS_TABLE),
        }
        logger.info(f"Datasets successfully exported: {stats}")
        return stats
