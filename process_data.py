"""
Data Processing, Cleaning, Augmentation, and Splitting Pipeline for Lao AI Datasets.

Transforms raw and extracted Lao linguistic data into production-ready AI training sets:
- Deep Unicode normalization (NFC, diacritic cleanup)
- Exact & near-duplicate elimination (Jaccard similarity)
- Synthetic grammar task generation (GEC, Cloze, Tone drills)
- Precise token counting via tiktoken
- Train / Validation / Test stratified splitting
- Comprehensive analytical reporting (Markdown & JSON)
"""

import argparse
import hashlib
import io
import json
import logging
import math
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure UTF-8 stdout on Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODER = None

from core.unicode_utils import (
    contains_lao,
    lao_character_ratio,
    normalize_lao_text,
    analyze_lao_script
)
from processors.synthetic_generator import SyntheticGrammarGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LaoDataProcessor")


class LaoDatasetProcessor:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        min_chars: int = 50,
        min_lao_ratio: float = 0.10,
        similarity_threshold: float = 0.85,
        seed: int = 42
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_chars = min_chars
        self.min_lao_ratio = min_lao_ratio
        self.similarity_threshold = similarity_threshold
        self.seed = seed
        random.seed(seed)

        self.stats: Dict[str, Any] = {
            "initial_instruct_count": 0,
            "cleaned_instruct_count": 0,
            "augmented_instruct_count": 0,
            "initial_pretrain_count": 0,
            "cleaned_pretrain_count": 0,
            "duplicate_records_removed": 0,
            "low_quality_records_removed": 0,
            "total_pretrain_tokens": 0,
            "total_instruct_tokens": 0,
            "category_distribution": {},
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken cl100k_base or Lao character approximation."""
        if not text:
            return 0
        if ENCODER:
            try:
                return len(ENCODER.encode(text))
            except Exception:
                pass
        # Fallback approximation for Lao script (approx 2.5 chars per subword token)
        return max(1, math.ceil(len(text) / 2.5))

    def _shingles(self, text: str, k: int = 5) -> Set[str]:
        """Generate k-character shingles for near-duplicate Jaccard similarity."""
        clean = re.sub(r"\s+", "", text.lower())
        if len(clean) < k:
            return {clean}
        return {clean[i:i + k] for i in range(len(clean) - k + 1)}

    def jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two shingle sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def clean_record_text(self, text: str) -> str:
        """Apply deep normalization and symbol cleanup."""
        if not text:
            return ""
        # 1. Unicode NFC
        norm = normalize_lao_text(text)
        # 2. Strip residual HTML tags or markdown image links
        norm = re.sub(r"<[^>]+>", " ", norm)
        norm = re.sub(r"!\[.*?\]\(.*?\)", "", norm)
        # 3. Strip non-printable / control characters (except newline, tab)
        norm = "".join(ch for ch in norm if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
        # 4. Standardize quotes
        norm = norm.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        # 5. Clean whitespace
        norm = re.sub(r"[ \t]+", " ", norm)
        norm = re.sub(r"\n{3,}", "\n\n", norm)
        return norm.strip()

    def process_instruct_data(self, input_file: Path, augment: bool = True) -> List[Dict[str, Any]]:
        """Load, clean, deduplicate, and augment instruction-tuning samples."""
        if not input_file.exists():
            logger.warning(f"Instruction input file does not exist: {input_file}")
            return []

        raw_records = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        raw_records.append(json.loads(line))
                    except Exception as e:
                        logger.warning(f"Malformed JSONL line in {input_file}: {e}")

        self.stats["initial_instruct_count"] = len(raw_records)
        logger.info(f"Loaded {len(raw_records)} raw instruction records.")

        # Deduplication & Quality Filtering
        seen_instructions = set()
        seen_shingles: List[Tuple[Set[str], Dict[str, Any]]] = []
        filtered_records = []

        for item in raw_records:
            inst = self.clean_record_text(item.get("instruction", ""))
            inp = self.clean_record_text(item.get("input", ""))
            out = self.clean_record_text(item.get("output", ""))
            cat = item.get("category", "general")

            if not inst or not out:
                self.stats["low_quality_records_removed"] += 1
                continue

            # Exact deduplication on instruction
            inst_hash = hashlib.sha256(inst.lower().encode("utf-8")).hexdigest()
            if inst_hash in seen_instructions:
                self.stats["duplicate_records_removed"] += 1
                continue
            seen_instructions.add(inst_hash)

            # Near-duplicate check with Jaccard similarity
            shingles = self._shingles(inst)
            is_near_dup = False
            for prev_shingles, _ in seen_shingles:
                if self.jaccard_similarity(shingles, prev_shingles) >= self.similarity_threshold:
                    is_near_dup = True
                    break

            if is_near_dup:
                self.stats["duplicate_records_removed"] += 1
                continue

            seen_shingles.append((shingles, item))

            # Format standardized record
            token_count = self.count_tokens(inst) + self.count_tokens(inp) + self.count_tokens(out)
            user_msg = f"{inst}\n\n{inp}".strip() if inp else inst

            cleaned_item = {
                "id": item.get("id", f"inst_{len(filtered_records) + 1:04d}"),
                "category": cat,
                "instruction": inst,
                "input": inp,
                "output": out,
                "token_estimate": token_count,
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": out}
                ]
            }
            filtered_records.append(cleaned_item)

        self.stats["cleaned_instruct_count"] = len(filtered_records)

        # Augmentation Phase
        if augment:
            logger.info("Augmenting instruction dataset with synthetic grammar tasks...")
            synth = SyntheticGrammarGenerator(seed=self.seed)
            cloze_tasks = synth.generate_classifier_cloze_tasks(count=15)
            gec_tasks = synth.generate_error_correction_tasks(count=10)
            tone_tasks = synth.generate_tone_and_phonology_tasks(count=10)

            augmented = cloze_tasks + gec_tasks + tone_tasks
            for task in augmented:
                tokens = self.count_tokens(task["instruction"]) + self.count_tokens(task["output"])
                task["token_estimate"] = tokens
                filtered_records.append(task)

            self.stats["augmented_instruct_count"] = len(augmented)
            logger.info(f"Added {len(augmented)} synthetic grammar instruction tasks.")

        # Update category distributions and tokens
        for rec in filtered_records:
            cat = rec.get("category", "general")
            self.stats["category_distribution"][cat] = self.stats["category_distribution"].get(cat, 0) + 1
            self.stats["total_instruct_tokens"] += rec.get("token_estimate", 0)

        return filtered_records

    def process_pretrain_data(self, input_file: Path) -> List[Dict[str, Any]]:
        """Load, clean, and deduplicate documents for the pretraining corpus."""
        if not input_file.exists():
            logger.warning(f"Pretrain input file does not exist: {input_file}")
            return []

        raw_docs = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        raw_docs.append(json.loads(line))
                    except Exception as e:
                        logger.warning(f"Malformed JSONL line in {input_file}: {e}")

        self.stats["initial_pretrain_count"] = len(raw_docs)
        logger.info(f"Loaded {len(raw_docs)} raw pretraining documents.")

        seen_hashes = set()
        cleaned_docs = []

        for doc in raw_docs:
            raw_text = doc.get("text", "")
            clean_text = self.clean_record_text(raw_text)

            # Length filter
            if len(clean_text) < self.min_chars:
                self.stats["low_quality_records_removed"] += 1
                continue

            # Script ratio filter
            ratio = lao_character_ratio(clean_text)
            is_grammar = any(k in clean_text.lower() for k in ["grammar", "syntax", "tone", "phonology", "classifier", "ໄວຍາກອນ"])
            if ratio < self.min_lao_ratio and not is_grammar:
                self.stats["low_quality_records_removed"] += 1
                continue

            # Exact deduplication on content snippet
            fingerprint = hashlib.sha256(clean_text[:500].encode("utf-8")).hexdigest()
            if fingerprint in seen_hashes:
                self.stats["duplicate_records_removed"] += 1
                continue
            seen_hashes.add(fingerprint)

            token_count = self.count_tokens(clean_text)
            self.stats["total_pretrain_tokens"] += token_count

            cleaned_docs.append({
                "id": doc.get("id", f"pretrain_{len(cleaned_docs) + 1:05d}"),
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "url": doc.get("url", ""),
                "char_length": len(clean_text),
                "token_estimate": token_count,
                "lao_ratio": round(ratio, 4),
                "text": clean_text
            })

        self.stats["cleaned_pretrain_count"] = len(cleaned_docs)
        return cleaned_docs

    def split_dataset(
        self,
        records: List[Dict[str, Any]],
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Partition records into train, validation, and test sets."""
        if not records:
            return [], [], []

        shuffled = list(records)
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_set = shuffled[:train_end]
        val_set = shuffled[train_end:val_end]
        test_set = shuffled[val_end:]

        return train_set, val_set, test_set

    def save_splits(
        self,
        prefix: str,
        train_set: List[Dict[str, Any]],
        val_set: List[Dict[str, Any]],
        test_set: List[Dict[str, Any]]
    ):
        """Save train/val/test splits to disk in JSONL format."""
        splits_dir = self.output_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        for name, data in [("train", train_set), ("val", val_set), ("test", test_set)]:
            path = splits_dir / f"{prefix}_{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"Saved {name} split: {path} ({len(data)} items)")

    def generate_report(self) -> Tuple[Path, Path]:
        """Generate detailed JSON and Markdown analytical reports."""
        json_report_path = self.output_dir / "processing_report.json"
        md_report_path = self.output_dir / "processing_report.md"

        # Save JSON
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        # Save Markdown Report
        categories_table = "\n".join(
            f"| `{cat}` | {count} |" for cat, count in sorted(self.stats["category_distribution"].items(), key=lambda x: -x[1])
        )

        md_content = f"""# 🇱🇦 Lao Language & Grammar AI Dataset Processing Report

Generated automatically by `process_data.py`.

---

## 📊 Summary Statistics

| Metric | Instruction Dataset | Pretraining Corpus |
| :--- | :--- | :--- |
| **Initial Records** | {self.stats['initial_instruct_count']} | {self.stats['initial_pretrain_count']} |
| **Duplicates Removed** | {self.stats['duplicate_records_removed']} | (included in total dedup) |
| **Low Quality Filtered** | {self.stats['low_quality_records_removed']} | (included in total filtered) |
| **Synthetic Tasks Added** | {self.stats['augmented_instruct_count']} | N/A |
| **Final Curated Items** | **{self.stats['cleaned_instruct_count'] + self.stats['augmented_instruct_count']}** | **{self.stats['cleaned_pretrain_count']}** |
| **Estimated Total Tokens** | **{self.stats['total_instruct_tokens']:,}** | **{self.stats['total_pretrain_tokens']:,}** |

---

## 🏷️ Grammar Category Distribution (Instruct Samples)

| Category | Sample Count |
| :--- | :--- |
{categories_table}

---

## 🚀 Recommended AI Training Configurations

### Fine-Tuning (LLaMA-3, Mistral, Gemma, Qwen)
- **Target File:** `data/splits/instruct_train.jsonl`
- **Validation File:** `data/splits/instruct_val.jsonl`
- **Format:** `messages` format with User/Assistant turns
- **Recommended Hyperparameters:**
  - `learning_rate`: 2e-5 (LoRA / QLoRA)
  - `lora_r`: 16, `lora_alpha`: 32
  - `max_seq_length`: 2048
  - `epochs`: 3

---
"""
        with open(md_report_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Report exported: {md_report_path}")
        return json_report_path, md_report_path


def main():
    parser = argparse.ArgumentParser(
        description="Lao AI Dataset Processing, Augmentation, and Splitting Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", type=str, default="data/processed", help="Path to input datasets")
    parser.add_argument("--output-dir", type=str, default="data/final", help="Path to export processed datasets")
    parser.add_argument("--augment", action="store_true", default=True, help="Generate synthetic grammar error correction & cloze tasks")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Do not generate synthetic tasks")
    parser.add_argument("--split", action="store_true", default=True, help="Create train/val/test splits")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Ratio for training set")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Ratio for validation set")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Ratio for test set")
    parser.add_argument("--similarity", type=float, default=0.85, help="Near-duplicate Jaccard similarity threshold")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    processor = LaoDatasetProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        similarity_threshold=args.similarity
    )

    print("\n" + "=" * 80)
    print("      🇱🇦 LAO AI DATASET PROCESSING & AUGMENTATION PIPELINE")
    print("=" * 80 + "\n")

    # 1. Process Instruction Dataset
    instruct_file = input_dir / "lao_grammar_instruct.jsonl"
    logger.info(f"Processing instruction data from {instruct_file}...")
    cleaned_instruct = processor.process_instruct_data(instruct_file, augment=args.augment)

    # 2. Process Pretraining Corpus
    pretrain_file = input_dir / "lao_pretrain_corpus.jsonl"
    logger.info(f"Processing pretraining corpus from {pretrain_file}...")
    cleaned_pretrain = processor.process_pretrain_data(pretrain_file)

    # 3. Save Unified Clean Datasets
    final_instruct_path = output_dir / "lao_instruct_curated.jsonl"
    with open(final_instruct_path, "w", encoding="utf-8") as f:
        for item in cleaned_instruct:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Exported master instruction dataset: {final_instruct_path} ({len(cleaned_instruct)} records)")

    final_pretrain_path = output_dir / "lao_pretrain_curated.jsonl"
    with open(final_pretrain_path, "w", encoding="utf-8") as f:
        for item in cleaned_pretrain:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Exported master pretrain corpus: {final_pretrain_path} ({len(cleaned_pretrain)} docs)")

    # 4. Partition Splits
    if args.split:
        logger.info("Partitioning datasets into train, validation, and test splits...")
        train_i, val_i, test_i = processor.split_dataset(
            cleaned_instruct,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        processor.save_splits("instruct", train_i, val_i, test_i)

        train_p, val_p, test_p = processor.split_dataset(
            cleaned_pretrain,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        processor.save_splits("pretrain", train_p, val_p, test_p)

    # 5. Generate Reports
    json_rep, md_rep = processor.generate_report()

    print("\n" + "=" * 80)
    print("                     PROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"  • Curated Instruction Dataset  : {final_instruct_path} ({len(cleaned_instruct)} items)")
    print(f"  • Curated Pretrain Corpus     : {final_pretrain_path} ({len(cleaned_pretrain)} docs)")
    print(f"  • Train / Val / Test Splits    : {output_dir / 'splits'}")
    print(f"  • Quality Analytical Report    : {md_rep}")
    print(f"  • Estimated Pretrain Tokens   : {processor.stats['total_pretrain_tokens']:,}")
    print(f"  • Estimated Instruct Tokens   : {processor.stats['total_instruct_tokens']:,}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
