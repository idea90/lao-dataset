"""
Configuration settings for the Lao Language and Grammar Harvester & Dataset Bot.
"""

from pathlib import Path
from typing import Dict, List

# Directory Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure runtime directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Output Dataset Files
DATASET_INSTRUCT_FILE = PROCESSED_DIR / "lao_grammar_instruct.jsonl"
DATASET_PRETRAIN_FILE = PROCESSED_DIR / "lao_pretrain_corpus.jsonl"
DATASET_RULES_FILE = PROCESSED_DIR / "lao_grammar_rules.json"
DATASET_CLASSIFIERS_FILE = PROCESSED_DIR / "lao_classifiers.json"

# Network & Crawler Settings
USER_AGENT = "LaoGrammarResearchBot/1.0 (+https://github.com/lao-linguistics-dataset; Educational and Linguistic Research)"
REQUEST_TIMEOUT = 15.0
RATE_LIMIT_DELAY = 1.0  # Respectful delay in seconds between requests
MAX_RETRIES = 3

# Wikipedia & Wikimedia Configuration
WIKIPEDIA_LAO_API = "https://lo.wikipedia.org/w/api.php"
WIKIPEDIA_EN_API = "https://en.wikipedia.org/w/api.php"
WIKTIONARY_EN_API = "https://en.wiktionary.org/w/api.php"
WIKTIONARY_LO_API = "https://lo.wiktionary.org/w/api.php"

# Wikipedia Seed Articles
WIKI_LAO_SEEDS = [
    "ພາສາລາວ",
    "ອັກສອນລາວ",
    "ໄວຍາກອນລາວ",
    "ພະຍັນຊະນະ",
    "ສະຫຼະ",
    "ວັນນະຍຸດ",
    "ລັກສະນະນາມ",
    "ປະໂຫຍກ",
    "ຄຳນາມ",
    "ຄຳກິລິຍາ",
    "ຄຳຄຸນນາມ",
    "ປະເທດລາວ",
    "ວຽງຈັນ",
    "ປະຫວັດສາດລາວ",
    "ວັດທະນະທຳລາວ"
]

WIKI_EN_SEEDS = [
    "Lao language",
    "Lao grammar",
    "Lao script",
    "Lao phonology",
    "Lao literature",
    "Comparison of Lao and Thai",
    "Languages of Laos",
    "Kra–Dai languages",
    "Southwestern Tai languages"
]

# Targeted Web Grammar & Educational Resources
CURATED_GRAMMAR_URLS = [
    "https://www.omniglot.com/writing/lao.htm",
    "https://seasite.niu.edu/lao/",
    "https://en.wikipedia.org/wiki/Lao_language",
    "https://en.wikipedia.org/wiki/Lao_grammar",
    "https://en.wikipedia.org/wiki/Lao_script",
    "https://en.wikipedia.org/wiki/Lao_phonology"
]

# Automated Search Harvester Queries (Lao & English)
SEARCH_QUERIES = [
    "ໄວຍາກອນລາວ ຫຼັກໄວຍາກອນ",
    "ລັກສະນະນາມ ພາສາລາວ ວິທີໃຊ້",
    "ກົດເກນ ຜັນສຽງ ວັນນະຍຸດ ລາວ",
    "ໂຄງສ້າງ ປະໂຫຍກ ພາສາລາວ",
    "ພະຍັນຊະນະ ສະຫຼະ ພາສາລາວ ຫຼັກການ",
    "Lao grammar rules syntax sentence structure",
    "Lao language classifiers list grammar",
    "Lao tones high middle low consonants rules",
    "Lao language learning grammar guide"
]

# Text Quality Thresholds
MIN_TEXT_CHARS = 80
MIN_LAO_CHAR_RATIO = 0.15  # At least 15% Lao unicode characters for raw Lao documents
