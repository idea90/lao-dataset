"""
Main CLI entrypoint for Lao Language and Grammar Harvester & AI Dataset Builder.
"""

import argparse
import io
import json
import logging
import sys
from pathlib import Path

# Force UTF-8 stdout and stderr encoding for Windows terminal compatibility
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from config import (
    DATASET_INSTRUCT_FILE,
    DATASET_PRETRAIN_FILE,
    DATASET_RULES_FILE,
    DATASET_CLASSIFIERS_FILE,
    RAW_DIR,
    PROCESSED_DIR
)
from core.knowledge_base import LAO_GRAMMAR_RULES, LAO_CLASSIFIERS_TABLE
from core.unicode_utils import analyze_lao_script
from scrapers.wikipedia import WikipediaScraper
from scrapers.wiktionary import WiktionaryScraper
from scrapers.web_crawler import WebCrawler
from scrapers.search_harvester import SearchHarvester
from processors.dataset_builder import DatasetBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LaoBot")


def print_banner():
    banner = """
================================================================================
   [Lao Language & Grammar Harvester & AI Dataset Bot] (ພາສາລາວ)
================================================================================
"""
    print(banner)


def cmd_crawl(args) -> list:
    """Execute scraping and web harvesting based on user-selected sources."""
    print_banner()
    logger.info("Starting crawl phase...")
    all_crawled_docs = []

    # 1. Wikipedia Scraper
    if args.source in ["all", "wiki"]:
        logger.info(">>> Initializing Wikipedia Scraper (Lao & English linguistics)...")
        wiki = WikipediaScraper()
        try:
            wiki_docs = wiki.harvest_all(max_search_per_seed=args.limit)
            all_crawled_docs.extend(wiki_docs)
            logger.info(f"Wikipedia finished: {len(wiki_docs)} articles harvested.")
        finally:
            wiki.close()

    # 2. Wiktionary Scraper
    if args.source in ["all", "wiktionary"]:
        logger.info(">>> Initializing Wiktionary Scraper (Lao lexical and grammar entries)...")
        wikt = WiktionaryScraper()
        try:
            wikt_entries = wikt.harvest_vocabulary_and_grammar(limit_per_category=args.limit)
            wikt.save_cache("wiktionary_all_entries.json", wikt_entries)
            logger.info(f"Wiktionary finished: {len(wikt_entries)} entries harvested.")
        finally:
            wikt.close()

    # 3. Curated Web Crawler
    crawler = WebCrawler()
    if args.source in ["all", "web"]:
        logger.info(">>> Initializing Curated Web Crawler (Grammar guides, SEASite, Omniglot)...")
        try:
            curated_docs = crawler.crawl_curated_resources()
            all_crawled_docs.extend(curated_docs)
            logger.info(f"Curated web crawler finished: {len(curated_docs)} documents harvested.")
        except Exception as e:
            logger.error(f"Web crawler error: {e}")

    # 4. Search Harvester (DuckDuckGo search across the internet)
    if args.source in ["all", "search"]:
        logger.info(">>> Initializing Internet Search Harvester (DuckDuckGo queries)...")
        searcher = SearchHarvester(crawler)
        try:
            search_docs = searcher.harvest(max_queries=args.limit, results_per_query=args.limit)
            all_crawled_docs.extend(search_docs)
            logger.info(f"Search harvester finished: {len(search_docs)} documents harvested.")
        except Exception as e:
            logger.error(f"Search harvester error: {e}")

    crawler.close()
    logger.info(f"Total crawled documents collected in this run: {len(all_crawled_docs)}")
    return all_crawled_docs


def cmd_build(args):
    """Process all cached and harvested data to produce final AI datasets."""
    print_banner()
    logger.info("Starting dataset processing & AI compilation phase...")

    builder = DatasetBuilder()

    # Load all cached Wikipedia articles
    wiki_cache = RAW_DIR / "wikipedia"
    loaded_docs = []
    if wiki_cache.exists():
        for file in wiki_cache.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    if isinstance(doc, dict) and "text" in doc:
                        loaded_docs.append(doc)
            except Exception as e:
                logger.warning(f"Could not load {file}: {e}")

    # Load all cached Web pages
    web_cache = RAW_DIR / "web_crawl"
    if web_cache.exists():
        for file in web_cache.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    if isinstance(doc, dict) and "text" in doc:
                        loaded_docs.append(doc)
            except Exception as e:
                logger.warning(f"Could not load {file}: {e}")

    # Process Wiktionary entries
    wikt_cache = RAW_DIR / "wiktionary" / "wiktionary_all_entries.json"
    if wikt_cache.exists():
        try:
            with open(wikt_cache, "r", encoding="utf-8") as f:
                wikt_entries = json.load(f)
                if isinstance(wikt_entries, list):
                    wikt_instructs = builder.process_wiktionary_entries(wikt_entries)
                    builder.instruct_records.extend(wikt_instructs)
                    logger.info(f"Processed {len(wikt_instructs)} Wiktionary instruction samples.")
        except Exception as e:
            logger.warning(f"Error loading Wiktionary cache: {e}")

    # Process all crawled documents
    logger.info(f"Processing {len(loaded_docs)} total crawled documents...")
    builder.process_crawled_documents(loaded_docs)

    # Save final datasets
    stats = builder.save_all_datasets()

    print("\n" + "=" * 60)
    print("           DATASET BUILD COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"  • Fine-Tuning / Instruct Dataset : {DATASET_INSTRUCT_FILE} ({stats['instruct_samples']} samples)")
    print(f"  • Raw Pretraining Corpus        : {DATASET_PRETRAIN_FILE} ({stats['pretrain_documents']} docs)")
    print(f"  • Grammar Knowledge Rules Base  : {DATASET_RULES_FILE} ({stats['grammar_rules']} rules)")
    print(f"  • Lao Classifiers Reference     : {DATASET_CLASSIFIERS_FILE} ({stats['classifiers']} classifiers)")
    print("=" * 60 + "\n")


def cmd_stats(args):
    """Display detailed statistics of the built datasets."""
    print_banner()
    print("=== Lao AI Dataset Summary ===")

    if DATASET_INSTRUCT_FILE.exists():
        with open(DATASET_INSTRUCT_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"[*] Instruct Samples count: {len(lines)}")
        if lines:
            sample = json.loads(lines[0])
            print(f"    Sample 1 ID       : {sample.get('id')}")
            print(f"    Sample 1 Category : {sample.get('category')}")
            print(f"    Sample 1 Question : {sample.get('instruction')[:80]}...")
    else:
        print("[-] Instruct dataset not generated yet. Run 'build' or 'run' first.")

    if DATASET_PRETRAIN_FILE.exists():
        with open(DATASET_PRETRAIN_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"[*] Pretrain Documents count: {len(lines)}")
        total_chars = sum(len(json.loads(l).get("text", "")) for l in lines)
        print(f"    Total text characters   : {total_chars:,}")
    else:
        print("[-] Pretraining corpus not generated yet. Run 'build' or 'run' first.")

    if DATASET_RULES_FILE.exists():
        with open(DATASET_RULES_FILE, "r", encoding="utf-8") as f:
            rules = json.load(f)
        print(f"[*] Structured Grammar Rules: {len(rules)}")
        for r in rules:
            print(f"    - [{r.get('category')}] {r.get('rule_name')} ({r.get('lao_name')})")

    if DATASET_CLASSIFIERS_FILE.exists():
        with open(DATASET_CLASSIFIERS_FILE, "r", encoding="utf-8") as f:
            clfs = json.load(f)
        print(f"[*] Lao Classifiers Catalog : {len(clfs)} classifiers mapped")


def cmd_run(args):
    """End-to-end: Crawl web sources, harvest grammar rules, and compile AI datasets."""
    cmd_crawl(args)
    cmd_build(args)
    cmd_stats(args)


def main():
    parser = argparse.ArgumentParser(
        description="Lao Language & Grammar Web Harvester & AI Dataset Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Command: run (crawl + build + stats)
    run_p = subparsers.add_parser("run", help="Run end-to-end harvest and build AI datasets")
    run_p.add_argument("--source", choices=["all", "wiki", "wiktionary", "web", "search"], default="all", help="Sources to crawl")
    run_p.add_argument("--limit", type=int, default=5, help="Max items per seed or category during crawl")

    # Command: crawl
    crawl_p = subparsers.add_parser("crawl", help="Only harvest data from the web")
    crawl_p.add_argument("--source", choices=["all", "wiki", "wiktionary", "web", "search"], default="all", help="Sources to crawl")
    crawl_p.add_argument("--limit", type=int, default=5, help="Max items per seed or category")

    # Command: build
    subparsers.add_parser("build", help="Build AI datasets from cached and built-in linguistic data")

    # Command: stats
    subparsers.add_parser("stats", help="Show statistics of generated datasets")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "crawl":
        cmd_crawl(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
