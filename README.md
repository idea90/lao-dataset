# 🇱🇦 Lao Language & Grammar AI Dataset Harvester & Knowledge Service

An automated Python system designed to crawl, harvest, clean, augment, and compile comprehensive linguistic data and grammar rules for the **Lao Language (ພາສາລາວ)** into:
1. **An AI-Ready Web Service & Knowledge API (`server.py`)**: Allows any LLM or AI agent (ChatGPT, Claude, Gemma, DeepSeek) to fetch and understand Lao grammar rules in real time without training models or spinning up fans.
2. **Standardized `/llms.txt` & `/llms-full.txt`**: Native markdown specifications following the official LLM website scraping protocol.
3. **Dedicated Data Processing & Augmentation Pipeline (`process_data.py`)**: Cleans, deduplicates, and splits data into train/val/test splits.
4. **PyTorch Fine-Tuning Engine (`train_pytorch.py` & `inference.py`)**: Optional LoRA fine-tuning for local models.

---

## 🌟 Key Capabilities

### 🌐 1. Live Web Service & API for LLMs (`server.py`)
- **Zero Heavy Compute / No Fan Spin**: Lightweight FastAPI server consuming < 40MB RAM and 0% GPU.
- **`/llms.txt` & `/llms-full.txt`**: Allows LLMs (like Perplexity, Cursor, Claude, or ChatGPT browsing) to read the full Lao grammar rules, tone charts, and classifier catalog in dense, structured markdown.
- **Semantic Search API (`/api/search?q=...`)**: LLMs can search for rules or classifiers using natural language (e.g. "classifier for car", "how negation works in Lao").
- **Classifier Lookup API (`/api/classifiers`)**: Returns the exact Lao classifier (ລັກສະນະນາມ) and usage rules for counting nouns.
- **Live Grammar Validator (`/api/validate`)**: Analyzes Lao sentences to check SVO word order, numbers + classifiers, negation placement (`ບໍ່`), and interrogative particles.
- **Interactive Web UI**: Modern dark-themed dashboard at `http://localhost:8000` with instant live search.

### 🤖 2. Internet Harvester & Data Pipeline
- **Wikipedia & Wiktionary APIs**: Crawls linguistic entries, IPA pronunciations, parts of speech, and tonal descriptions.
- **Synthetic Task Generator**: Automatically generates Grammar Error Correction (GEC) pairs, Cloze exercises, and tone determination drills.
- **Deduplication**: Exact SHA-256 and Jaccard shingle similarity deduplication.

---

## 📁 Directory Structure

```
lao-dataset/
├── server.py                  # 🚀 FastAPI knowledge service & /llms.txt web portal
├── main.py                    # Unified crawling and building CLI tool
├── process_data.py            # Dedicated data cleaning, augmentation, & splitting tool
├── train_pytorch.py           # Optional PyTorch LoRA fine-tuning script
├── inference.py               # Interactive inference CLI and benchmark tester
├── config.py                  # URLs, seeds, search queries, and thresholds
├── requirements.txt           # Python dependencies
├── core/
│   ├── unicode_utils.py       # Lao Unicode detection, ratio scoring, NFC normalization
│   ├── cleaner.py             # Trafilatura HTML cleaner & deduplication
│   └── knowledge_base.py      # Core Lao grammar rules, tones & classifiers
├── scrapers/
│   ├── base.py                # Base HTTP scraper with rate limiting and caching
│   ├── wikipedia.py           # Lao & English MediaWiki API scraper
│   ├── wiktionary.py          # Wiktionary grammatical entries harvester
│   ├── web_crawler.py         # Polite web page crawler
│   └── search_harvester.py    # DuckDuckGo search harvester
├── processors/
│   ├── grammar_extractor.py   # Pattern recognition for classifiers and rules
│   ├── synthetic_generator.py # Synthetic GEC, Cloze, and tone drill generator
│   └── dataset_builder.py     # JSONL/JSON dataset generators
└── data/
    ├── raw/                   # Raw crawled cache (Wikipedia, Wiktionary, Web)
    ├── processed/             # Base datasets extracted from raw crawl
    └── final/                 # Final processed datasets for AI training
        ├── lao_instruct_curated.jsonl
        ├── lao_pretrain_curated.jsonl
        ├── processing_report.md
        └── splits/            # Train, val, and test splits
```

---

## 🚀 Running the Web Service for LLMs

Start the server:

```powershell
.\.venv\Scripts\python server.py
```
Or with `uvicorn` live reloading:
```powershell
.\.venv\Scripts\uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Access Points
- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **LLM Summary (`/llms.txt`)**: [http://localhost:8000/llms.txt](http://localhost:8000/llms.txt)
- **Full LLM Manual (`/llms-full.txt`)**: [http://localhost:8000/llms-full.txt](http://localhost:8000/llms-full.txt)

---

## 🤖 How Any LLM Uses This Website

### 1. Ingest via System Prompt / RAG
Give your LLM prompt:
> *"Before answering questions about Lao, read http://localhost:8000/llms-full.txt for the authoritative grammar rules, classifiers, and tone matrix."*

### 2. Tool / Function Calling
Add tools to your LLM agent:
```json
{
  "name": "lookup_lao_classifier",
  "description": "Finds the correct Lao classifier for counting or specifying an object.",
  "parameters": {
    "type": "object",
    "properties": {
      "noun": {"type": "string", "description": "The item or concept, e.g. 'car' or 'dog'"}
    }
  }
}
```
The agent calls: `GET http://localhost:8000/api/classifiers?q=car`  
Response:
```json
{
  "query": "car",
  "count": 1,
  "results": [
    {
      "classifier": "ຄັນ",
      "transcription": "khan",
      "usage": "Vehicles (cars, bicycles), umbrellas, spoons, forks"
    }
  ]
}
```

### 3. Sentence Validation
Validate any Lao sentence structure:
```bash
curl -X POST http://localhost:8000/api/validate \
  -H "Content-Type: application/json" \
  -d '{"sentence": "ຂ້ອຍມີຫມາ 2 ໂຕ"}'
```

---

## 🛠️ Data Pipeline & Local Fine-Tuning (Optional)

### Run Harvester & Process Data
```bash
# Crawl web and Wikipedia
python main.py run

# Process, clean, augment with synthetic drills, and create 80/10/10 splits
python process_data.py
```

### Optional PyTorch Fine-Tuning
If you ever want to train a dedicated offline weights adapter:
```bash
python train_pytorch.py --model-name Qwen/Qwen2.5-0.5B-Instruct --output-dir models/lao-adapter
python inference.py --base-model Qwen/Qwen2.5-0.5B-Instruct --adapter-path models/lao-adapter/final_model
```
