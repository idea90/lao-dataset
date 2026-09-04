# 🇱🇦 Lao Language & Grammar AI Dataset Harvester & Knowledge Service

An automated Python system designed to crawl, harvest, clean, augment, and compile comprehensive linguistic data and grammar rules for the **Lao Language (ພາສາລາວ)** into:
1. **GitHub Pages Web Portal & Static API (`docs/`)**: Host 100% free on GitHub Pages with zero servers or maintenance.
2. **AI-Ready Local Web Service (`server.py`)**: Real-time FastAPI server with live grammar verification and `/llms.txt`.
3. **Dedicated Data Processing & Augmentation Pipeline (`process_data.py`)**: Cleans, deduplicates, and splits data into train/val/test partitions.
4. **PyTorch Fine-Tuning Engine (`train_pytorch.py` & `inference.py`)**: Optional LoRA fine-tuning for local models.

---

## 🌟 GitHub Pages Deployment Guide

The static site and API have been built into the [`docs/`](file:///C:/Users/advice/Downloads/CODE/lao-dataset/docs) folder. Once pushed to GitHub, it is completely free, 100% serverless, and accessible by any LLM.

### 1. Push to Your GitHub Repository

Create a new repository on [GitHub.com](https://github.com/new) (e.g. named `lao-dataset`), then link and push:

```bash
# Rename branch to main
git branch -M main

# Add your GitHub remote URL (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/lao-dataset.git

# Push the codebase
git push -u origin main
```

### 2. Enable GitHub Pages in 2 Clicks
1. Go to your repository on GitHub: `https://github.com/YOUR_USERNAME/lao-dataset`
2. Click **Settings** (top tabs) &rarr; **Pages** (left sidebar).
3. Under **Build and deployment** &rarr; **Branch**:
   - Select branch: `main`
   - Select folder: `/docs`
   - Click **Save**.

Your site will be live within ~60 seconds at:
```
https://YOUR_USERNAME.github.io/lao-dataset/
```

### 3. Endpoints Available to LLMs on GitHub Pages

| Resource | URL Path | What LLMs Get |
| :--- | :--- | :--- |
| **LLM Index** | `/llms.txt` | Standard index of Lao language knowledge base |
| **Full LLM Manual** | `/llms-full.txt` | Complete Lao grammar manual in Markdown for context ingestion |
| **Classifiers API** | `/api/classifiers.json` | JSON catalog of all numerical classifiers (`ລັກສະນະນາມ`) |
| **Grammar Rules API** | `/api/rules.json` | Formal SVO syntax, negation, and question rules in JSON |
| **Tone Determination API** | `/api/tones.json` | 6-tone determination rules for High, Middle, and Low consonants |
| **Consonant Classes API** | `/api/consonants.json` | High, Middle, Low consonant classification |

---

## 📁 Directory Structure

```
lao-dataset/
├── docs/                      # 🌐 GitHub Pages Root Directory
│   ├── .nojekyll              # Bypasses Jekyll processing on GitHub Pages
│   ├── index.html             # Modern interactive UI with instant search
│   ├── llms.txt               # LLM standard discovery file
│   ├── llms-full.txt          # Full Lao grammar manual for LLMs
│   └── api/                   # Static REST JSON APIs for AI Agents
│       ├── rules.json         # All grammar rules
│       ├── classifiers.json   # All classifiers
│       ├── tones.json         # Tone matrix
│       └── consonants.json    # Consonant classes
├── build_pages.py             # Script that generates/updates docs/ bundle
├── server.py                  # Local FastAPI service with live validation
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
└── data/
    └── final/                 # Curated AI datasets & train/val/test splits
```

---

## 🔄 Rebuilding the GitHub Pages Bundle
Whenever you harvest more grammar rules or update classifiers, regenerate the static bundle:
```bash
python build_pages.py
git add docs/
git commit -m "Update GitHub Pages site and static APIs"
git push
```
