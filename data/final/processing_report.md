# 🇱🇦 Lao Language & Grammar AI Dataset Processing Report

Generated automatically by `process_data.py`.

---

## 📊 Summary Statistics

| Metric | Instruction Dataset | Pretraining Corpus |
| :--- | :--- | :--- |
| **Initial Records** | 79 | 29 |
| **Duplicates Removed** | 10 | (included in total dedup) |
| **Low Quality Filtered** | 0 | (included in total filtered) |
| **Synthetic Tasks Added** | 33 | N/A |
| **Final Curated Items** | **104** | **27** |
| **Estimated Total Tokens** | **19,117** | **262,480** |

---

## 🏷️ Grammar Category Distribution (Instruct Samples)

| Category | Sample Count |
| :--- | :--- |
| `lexicon` | 31 |
| `classifiers` | 15 |
| `classifier_cloze` | 15 |
| `phonology_tone_drill` | 10 |
| `grammar_error_correction` | 8 |
| `vocabulary` | 6 |
| `phonology` | 3 |
| `Syntax & Word Order` | 2 |
| `Noun Phrase Structure` | 2 |
| `Classifiers (ລັກສະນະນາມ)` | 2 |
| `Tense & Aspect Markers` | 2 |
| `Negation` | 2 |
| `Interrogatives (Questions)` | 2 |
| `Nominalization (ການສ້າງຄຳນາມ)` | 2 |
| `Phonology & Tone Determination` | 2 |

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
