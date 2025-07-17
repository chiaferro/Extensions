# Extensions
# 🔍 Beyond KeyBERT: Topic Modeling Extensions

This repository contains two advanced extensions for the [KeyBERT](https://github.com/MaartenGr/KeyBERT) framework, developed as part of a Deep NLP project at the Polytechnic of Turin.

---

## 🧩 Extensions

### 1️⃣ Ablation Study – Candidate Filtering
- Dataset: CNN/DailyMail summaries (via Hugging Face)
- Goal: Evaluate KeyBERT with vs. without syntactic candidate filtering
- Output: Precision, Jaccard overlap, keyword length comparison
- Notebook: `ablation_study.ipynb`

### 2️⃣ Semantic Enrichment – WordNet Expansion
- Dataset: BBC News articles
- Goal: Improve topic coverage via synonym expansion of extracted keywords
- Output: Coverage scores, enriched keyword sets
- Notebook: `wordnet_enrichment.ipynb`

---

## 📊 Files

| File                          | Description                            |
|-------------------------------|----------------------------------------|
| `report.tex`                  | Final LaTeX report                     |
| `ablation_study.ipynb`        | Colab-ready notebook for Extension 1   |
| `wordnet_enrichment.ipynb`    | Colab-ready notebook for Extension 2   |
| `wordnet_expansion_results.csv` | Example output (expanded keywords)   |

---

## 📦 Setup & Requirements

Install dependencies:
```bash
pip install keybert sentence-transformers nltk spacy wordcloud pandas
python -m spacy download en_core_web_sm
