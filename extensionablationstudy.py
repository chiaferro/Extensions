!pip install datasets keybert sentence-transformers spacy
!python -m spacy download en_core_web_sm

!pip install --upgrade datasets fsspec

import pandas as pd
from datasets import load_dataset
dataset =load_dataset("cnn_dailymail", "3.0.0", split="test", download_mode="force_redownload")

# Load 200 English news summaries
#dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
#texts = [item['highlights'] for item in dataset[:200]]
texts = [item['highlights'] for item in dataset.select(range(200))]

print(texts[:1])

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import spacy

nlp = spacy.load("en_core_web_sm")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=embedding_model)

def extract_with_candidates(text, top_n=5):
    doc = nlp(text)
    noun_phrases = list(set(chunk.text.strip() for chunk in doc.noun_chunks))
    return kw_model.extract_keywords(text, candidates=noun_phrases, top_n=top_n)

def extract_without_filter(text, top_n=5):
    return kw_model.extract_keywords(text, top_n=top_n)

#comparison example

i = 0
print(f"\nTEXT:\n{texts[i]}\n")

print(" With Candidate Filtering:\n")
for kw, score in extract_with_candidates(texts[i]):
    print(f" - {kw} ({score:.4f})")

print("\n Without Filtering:\n")
for kw, score in extract_without_filter(texts[i]):
    print(f" - {kw} ({score:.4f})")

import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

def evaluate_keywords(texts, top_n=5):
    results = []

    for text in texts[:50]:  # You can increase the range
        try:
            # Extract keywords
            filtered = [kw[0] for kw in extract_with_candidates(text, top_n=top_n)]
            unfiltered = [kw[0] for kw in extract_without_filter(text, top_n=top_n)]

            # Token overlap
            set_filt = set(filtered)
            set_unfilt = set(unfiltered)
            jaccard = len(set_filt & set_unfilt) / len(set_filt | set_unfilt) if set_filt | set_unfilt else 0

            # Noun-based ground truth (proxy)
            doc = nlp(text)
            noun_tokens = [token.text.lower() for token in doc if token.pos_ == "NOUN"]
            most_common_nouns = set([w for w, _ in Counter(noun_tokens).most_common(10)])

            # Precision@N
            precision_filt = len(set_filt & most_common_nouns) / top_n
            precision_unfilt = len(set_unfilt & most_common_nouns) / top_n

            # Average keyword length
            avg_len_filt = np.mean([len(k.split()) for k in filtered])
            avg_len_unfilt = np.mean([len(k.split()) for k in unfiltered])

            results.append({
                "precision_filt": precision_filt,
                "precision_unfilt": precision_unfilt,
                "jaccard": jaccard,
                "len_filt": avg_len_filt,
                "len_unfilt": avg_len_unfilt
            })

        except Exception:
            continue

    return pd.DataFrame(results)

# Run evaluation
metrics_df = evaluate_keywords(texts)
print(metrics_df.mean().round(3))
