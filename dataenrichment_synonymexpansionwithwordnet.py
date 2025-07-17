!pip install keybert sentence-transformers nltk pandas
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')

import pandas as pd

# Correct URL for the BBC text classification dataset
!wget https://huggingface.co/datasets/SetFit/bbc-news/resolve/main/bbc-text.csv -O bbc-text.csv

import pandas as pd

df = pd.read_csv("bbc-text.csv")
print(df.columns)           # Should show 'category', 'text'
texts = df['text'].fillna("").tolist()
print("Loaded", len(texts), "articles. Example:\n", texts[:2])

from keybert import KeyBERT
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=embedding_model)

# Extract keywords
def extract_keywords(text, top_n=5):
    return [kw[0] for kw in kw_model.extract_keywords(text, top_n=top_n)]

# Expand keywords using WordNet
def expand_keywords(keywords):
    expanded = set(keywords)
    for kw in keywords:
        for syn in wn.synsets(kw):
            for lemma in syn.lemmas():
                word = lemma.name().replace("_", " ").lower()
                if word != kw:
                    expanded.add(word)
    return list(expanded)

from nltk.tokenize import word_tokenize
from collections import Counter

def evaluate_expansion(text, original_keywords, expanded_keywords):
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    common = [w for w, _ in Counter(tokens).most_common(20)]

    original_covered = len(set(original_keywords) & set(common)) / len(common)
    expanded_covered = len(set(expanded_keywords) & set(common)) / len(common)

    return original_covered, expanded_covered

# Run on sample
text = texts[1] # Changed to use the second text
original = extract_keywords(text, top_n=10)
expanded = expand_keywords(original)
orig_cov, exp_cov = evaluate_expansion(text, original, expanded)

print("Original Keywords:", original)
print("Expanded Keywords:", expanded[:10])
print(f"\nCoverage before: {orig_cov:.2f}, after expansion: {exp_cov:.2f}")

# Evaluate Diversity of Keywords
original_unique_count = len(set(original))
expanded_unique_count = len(set(expanded))

print(f"Number of unique original keywords: {original_unique_count}")
print(f"Number of unique expanded keywords: {expanded_unique_count}")

print("\nOriginal Keywords:", original)
print("Expanded Keywords (first 20):", list(expanded)[:20])

from sentence_transformers import util

# Get the embedding for the entire text
text_embedding = embedding_model.encode(text, convert_to_tensor=True)

# Get embeddings for the expanded keywords
expanded_keyword_embeddings = embedding_model.encode(expanded, convert_to_tensor=True)

# Calculate the average embedding of the expanded keywords
average_expanded_keyword_embedding = expanded_keyword_embeddings.mean(dim=0)

# Calculate the cosine similarity between the text embedding and the average expanded keyword embedding
similarity = util.cos_sim(text_embedding, average_expanded_keyword_embedding)

print(f"Cosine similarity between text and average expanded keyword embedding: {similarity.item():.4f}")

# Get embeddings for the original keywords
original_keyword_embeddings = embedding_model.encode(original, convert_to_tensor=True)

# Calculate the average embedding of the original keywords
average_original_keyword_embedding = original_keyword_embeddings.mean(dim=0)

# Calculate the cosine similarity between the text embedding and the average original keyword embedding
original_similarity = util.cos_sim(text_embedding, average_original_keyword_embedding)

print(f"Cosine similarity between text and average original keyword embedding: {original_similarity.item():.4f}")
print(f"Cosine similarity between text and average expanded keyword embedding: {similarity.item():.4f}") # Display expanded similarity again for comparison
