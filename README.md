# Semantic Similarity Engine

Three approaches to measuring text similarity — from keyword matching to transformer embeddings.
Built to understand how semantic representation evolves across methods.

---

## Problem

How do we measure if two sentences mean the same thing?
"I love dogs" and "I adore puppies" — similar meaning, zero shared words.
TF-IDF fails here. Embeddings don't.

---

## Versions

| Version | Method | Captures Meaning? |
|---------|--------|-------------------|
| v1 | TF-IDF + Cosine | No — word overlap only |
| v2 | Sentence Embeddings (SBERT) | Yes — semantic similarity |
| v3 | Transformer Embeddings | Yes — context-aware |

---

## Features

- Similarity ranking — given a query, rank all sentences by similarity
- Semantic matching — find sentences with same meaning, different words
- Embedding comparison — visualize how methods differ
- Query testing — interactive similarity search

---

## Project Structure

```
semantic-similarity-engine/
├── api/
│   ├── main.py
│   └── routers/
│       ├── health.py
│       └── search.py
├── notebooks/
│   ├── 01_tfidf_baseline.ipynb
│   ├── 02_sentence_embeddings.ipynb
│   ├── 03_transformer_embeddings.ipynb
│   └── 04_comparison.ipynb
├── src/
│   └── similarity.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Visualization

PCA and t-SNE plots to visualize how different methods
represent sentences in vector space.

---

## Getting Started

### Run locally
```bash
pip install -r requirements.txt
cd api
uvicorn main:app --reload
```

### Run with Docker
```bash
docker build -t semantic-similarity-engine .
docker run -p 8000:8000 semantic-similarity-engine
```

### Run tests
```bash
pytest tests/test_api.py -v
```

---

## What I Learned

TF-IDF measures word overlap, not meaning.
Two sentences can be semantically identical but score 0.00 similarity
if they share no words — "Machine learning" vs "AI systems".

SBERT solves this by mapping sentences to a space where meaning = proximity.
Fine-tuning on sentence pairs is what makes the difference,
not just the transformer architecture itself.

Mean pooling consistently outperforms CLS token for similarity tasks
because it captures the full sentence, not just the classification token.

---

## Results

| Method | Within-Group Similarity | Between-Group Similarity |
|--------|------------------------|--------------------------|
| TF-IDF | 0.00 - 0.14 | ~0.00 |
| SBERT | 0.31 - 0.64 | ~0.05 |
| DistilBERT Mean | 0.78 - 0.90 | ~0.57 |

---

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Sentence Transformers](https://www.sbert.net/)

---

## Stack

Python · scikit-learn · sentence-transformers · matplotlib

## Usage

```python
from src.similarity import TFIDFSimilarity, SBERTSimilarity

# TF-IDF baseline
tfidf = TFIDFSimilarity()
results = tfidf.query("machine learning", corpus, top_n=3)

# SBERT semantic similarity
sbert = SBERTSimilarity()
results = sbert.query("machine learning", corpus, top_n=3)
```

---

## Author

[Honaxen](https://github.com/Honaxen)