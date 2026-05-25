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
├── notebooks/
│   ├── 01_tfidf_baseline.ipynb
│   ├── 02_sentence_embeddings.ipynb
│   ├── 03_transformer_embeddings.ipynb
│   └── 04_comparison.ipynb
├── src/
│   └── similarity.py
├── data/
└── README.md
```

---

## Visualization

PCA and t-SNE plots to visualize how different methods
represent sentences in vector space.

---

## What I Learned

TBD — will be updated after all versions are complete.

---

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Sentence Transformers](https://www.sbert.net/)

---

## Stack

Python · scikit-learn · sentence-transformers · matplotlib

---

## Author

[Honaxen](https://github.com/Honaxen)