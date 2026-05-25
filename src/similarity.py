"""
similarity.py
-------------
Reusable semantic similarity functions.

Three approaches:
    - TF-IDF + Cosine Similarity (baseline)
    - SBERT Sentence Embeddings
    - Transformer Mean Pooling

Usage:
    from similarity import TFIDFSimilarity, SBERTSimilarity
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFSimilarity:
    """TF-IDF based sentence similarity — fast baseline."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.fitted = False

    def fit(self, corpus: list) -> None:
        """Fit vectorizer on a corpus."""
        self.vectorizer.fit(corpus)
        self.fitted = True

    def similarity(self, sentences: list) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        if not self.fitted:
            self.vectorizer.fit(sentences)
        vectors = self.vectorizer.transform(sentences)
        return cosine_similarity(vectors)

    def query(self, query: str, corpus: list, top_n: int = 3) -> list:
        """Find most similar sentences to a query."""
        if not self.fitted:
            self.vectorizer.fit(corpus)
        query_vec = self.vectorizer.transform([query])
        corpus_vecs = self.vectorizer.transform(corpus)
        scores = cosine_similarity(query_vec, corpus_vecs)[0]
        top_indices = scores.argsort()[::-1][:top_n]
        return [(corpus[i], float(scores[i])) for i in top_indices]


class SBERTSimilarity:
    """SBERT sentence embeddings — semantic similarity."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("pip install sentence-transformers")

    def encode(self, sentences: list) -> np.ndarray:
        """Encode sentences to dense vectors."""
        return self.model.encode(sentences)

    def similarity(self, sentences: list) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        embeddings = self.encode(sentences)
        return cosine_similarity(embeddings)

    def query(self, query: str, corpus: list, top_n: int = 3) -> list:
        """Find most similar sentences to a query."""
        query_embedding = self.encode([query])
        corpus_embeddings = self.encode(corpus)
        scores = cosine_similarity(query_embedding, corpus_embeddings)[0]
        top_indices = scores.argsort()[::-1][:top_n]
        return [(corpus[i], float(scores[i])) for i in top_indices]