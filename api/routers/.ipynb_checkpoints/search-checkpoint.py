"""
search.py
---------
Semantic Similarity endpoints.

Endpoints:
    POST /search/query  — find similar sentences to a query
    POST /search/add    — add sentences to the corpus
    GET  /search/corpus — view current corpus
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

router = APIRouter()

# Load model once at startup
model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory corpus
corpus = [
    "Machine learning models learn from data",
    "AI systems improve through training on examples",
    "The weather today is sunny and warm",
    "I love eating pizza and pasta",
    "Deep learning uses neural networks",
    "Natural language processing is fascinating",
]
corpus_embeddings = model.encode(corpus)


# --- Request/Response Models ---

class QueryRequest(BaseModel):
    query: str
    top_n: int = 3


class AddRequest(BaseModel):
    sentences: list[str]


class SearchResult(BaseModel):
    sentence: str
    score: float


class QueryResponse(BaseModel):
    query: str
    results: list[SearchResult]


# --- Endpoints ---

@router.post("/query", response_model=QueryResponse)
async def semantic_search(request: QueryRequest):
    """
    Find the most semantically similar sentences to a query.
    
    Args:
        query: The search query
        top_n: Number of results to return (default: 3)
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_embedding = model.encode([request.query])
    scores = cosine_similarity(query_embedding, corpus_embeddings)[0]

    top_indices = scores.argsort()[::-1][:request.top_n]
    results = [
        SearchResult(sentence=corpus[i], score=float(scores[i]))
        for i in top_indices
    ]

    return QueryResponse(query=request.query, results=results)


@router.post("/add")
async def add_sentences(request: AddRequest):
    """
    Add new sentences to the corpus.
    Embeddings are computed and stored immediately.
    """
    global corpus, corpus_embeddings

    if not request.sentences:
        raise HTTPException(status_code=400, detail="No sentences provided")

    new_embeddings = model.encode(request.sentences)
    corpus.extend(request.sentences)
    corpus_embeddings = np.vstack([corpus_embeddings, new_embeddings])

    return {
        "added": len(request.sentences),
        "total_corpus_size": len(corpus)
    }


@router.get("/corpus")
async def get_corpus():
    """Return all sentences currently in the corpus."""
    return {
        "total": len(corpus),
        "sentences": corpus
    }