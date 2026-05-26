"""
main.py
-------
FastAPI application entry point.

Run with:
    uvicorn main:app --reload
"""

from fastapi import FastAPI
from routers import search, health

app = FastAPI(
    title="Semantic Similarity Engine",
    description="Compare and search sentences by meaning, not just keywords.",
    version="1.0.0"
)

# Register routers
app.include_router(health.router, tags=["Health"])
app.include_router(search.router, prefix="/search", tags=["Search"])


@app.get("/")
async def root():
    return {
        "name": "Semantic Search Engine",
        "version": "1.0.0",
        "docs": "/docs"
    }