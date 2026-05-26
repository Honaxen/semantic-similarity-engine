"""
health.py
---------
Health check endpoints.
Used to verify the API is running and ready.
"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check — is the API alive?"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check — is the model loaded and ready?"""
    return {
        "status": "ready",
        "model": "all-MiniLM-L6-v2",
        "timestamp": datetime.utcnow().isoformat()
    }