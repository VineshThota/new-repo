from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from datetime import datetime
import logging
import time

from slack_semantic_search import SlackSemanticSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Slack Search API",
    description="Semantic search enhancement for Slack with AI-powered features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search engine
search_engine = None

# Pydantic models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    user_id: str = Field(default="default_user")
    search_type: str = Field(default="hybrid")
    limit: int = Field(default=10, ge=1, le=50)

class SearchResult(BaseModel):
    message_id: str
    score: float
    message: str
    user: str
    channel: str
    timestamp: str
    reactions: List[str]

@app.on_event("startup")
async def startup_event():
    global search_engine
    logger.info("Initializing search engine...")
    search_engine = SlackSemanticSearch()
    search_engine.load_sample_data()
    logger.info("Search engine ready")

@app.get("/")
async def root():
    return {"message": "AI-Powered Slack Search API", "version": "1.0.0"}

@app.post("/search")
async def search_messages(request: SearchRequest):
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not ready")
    
    start_time = time.time()
    results = search_engine.search(
        query=request.query,
        user_id=request.user_id,
        search_type=request.search_type,
        limit=request.limit
    )
    search_time = (time.time() - start_time) * 1000
    
    return {
        "query": request.query,
        "total_results": len(results),
        "search_time_ms": search_time,
        "results": results
    }

@app.get("/analytics")
async def get_analytics():
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not ready")
    
    return search_engine.get_search_analytics()

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)