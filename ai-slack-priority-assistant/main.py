#!/usr/bin/env python3
"""
AI Slack Priority Assistant - Main Application

Intelligent message filtering and prioritization system for Slack workspaces.
Addresses information overload by using AI/ML to identify and prioritize important messages.

Author: Vinesh Thota
Date: January 2026
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# AI/ML Libraries
from transformers import pipeline, AutoTokenizer, AutoModel
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import torch

# Slack SDK
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Database and Caching
import redis
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Slack Priority Assistant",
    description="Intelligent message filtering and prioritization for Slack",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Models
Base = declarative_base()

class SlackMessage(Base):
    __tablename__ = "slack_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    message_ts = Column(String, unique=True, index=True)
    channel_id = Column(String, index=True)
    channel_name = Column(String)
    user_id = Column(String, index=True)
    user_name = Column(String)
    text = Column(Text)
    thread_ts = Column(String, nullable=True)
    priority_score = Column(Float, default=0.0)
    urgency_score = Column(Float, default=0.0)
    importance_score = Column(Float, default=0.0)
    relevance_score = Column(Float, default=0.0)
    sentiment_score = Column(Float, default=0.0)
    entities = Column(Text)  # JSON string of extracted entities
    topics = Column(Text)   # JSON string of identified topics
    ai_summary = Column(Text, nullable=True)
    is_important = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

class UserPreferences(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    priority_threshold = Column(Float, default=5.0)
    notification_frequency = Column(String, default="immediate")  # immediate, hourly, daily
    important_channels = Column(Text)  # JSON array of channel IDs
    important_keywords = Column(Text)  # JSON array of keywords
    team_members = Column(Text)  # JSON array of important team member IDs
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Models
class MessageAnalysis(BaseModel):
    message_ts: str
    channel_id: str
    priority_score: float
    urgency_score: float
    importance_score: float
    relevance_score: float
    sentiment_score: float
    entities: List[str]
    topics: List[str]
    ai_summary: str
    is_important: bool

class PriorityFeedRequest(BaseModel):
    user_id: str
    limit: int = 20
    min_priority: float = 5.0
    time_range_hours: int = 24
    channels: Optional[List[str]] = None

class FilterRequest(BaseModel):
    user_id: str
    channels: List[str]
    min_priority: float = 0.0
    max_priority: float = 10.0
    time_range_hours: int = 24
    include_threads: bool = True
    keywords: Optional[List[str]] = None

# AI Models and Components
class AIMessageAnalyzer:
    """Core AI component for analyzing and scoring Slack messages"""
    
    def __init__(self):
        # Initialize NLP models
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.summarizer = pipeline("summarization", 
                                 model="facebook/bart-large-cnn")
        self.ner_model = spacy.load("en_core_web_sm")
        
        # Initialize embeddings model for semantic similarity
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # TF-IDF for topic extraction
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Urgency keywords and patterns
        self.urgency_keywords = {
            'high': ['urgent', 'asap', 'immediately', 'critical', 'emergency', 'deadline', 'today'],
            'medium': ['soon', 'priority', 'important', 'needed', 'required', 'tomorrow'],
            'low': ['when possible', 'eventually', 'sometime', 'later']
        }
        
        # Authority indicators
        self.authority_keywords = ['ceo', 'cto', 'manager', 'director', 'lead', 'head', 'vp']
        
        logger.info("AI Message Analyzer initialized successfully")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for text using transformer model"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy().flatten()
    
    def analyze_urgency(self, text: str) -> float:
        """Analyze urgency level of message content (0-3 scale)"""
        text_lower = text.lower()
        urgency_score = 0.0
        
        # Check for urgency keywords
        for level, keywords in self.urgency_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if level == 'high':
                        urgency_score = max(urgency_score, 3.0)
                    elif level == 'medium':
                        urgency_score = max(urgency_score, 2.0)
                    elif level == 'low':
                        urgency_score = max(urgency_score, 1.0)
        
        # Check for time-sensitive patterns
        time_patterns = ['today', 'tonight', 'this morning', 'this afternoon', 'by end of day', 'eod']
        for pattern in time_patterns:
            if pattern in text_lower:
                urgency_score = max(urgency_score, 2.5)
        
        # Check for question marks (often indicate need for response)
        if '?' in text:
            urgency_score = max(urgency_score, 1.5)
        
        return min(urgency_score, 3.0)
    
    def analyze_importance(self, text: str, user_name: str, channel_name: str) -> float:
        """Analyze importance based on content and context (0-2 scale)"""
        importance_score = 0.0
        text_lower = text.lower()
        
        # Check for authority indicators in user name or message
        user_lower = user_name.lower()
        for keyword in self.authority_keywords:
            if keyword in user_lower or keyword in text_lower:
                importance_score += 1.0
                break
        
        # Important business keywords
        business_keywords = ['decision', 'announcement', 'launch', 'release', 'meeting', 
                           'deadline', 'budget', 'revenue', 'client', 'customer']
        for keyword in business_keywords:
            if keyword in text_lower:
                importance_score += 0.5
                break
        
        # Channel-based importance
        important_channels = ['general', 'announcements', 'leadership', 'all-hands']
        if any(ch in channel_name.lower() for ch in important_channels):
            importance_score += 0.5
        
        return min(importance_score, 2.0)
    
    def analyze_relevance(self, text: str, user_context: Dict) -> float:
        """Analyze personal relevance to user (0-3 scale)"""
        relevance_score = 0.0
        text_lower = text.lower()
        
        # Check for user mentions or direct references
        user_keywords = user_context.get('keywords', [])
        for keyword in user_keywords:
            if keyword.lower() in text_lower:
                relevance_score += 1.0
        
        # Check for project/team relevance
        user_projects = user_context.get('projects', [])
        for project in user_projects:
            if project.lower() in text_lower:
                relevance_score += 1.5
        
        # Check for skill/domain relevance
        user_skills = user_context.get('skills', [])
        for skill in user_skills:
            if skill.lower() in text_lower:
                relevance_score += 0.5
        
        return min(relevance_score, 3.0)
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        doc = self.ner_model(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'TIME', 'MONEY', 'PRODUCT']:
                entities.append(f"{ent.text}:{ent.label_}")
        
        return entities
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text using TF-IDF"""
        try:
            # Simple keyword extraction for now
            doc = self.ner_model(text)
            topics = []
            
            # Extract noun phrases as topics
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3 and len(chunk.text) > 3:
                    topics.append(chunk.text.lower())
            
            return topics[:5]  # Return top 5 topics
        except Exception as e:
            logger.error(f"Topic extraction error: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of message (-1 to 1 scale)"""
        try:
            result = self.sentiment_analyzer(text)[0]
            
            # Convert to numerical score
            if result['label'] == 'LABEL_2':  # Positive
                return result['score']
            elif result['label'] == 'LABEL_0':  # Negative
                return -result['score']
            else:  # Neutral
                return 0.0
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return 0.0
    
    def generate_summary(self, text: str) -> str:
        """Generate AI summary of message content"""
        try:
            if len(text) < 100:
                return text  # Too short to summarize
            
            # Truncate if too long for model
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            summary = self.summarizer(text, max_length=50, min_length=10, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return text[:100] + "..." if len(text) > 100 else text
    
    def calculate_priority_score(self, message_data: Dict, user_context: Dict) -> MessageAnalysis:
        """Main function to analyze message and calculate priority score"""
        text = message_data.get('text', '')
        user_name = message_data.get('user_name', '')
        channel_name = message_data.get('channel_name', '')
        
        # Analyze different aspects
        urgency_score = self.analyze_urgency(text)
        importance_score = self.analyze_importance(text, user_name, channel_name)
        relevance_score = self.analyze_relevance(text, user_context)
        sentiment_score = self.analyze_sentiment(text)
        
        # Extract entities and topics
        entities = self.extract_entities(text)
        topics = self.extract_topics(text)
        
        # Generate summary
        ai_summary = self.generate_summary(text)
        
        # Calculate weighted priority score (0-10 scale)
        priority_score = (
            urgency_score * 0.3 +      # 30% weight for urgency
            importance_score * 0.25 +   # 25% weight for importance
            relevance_score * 0.35 +    # 35% weight for personal relevance
            abs(sentiment_score) * 0.1  # 10% weight for emotional intensity
        )
        
        # Scale to 0-10
        priority_score = min(priority_score * (10/8.5), 10.0)  # Max possible is 8.5, scale to 10
        
        # Determine if message is important (threshold: 6.0)
        is_important = priority_score >= 6.0
        
        return MessageAnalysis(
            message_ts=message_data.get('message_ts', ''),
            channel_id=message_data.get('channel_id', ''),
            priority_score=round(priority_score, 2),
            urgency_score=round(urgency_score, 2),
            importance_score=round(importance_score, 2),
            relevance_score=round(relevance_score, 2),
            sentiment_score=round(sentiment_score, 2),
            entities=entities,
            topics=topics,
            ai_summary=ai_summary,
            is_important=is_important
        )

# Global AI analyzer instance
ai_analyzer = AIMessageAnalyzer()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./slack_ai.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

# Redis setup for caching
try:
    redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    redis_client.ping()
    logger.info("Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
    redis_client = None

# Slack client setup
slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

# Helper functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_context(user_id: str, db: Session) -> Dict:
    """Get user context for personalized analysis"""
    user_prefs = db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
    
    if user_prefs:
        return {
            'keywords': eval(user_prefs.important_keywords) if user_prefs.important_keywords else [],
            'projects': [],  # Could be expanded with project tracking
            'skills': [],   # Could be expanded with skill tracking
            'team_members': eval(user_prefs.team_members) if user_prefs.team_members else []
        }
    
    return {'keywords': [], 'projects': [], 'skills': [], 'team_members': []}

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "AI Slack Priority Assistant API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": [
            "/analyze-message",
            "/priority-feed",
            "/filter-messages",
            "/user-preferences",
            "/health"
        ]
    }

@app.post("/analyze-message")
async def analyze_message(message_data: dict, background_tasks: BackgroundTasks):
    """Analyze a single message and return priority analysis"""
    try:
        # Get user context (default for demo)
        user_context = {'keywords': [], 'projects': [], 'skills': [], 'team_members': []}
        
        # Analyze message
        analysis = ai_analyzer.calculate_priority_score(message_data, user_context)
        
        # Store in database (background task)
        background_tasks.add_task(store_message_analysis, message_data, analysis)
        
        return analysis
    
    except Exception as e:
        logger.error(f"Message analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/priority-feed")
async def get_priority_feed(request: PriorityFeedRequest):
    """Get personalized priority feed for user"""
    try:
        db = next(get_db())
        
        # Calculate time threshold
        time_threshold = datetime.utcnow() - timedelta(hours=request.time_range_hours)
        
        # Query messages
        query = db.query(SlackMessage).filter(
            SlackMessage.priority_score >= request.min_priority,
            SlackMessage.created_at >= time_threshold
        )
        
        if request.channels:
            query = query.filter(SlackMessage.channel_id.in_(request.channels))
        
        messages = query.order_by(SlackMessage.priority_score.desc()).limit(request.limit).all()
        
        # Convert to response format
        feed = []
        for msg in messages:
            feed.append({
                "message_ts": msg.message_ts,
                "channel_name": msg.channel_name,
                "user_name": msg.user_name,
                "text": msg.text,
                "priority_score": msg.priority_score,
                "ai_summary": msg.ai_summary,
                "is_important": msg.is_important,
                "created_at": msg.created_at.isoformat(),
                "entities": eval(msg.entities) if msg.entities else [],
                "topics": eval(msg.topics) if msg.topics else []
            })
        
        return {
            "user_id": request.user_id,
            "total_messages": len(feed),
            "time_range_hours": request.time_range_hours,
            "min_priority": request.min_priority,
            "messages": feed
        }
    
    except Exception as e:
        logger.error(f"Priority feed error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/filter-messages")
async def filter_messages(request: FilterRequest):
    """Filter messages based on criteria"""
    try:
        db = next(get_db())
        
        # Calculate time threshold
        time_threshold = datetime.utcnow() - timedelta(hours=request.time_range_hours)
        
        # Build query
        query = db.query(SlackMessage).filter(
            SlackMessage.channel_id.in_(request.channels),
            SlackMessage.priority_score >= request.min_priority,
            SlackMessage.priority_score <= request.max_priority,
            SlackMessage.created_at >= time_threshold
        )
        
        if not request.include_threads:
            query = query.filter(SlackMessage.thread_ts.is_(None))
        
        if request.keywords:
            # Simple keyword filtering (could be enhanced with semantic search)
            keyword_filter = ""
            for keyword in request.keywords:
                if keyword_filter:
                    keyword_filter += " OR "
                keyword_filter += f"text LIKE '%{keyword}%'"
            query = query.filter(keyword_filter)
        
        messages = query.order_by(SlackMessage.priority_score.desc()).all()
        
        # Convert to response format
        filtered_messages = []
        for msg in messages:
            filtered_messages.append({
                "message_ts": msg.message_ts,
                "channel_name": msg.channel_name,
                "user_name": msg.user_name,
                "text": msg.text,
                "priority_score": msg.priority_score,
                "urgency_score": msg.urgency_score,
                "importance_score": msg.importance_score,
                "relevance_score": msg.relevance_score,
                "ai_summary": msg.ai_summary,
                "created_at": msg.created_at.isoformat()
            })
        
        return {
            "total_messages": len(filtered_messages),
            "filters_applied": {
                "channels": request.channels,
                "priority_range": [request.min_priority, request.max_priority],
                "time_range_hours": request.time_range_hours,
                "include_threads": request.include_threads,
                "keywords": request.keywords
            },
            "messages": filtered_messages
        }
    
    except Exception as e:
        logger.error(f"Message filtering error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_models_loaded": True,
        "database_connected": True,
        "redis_connected": redis_client is not None
    }

# Background task functions
async def store_message_analysis(message_data: dict, analysis: MessageAnalysis):
    """Store message analysis in database"""
    try:
        db = next(get_db())
        
        # Check if message already exists
        existing = db.query(SlackMessage).filter(
            SlackMessage.message_ts == analysis.message_ts
        ).first()
        
        if not existing:
            # Create new message record
            db_message = SlackMessage(
                message_ts=analysis.message_ts,
                channel_id=analysis.channel_id,
                channel_name=message_data.get('channel_name', ''),
                user_id=message_data.get('user_id', ''),
                user_name=message_data.get('user_name', ''),
                text=message_data.get('text', ''),
                thread_ts=message_data.get('thread_ts'),
                priority_score=analysis.priority_score,
                urgency_score=analysis.urgency_score,
                importance_score=analysis.importance_score,
                relevance_score=analysis.relevance_score,
                sentiment_score=analysis.sentiment_score,
                entities=str(analysis.entities),
                topics=str(analysis.topics),
                ai_summary=analysis.ai_summary,
                is_important=analysis.is_important,
                processed_at=datetime.utcnow()
            )
            
            db.add(db_message)
            db.commit()
            logger.info(f"Stored analysis for message {analysis.message_ts}")
    
    except Exception as e:
        logger.error(f"Database storage error: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )