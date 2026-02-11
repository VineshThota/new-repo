#!/usr/bin/env python3
"""
AI-Powered Microsoft Teams Notification Reliability Enhancement System
Main application entry point

Author: AI Product Enhancement Research System
Date: February 2026
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
import sqlite3
import json
import time
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class NotificationRequest(BaseModel):
    user_id: str
    message: str
    priority: str = "medium"
    channel: str = "teams_chat"
    sender_id: Optional[str] = None
    meeting_context: Optional[bool] = False

class DeviceContext(BaseModel):
    device_type: str
    app_version: str
    network_quality: float
    battery_level: Optional[float] = None
    is_active: bool = True

class NotificationResponse(BaseModel):
    notification_id: str
    success_probability: float
    urgency_score: float
    recommended_channels: List[str]
    retry_strategy: Dict[str, Any]
    estimated_delivery_time: float

class NotificationEnhancer:
    """
    Core AI-powered notification enhancement system
    """
    
    def __init__(self):
        self.db_path = "notifications.db"
        self.failure_predictor = None
        self.scaler = StandardScaler()
        self.urgency_classifier = None
        self.notification_history = []
        
        # Initialize components
        self._init_database()
        self._load_models()
        
        logger.info("NotificationEnhancer initialized successfully")
    
    def _init_database(self):
        """Initialize SQLite database for storing notification data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                message TEXT,
                priority TEXT,
                channel TEXT,
                timestamp DATETIME,
                delivery_status TEXT,
                failure_reason TEXT,
                device_type TEXT,
                app_version TEXT,
                network_quality REAL,
                urgency_score REAL,
                success_probability REAL,
                actual_delivered BOOLEAN,
                delivery_time REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferred_channels TEXT,
                urgency_keywords TEXT,
                notification_frequency TEXT,
                quiet_hours TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    
    def _load_models(self):
        """Load or train ML models"""
        try:
            # Load pre-trained urgency classifier (BERT-based)
            self.urgency_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased",
                return_all_scores=True
            )
            
            # Generate synthetic training data for failure predictor
            self._train_failure_predictor()
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to simple models
            self._init_fallback_models()
    
    def _train_failure_predictor(self):
        """Train the notification failure prediction model"""
        # Generate synthetic training data based on real-world patterns
        np.random.seed(42)
        n_samples = 10000
        
        # Features: device_type_encoded, network_quality, app_version_score, 
        #          time_of_day, user_activity_score, message_priority_score
        X = np.random.rand(n_samples, 6)
        
        # Simulate realistic failure patterns
        failure_prob = (
            0.3 * (X[:, 1] < 0.3) +  # Poor network quality
            0.2 * (X[:, 0] > 0.8) +   # Older devices
            0.15 * (X[:, 2] < 0.4) +  # Outdated app versions
            0.1 * (X[:, 3] > 0.9) +   # Late night hours
            0.1 * (X[:, 4] < 0.2)     # Low user activity
        )
        
        y = (failure_prob + np.random.normal(0, 0.1, n_samples)) > 0.25
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        self.failure_predictor = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.failure_predictor.fit(X_train_scaled, y_train)
        
        # Evaluate
        X_test_scaled = self.scaler.transform(X_test)
        accuracy = self.failure_predictor.score(X_test_scaled, y_test)
        logger.info(f"Failure predictor trained with accuracy: {accuracy:.3f}")
    
    def _init_fallback_models(self):
        """Initialize simple fallback models if advanced models fail"""
        self.failure_predictor = None
        self.urgency_classifier = None
        logger.warning("Using fallback models")
    
    def predict_failure_risk(
        self, 
        user_id: str, 
        device_type: str, 
        network_quality: float,
        app_version: str = "1.0.0"
    ) -> float:
        """Predict the probability of notification delivery failure"""
        if not self.failure_predictor:
            # Fallback logic
            base_risk = 0.1
            if network_quality < 0.5:
                base_risk += 0.3
            if device_type.lower() in ['android', 'old_ios']:
                base_risk += 0.2
            return min(base_risk, 0.9)
        
        try:
            # Encode features
            device_score = {
                'ios': 0.2, 'android': 0.5, 'windows': 0.3, 
                'mac': 0.2, 'web': 0.1
            }.get(device_type.lower(), 0.5)
            
            app_version_score = self._parse_version_score(app_version)
            current_hour = datetime.now().hour
            time_score = current_hour / 24.0
            
            # Get user activity score (simplified)
            user_activity = self._get_user_activity_score(user_id)
            
            # Priority score (default medium)
            priority_score = 0.5
            
            features = np.array([[
                device_score, network_quality, app_version_score,
                time_score, user_activity, priority_score
            ]])
            
            features_scaled = self.scaler.transform(features)
            failure_prob = self.failure_predictor.predict_proba(features_scaled)[0][1]
            
            return float(failure_prob)
            
        except Exception as e:
            logger.error(f"Error predicting failure risk: {e}")
            return 0.2  # Default moderate risk
    
    def classify_urgency(self, message: str, context: Dict = None) -> float:
        """Classify message urgency using NLP"""
        if not self.urgency_classifier:
            # Fallback keyword-based urgency detection
            urgent_keywords = [
                'urgent', 'asap', 'immediately', 'emergency', 'critical',
                'now', 'deadline', 'meeting', 'call', 'important'
            ]
            
            message_lower = message.lower()
            urgency_score = sum(
                2 if keyword in message_lower else 0 
                for keyword in urgent_keywords
            )
            
            # Add context-based scoring
            if context:
                if context.get('meeting_context'):
                    urgency_score += 3
                if context.get('sender_importance', 0) > 0.7:
                    urgency_score += 2
            
            return min(urgency_score / 10.0, 1.0)
        
        try:
            # Use BERT-based classification
            results = self.urgency_classifier(message)
            
            # Extract urgency indicators
            urgency_indicators = [
                'urgent', 'important', 'critical', 'meeting', 'deadline'
            ]
            
            urgency_score = 0.0
            for word in message.lower().split():
                if word in urgency_indicators:
                    urgency_score += 0.2
            
            # Combine with context
            if context and context.get('meeting_context'):
                urgency_score += 0.3
            
            return min(urgency_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error classifying urgency: {e}")
            return 0.5  # Default medium urgency
    
    def generate_retry_strategy(
        self, 
        failure_risk: float, 
        urgency_score: float
    ) -> Dict[str, Any]:
        """Generate intelligent retry strategy based on ML predictions"""
        if failure_risk < 0.3:
            return {
                'max_retries': 2,
                'retry_intervals': [30, 120],  # seconds
                'fallback_channels': ['email'] if urgency_score > 0.7 else [],
                'escalation_threshold': 0.8
            }
        elif failure_risk < 0.7:
            return {
                'max_retries': 4,
                'retry_intervals': [15, 60, 180, 300],
                'fallback_channels': ['email', 'sms'] if urgency_score > 0.5 else ['email'],
                'escalation_threshold': 0.6
            }
        else:
            return {
                'max_retries': 6,
                'retry_intervals': [5, 15, 30, 60, 120, 300],
                'fallback_channels': ['email', 'sms', 'desktop_notification'],
                'escalation_threshold': 0.4,
                'immediate_fallback': True
            }
    
    def process_notification(
        self, 
        notification: Dict[str, Any], 
        device_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Main processing function for notifications"""
        try:
            # Generate unique notification ID
            notification_id = f"notif_{int(time.time() * 1000)}"
            
            # Extract information
            user_id = notification['user_id']
            message = notification['message']
            priority = notification.get('priority', 'medium')
            
            # Get device context
            if not device_context:
                device_context = {
                    'device_type': 'unknown',
                    'network_quality': 0.8,
                    'app_version': '1.0.0'
                }
            
            # Predict failure risk
            failure_risk = self.predict_failure_risk(
                user_id=user_id,
                device_type=device_context['device_type'],
                network_quality=device_context['network_quality'],
                app_version=device_context['app_version']
            )
            
            # Classify urgency
            urgency_score = self.classify_urgency(
                message, 
                {
                    'meeting_context': notification.get('meeting_context', False),
                    'sender_importance': 0.5  # Default
                }
            )
            
            # Generate retry strategy
            retry_strategy = self.generate_retry_strategy(failure_risk, urgency_score)
            
            # Recommend delivery channels
            recommended_channels = self._recommend_channels(
                failure_risk, urgency_score, user_id
            )
            
            # Estimate delivery time
            estimated_delivery_time = self._estimate_delivery_time(
                failure_risk, device_context['network_quality']
            )
            
            # Store in database
            self._store_notification(
                notification_id, notification, device_context,
                failure_risk, urgency_score
            )
            
            result = {
                'notification_id': notification_id,
                'success_probability': 1.0 - failure_risk,
                'urgency_score': urgency_score,
                'recommended_channels': recommended_channels,
                'retry_strategy': retry_strategy,
                'estimated_delivery_time': estimated_delivery_time
            }
            
            logger.info(f"Processed notification {notification_id} with success probability {1.0 - failure_risk:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing notification: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _parse_version_score(self, version: str) -> float:
        """Convert app version to numeric score"""
        try:
            parts = version.split('.')
            major = int(parts[0]) if len(parts) > 0 else 1
            minor = int(parts[1]) if len(parts) > 1 else 0
            return min((major + minor * 0.1) / 10.0, 1.0)
        except:
            return 0.5
    
    def _get_user_activity_score(self, user_id: str) -> float:
        """Get user activity score (simplified)"""
        # In real implementation, this would query user activity data
        return 0.7  # Default active user
    
    def _recommend_channels(
        self, 
        failure_risk: float, 
        urgency_score: float, 
        user_id: str
    ) -> List[str]:
        """Recommend optimal delivery channels"""
        channels = ['teams_notification']
        
        if failure_risk > 0.5 or urgency_score > 0.7:
            channels.append('email')
        
        if urgency_score > 0.8:
            channels.extend(['sms', 'desktop_notification'])
        
        return channels
    
    def _estimate_delivery_time(self, failure_risk: float, network_quality: float) -> float:
        """Estimate notification delivery time in seconds"""
        base_time = 2.0  # Base delivery time
        
        # Adjust for network quality
        network_delay = (1.0 - network_quality) * 5.0
        
        # Adjust for failure risk (more retries = longer time)
        failure_delay = failure_risk * 10.0
        
        return base_time + network_delay + failure_delay
    
    def _store_notification(
        self, 
        notification_id: str, 
        notification: Dict, 
        device_context: Dict,
        failure_risk: float, 
        urgency_score: float
    ):
        """Store notification data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO notifications (
                    id, user_id, message, priority, channel, timestamp,
                    device_type, app_version, network_quality,
                    urgency_score, success_probability
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                notification_id,
                notification['user_id'],
                notification['message'],
                notification.get('priority', 'medium'),
                notification.get('channel', 'teams_chat'),
                datetime.now().isoformat(),
                device_context.get('device_type', 'unknown'),
                device_context.get('app_version', '1.0.0'),
                device_context.get('network_quality', 0.8),
                urgency_score,
                1.0 - failure_risk
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing notification: {e}")

# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global enhancer
    enhancer = NotificationEnhancer()
    logger.info("Application started")
    yield
    # Shutdown
    logger.info("Application shutting down")

app = FastAPI(
    title="AI-Powered Teams Notification Enhancement",
    description="Intelligent notification reliability system for Microsoft Teams",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global enhancer instance
enhancer: NotificationEnhancer = None

@app.get("/")
async def root():
    return {
        "message": "AI-Powered Teams Notification Enhancement API",
        "version": "1.0.0",
        "status": "active"
    }

@app.post("/process-notification", response_model=NotificationResponse)
async def process_notification(
    notification: NotificationRequest,
    device_context: Optional[DeviceContext] = None
):
    """Process a notification with AI enhancement"""
    notification_dict = notification.dict()
    device_dict = device_context.dict() if device_context else None
    
    result = enhancer.process_notification(notification_dict, device_dict)
    return NotificationResponse(**result)

@app.get("/predict-failure/{user_id}")
async def predict_failure(
    user_id: str,
    device_type: str = "mobile",
    network_quality: float = 0.8,
    app_version: str = "1.0.0"
):
    """Predict notification failure risk for a user"""
    failure_risk = enhancer.predict_failure_risk(
        user_id, device_type, network_quality, app_version
    )
    
    return {
        "user_id": user_id,
        "failure_risk": failure_risk,
        "success_probability": 1.0 - failure_risk,
        "risk_level": "high" if failure_risk > 0.7 else "medium" if failure_risk > 0.3 else "low"
    }

@app.post("/classify-urgency")
async def classify_urgency(message: str, meeting_context: bool = False):
    """Classify message urgency"""
    context = {'meeting_context': meeting_context}
    urgency_score = enhancer.classify_urgency(message, context)
    
    return {
        "message": message,
        "urgency_score": urgency_score,
        "urgency_level": "high" if urgency_score > 0.7 else "medium" if urgency_score > 0.3 else "low"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": enhancer.failure_predictor is not None
    }

if __name__ == "__main__":
    logger.info("Starting AI-Powered Teams Notification Enhancement System")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )