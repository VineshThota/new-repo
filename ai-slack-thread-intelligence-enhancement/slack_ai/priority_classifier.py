"""Priority Classifier Module

Provides AI-powered priority classification for Slack messages.
Classifies messages as HIGH, MEDIUM, or LOW priority based on content analysis.
"""

import re
from typing import Dict, List, Any
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import os

class PriorityClassifier:
    """AI-powered priority classification for Slack messages."""
    
    def __init__(self):
        """Initialize the priority classifier."""
        
        # High priority indicators
        self.high_priority_keywords = [
            'urgent', 'asap', 'emergency', 'critical', 'immediately', 'now',
            'deadline', 'important', 'priority', 'escalation', 'issue', 'problem',
            'down', 'broken', 'failed', 'error', 'bug', 'outage', 'incident',
            'help', 'stuck', 'blocked', 'meeting', 'client', 'customer'
        ]
        
        # Medium priority indicators
        self.medium_priority_keywords = [
            'review', 'feedback', 'update', 'status', 'progress', 'question',
            'discuss', 'plan', 'schedule', 'tomorrow', 'next week', 'follow up',
            'reminder', 'check', 'confirm', 'approve', 'decision'
        ]
        
        # Low priority indicators
        self.low_priority_keywords = [
            'fyi', 'info', 'heads up', 'btw', 'by the way', 'just so you know',
            'sharing', 'interesting', 'cool', 'nice', 'great', 'awesome',
            'thanks', 'thank you', 'good job', 'well done'
        ]
        
        # Urgency patterns
        self.urgency_patterns = [
            r'\b(asap|urgent|emergency|critical|immediately)\b',
            r'\b(need.{0,10}(now|today|asap))\b',
            r'\b(deadline.{0,20}(today|tomorrow))\b',
            r'\b(production.{0,10}(down|issue|problem))\b',
            r'\b(server.{0,10}(down|error|issue))\b'
        ]
        
        # Initialize or load the ML model
        self.model = self._initialize_model()
    
    def classify_priority(self, message_text: str, context: Dict = None) -> Dict[str, Any]:
        """Classify the priority of a Slack message.
        
        Args:
            message_text: The text content of the message
            context: Optional context information (channel, user, time, etc.)
            
        Returns:
            Dictionary with priority level, confidence score, and reasoning
        """
        if not message_text or not message_text.strip():
            return {
                'level': 'LOW',
                'confidence': 0.5,
                'reasoning': 'Empty or whitespace-only message'
            }
        
        # Calculate various priority indicators
        keyword_score = self._calculate_keyword_score(message_text)
        urgency_score = self._calculate_urgency_score(message_text)
        sentiment_score = self._calculate_sentiment_score(message_text)
        pattern_score = self._calculate_pattern_score(message_text)
        context_score = self._calculate_context_score(context) if context else 0.0
        
        # Combine scores with weights
        total_score = (
            keyword_score * 0.3 +
            urgency_score * 0.25 +
            pattern_score * 0.2 +
            sentiment_score * 0.15 +
            context_score * 0.1
        )
        
        # Determine priority level and confidence
        if total_score >= 0.7:
            level = 'HIGH'
            confidence = min(0.9, 0.6 + total_score * 0.3)
        elif total_score >= 0.4:
            level = 'MEDIUM'
            confidence = min(0.8, 0.5 + total_score * 0.3)
        else:
            level = 'LOW'
            confidence = min(0.7, 0.4 + (1 - total_score) * 0.3)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(message_text, total_score, {
            'keywords': keyword_score,
            'urgency': urgency_score,
            'patterns': pattern_score,
            'sentiment': sentiment_score,
            'context': context_score
        })
        
        return {
            'level': level,
            'confidence': confidence,
            'reasoning': reasoning,
            'scores': {
                'total': total_score,
                'keyword': keyword_score,
                'urgency': urgency_score,
                'pattern': pattern_score,
                'sentiment': sentiment_score,
                'context': context_score
            }
        }
    
    def _calculate_keyword_score(self, text: str) -> float:
        """Calculate priority score based on keyword presence."""
        text_lower = text.lower()
        
        high_count = sum(1 for keyword in self.high_priority_keywords if keyword in text_lower)
        medium_count = sum(1 for keyword in self.medium_priority_keywords if keyword in text_lower)
        low_count = sum(1 for keyword in self.low_priority_keywords if keyword in text_lower)
        
        # Weighted scoring
        score = (high_count * 1.0 + medium_count * 0.5 + low_count * 0.1) / max(len(text.split()) / 10, 1)
        
        return min(score, 1.0)
    
    def _calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency score based on text patterns and formatting."""
        score = 0.0
        text_lower = text.lower()
        
        # Check urgency patterns
        for pattern in self.urgency_patterns:
            if re.search(pattern, text_lower):
                score += 0.3
        
        # Check for exclamation marks
        exclamation_count = text.count('!')
        score += min(exclamation_count * 0.1, 0.3)
        
        # Check for all caps words (excluding short words)
        words = text.split()
        caps_words = [word for word in words if word.isupper() and len(word) > 2]
        score += min(len(caps_words) * 0.1, 0.2)
        
        # Check for question marks (questions often need responses)
        if '?' in text:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate priority score based on sentiment analysis."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Negative sentiment often indicates problems/issues (higher priority)
            if polarity < -0.3:
                return 0.7
            elif polarity < -0.1:
                return 0.5
            elif polarity > 0.3:
                return 0.2  # Very positive messages are often low priority
            else:
                return 0.3  # Neutral sentiment
        except:
            return 0.3
    
    def _calculate_pattern_score(self, text: str) -> float:
        """Calculate priority score based on message patterns."""
        score = 0.0
        
        # Check for @mentions (often indicates assignment or direct communication)
        mentions = len(re.findall(r'@\w+', text))
        score += min(mentions * 0.2, 0.4)
        
        # Check for time-sensitive language
        time_patterns = [
            r'\b(today|tonight|this morning|this afternoon)\b',
            r'\b(by \d+|before \d+|until \d+)\b',
            r'\b(deadline|due date|expires?)\b'
        ]
        
        for pattern in time_patterns:
            if re.search(pattern, text.lower()):
                score += 0.3
                break
        
        # Check for action-oriented language
        action_patterns = [
            r'\b(need to|have to|must|should|required?)\b',
            r'\b(please|can you|could you|would you)\b',
            r'\b(fix|resolve|handle|address)\b'
        ]
        
        for pattern in action_patterns:
            if re.search(pattern, text.lower()):
                score += 0.2
                break
        
        return min(score, 1.0)
    
    def _calculate_context_score(self, context: Dict) -> float:
        """Calculate priority score based on message context."""
        if not context:
            return 0.0
        
        score = 0.0
        
        # Channel-based scoring
        channel = context.get('channel', '').lower()
        if any(keyword in channel for keyword in ['urgent', 'critical', 'incident', 'alert']):
            score += 0.5
        elif any(keyword in channel for keyword in ['general', 'random', 'social']):
            score -= 0.2
        
        # Time-based scoring (messages outside business hours might be more urgent)
        timestamp = context.get('timestamp')
        if timestamp:
            # This would need proper timestamp parsing in real implementation
            # For demo, assume some messages are after hours
            if hash(str(timestamp)) % 4 == 0:  # Simulate 25% after-hours messages
                score += 0.3
        
        # User-based scoring
        user = context.get('user', '').lower()
        if any(keyword in user for keyword in ['ceo', 'cto', 'manager', 'lead']):
            score += 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def _generate_reasoning(self, text: str, total_score: float, scores: Dict) -> str:
        """Generate human-readable reasoning for the priority classification."""
        reasons = []
        
        if scores['urgency'] > 0.5:
            reasons.append("Contains urgency indicators")
        
        if scores['keywords'] > 0.5:
            reasons.append("High-priority keywords detected")
        
        if scores['patterns'] > 0.4:
            reasons.append("Action-oriented or time-sensitive language")
        
        if scores['sentiment'] > 0.6:
            reasons.append("Negative sentiment suggests issues")
        
        if scores['context'] > 0.3:
            reasons.append("Context indicates higher priority")
        
        if '!' in text:
            reasons.append("Exclamation marks indicate emphasis")
        
        if re.search(r'@\w+', text):
            reasons.append("Direct mentions require attention")
        
        if not reasons:
            if total_score < 0.3:
                reasons.append("Informational content with low urgency")
            else:
                reasons.append("Standard business communication")
        
        return "; ".join(reasons)
    
    def _initialize_model(self):
        """Initialize or load the machine learning model."""
        # For this demo, we'll use a simple rule-based approach
        # In a production system, you would train an ML model on labeled data
        
        # This is a placeholder for a more sophisticated ML model
        # that would be trained on historical Slack data with priority labels
        return None
    
    def batch_classify(self, messages: List[Dict]) -> List[Dict]:
        """Classify priority for multiple messages at once."""
        results = []
        
        for msg in messages:
            text = msg.get('text', '')
            context = {
                'channel': msg.get('channel'),
                'user': msg.get('user'),
                'timestamp': msg.get('timestamp')
            }
            
            priority = self.classify_priority(text, context)
            
            results.append({
                'message': msg,
                'priority': priority
            })
        
        return results
    
    def get_priority_stats(self, messages: List[Dict]) -> Dict[str, Any]:
        """Get priority distribution statistics for a set of messages."""
        classifications = self.batch_classify(messages)
        
        priority_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        total_confidence = 0.0
        
        for result in classifications:
            level = result['priority']['level']
            confidence = result['priority']['confidence']
            
            priority_counts[level] += 1
            total_confidence += confidence
        
        total_messages = len(classifications)
        avg_confidence = total_confidence / total_messages if total_messages > 0 else 0.0
        
        return {
            'total_messages': total_messages,
            'priority_distribution': priority_counts,
            'priority_percentages': {
                level: (count / total_messages * 100) if total_messages > 0 else 0
                for level, count in priority_counts.items()
            },
            'average_confidence': avg_confidence,
            'high_priority_ratio': priority_counts['HIGH'] / total_messages if total_messages > 0 else 0
        }