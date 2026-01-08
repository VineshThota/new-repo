import re
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from collections import Counter
import math

# For production, these would be actual ML models
# For demo purposes, we'll use rule-based approaches with realistic scoring

class MessagePrioritizer:
    """AI-powered message prioritization system"""
    
    def __init__(self):
        self.urgency_keywords = {
            'critical': 95,
            'urgent': 90,
            'emergency': 95,
            'asap': 85,
            'immediately': 90,
            'deadline': 80,
            'eod': 75,
            'today': 70,
            'tomorrow': 60,
            'breaking': 85,
            'down': 80,
            'outage': 90,
            'issue': 65,
            'problem': 70,
            'bug': 75,
            'error': 70,
            'failed': 75,
            'crash': 80,
            'alert': 85,
            'warning': 70,
            'security': 85,
            'breach': 95,
            'compliance': 80,
            'mandatory': 75,
            'required': 65,
            'meeting': 60,
            'cancelled': 70,
            'moved': 65,
            'rescheduled': 65,
            'action': 70,
            'todo': 65,
            'task': 60,
            'review': 55,
            'approve': 70,
            'decision': 75,
            'feedback': 50,
            'fyi': 30,
            'heads up': 45,
            'update': 50
        }
        
        self.channel_weights = {
            'alerts': 1.0,
            'incidents': 0.95,
            'security': 0.9,
            'announcements': 0.85,
            'engineering': 0.8,
            'product': 0.75,
            'bugs': 0.8,
            'infrastructure': 0.85,
            'devops': 0.8,
            'general': 0.6,
            'random': 0.3,
            'social': 0.2,
            'lunch': 0.1,
            'coffee': 0.1
        }
        
        self.user_roles = {
            'ceo': 1.0,
            'cto': 0.95,
            'vp': 0.9,
            'director': 0.85,
            'manager': 0.8,
            'lead': 0.75,
            'senior': 0.7,
            'engineer': 0.65,
            'developer': 0.65,
            'designer': 0.6,
            'analyst': 0.6,
            'coordinator': 0.55,
            'intern': 0.4
        }
        
        self.mention_patterns = {
            '@channel': 90,
            '@here': 85,
            '@everyone': 80,
            'urgent': 85,
            'please': 60,
            'help': 70,
            'need': 65,
            'can you': 60,
            'could you': 55
        }
    
    def calculate_priority(self, message) -> float:
        """Calculate comprehensive priority score for a message"""
        try:
            # Base score
            score = 50.0
            
            # 1. Urgency keyword analysis (0-40 points)
            urgency_score = self._analyze_urgency(message.text)
            score += urgency_score * 0.4
            
            # 2. Sender importance (0-20 points)
            sender_score = self._analyze_sender_importance(message.user)
            score += sender_score * 0.2
            
            # 3. Channel relevance (0-15 points)
            channel_score = self._analyze_channel_importance(message.channel)
            score += channel_score * 0.15
            
            # 4. Temporal factors (0-10 points)
            temporal_score = self._analyze_temporal_factors(message.timestamp)
            score += temporal_score * 0.1
            
            # 5. Social signals (0-10 points)
            social_score = self._analyze_social_signals(message.reactions, message.mentions)
            score += social_score * 0.1
            
            # 6. Content analysis (0-5 points)
            content_score = self._analyze_content_complexity(message.text)
            score += content_score * 0.05
            
            # Normalize to 0-100 range
            score = max(0, min(100, score))
            
            return round(score, 1)
            
        except Exception as e:
            # Fallback to medium priority if analysis fails
            return 50.0
    
    def _analyze_urgency(self, text: str) -> float:
        """Analyze urgency keywords and patterns in message text"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        urgency_score = 0.0
        
        # Check for urgency keywords
        for keyword, weight in self.urgency_keywords.items():
            if keyword in text_lower:
                urgency_score = max(urgency_score, weight)
        
        # Check for time-sensitive patterns
        time_patterns = [
            r'\b(today|tonight|now|asap)\b',
            r'\b(deadline|due|eod|end of day)\b',
            r'\b(urgent|emergency|critical)\b',
            r'\b(down|outage|broken|failed)\b',
            r'\b(meeting.*(today|now|soon))\b'
        ]
        
        for pattern in time_patterns:
            if re.search(pattern, text_lower):
                urgency_score += 15
        
        # Check for escalation language
        escalation_patterns = [
            r'[!]{2,}',  # Multiple exclamation marks
            r'\b(help|sos|mayday)\b',
            r'\b(please.*(urgent|asap|now))\b',
            r'\b(need.*(immediate|urgent|asap))\b'
        ]
        
        for pattern in escalation_patterns:
            if re.search(pattern, text_lower):
                urgency_score += 10
        
        return min(100, urgency_score)
    
    def _analyze_sender_importance(self, user: str) -> float:
        """Analyze sender importance based on role and interaction history"""
        if not user:
            return 50.0
        
        user_lower = user.lower()
        importance_score = 50.0
        
        # Check for role indicators in username
        for role, weight in self.user_roles.items():
            if role in user_lower:
                importance_score = weight * 100
                break
        
        # Check for specific high-importance users
        high_importance_indicators = ['admin', 'security', 'devops', 'oncall', 'support']
        for indicator in high_importance_indicators:
            if indicator in user_lower:
                importance_score = max(importance_score, 80)
        
        return importance_score
    
    def _analyze_channel_importance(self, channel: str) -> float:
        """Analyze channel importance and relevance"""
        if not channel:
            return 50.0
        
        channel_lower = channel.lower()
        
        # Direct channel weight lookup
        for channel_name, weight in self.channel_weights.items():
            if channel_name in channel_lower:
                return weight * 100
        
        # Default scoring based on channel patterns
        if any(keyword in channel_lower for keyword in ['alert', 'incident', 'emergency']):
            return 95.0
        elif any(keyword in channel_lower for keyword in ['announce', 'important', 'critical']):
            return 85.0
        elif any(keyword in channel_lower for keyword in ['eng', 'dev', 'tech', 'prod']):
            return 75.0
        elif any(keyword in channel_lower for keyword in ['general', 'team', 'project']):
            return 60.0
        else:
            return 50.0
    
    def _analyze_temporal_factors(self, timestamp: datetime) -> float:
        """Analyze time-based urgency factors"""
        if not timestamp:
            return 50.0
        
        now = datetime.now()
        time_diff = now - timestamp
        
        # Recent messages get higher priority
        if time_diff < timedelta(minutes=5):
            return 90.0
        elif time_diff < timedelta(minutes=15):
            return 80.0
        elif time_diff < timedelta(hours=1):
            return 70.0
        elif time_diff < timedelta(hours=4):
            return 60.0
        elif time_diff < timedelta(hours=24):
            return 50.0
        else:
            return 30.0
    
    def _analyze_social_signals(self, reactions: List[Dict], mentions: List[str]) -> float:
        """Analyze social engagement signals"""
        social_score = 0.0
        
        # Analyze reactions
        if reactions:
            total_reactions = sum(r.get('count', 0) for r in reactions)
            
            # High-priority reaction emojis
            priority_reactions = ['fire', 'warning', 'exclamation', 'sos', 'alert', 'eyes']
            priority_reaction_count = sum(
                r.get('count', 0) for r in reactions 
                if r.get('name', '') in priority_reactions
            )
            
            social_score += min(30, total_reactions * 2)
            social_score += priority_reaction_count * 5
        
        # Analyze mentions
        if mentions:
            mention_score = 0
            for mention in mentions:
                mention_lower = mention.lower()
                
                # High-priority mentions
                if mention_lower in ['@channel', '@here', '@everyone']:
                    mention_score += 25
                elif any(role in mention_lower for role in ['admin', 'manager', 'lead']):
                    mention_score += 15
                else:
                    mention_score += 10
            
            social_score += min(40, mention_score)
        
        return min(100, social_score)
    
    def _analyze_content_complexity(self, text: str) -> float:
        """Analyze content complexity and information density"""
        if not text:
            return 0.0
        
        complexity_score = 0.0
        
        # Length-based scoring
        word_count = len(text.split())
        if word_count > 100:
            complexity_score += 20
        elif word_count > 50:
            complexity_score += 15
        elif word_count > 20:
            complexity_score += 10
        
        # Technical content indicators
        technical_patterns = [
            r'\b(error|exception|stack trace|log|debug)\b',
            r'\b(server|database|api|endpoint|service)\b',
            r'\b(deploy|release|build|version|commit)\b',
            r'\b(config|setting|parameter|variable)\b'
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, text.lower()):
                complexity_score += 10
        
        # Question indicators (may need response)
        if '?' in text or any(word in text.lower() for word in ['how', 'what', 'when', 'where', 'why', 'who']):
            complexity_score += 15
        
        return min(100, complexity_score)
    
    def get_priority_explanation(self, message) -> Dict[str, float]:
        """Get detailed breakdown of priority scoring factors"""
        explanation = {
            'urgency_score': self._analyze_urgency(message.text),
            'sender_importance': self._analyze_sender_importance(message.user),
            'channel_relevance': self._analyze_channel_importance(message.channel),
            'temporal_factor': self._analyze_temporal_factors(message.timestamp),
            'social_signals': self._analyze_social_signals(message.reactions, message.mentions),
            'content_complexity': self._analyze_content_complexity(message.text)
        }
        
        explanation['total_score'] = self.calculate_priority(message)
        return explanation

class ThreadSummarizer:
    """AI-powered thread summarization system"""
    
    def __init__(self):
        self.action_keywords = [
            'todo', 'task', 'action', 'follow up', 'next step',
            'assign', 'responsible', 'deadline', 'due', 'complete',
            'finish', 'implement', 'fix', 'resolve', 'address'
        ]
        
        self.decision_keywords = [
            'decide', 'decision', 'agreed', 'consensus', 'approved',
            'rejected', 'chosen', 'selected', 'final', 'conclusion'
        ]
    
    def generate_summary(self, messages: List) -> str:
        """Generate intelligent summary of thread messages"""
        if not messages:
            return "No messages to summarize."
        
        if len(messages) == 1:
            return f"Single message from {messages[0].user}: {messages[0].text[:100]}..."
        
        # Extract key information
        participants = list(set(msg.user for msg in messages))
        key_topics = self._extract_key_topics(messages)
        decisions = self._extract_decisions(messages)
        action_items = self._extract_action_items(messages)
        
        # Build summary
        summary_parts = []
        
        summary_parts.append(f"Thread with {len(participants)} participants: {', '.join(participants[:3])}{'...' if len(participants) > 3 else ''}")
        
        if key_topics:
            summary_parts.append(f"Key topics: {', '.join(key_topics[:3])}")
        
        if decisions:
            summary_parts.append(f"Decisions made: {'; '.join(decisions[:2])}")
        
        if action_items:
            summary_parts.append(f"Action items: {'; '.join(action_items[:3])}")
        
        return " | ".join(summary_parts)
    
    def extract_action_items(self, messages: List) -> List[str]:
        """Extract action items from thread messages"""
        action_items = []
        
        for message in messages:
            text = message.text.lower()
            
            # Look for action-oriented sentences
            sentences = text.split('.')
            for sentence in sentences:
                if any(keyword in sentence for keyword in self.action_keywords):
                    # Clean and format the action item
                    action_item = sentence.strip().capitalize()
                    if len(action_item) > 10 and action_item not in action_items:
                        action_items.append(action_item[:100] + '...' if len(action_item) > 100 else action_item)
        
        return action_items[:5]  # Return top 5 action items
    
    def _extract_key_topics(self, messages: List) -> List[str]:
        """Extract key topics from messages using simple NLP"""
        # Combine all message text
        all_text = ' '.join(msg.text.lower() for msg in messages)
        
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b\w{3,}\b', all_text)
        filtered_words = [word for word in words if word not in stop_words]
        
        # Get most common meaningful words
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(5) if count > 1]
    
    def _extract_decisions(self, messages: List) -> List[str]:
        """Extract decisions made in the thread"""
        decisions = []
        
        for message in messages:
            text = message.text.lower()
            sentences = text.split('.')
            
            for sentence in sentences:
                if any(keyword in sentence for keyword in self.decision_keywords):
                    decision = sentence.strip().capitalize()
                    if len(decision) > 10 and decision not in decisions:
                        decisions.append(decision[:80] + '...' if len(decision) > 80 else decision)
        
        return decisions[:3]  # Return top 3 decisions
    
    def _extract_action_items(self, messages: List) -> List[str]:
        """Extract action items (same as public method for internal use)"""
        return self.extract_action_items(messages)

class SmartNotificationManager:
    """Manages intelligent notifications based on priority and user context"""
    
    def __init__(self):
        self.notification_history = []
        self.user_preferences = {
            'quiet_hours': (22, 8),  # 10 PM to 8 AM
            'max_notifications_per_hour': 10,
            'priority_threshold': 75
        }
    
    def should_notify(self, message, priority_score: float) -> bool:
        """Determine if a notification should be sent"""
        # Check priority threshold
        if priority_score < self.user_preferences['priority_threshold']:
            return False
        
        # Check quiet hours
        current_hour = datetime.now().hour
        quiet_start, quiet_end = self.user_preferences['quiet_hours']
        
        if quiet_start > quiet_end:  # Overnight quiet hours
            if current_hour >= quiet_start or current_hour < quiet_end:
                return priority_score >= 90  # Only critical messages during quiet hours
        else:  # Same day quiet hours
            if quiet_start <= current_hour < quiet_end:
                return priority_score >= 90
        
        # Check notification frequency
        recent_notifications = [
            n for n in self.notification_history 
            if (datetime.now() - n['timestamp']).total_seconds() < 3600
        ]
        
        if len(recent_notifications) >= self.user_preferences['max_notifications_per_hour']:
            return priority_score >= 95  # Only highest priority if too many recent notifications
        
        return True
    
    def log_notification(self, message, priority_score: float):
        """Log a sent notification"""
        self.notification_history.append({
            'message_id': message.id,
            'priority_score': priority_score,
            'timestamp': datetime.now(),
            'channel': message.channel,
            'user': message.user
        })
        
        # Keep only recent history (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.notification_history = [
            n for n in self.notification_history 
            if n['timestamp'] > cutoff
        ]

# Utility functions for integration
def batch_process_messages(messages: List, prioritizer: MessagePrioritizer) -> List[Tuple[any, float]]:
    """Process multiple messages in batch for efficiency"""
    results = []
    for message in messages:
        priority_score = prioritizer.calculate_priority(message)
        results.append((message, priority_score))
    
    # Sort by priority (highest first)
    return sorted(results, key=lambda x: x[1], reverse=True)

def filter_by_priority(messages: List[Tuple[any, float]], min_priority: float = 50.0) -> List[Tuple[any, float]]:
    """Filter messages by minimum priority threshold"""
    return [(msg, score) for msg, score in messages if score >= min_priority]

def get_priority_distribution(messages: List[Tuple[any, float]]) -> Dict[str, int]:
    """Get distribution of messages by priority level"""
    distribution = {'high': 0, 'medium': 0, 'low': 0}
    
    for _, score in messages:
        if score >= 75:
            distribution['high'] += 1
        elif score >= 50:
            distribution['medium'] += 1
        else:
            distribution['low'] += 1
    
    return distribution