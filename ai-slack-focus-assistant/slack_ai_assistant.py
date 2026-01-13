import re
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict
import json
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class MessageClassifier:
    """AI-powered message classifier for priority and urgency detection"""
    
    def __init__(self):
        self.urgent_keywords = [
            'urgent', 'asap', 'emergency', 'critical', 'immediate', 'now',
            'breaking', 'down', 'outage', 'failed', 'error', 'bug', 'issue',
            'problem', 'help', 'stuck', 'blocked', 'deadline', 'today'
        ]
        
        self.important_keywords = [
            'meeting', 'review', 'approval', 'decision', 'budget', 'launch',
            'release', 'milestone', 'deliverable', 'requirement', 'specification',
            'proposal', 'strategy', 'planning', 'roadmap', 'priority'
        ]
        
        self.noise_keywords = [
            'coffee', 'lunch', 'weather', 'weekend', 'birthday', 'congratulations',
            'thanks', 'thank you', 'good morning', 'good afternoon', 'hello',
            'hi', 'hey', 'lol', 'haha', 'emoji', '😀', '😊', '👍', '🎉'
        ]
        
        self.authority_indicators = [
            'ceo', 'cto', 'vp', 'director', 'manager', 'lead', 'senior',
            'head', 'chief', 'president', 'founder'
        ]
    
    def classify_priority(self, message: str, sender: str = None, channel: str = None) -> Dict[str, Any]:
        """Classify message priority using multiple signals"""
        message_lower = message.lower()
        
        # Initialize scores
        urgency_score = 0
        importance_score = 0
        noise_score = 0
        
        # Keyword-based scoring
        for keyword in self.urgent_keywords:
            if keyword in message_lower:
                urgency_score += 2
        
        for keyword in self.important_keywords:
            if keyword in message_lower:
                importance_score += 1.5
        
        for keyword in self.noise_keywords:
            if keyword in message_lower:
                noise_score += 1
        
        # Sentiment analysis
        blob = TextBlob(message)
        sentiment = blob.sentiment
        
        # Negative sentiment often indicates urgency
        if sentiment.polarity < -0.3:
            urgency_score += 1
        
        # Length and structure analysis
        word_count = len(message.split())
        
        # Very short messages might be noise
        if word_count < 3:
            noise_score += 1
        
        # Very long messages might be important
        if word_count > 50:
            importance_score += 0.5
        
        # Capitalization analysis
        caps_ratio = sum(1 for c in message if c.isupper()) / len(message) if message else 0
        if caps_ratio > 0.3:  # High caps ratio indicates urgency
            urgency_score += 1.5
        
        # Punctuation analysis
        exclamation_count = message.count('!')
        question_count = message.count('?')
        
        urgency_score += exclamation_count * 0.5
        importance_score += question_count * 0.3
        
        # Time-based keywords
        time_urgent = ['now', 'asap', 'immediately', 'urgent', 'today']
        for keyword in time_urgent:
            if keyword in message_lower:
                urgency_score += 1.5
        
        # Authority-based scoring (if sender info available)
        if sender:
            sender_lower = sender.lower()
            for indicator in self.authority_indicators:
                if indicator in sender_lower:
                    importance_score += 1
                    break
        
        # Channel-based scoring
        if channel:
            channel_lower = channel.lower()
            if any(word in channel_lower for word in ['urgent', 'critical', 'emergency']):
                urgency_score += 2
            elif any(word in channel_lower for word in ['general', 'random', 'social']):
                noise_score += 0.5
        
        # Determine final classification
        total_score = urgency_score + importance_score - noise_score
        
        if urgency_score >= 3:
            level = 'URGENT'
            confidence = min(0.95, 0.6 + urgency_score * 0.1)
            reasoning = f"High urgency signals detected (score: {urgency_score:.1f})"
        elif importance_score >= 2 or total_score >= 2:
            level = 'IMPORTANT'
            confidence = min(0.9, 0.5 + importance_score * 0.1)
            reasoning = f"Important content identified (score: {importance_score:.1f})"
        elif noise_score >= 2 or total_score <= 0:
            level = 'NOISE'
            confidence = min(0.85, 0.4 + noise_score * 0.1)
            reasoning = f"Low-priority social content (noise score: {noise_score:.1f})"
        else:
            level = 'FYI'
            confidence = 0.7
            reasoning = "Informational content for awareness"
        
        return {
            'level': level,
            'confidence': confidence,
            'reasoning': reasoning,
            'scores': {
                'urgency': urgency_score,
                'importance': importance_score,
                'noise': noise_score,
                'total': total_score
            }
        }

class ThreadSummarizer:
    """AI-powered thread summarization with action item extraction"""
    
    def __init__(self):
        self.action_keywords = [
            'will', 'should', 'need to', 'have to', 'must', 'let\'s',
            'schedule', 'plan', 'create', 'update', 'review', 'send',
            'call', 'meeting', 'deadline', 'by', 'tomorrow', 'next week'
        ]
        
        self.decision_keywords = [
            'decided', 'agreed', 'confirmed', 'approved', 'rejected',
            'chosen', 'selected', 'final', 'conclusion', 'resolution'
        ]
    
    def summarize_thread(self, messages: List[str]) -> Dict[str, Any]:
        """Generate comprehensive thread summary with key insights"""
        if not messages:
            return {
                'summary': 'No messages to summarize',
                'action_items': [],
                'key_decisions': [],
                'participants': [],
                'topics': []
            }
        
        # Combine all messages
        full_text = ' '.join(messages)
        
        # Extract key topics using TF-IDF
        topics = self._extract_topics(messages)
        
        # Generate summary using extractive summarization
        summary = self._generate_extractive_summary(messages)
        
        # Extract action items
        action_items = self._extract_action_items(messages)
        
        # Extract key decisions
        key_decisions = self._extract_decisions(messages)
        
        # Identify participants (simulated)
        participants = self._identify_participants(messages)
        
        return {
            'summary': summary,
            'action_items': action_items,
            'key_decisions': key_decisions,
            'participants': participants,
            'topics': topics,
            'message_count': len(messages),
            'estimated_read_time': len(messages) * 0.5  # 30 seconds per message
        }
    
    def _extract_topics(self, messages: List[str], max_topics: int = 5) -> List[str]:
        """Extract key topics using TF-IDF"""
        if len(messages) < 2:
            return []
        
        try:
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            # Fit and transform messages
            tfidf_matrix = vectorizer.fit_transform(messages)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top topics
            top_indices = np.argsort(mean_scores)[-max_topics:]
            topics = [feature_names[i] for i in reversed(top_indices) if mean_scores[i] > 0]
            
            return topics[:max_topics]
        
        except Exception:
            # Fallback: extract common words
            all_words = ' '.join(messages).lower().split()
            word_counts = Counter(all_words)
            common_words = [word for word, count in word_counts.most_common(5) 
                          if len(word) > 3 and word.isalpha()]
            return common_words
    
    def _generate_extractive_summary(self, messages: List[str], max_sentences: int = 3) -> str:
        """Generate extractive summary by selecting most important sentences"""
        if len(messages) <= 2:
            return ' '.join(messages)
        
        try:
            # Calculate sentence importance scores
            sentence_scores = []
            
            for i, message in enumerate(messages):
                score = 0
                message_lower = message.lower()
                
                # Position-based scoring (first and last messages are important)
                if i == 0 or i == len(messages) - 1:
                    score += 2
                
                # Length-based scoring
                word_count = len(message.split())
                if 10 <= word_count <= 50:  # Optimal length
                    score += 1
                
                # Keyword-based scoring
                important_words = ['decision', 'action', 'next', 'plan', 'meeting', 'deadline']
                for word in important_words:
                    if word in message_lower:
                        score += 1
                
                # Question/answer patterns
                if '?' in message:
                    score += 0.5
                if any(word in message_lower for word in ['yes', 'no', 'agreed', 'confirmed']):
                    score += 1
                
                sentence_scores.append((score, message))
            
            # Select top sentences
            sentence_scores.sort(reverse=True, key=lambda x: x[0])
            top_sentences = [msg for score, msg in sentence_scores[:max_sentences]]
            
            # Maintain chronological order
            summary_sentences = []
            for message in messages:
                if message in top_sentences:
                    summary_sentences.append(message)
            
            return ' '.join(summary_sentences)
        
        except Exception:
            # Fallback: return first, middle, and last messages
            if len(messages) >= 3:
                return f"{messages[0]} {messages[len(messages)//2]} {messages[-1]}"
            else:
                return ' '.join(messages)
    
    def _extract_action_items(self, messages: List[str]) -> List[str]:
        """Extract action items from messages"""
        action_items = []
        
        for message in messages:
            message_lower = message.lower()
            
            # Look for action-oriented sentences
            for keyword in self.action_keywords:
                if keyword in message_lower:
                    # Extract the sentence containing the action keyword
                    sentences = message.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower() and len(sentence.strip()) > 10:
                            action_items.append(sentence.strip())
                            break
                    break
        
        # Remove duplicates and limit to top 5
        unique_actions = list(dict.fromkeys(action_items))[:5]
        return unique_actions
    
    def _extract_decisions(self, messages: List[str]) -> List[str]:
        """Extract key decisions from messages"""
        decisions = []
        
        for message in messages:
            message_lower = message.lower()
            
            # Look for decision-oriented sentences
            for keyword in self.decision_keywords:
                if keyword in message_lower:
                    # Extract the sentence containing the decision keyword
                    sentences = message.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower() and len(sentence.strip()) > 10:
                            decisions.append(sentence.strip())
                            break
                    break
        
        # Remove duplicates and limit to top 3
        unique_decisions = list(dict.fromkeys(decisions))[:3]
        return unique_decisions
    
    def _identify_participants(self, messages: List[str]) -> List[str]:
        """Identify thread participants (simulated)"""
        # In a real implementation, this would extract actual user mentions
        # For demo purposes, we'll simulate participants
        participant_count = min(len(messages) // 2 + 1, 5)
        participants = [f"user_{i+1}" for i in range(participant_count)]
        return participants

class FocusTimeAnalyzer:
    """Analyze user activity patterns and suggest optimal focus times"""
    
    def __init__(self):
        self.optimal_focus_duration = 90  # minutes
        self.min_focus_duration = 30  # minutes
    
    def analyze_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user activity patterns (simulated)"""
        # In a real implementation, this would analyze actual Slack data
        # For demo purposes, we'll generate realistic patterns
        
        patterns = {
            'peak_hours': [9, 10, 14, 15],  # Most productive hours
            'low_activity_hours': [12, 13, 17],  # Lunch and end of day
            'average_interruptions_per_hour': 8.5,
            'average_response_time': 12,  # minutes
            'preferred_communication_style': 'async',
            'focus_score': 0.72  # 0-1 scale
        }
        
        return patterns
    
    def suggest_focus_blocks(self, activity_data: List[Dict]) -> List[Dict[str, Any]]:
        """Suggest optimal focus time blocks based on activity patterns"""
        recommendations = []
        
        # Analyze activity data to find low-interruption periods
        for hour_data in activity_data:
            hour = hour_data['hour']
            interruptions = hour_data['interruptions']
            productivity = hour_data.get('productivity_score', 0.5)
            
            # Suggest focus blocks for hours with low interruptions and high productivity
            if interruptions <= 6 and productivity >= 0.6:
                # Determine block duration based on productivity
                if productivity >= 0.8:
                    duration = 90
                    quality = 'High'
                elif productivity >= 0.7:
                    duration = 60
                    quality = 'Medium'
                else:
                    duration = 45
                    quality = 'Good'
                
                recommendations.append({
                    'start': f"{hour}:00",
                    'end': f"{hour + duration//60}:{duration%60:02d}",
                    'duration': duration,
                    'quality': quality,
                    'interruption_risk': 'Low' if interruptions <= 4 else 'Medium',
                    'productivity_score': productivity
                })
        
        # Sort by productivity score and limit to top 3
        recommendations.sort(key=lambda x: x['productivity_score'], reverse=True)
        return recommendations[:3]
    
    def calculate_focus_score(self, interruptions: int, message_count: int, 
                            focus_time: int) -> float:
        """Calculate focus score based on activity metrics"""
        # Normalize metrics
        interruption_penalty = min(interruptions / 20, 1.0)  # Max penalty at 20 interruptions
        message_penalty = min(message_count / 100, 0.5)  # Max penalty at 100 messages
        focus_bonus = min(focus_time / 240, 1.0)  # Max bonus at 4 hours
        
        # Calculate score (0-1 scale)
        score = max(0, 1.0 - interruption_penalty - message_penalty + focus_bonus * 0.5)
        return round(score, 2)

class DailyDigestGenerator:
    """Generate comprehensive daily activity digests"""
    
    def __init__(self):
        self.summarizer = ThreadSummarizer()
        self.classifier = MessageClassifier()
    
    def generate_digest(self, messages: List[Dict], date: datetime.date) -> Dict[str, Any]:
        """Generate comprehensive daily digest"""
        # Filter messages for the specific date (simulated)
        daily_messages = messages  # In real implementation, filter by date
        
        # Basic statistics
        total_messages = len(daily_messages)
        channels = set(msg.get('channel', 'general') for msg in daily_messages)
        active_channels = len(channels)
        
        # Classify all messages
        classified_messages = []
        for msg in daily_messages:
            classification = self.classifier.classify_priority(
                msg['content'], 
                msg.get('sender'), 
                msg.get('channel')
            )
            msg_with_class = {**msg, **classification}
            classified_messages.append(msg_with_class)
        
        # Extract priority distribution
        priority_counts = Counter(msg['level'] for msg in classified_messages)
        
        # Generate top conversations
        top_conversations = self._generate_top_conversations(classified_messages)
        
        # Extract action items and decisions
        all_messages_text = [msg['content'] for msg in daily_messages]
        thread_summary = self.summarizer.summarize_thread(all_messages_text)
        
        # Generate trending topics
        trending_topics = self._generate_trending_topics(all_messages_text)
        
        # Calculate productivity metrics
        productivity_metrics = self._calculate_productivity_metrics(classified_messages)
        
        return {
            'date': date.strftime('%Y-%m-%d'),
            'total_messages': total_messages,
            'active_channels': active_channels,
            'priority_distribution': dict(priority_counts),
            'top_conversations': top_conversations,
            'action_items': thread_summary['action_items'],
            'key_decisions': thread_summary['key_decisions'],
            'trending_topics': trending_topics,
            'productivity_metrics': productivity_metrics,
            'summary': f"Processed {total_messages} messages across {active_channels} channels"
        }
    
    def _generate_top_conversations(self, messages: List[Dict], limit: int = 3) -> List[Dict]:
        """Generate top conversations by channel activity"""
        channel_activity = defaultdict(list)
        
        # Group messages by channel
        for msg in messages:
            channel = msg.get('channel', 'general')
            channel_activity[channel].append(msg)
        
        # Sort channels by message count and importance
        top_channels = []
        for channel, msgs in channel_activity.items():
            message_count = len(msgs)
            importance_score = sum(1 for msg in msgs if msg.get('level') in ['URGENT', 'IMPORTANT'])
            
            # Generate summary for this channel
            channel_messages = [msg['content'] for msg in msgs]
            summary = self.summarizer._generate_extractive_summary(channel_messages, 1)
            
            top_channels.append({
                'channel': channel,
                'message_count': message_count,
                'importance_score': importance_score,
                'summary': summary[:100] + '...' if len(summary) > 100 else summary
            })
        
        # Sort by combined score and return top conversations
        top_channels.sort(key=lambda x: x['message_count'] + x['importance_score'] * 2, reverse=True)
        return top_channels[:limit]
    
    def _generate_trending_topics(self, messages: List[str], limit: int = 5) -> List[Dict]:
        """Generate trending topics from messages"""
        # Extract topics using the summarizer
        topics = self.summarizer._extract_topics(messages, limit * 2)
        
        # Simulate mention counts
        trending_topics = []
        for topic in topics[:limit]:
            mentions = random.randint(3, 15)  # Simulate mention count
            trending_topics.append({
                'topic': topic,
                'mentions': mentions
            })
        
        # Sort by mentions
        trending_topics.sort(key=lambda x: x['mentions'], reverse=True)
        return trending_topics
    
    def _calculate_productivity_metrics(self, messages: List[Dict]) -> Dict[str, Any]:
        """Calculate productivity-related metrics"""
        total_messages = len(messages)
        
        if total_messages == 0:
            return {
                'focus_score': 0,
                'interruption_rate': 0,
                'response_efficiency': 0
            }
        
        # Calculate metrics
        urgent_count = sum(1 for msg in messages if msg.get('level') == 'URGENT')
        noise_count = sum(1 for msg in messages if msg.get('level') == 'NOISE')
        
        focus_score = max(0, 1 - (urgent_count + noise_count) / total_messages)
        interruption_rate = urgent_count / total_messages if total_messages > 0 else 0
        response_efficiency = 1 - noise_count / total_messages if total_messages > 0 else 0
        
        return {
            'focus_score': round(focus_score, 2),
            'interruption_rate': round(interruption_rate, 2),
            'response_efficiency': round(response_efficiency, 2),
            'signal_to_noise_ratio': round((total_messages - noise_count) / total_messages, 2) if total_messages > 0 else 0
        }

def generate_sample_messages(count: int = 50) -> List[Dict[str, Any]]:
    """Generate sample Slack messages for demonstration"""
    
    message_templates = {
        'urgent': [
            "URGENT: Production server is experiencing high latency",
            "Critical bug found in the payment processing system",
            "Client demo is in 30 minutes and the app is down",
            "Security alert: Unusual login activity detected",
            "ASAP: Need approval for emergency server maintenance",
            "Breaking: Major competitor just announced similar feature",
            "Help needed: Database connection failing",
            "Immediate action required: Customer data breach suspected"
        ],
        'important': [
            "Please review the Q4 budget proposal by end of day",
            "New feature requirements from our biggest client",
            "Team meeting scheduled for tomorrow at 2 PM",
            "Code review needed for the upcoming release",
            "Stakeholder feedback on the product roadmap",
            "Performance metrics show 20% improvement this quarter",
            "New hire starting next week - need onboarding plan",
            "Legal team needs to review the new user agreement"
        ],
        'fyi': [
            "FYI: Office will be closed next Friday for maintenance",
            "Updated company policies available in the handbook",
            "New parking regulations effective next month",
            "Lunch and learn session scheduled for next Tuesday",
            "IT department upgrading network infrastructure",
            "Employee survey results will be shared next week",
            "New coffee machine installed in the break room",
            "Quarterly all-hands meeting moved to next Thursday"
        ],
        'noise': [
            "Anyone know a good coffee shop near the office?",
            "Happy birthday to Sarah! 🎉",
            "Beautiful weather today! Perfect for a walk",
            "Did anyone watch the game last night?",
            "Thanks for the help with the presentation!",
            "Good morning everyone! Hope you have a great day",
            "LOL that meme in #random was hilarious",
            "Congratulations on the promotion, Mike!"
        ]
    }
    
    channels = ['general', 'dev-team', 'marketing', 'support', 'random', 'product', 'design']
    senders = [f'user_{i}' for i in range(1, 21)]
    
    messages = []
    
    for _ in range(count):
        # Choose message type with realistic distribution
        msg_type = random.choices(
            ['urgent', 'important', 'fyi', 'noise'],
            weights=[0.1, 0.3, 0.4, 0.2]
        )[0]
        
        content = random.choice(message_templates[msg_type])
        sender = random.choice(senders)
        channel = random.choice(channels)
        
        # Add some variation to messages
        if random.random() < 0.3:  # 30% chance to add variation
            variations = [
                f"{content} - thoughts?",
                f"{content} Let me know what you think.",
                f"{content} Thanks!",
                f"Update: {content}",
                f"FYI - {content}"
            ]
            content = random.choice(variations)
        
        messages.append({
            'content': content,
            'sender': sender,
            'channel': channel,
            'timestamp': datetime.now() - timedelta(minutes=random.randint(0, 1440))  # Random time in last 24h
        })
    
    return messages

# Example usage and testing
if __name__ == "__main__":
    # Test the classifier
    classifier = MessageClassifier()
    test_messages = [
        "URGENT: Production server is down!",
        "Please review the budget proposal",
        "FYI: New team member starting Monday",
        "Anyone want to grab coffee?"
    ]
    
    print("Message Classification Test:")
    for msg in test_messages:
        result = classifier.classify_priority(msg)
        print(f"Message: {msg}")
        print(f"Priority: {result['level']} (Confidence: {result['confidence']:.2f})")
        print(f"Reasoning: {result['reasoning']}")
        print("-" * 50)
    
    # Test the summarizer
    summarizer = ThreadSummarizer()
    test_thread = [
        "We need to discuss the Q4 roadmap",
        "I think we should prioritize mobile features",
        "Agreed, but we also need to fix performance issues",
        "Let's schedule a meeting for Thursday at 2 PM",
        "I'll prepare the requirements document"
    ]
    
    print("\nThread Summarization Test:")
    summary = summarizer.summarize_thread(test_thread)
    print(f"Summary: {summary['summary']}")
    print(f"Action Items: {summary['action_items']}")
    print(f"Key Decisions: {summary['key_decisions']}")