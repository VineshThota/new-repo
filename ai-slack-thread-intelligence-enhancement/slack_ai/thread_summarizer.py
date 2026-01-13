"""Thread Summarizer Module

Provides AI-powered thread summarization capabilities for Slack conversations.
Extracts key information including summaries, decisions, action items, and participants.
"""

import re
from typing import Dict, List, Any
from textblob import TextBlob
import nltk
from collections import Counter
import datetime

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

class ThreadSummarizer:
    """AI-powered thread summarization for Slack conversations."""
    
    def __init__(self):
        """Initialize the thread summarizer."""
        self.stop_words = set(stopwords.words('english'))
        
        # Keywords for identifying different types of content
        self.decision_keywords = [
            'decided', 'decision', 'agreed', 'conclude', 'final', 'chosen',
            'selected', 'approved', 'confirmed', 'resolved', 'settled'
        ]
        
        self.action_keywords = [
            'todo', 'to do', 'action', 'task', 'need to', 'should', 'must',
            'will do', 'assign', 'responsible', 'deadline', 'by when', 'follow up'
        ]
        
        self.urgency_keywords = [
            'urgent', 'asap', 'immediately', 'critical', 'emergency', 'now',
            'today', 'deadline', 'important', 'priority'
        ]
    
    def summarize_thread(self, thread_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize a Slack thread and extract key information.
        
        Args:
            thread_messages: List of message dictionaries with 'text', 'user', 'timestamp'
            
        Returns:
            Dictionary containing summary, decisions, action items, and participants
        """
        if not thread_messages:
            return self._empty_summary()
        
        # Extract text content and metadata
        all_text = " ".join([msg.get('text', '') for msg in thread_messages])
        participants = list(set([msg.get('user', 'Unknown') for msg in thread_messages]))
        
        # Generate summary
        summary = self._generate_summary(all_text, thread_messages)
        
        # Extract key decisions
        decisions = self._extract_decisions(thread_messages)
        
        # Extract action items
        action_items = self._extract_action_items(thread_messages)
        
        # Analyze sentiment and urgency
        sentiment = self._analyze_sentiment(all_text)
        urgency_score = self._calculate_urgency(all_text)
        
        return {
            'summary': summary,
            'key_decisions': decisions,
            'action_items': action_items,
            'participants': participants,
            'message_count': len(thread_messages),
            'sentiment': sentiment,
            'urgency_score': urgency_score,
            'thread_duration': self._calculate_duration(thread_messages)
        }
    
    def _generate_summary(self, text: str, messages: List[Dict]) -> str:
        """Generate a concise summary of the thread content."""
        if not text.strip():
            return "Empty thread with no content."
        
        # Clean and tokenize text
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 2:
            return text.strip()
        
        # Score sentences based on word frequency and position
        word_freq = self._calculate_word_frequency(text)
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word.isalnum() and word not in self.stop_words]
            
            if len(words) > 0:
                score = sum(word_freq.get(word, 0) for word in words) / len(words)
                # Boost score for sentences with decision/action keywords
                if any(keyword in sentence.lower() for keyword in self.decision_keywords + self.action_keywords):
                    score *= 1.5
                sentence_scores[sentence] = score
        
        # Select top sentences for summary
        if len(sentence_scores) == 0:
            return "Thread contains mostly non-text content."
        
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        summary_length = min(3, len(top_sentences))
        
        summary_sentences = [sent[0] for sent in top_sentences[:summary_length]]
        
        # Reorder sentences by original appearance
        ordered_summary = []
        for sentence in sentences:
            if sentence in summary_sentences:
                ordered_summary.append(sentence)
        
        return " ".join(ordered_summary)
    
    def _extract_decisions(self, messages: List[Dict]) -> List[str]:
        """Extract key decisions from the thread."""
        decisions = []
        
        for msg in messages:
            text = msg.get('text', '').lower()
            
            # Look for decision-related keywords
            for keyword in self.decision_keywords:
                if keyword in text:
                    # Extract the sentence containing the decision
                    sentences = sent_tokenize(msg.get('text', ''))
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            decisions.append(f"{msg.get('user', 'Someone')}: {sentence.strip()}")
                            break
        
        return list(set(decisions))  # Remove duplicates
    
    def _extract_action_items(self, messages: List[Dict]) -> List[str]:
        """Extract action items and tasks from the thread."""
        action_items = []
        
        for msg in messages:
            text = msg.get('text', '').lower()
            original_text = msg.get('text', '')
            
            # Look for action-related keywords
            for keyword in self.action_keywords:
                if keyword in text:
                    sentences = sent_tokenize(original_text)
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            action_items.append(f"{msg.get('user', 'Someone')}: {sentence.strip()}")
                            break
            
            # Look for @mentions (assignments)
            mentions = re.findall(r'@\w+', original_text)
            if mentions and any(action_word in text for action_word in ['need', 'should', 'can you', 'please']):
                action_items.append(f"Assignment: {original_text.strip()}")
        
        return list(set(action_items))  # Remove duplicates
    
    def _calculate_word_frequency(self, text: str) -> Dict[str, float]:
        """Calculate word frequency for text summarization."""
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        if not words:
            return {}
        
        word_count = Counter(words)
        max_freq = max(word_count.values())
        
        # Normalize frequencies
        word_freq = {word: count/max_freq for word, count in word_count.items()}
        
        return word_freq
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze the overall sentiment of the thread."""
        if not text.strip():
            return {'polarity': 0.0, 'subjectivity': 0.0, 'label': 'neutral'}
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.1:
            label = 'positive'
        elif polarity < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'label': label
        }
    
    def _calculate_urgency(self, text: str) -> float:
        """Calculate urgency score based on keywords and patterns."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        urgency_score = 0.0
        
        # Check for urgency keywords
        for keyword in self.urgency_keywords:
            if keyword in text_lower:
                urgency_score += 0.2
        
        # Check for exclamation marks
        urgency_score += min(text.count('!') * 0.1, 0.3)
        
        # Check for all caps words
        words = text.split()
        caps_words = [word for word in words if word.isupper() and len(word) > 2]
        urgency_score += min(len(caps_words) * 0.1, 0.2)
        
        return min(urgency_score, 1.0)  # Cap at 1.0
    
    def _calculate_duration(self, messages: List[Dict]) -> str:
        """Calculate the duration of the thread conversation."""
        if len(messages) < 2:
            return "Single message"
        
        try:
            # For demo purposes, return a sample duration
            # In real implementation, parse actual timestamps
            return "2 hours 15 minutes"
        except:
            return "Unknown duration"
    
    def _empty_summary(self) -> Dict[str, Any]:
        """Return empty summary structure."""
        return {
            'summary': "No messages to summarize.",
            'key_decisions': [],
            'action_items': [],
            'participants': [],
            'message_count': 0,
            'sentiment': {'polarity': 0.0, 'subjectivity': 0.0, 'label': 'neutral'},
            'urgency_score': 0.0,
            'thread_duration': "0 minutes"
        }