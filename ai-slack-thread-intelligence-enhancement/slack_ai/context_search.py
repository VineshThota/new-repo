"""Context Search Module

Provides AI-powered natural language search capabilities for Slack messages.
Uses semantic similarity and vector embeddings for intelligent information retrieval.
"""

import re
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import json
from collections import defaultdict
import datetime

class ContextSearch:
    """AI-powered context search for Slack conversations."""
    
    def __init__(self, messages: List[Dict] = None):
        """Initialize the context search engine.
        
        Args:
            messages: List of message dictionaries to index
        """
        self.messages = messages or []
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Search enhancement keywords
        self.topic_keywords = {
            'pricing': ['price', 'cost', 'pricing', 'budget', 'money', 'payment', 'billing'],
            'production': ['production', 'prod', 'server', 'deployment', 'live', 'outage'],
            'meeting': ['meeting', 'call', 'schedule', 'calendar', 'zoom', 'conference'],
            'bug': ['bug', 'issue', 'problem', 'error', 'broken', 'fix', 'debug'],
            'feature': ['feature', 'functionality', 'requirement', 'spec', 'development'],
            'customer': ['customer', 'client', 'user', 'feedback', 'support', 'complaint']
        }
        
        # Initialize search index
        self.search_index = None
        self.message_vectors = None
        
        if self.messages:
            self._build_search_index()
    
    def add_messages(self, new_messages: List[Dict]):
        """Add new messages to the search index."""
        self.messages.extend(new_messages)
        self._build_search_index()
    
    def query(self, search_query: str, limit: int = 10, min_score: float = 0.1) -> List[Dict]:
        """Search for messages using natural language query.
        
        Args:
            search_query: Natural language search query
            limit: Maximum number of results to return
            min_score: Minimum relevance score threshold
            
        Returns:
            List of relevant messages with relevance scores
        """
        if not search_query.strip() or not self.messages:
            return []
        
        # Preprocess query
        processed_query = self._preprocess_query(search_query)
        
        # Get semantic matches
        semantic_results = self._semantic_search(processed_query, limit * 2)
        
        # Get keyword matches
        keyword_results = self._keyword_search(search_query, limit * 2)
        
        # Combine and rank results
        combined_results = self._combine_results(semantic_results, keyword_results)
        
        # Filter by minimum score and limit
        filtered_results = [
            result for result in combined_results 
            if result['score'] >= min_score
        ][:limit]
        
        return filtered_results
    
    def _build_search_index(self):
        """Build the search index from messages."""
        if not self.messages:
            return
        
        # Extract text content for vectorization
        texts = []
        for msg in self.messages:
            text = msg.get('text', '')
            # Include channel and user context
            channel = msg.get('channel', '')
            user = msg.get('user', '')
            
            # Combine text with context
            full_text = f"{text} channel:{channel} user:{user}"
            texts.append(full_text)
        
        # Build TF-IDF vectors
        try:
            self.message_vectors = self.vectorizer.fit_transform(texts)
            self.search_index = True
        except Exception as e:
            print(f"Error building search index: {e}")
            self.search_index = False
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess the search query for better matching."""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Expand topic-related queries
        for topic, keywords in self.topic_keywords.items():
            if topic in query:
                query += " " + " ".join(keywords)
        
        # Handle question words
        question_patterns = {
            r'\bwhat.*decide[d]?\b': 'decision decided agreed',
            r'\bwhen.*meeting\b': 'meeting schedule calendar',
            r'\bwho.*responsible\b': 'assigned responsible owner',
            r'\bhow.*fix\b': 'solution fix resolve',
            r'\bwhy.*choose\b': 'reason decision rationale'
        }
        
        for pattern, expansion in question_patterns.items():
            if re.search(pattern, query):
                query += " " + expansion
        
        return query
    
    def _semantic_search(self, query: str, limit: int) -> List[Dict]:
        """Perform semantic search using TF-IDF vectors."""
        if not self.search_index or not self.message_vectors:
            return []
        
        try:
            # Vectorize the query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.message_vectors).flatten()
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:limit]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    result = self.messages[idx].copy()
                    result['score'] = float(similarities[idx])
                    result['match_type'] = 'semantic'
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def _keyword_search(self, query: str, limit: int) -> List[Dict]:
        """Perform keyword-based search with fuzzy matching."""
        query_words = set(query.lower().split())
        results = []
        
        for msg in self.messages:
            text = msg.get('text', '').lower()
            channel = msg.get('channel', '').lower()
            user = msg.get('user', '').lower()
            
            # Calculate keyword match score
            text_words = set(text.split())
            
            # Exact matches
            exact_matches = len(query_words.intersection(text_words))
            
            # Partial matches (substring)
            partial_matches = 0
            for query_word in query_words:
                if any(query_word in text_word for text_word in text_words):
                    partial_matches += 1
            
            # Context matches (channel, user)
            context_matches = 0
            for query_word in query_words:
                if query_word in channel or query_word in user:
                    context_matches += 1
            
            # Calculate total score
            if exact_matches > 0 or partial_matches > 0 or context_matches > 0:
                score = (
                    exact_matches * 1.0 + 
                    partial_matches * 0.5 + 
                    context_matches * 0.3
                ) / len(query_words)
                
                result = msg.copy()
                result['score'] = score
                result['match_type'] = 'keyword'
                results.append(result)
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def _combine_results(self, semantic_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """Combine and deduplicate results from different search methods."""
        # Create a dictionary to track unique messages
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            msg_id = self._get_message_id(result)
            if msg_id not in combined:
                combined[msg_id] = result
            else:
                # Boost score if found by multiple methods
                combined[msg_id]['score'] = max(combined[msg_id]['score'], result['score']) * 1.2
                combined[msg_id]['match_type'] = 'combined'
        
        # Add keyword results
        for result in keyword_results:
            msg_id = self._get_message_id(result)
            if msg_id not in combined:
                combined[msg_id] = result
            else:
                # Boost score if found by multiple methods
                combined[msg_id]['score'] = max(combined[msg_id]['score'], result['score']) * 1.2
                combined[msg_id]['match_type'] = 'combined'
        
        # Convert back to list and sort by score
        results = list(combined.values())
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def _get_message_id(self, message: Dict) -> str:
        """Generate a unique ID for a message."""
        return f"{message.get('channel', '')}_{message.get('ts', '')}_{message.get('user', '')}"
    
    def search_by_topic(self, topic: str, limit: int = 10) -> List[Dict]:
        """Search for messages related to a specific topic."""
        if topic.lower() in self.topic_keywords:
            keywords = self.topic_keywords[topic.lower()]
            query = " ".join(keywords)
            return self.query(query, limit)
        else:
            return self.query(topic, limit)
    
    def search_by_user(self, username: str, query: str = "", limit: int = 10) -> List[Dict]:
        """Search for messages from a specific user."""
        user_messages = [msg for msg in self.messages if msg.get('user', '').lower() == username.lower()]
        
        if not query:
            # Return recent messages from user
            return sorted(user_messages, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]
        
        # Search within user's messages
        temp_search = ContextSearch(user_messages)
        return temp_search.query(query, limit)
    
    def search_by_channel(self, channel: str, query: str = "", limit: int = 10) -> List[Dict]:
        """Search for messages in a specific channel."""
        channel_messages = [msg for msg in self.messages if msg.get('channel', '').lower() == channel.lower()]
        
        if not query:
            # Return recent messages from channel
            return sorted(channel_messages, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]
        
        # Search within channel messages
        temp_search = ContextSearch(channel_messages)
        return temp_search.query(query, limit)
    
    def search_by_date_range(self, start_date: str, end_date: str, query: str = "", limit: int = 10) -> List[Dict]:
        """Search for messages within a date range."""
        # For demo purposes, we'll use a simple timestamp comparison
        # In a real implementation, you'd parse actual timestamps
        
        filtered_messages = []
        for msg in self.messages:
            timestamp = msg.get('timestamp', '')
            # Simple demo filter - in reality you'd parse dates properly
            if start_date <= timestamp <= end_date:
                filtered_messages.append(msg)
        
        if not query:
            return sorted(filtered_messages, key=lambda x: x.get('timestamp', ''), reverse=True)[:limit]
        
        # Search within date-filtered messages
        temp_search = ContextSearch(filtered_messages)
        return temp_search.query(query, limit)
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Get search suggestions based on partial query."""
        suggestions = []
        
        # Topic-based suggestions
        for topic in self.topic_keywords.keys():
            if topic.startswith(partial_query.lower()):
                suggestions.append(f"Search for {topic} discussions")
        
        # Common search patterns
        common_patterns = [
            "What did we decide about",
            "When is the meeting for",
            "Who is responsible for",
            "How do we fix",
            "Why did we choose",
            "Status update on",
            "Any issues with"
        ]
        
        for pattern in common_patterns:
            if pattern.lower().startswith(partial_query.lower()):
                suggestions.append(pattern)
        
        return suggestions[:5]
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the search index."""
        if not self.messages:
            return {'total_messages': 0, 'indexed': False}
        
        # Calculate basic stats
        channels = set(msg.get('channel', '') for msg in self.messages)
        users = set(msg.get('user', '') for msg in self.messages)
        
        # Calculate average message length
        total_length = sum(len(msg.get('text', '')) for msg in self.messages)
        avg_length = total_length / len(self.messages) if self.messages else 0
        
        return {
            'total_messages': len(self.messages),
            'unique_channels': len(channels),
            'unique_users': len(users),
            'average_message_length': avg_length,
            'indexed': bool(self.search_index),
            'vocabulary_size': len(self.vectorizer.vocabulary_) if self.search_index else 0
        }