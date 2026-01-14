import os
import json
import sqlite3
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlackSemanticSearch:
    """
    AI-powered semantic search engine for Slack messages.
    
    Features:
    - Semantic search using sentence transformers
    - Vector similarity search with FAISS
    - Personalized results based on user behavior
    - Hybrid search combining semantic and keyword matching
    - Context-aware query processing with GPT-4
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_path: str = "slack_messages.db"):
        """
        Initialize the semantic search engine.
        
        Args:
            model_name: Sentence transformer model name
            db_path: SQLite database path for message storage
        """
        self.model_name = model_name
        self.db_path = db_path
        
        # Initialize AI models
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.sentence_model = SentenceTransformer(model_name)
        self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
        
        # Initialize OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize FAISS index
        self.faiss_index = None
        self.message_embeddings = None
        self.messages_df = None
        
        # Initialize TF-IDF for hybrid search
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        
        # User profiling for personalization
        self.user_profiles = {}
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for message storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE,
                user_id TEXT,
                username TEXT,
                channel_id TEXT,
                channel_name TEXT,
                message_text TEXT,
                timestamp DATETIME,
                thread_ts TEXT,
                message_type TEXT,
                reactions TEXT,
                file_attachments TEXT
            )
        """)
        
        # Create user interactions table for personalization
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                message_id TEXT,
                interaction_type TEXT,
                timestamp DATETIME
            )
        """)
        
        # Create search history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                query TEXT,
                search_type TEXT,
                results_count INTEGER,
                timestamp DATETIME
            )
        """)
        
        conn.commit()
        conn.close()
        
    def load_sample_data(self):
        """Load sample Slack messages for demonstration."""
        sample_messages = [
            {
                "message_id": "msg_001",
                "user_id": "user_alice",
                "username": "Alice Johnson",
                "channel_id": "ch_general",
                "channel_name": "general",
                "message_text": "Hey team, we need to discuss the Q4 budget planning meeting scheduled for next week. Please review the financial projections.",
                "timestamp": datetime.now() - timedelta(days=2),
                "thread_ts": None,
                "message_type": "message",
                "reactions": json.dumps(["thumbs_up", "eyes"]),
                "file_attachments": None
            },
            {
                "message_id": "msg_002",
                "user_id": "user_bob",
                "username": "Bob Smith",
                "channel_id": "ch_engineering",
                "channel_name": "engineering",
                "message_text": "The new API deployment is complete. All endpoints are working correctly. Performance metrics look good.",
                "timestamp": datetime.now() - timedelta(days=1),
                "thread_ts": None,
                "message_type": "message",
                "reactions": json.dumps(["rocket", "thumbs_up"]),
                "file_attachments": None
            },
            {
                "message_id": "msg_003",
                "user_id": "user_carol",
                "username": "Carol Davis",
                "channel_id": "ch_marketing",
                "channel_name": "marketing",
                "message_text": "Campaign performance update: CTR increased by 15% this month. The new creative assets are performing well.",
                "timestamp": datetime.now() - timedelta(hours=12),
                "thread_ts": None,
                "message_type": "message",
                "reactions": json.dumps(["chart_with_upwards_trend"]),
                "file_attachments": None
            },
            {
                "message_id": "msg_004",
                "user_id": "user_david",
                "username": "David Wilson",
                "channel_id": "ch_general",
                "channel_name": "general",
                "message_text": "Reminder: All-hands meeting tomorrow at 2 PM. We'll discuss project deadlines and resource allocation.",
                "timestamp": datetime.now() - timedelta(hours=6),
                "thread_ts": None,
                "message_type": "message",
                "reactions": json.dumps(["calendar"]),
                "file_attachments": None
            },
            {
                "message_id": "msg_005",
                "user_id": "user_eve",
                "username": "Eve Brown",
                "channel_id": "ch_design",
                "channel_name": "design",
                "message_text": "New UI mockups are ready for review. The user experience flow has been improved based on feedback.",
                "timestamp": datetime.now() - timedelta(hours=3),
                "thread_ts": None,
                "message_type": "message",
                "reactions": json.dumps(["art", "eyes"]),
                "file_attachments": json.dumps(["mockup_v2.figma"])
            },
            {
                "message_id": "msg_006",
                "user_id": "user_frank",
                "username": "Frank Miller",
                "channel_id": "ch_sales",
                "channel_name": "sales",
                "message_text": "Great news! We closed the enterprise deal with TechCorp. Revenue target for Q4 is looking achievable.",
                "timestamp": datetime.now() - timedelta(hours=1),
                "thread_ts": None,
                "message_type": "message",
                "reactions": json.dumps(["money_with_wings", "tada"]),
                "file_attachments": None
            }
        ]
        
        # Insert sample messages into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for msg in sample_messages:
            cursor.execute("""
                INSERT OR REPLACE INTO messages 
                (message_id, user_id, username, channel_id, channel_name, 
                 message_text, timestamp, thread_ts, message_type, reactions, file_attachments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                msg["message_id"], msg["user_id"], msg["username"],
                msg["channel_id"], msg["channel_name"], msg["message_text"],
                msg["timestamp"], msg["thread_ts"], msg["message_type"],
                msg["reactions"], msg["file_attachments"]
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Loaded {len(sample_messages)} sample messages")
        
        # Build search indices
        self.build_search_indices()
        
    def build_search_indices(self):
        """Build semantic and keyword search indices."""
        # Load messages from database
        conn = sqlite3.connect(self.db_path)
        self.messages_df = pd.read_sql_query(
            "SELECT * FROM messages ORDER BY timestamp DESC", conn
        )
        conn.close()
        
        if len(self.messages_df) == 0:
            logger.warning("No messages found in database")
            return
        
        logger.info(f"Building search indices for {len(self.messages_df)} messages")
        
        # Generate semantic embeddings
        message_texts = self.messages_df['message_text'].tolist()
        self.message_embeddings = self.sentence_model.encode(
            message_texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index for vector similarity search
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.message_embeddings / np.linalg.norm(
            self.message_embeddings, axis=1, keepdims=True
        )
        self.faiss_index.add(normalized_embeddings.astype('float32'))
        
        # Build TF-IDF index for keyword search
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(message_texts)
        
        logger.info("Search indices built successfully")
        
    def _process_query_with_gpt4(self, query: str, user_id: str) -> Dict[str, Any]:
        """Process query with GPT-4 for better understanding and expansion."""
        try:
            # Get user context
            user_context = self._get_user_context(user_id)
            
            prompt = f"""
            Analyze this Slack search query and provide structured information:
            
            Query: "{query}"
            User Context: {user_context}
            
            Please provide:
            1. Intent classification (information_seeking, troubleshooting, meeting_related, project_related, etc.)
            2. Key entities and topics
            3. Expanded query terms (synonyms, related concepts)
            4. Suggested filters (channels, users, time ranges)
            
            Respond in JSON format.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes search queries for better information retrieval."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error processing query with GPT-4: {e}")
            return {
                "intent": "information_seeking",
                "entities": [],
                "expanded_terms": [query],
                "suggested_filters": {}
            }
    
    def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context for personalization."""
        conn = sqlite3.connect(self.db_path)
        
        # Get user's recent activity
        recent_messages = pd.read_sql_query(
            "SELECT channel_name, message_text FROM messages WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10",
            conn, params=(user_id,)
        )
        
        # Get user's search history
        search_history = pd.read_sql_query(
            "SELECT query, search_type FROM search_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5",
            conn, params=(user_id,)
        )
        
        conn.close()
        
        return {
            "recent_channels": recent_messages['channel_name'].unique().tolist() if not recent_messages.empty else [],
            "recent_topics": recent_messages['message_text'].tolist()[:3] if not recent_messages.empty else [],
            "search_history": search_history['query'].tolist() if not search_history.empty else []
        }
    
    def _semantic_search(self, query: str, limit: int = 10) -> List[Tuple[int, float]]:
        """Perform semantic search using sentence transformers and FAISS."""
        if self.faiss_index is None:
            return []
        
        # Generate query embedding
        query_embedding = self.sentence_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search in FAISS index
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), limit)
        
        return list(zip(indices[0], scores[0]))
    
    def _keyword_search(self, query: str, limit: int = 10) -> List[Tuple[int, float]]:
        """Perform keyword search using TF-IDF."""
        if self.tfidf_matrix is None:
            return []
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-limit:][::-1]
        
        return [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0]
    
    def _hybrid_search(self, query: str, limit: int = 10, semantic_weight: float = 0.7) -> List[Tuple[int, float]]:
        """Combine semantic and keyword search results."""
        semantic_results = self._semantic_search(query, limit * 2)
        keyword_results = self._keyword_search(query, limit * 2)
        
        # Combine and rerank results
        combined_scores = {}
        
        # Add semantic scores
        for idx, score in semantic_results:
            combined_scores[idx] = semantic_weight * score
        
        # Add keyword scores
        for idx, score in keyword_results:
            if idx in combined_scores:
                combined_scores[idx] += (1 - semantic_weight) * score
            else:
                combined_scores[idx] = (1 - semantic_weight) * score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results[:limit]
    
    def _personalize_results(self, results: List[Tuple[int, float]], user_id: str) -> List[Tuple[int, float]]:
        """Apply personalization to search results."""
        if not results:
            return results
        
        user_context = self._get_user_context(user_id)
        recent_channels = set(user_context.get("recent_channels", []))
        
        # Boost results from user's frequently used channels
        personalized_results = []
        for idx, score in results:
            if idx < len(self.messages_df):
                message_channel = self.messages_df.iloc[idx]['channel_name']
                if message_channel in recent_channels:
                    score *= 1.2  # Boost score by 20%
            personalized_results.append((idx, score))
        
        return sorted(personalized_results, key=lambda x: x[1], reverse=True)
    
    def search(self, query: str, user_id: str = "default_user", 
               search_type: str = "hybrid", limit: int = 10) -> List[Dict[str, Any]]:
        """Main search function.
        
        Args:
            query: Search query
            user_id: User identifier for personalization
            search_type: 'semantic', 'keyword', or 'hybrid'
            limit: Maximum number of results
            
        Returns:
            List of search results with metadata
        """
        if self.messages_df is None or len(self.messages_df) == 0:
            logger.warning("No messages available for search")
            return []
        
        # Process query with GPT-4 for better understanding
        query_analysis = self._process_query_with_gpt4(query, user_id)
        
        # Perform search based on type
        if search_type == "semantic":
            raw_results = self._semantic_search(query, limit * 2)
        elif search_type == "keyword":
            raw_results = self._keyword_search(query, limit * 2)
        else:  # hybrid
            raw_results = self._hybrid_search(query, limit * 2)
        
        # Apply personalization
        personalized_results = self._personalize_results(raw_results, user_id)
        
        # Format results
        formatted_results = []
        for idx, score in personalized_results[:limit]:
            if idx < len(self.messages_df):
                message_row = self.messages_df.iloc[idx]
                result = {
                    "message_id": message_row['message_id'],
                    "score": float(score),
                    "message": message_row['message_text'],
                    "user": message_row['username'],
                    "user_id": message_row['user_id'],
                    "channel": message_row['channel_name'],
                    "timestamp": message_row['timestamp'],
                    "reactions": json.loads(message_row['reactions']) if message_row['reactions'] else [],
                    "file_attachments": json.loads(message_row['file_attachments']) if message_row['file_attachments'] else [],
                    "query_analysis": query_analysis
                }
                formatted_results.append(result)
        
        # Log search for analytics
        self._log_search(user_id, query, search_type, len(formatted_results))
        
        return formatted_results
    
    def _log_search(self, user_id: str, query: str, search_type: str, results_count: int):
        """Log search for analytics and personalization."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO search_history (user_id, query, search_type, results_count, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, query, search_type, results_count, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_search_analytics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get search analytics and insights."""
        conn = sqlite3.connect(self.db_path)
        
        if user_id:
            search_data = pd.read_sql_query(
                "SELECT * FROM search_history WHERE user_id = ? ORDER BY timestamp DESC",
                conn, params=(user_id,)
            )
        else:
            search_data = pd.read_sql_query(
                "SELECT * FROM search_history ORDER BY timestamp DESC", conn
            )
        
        conn.close()
        
        if search_data.empty:
            return {"total_searches": 0, "avg_results": 0, "popular_queries": []}
        
        analytics = {
            "total_searches": len(search_data),
            "avg_results": search_data['results_count'].mean(),
            "popular_queries": search_data['query'].value_counts().head(10).to_dict(),
            "search_types": search_data['search_type'].value_counts().to_dict(),
            "recent_searches": search_data.head(10)[['query', 'search_type', 'timestamp']].to_dict('records')
        }
        
        return analytics
    
    def save_indices(self, filepath: str = "search_indices.pkl"):
        """Save search indices to disk."""
        indices_data = {
            "message_embeddings": self.message_embeddings,
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "messages_df": self.messages_df
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(indices_data, f)
        
        # Save FAISS index separately
        if self.faiss_index:
            faiss.write_index(self.faiss_index, filepath.replace('.pkl', '_faiss.index'))
        
        logger.info(f"Search indices saved to {filepath}")
    
    def load_indices(self, filepath: str = "search_indices.pkl"):
        """Load search indices from disk."""
        try:
            with open(filepath, 'rb') as f:
                indices_data = pickle.load(f)
            
            self.message_embeddings = indices_data["message_embeddings"]
            self.tfidf_vectorizer = indices_data["tfidf_vectorizer"]
            self.tfidf_matrix = indices_data["tfidf_matrix"]
            self.messages_df = indices_data["messages_df"]
            
            # Load FAISS index
            faiss_path = filepath.replace('.pkl', '_faiss.index')
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
            
            logger.info(f"Search indices loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading indices: {e}")
            self.build_search_indices()


if __name__ == "__main__":
    # Example usage
    search_engine = SlackSemanticSearch()
    
    # Load sample data
    search_engine.load_sample_data()
    
    # Perform searches
    queries = [
        "budget planning meeting",
        "API deployment status",
        "marketing campaign performance",
        "project deadlines",
        "UI design review"
    ]
    
    for query in queries:
        print(f"\n=== Search: '{query}' ===")
        results = search_engine.search(query, user_id="test_user", search_type="hybrid")
        
        for i, result in enumerate(results[:3], 1):
            print(f"{i}. [{result['score']:.3f}] {result['user']} in #{result['channel']}:")
            print(f"   {result['message'][:100]}...")
    
    # Show analytics
    print("\n=== Search Analytics ===")
    analytics = search_engine.get_search_analytics()
    print(f"Total searches: {analytics['total_searches']}")
    print(f"Average results per search: {analytics['avg_results']:.1f}")
    print(f"Popular queries: {list(analytics['popular_queries'].keys())[:3]}")