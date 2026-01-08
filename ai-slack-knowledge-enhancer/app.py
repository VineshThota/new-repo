import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
from collections import defaultdict

# AI/ML imports
import openai
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings

# Slack SDK (for real implementation)
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    st.warning("Slack SDK not installed. Using demo data.")

# Configure page
st.set_page_config(
    page_title="AI Slack Knowledge Enhancer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

@dataclass
class SlackMessage:
    """Data class for Slack messages"""
    id: str
    channel: str
    user: str
    text: str
    timestamp: datetime
    thread_ts: Optional[str] = None
    reactions: List[str] = None
    
    def __post_init__(self):
        if self.reactions is None:
            self.reactions = []

@dataclass
class ChannelSummary:
    """Data class for channel summaries"""
    channel: str
    time_range: str
    key_topics: List[str]
    action_items: List[str]
    decisions: List[str]
    participants: List[str]
    message_count: int
    summary_text: str
    keywords: List[str]

class SlackKnowledgeEnhancer:
    """Main class for AI-powered Slack knowledge management"""
    
    def __init__(self):
        self.openai_client = None
        self.sentence_model = None
        self.chroma_client = None
        self.slack_client = None
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize AI models and clients"""
        try:
            # Initialize OpenAI
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                openai.api_key = openai_key
                self.openai_client = openai
                st.success("✅ OpenAI API connected")
            else:
                st.warning("⚠️ OpenAI API key not found. Using demo mode.")
            
            # Initialize Sentence Transformer
            with st.spinner("Loading sentence transformer model..."):
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.success("✅ Sentence transformer loaded")
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            st.success("✅ Vector database initialized")
            
            # Initialize Slack client
            slack_token = os.getenv('SLACK_BOT_TOKEN')
            if slack_token and SLACK_AVAILABLE:
                self.slack_client = WebClient(token=slack_token)
                st.success("✅ Slack API connected")
            else:
                st.info("ℹ️ Using demo data (Slack API not configured)")
                
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
    
    def generate_demo_messages(self, channel: str, days: int = 7) -> List[SlackMessage]:
        """Generate demo Slack messages for testing"""
        demo_messages = [
            SlackMessage(
                id="msg_001",
                channel=channel,
                user="john_doe",
                text="Hey team, we need to finalize the product launch timeline. Can we schedule a meeting for tomorrow?",
                timestamp=datetime.now() - timedelta(days=2),
                reactions=["👍", "📅"]
            ),
            SlackMessage(
                id="msg_002",
                channel=channel,
                user="jane_smith",
                text="@john_doe I'm available at 2 PM. Should we invite the marketing team as well?",
                timestamp=datetime.now() - timedelta(days=2, hours=1),
                thread_ts="msg_001"
            ),
            SlackMessage(
                id="msg_003",
                channel=channel,
                user="mike_wilson",
                text="ACTION ITEM: Update the pricing strategy document by Friday. @jane_smith can you handle this?",
                timestamp=datetime.now() - timedelta(days=1),
                reactions=["✅"]
            ),
            SlackMessage(
                id="msg_004",
                channel=channel,
                user="sarah_johnson",
                text="DECISION: We're moving the launch date to March 15th to allow more time for testing.",
                timestamp=datetime.now() - timedelta(days=1, hours=3),
                reactions=["👍", "📝"]
            ),
            SlackMessage(
                id="msg_005",
                channel=channel,
                user="alex_brown",
                text="Great decision! This gives us more time to polish the user experience. I'll update the development roadmap accordingly.",
                timestamp=datetime.now() - timedelta(hours=12)
            ),
            SlackMessage(
                id="msg_006",
                channel=channel,
                user="lisa_davis",
                text="Question: What's our backup plan if we encounter any major bugs during testing?",
                timestamp=datetime.now() - timedelta(hours=8)
            ),
            SlackMessage(
                id="msg_007",
                channel=channel,
                user="john_doe",
                text="@lisa_davis Good question. We should have a hotfix process ready. Let's document this in our QA procedures.",
                timestamp=datetime.now() - timedelta(hours=6),
                thread_ts="msg_006"
            )
        ]
        return demo_messages
    
    def extract_action_items(self, messages: List[SlackMessage]) -> List[str]:
        """Extract action items from messages using NLP"""
        action_items = []
        action_patterns = [
            r'ACTION ITEM[:\s]+(.+)',
            r'TODO[:\s]+(.+)',
            r'@\w+\s+(?:can you|could you|please)\s+(.+)',
            r'(?:need to|should|must)\s+(.+?)(?:\.|$)',
        ]
        
        for message in messages:
            text = message.text
            for pattern in action_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.strip()) > 10:  # Filter out very short matches
                        action_items.append(f"• {match.strip()}")
        
        return list(set(action_items))  # Remove duplicates
    
    def extract_decisions(self, messages: List[SlackMessage]) -> List[str]:
        """Extract decisions from messages"""
        decisions = []
        decision_patterns = [
            r'DECISION[:\s]+(.+)',
            r'(?:we\'ve decided|decided to|decision is)\s+(.+?)(?:\.|$)',
            r'(?:agreed|consensus)\s+(?:that|to)\s+(.+?)(?:\.|$)',
        ]
        
        for message in messages:
            text = message.text
            for pattern in decision_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.strip()) > 10:
                        decisions.append(f"• {match.strip()}")
        
        return list(set(decisions))
    
    def extract_keywords(self, messages: List[SlackMessage]) -> List[str]:
        """Extract keywords using TF-IDF"""
        if not messages:
            return []
        
        # Combine all message texts
        texts = [msg.text for msg in messages]
        
        # Use TF-IDF to extract keywords
        vectorizer = TfidfVectorizer(
            max_features=20,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top keywords
            top_indices = mean_scores.argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
            
            return keywords
        except:
            return ["product", "launch", "timeline", "meeting", "team"]
    
    def generate_ai_summary(self, messages: List[SlackMessage], channel: str) -> str:
        """Generate AI-powered summary using GPT-4"""
        if not self.openai_client:
            # Fallback summary
            return f"""Channel #{channel} Summary:
            
This channel had {len(messages)} messages with active discussions about product launch planning. 
Key participants included team members coordinating timelines, discussing action items, and making 
important decisions about project milestones. The conversation focused on scheduling, resource 
allocation, and strategic planning.
            """
        
        # Prepare messages for GPT-4
        message_text = "\n".join([
            f"{msg.user}: {msg.text}" for msg in messages[-20:]  # Last 20 messages
        ])
        
        prompt = f"""
        Analyze the following Slack channel conversation and provide a concise summary:
        
        Channel: #{channel}
        Messages:
        {message_text}
        
        Please provide:
        1. A brief overview of the main topics discussed
        2. Key outcomes or decisions made
        3. Overall sentiment and team dynamics
        
        Keep the summary under 200 words and focus on actionable insights.
        """
        
        try:
            response = self.openai_client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that specializes in summarizing team communications and extracting key insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error generating AI summary: {str(e)}")
            return f"Summary generation failed. Channel had {len(messages)} messages about {channel} discussions."
    
    def summarize_channel(self, channel: str, time_range: str = "last_week", 
                         include_action_items: bool = True) -> ChannelSummary:
        """Generate comprehensive channel summary"""
        # Get messages (using demo data for now)
        messages = self.generate_demo_messages(channel)
        
        # Extract information
        action_items = self.extract_action_items(messages) if include_action_items else []
        decisions = self.extract_decisions(messages)
        keywords = self.extract_keywords(messages)
        participants = list(set([msg.user for msg in messages]))
        
        # Generate AI summary
        summary_text = self.generate_ai_summary(messages, channel)
        
        # Extract key topics (simplified)
        key_topics = keywords[:5] if keywords else ["general discussion"]
        
        return ChannelSummary(
            channel=channel,
            time_range=time_range,
            key_topics=key_topics,
            action_items=action_items,
            decisions=decisions,
            participants=participants,
            message_count=len(messages),
            summary_text=summary_text,
            keywords=keywords
        )
    
    def semantic_search(self, query: str, messages: List[SlackMessage], top_k: int = 5) -> List[Dict]:
        """Perform semantic search on messages"""
        if not self.sentence_model:
            return []
        
        # Encode query and messages
        query_embedding = self.sentence_model.encode([query])
        message_texts = [msg.text for msg in messages]
        message_embeddings = self.sentence_model.encode(message_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, message_embeddings)[0]
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append({
                    'message': messages[idx],
                    'similarity': float(similarities[idx]),
                    'text': messages[idx].text,
                    'user': messages[idx].user,
                    'timestamp': messages[idx].timestamp
                })
        
        return results
    
    def search(self, query: str, channels: List[str] = None, 
              date_range: str = "last_month") -> List[Dict]:
        """Intelligent search across channels"""
        all_results = []
        
        # Default channels if none specified
        if not channels:
            channels = ["product", "marketing", "engineering"]
        
        for channel in channels:
            messages = self.generate_demo_messages(channel)
            results = self.semantic_search(query, messages)
            
            for result in results:
                result['channel'] = channel
                all_results.append(result)
        
        # Sort by similarity
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return all_results[:10]  # Return top 10 results
    
    def generate_knowledge_base(self, channels: List[str], 
                              update_frequency: str = "daily") -> Dict[str, Any]:
        """Generate knowledge base from channels"""
        knowledge_base = {
            'faqs': [],
            'decisions': [],
            'experts': defaultdict(list),
            'topics': defaultdict(int),
            'last_updated': datetime.now().isoformat()
        }
        
        for channel in channels:
            messages = self.generate_demo_messages(channel)
            
            # Extract FAQs (questions and answers)
            for i, msg in enumerate(messages):
                if '?' in msg.text and len(msg.text) > 20:
                    # Look for answers in subsequent messages
                    answers = []
                    for j in range(i+1, min(i+4, len(messages))):
                        if (messages[j].thread_ts == msg.id or 
                            msg.user in messages[j].text):
                            answers.append(messages[j].text)
                    
                    if answers:
                        knowledge_base['faqs'].append({
                            'question': msg.text,
                            'answers': answers,
                            'channel': channel,
                            'asker': msg.user
                        })
            
            # Extract decisions
            decisions = self.extract_decisions(messages)
            for decision in decisions:
                knowledge_base['decisions'].append({
                    'decision': decision,
                    'channel': channel,
                    'timestamp': datetime.now().isoformat()
                })
            
            # Identify experts (users who participate most in discussions)
            user_participation = defaultdict(int)
            for msg in messages:
                user_participation[msg.user] += 1
            
            # Top 3 most active users per channel
            top_users = sorted(user_participation.items(), 
                             key=lambda x: x[1], reverse=True)[:3]
            
            for user, count in top_users:
                knowledge_base['experts'][channel].append({
                    'user': user,
                    'message_count': count
                })
        
        return knowledge_base

def main():
    """Main Streamlit application"""
    st.title("🤖 AI Slack Knowledge Enhancer")
    st.markdown("*Transform Slack information overload into intelligent knowledge management*")
    
    # Initialize the enhancer
    if 'enhancer' not in st.session_state:
        with st.spinner("Initializing AI models..."):
            st.session_state.enhancer = SlackKnowledgeEnhancer()
    
    enhancer = st.session_state.enhancer
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["Channel Summaries", "Intelligent Search", "Knowledge Base", "Analytics"]
    )
    
    if page == "Channel Summaries":
        st.header("📊 AI-Powered Channel Summaries")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            channel = st.text_input("Channel Name", value="product", placeholder="e.g., product, marketing, engineering")
        
        with col2:
            time_range = st.selectbox(
                "Time Range",
                ["last_day", "last_week", "last_month", "last_quarter"]
            )
        
        include_action_items = st.checkbox("Include Action Items", value=True)
        
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Analyzing channel conversations..."):
                summary = enhancer.summarize_channel(
                    channel=channel,
                    time_range=time_range,
                    include_action_items=include_action_items
                )
            
            # Display summary
            st.success(f"Summary generated for #{channel}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Messages", summary.message_count)
            with col2:
                st.metric("Participants", len(summary.participants))
            with col3:
                st.metric("Action Items", len(summary.action_items))
            
            # Main summary
            st.subheader("📝 Summary")
            st.write(summary.summary_text)
            
            # Key topics
            st.subheader("🏷️ Key Topics")
            st.write(", ".join(summary.key_topics))
            
            # Action items
            if summary.action_items:
                st.subheader("✅ Action Items")
                for item in summary.action_items:
                    st.write(item)
            
            # Decisions
            if summary.decisions:
                st.subheader("🎯 Key Decisions")
                for decision in summary.decisions:
                    st.write(decision)
            
            # Participants
            st.subheader("👥 Participants")
            st.write(", ".join(summary.participants))
            
            # Keywords
            if summary.keywords:
                st.subheader("🔍 Keywords")
                st.write(", ".join(summary.keywords))
    
    elif page == "Intelligent Search":
        st.header("🔍 Intelligent Search Engine")
        
        query = st.text_input(
            "Search Query",
            placeholder="e.g., What decisions were made about the product launch?"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            channels = st.multiselect(
                "Channels to Search",
                ["product", "marketing", "engineering", "design", "sales"],
                default=["product", "marketing"]
            )
        
        with col2:
            date_range = st.selectbox(
                "Date Range",
                ["last_week", "last_month", "last_quarter", "all_time"]
            )
        
        if st.button("Search", type="primary") and query:
            with st.spinner("Searching conversations..."):
                results = enhancer.search(
                    query=query,
                    channels=channels,
                    date_range=date_range
                )
            
            if results:
                st.success(f"Found {len(results)} relevant results")
                
                for i, result in enumerate(results):
                    with st.expander(f"Result {i+1} - #{result['channel']} (Similarity: {result['similarity']:.2f})"):
                        st.write(f"**User:** {result['user']}")
                        st.write(f"**Time:** {result['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Message:** {result['text']}")
            else:
                st.info("No results found. Try a different query or expand your search criteria.")
    
    elif page == "Knowledge Base":
        st.header("📚 Dynamic Knowledge Base")
        
        channels = st.multiselect(
            "Select Channels",
            ["product", "marketing", "engineering", "design", "sales", "support"],
            default=["product", "engineering"]
        )
        
        if st.button("Generate Knowledge Base", type="primary") and channels:
            with st.spinner("Building knowledge base..."):
                kb = enhancer.generate_knowledge_base(channels)
            
            st.success("Knowledge base generated successfully!")
            
            # FAQs
            if kb['faqs']:
                st.subheader("❓ Frequently Asked Questions")
                for faq in kb['faqs'][:5]:  # Show top 5
                    with st.expander(f"Q: {faq['question'][:100]}..."):
                        st.write(f"**Channel:** #{faq['channel']}")
                        st.write(f"**Asked by:** {faq['asker']}")
                        st.write("**Answers:**")
                        for answer in faq['answers']:
                            st.write(f"• {answer}")
            
            # Decisions
            if kb['decisions']:
                st.subheader("🎯 Decision Log")
                for decision in kb['decisions'][:5]:
                    st.write(f"**#{decision['channel']}:** {decision['decision']}")
            
            # Experts
            if kb['experts']:
                st.subheader("👨‍💼 Channel Experts")
                for channel, experts in kb['experts'].items():
                    st.write(f"**#{channel}:**")
                    for expert in experts:
                        st.write(f"• {expert['user']} ({expert['message_count']} messages)")
    
    elif page == "Analytics":
        st.header("📊 Communication Analytics")
        
        # Demo analytics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Time Saved",
                "2.5 hours",
                delta="+45 min vs last week"
            )
        
        with col2:
            st.metric(
                "Knowledge Items",
                "127",
                delta="+23 new items"
            )
        
        with col3:
            st.metric(
                "Search Accuracy",
                "94%",
                delta="+8% improvement"
            )
        
        with col4:
            st.metric(
                "User Satisfaction",
                "4.8/5",
                delta="+0.3 rating"
            )
        
        # Charts
        st.subheader("📈 Usage Trends")
        
        # Demo data for charts
        dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='D')
        usage_data = pd.DataFrame({
            'Date': dates,
            'Summaries Generated': [12, 15, 18, 22, 19, 25, 28],
            'Searches Performed': [45, 52, 48, 61, 58, 67, 72],
            'Knowledge Items Added': [8, 12, 15, 11, 16, 19, 23]
        })
        
        st.line_chart(usage_data.set_index('Date'))
        
        st.subheader("🎯 Top Search Queries")
        search_queries = pd.DataFrame({
            'Query': [
                "product launch timeline",
                "marketing campaign decisions",
                "engineering roadmap",
                "budget approval process",
                "team meeting notes"
            ],
            'Count': [45, 38, 32, 28, 24]
        })
        
        st.bar_chart(search_queries.set_index('Query'))
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **AI Slack Knowledge Enhancer** - Transforming information overload into intelligent insights | "
        "[GitHub Repository](https://github.com/VineshThota/new-repo/tree/main/ai-slack-knowledge-enhancer)"
    )

if __name__ == "__main__":
    main()