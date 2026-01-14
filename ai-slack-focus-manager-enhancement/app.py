import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import re
import json
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import sqlite3
import hashlib
import time

# Configure Streamlit page
st.set_page_config(
    page_title="SlackFocus AI - Intelligent Message Prioritization",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class SlackMessage:
    """Represents a Slack message with AI-enhanced metadata"""
    id: str
    text: str
    sender: str
    channel: str
    timestamp: datetime
    priority_score: float
    urgency_level: str
    requires_response: bool
    thread_ts: Optional[str] = None
    mentions: List[str] = None
    
@dataclass
class FocusSession:
    """Represents a focus session with filtering rules"""
    id: str
    user_id: str
    start_time: datetime
    duration_minutes: int
    allow_critical: bool
    filtered_count: int = 0
    critical_count: int = 0
    is_active: bool = True

class SlackFocusAI:
    """Main AI engine for Slack message prioritization and focus management"""
    
    def __init__(self):
        self.priority_classifier = None
        self.urgency_detector = None
        self.summarizer = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.init_ai_models()
        self.init_database()
        
    def init_ai_models(self):
        """Initialize AI models for message analysis"""
        # Initialize priority classifier (Random Forest)
        self.priority_classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10
        )
        
        # Initialize summarization pipeline
        try:
            self.summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                max_length=130,
                min_length=30,
                do_sample=False
            )
        except Exception:
            # Fallback to a lighter model if BART is not available
            self.summarizer = None
            
        # Train models with sample data
        self._train_priority_model()
        
    def init_database(self):
        """Initialize SQLite database for storing messages and sessions"""
        conn = sqlite3.connect('slack_focus.db')
        cursor = conn.cursor()
        
        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                text TEXT,
                sender TEXT,
                channel TEXT,
                timestamp DATETIME,
                priority_score REAL,
                urgency_level TEXT,
                requires_response BOOLEAN
            )
        ''')
        
        # Create focus sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS focus_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                start_time DATETIME,
                duration_minutes INTEGER,
                allow_critical BOOLEAN,
                filtered_count INTEGER,
                critical_count INTEGER,
                is_active BOOLEAN
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _train_priority_model(self):
        """Train the priority classification model with sample data"""
        # Sample training data (in production, this would be real Slack data)
        sample_messages = [
            ("Server is down! Need immediate help", "critical"),
            ("URGENT: Customer complaint about billing", "critical"),
            ("Meeting in 5 minutes, are you joining?", "important"),
            ("Please review the PR when you have time", "important"),
            ("How was your weekend?", "low"),
            ("Coffee anyone?", "low"),
            ("Deployment failed, investigating", "critical"),
            ("New feature request from client", "important"),
            ("Team lunch tomorrow at 12", "normal"),
            ("Great job on the presentation!", "low")
        ]
        
        texts = [msg[0] for msg in sample_messages]
        labels = [msg[1] for msg in sample_messages]
        
        # Convert text to features
        X = self.vectorizer.fit_transform(texts)
        
        # Map labels to numbers
        label_map = {'low': 0, 'normal': 1, 'important': 2, 'critical': 3}
        y = [label_map[label] for label in labels]
        
        # Train the model
        self.priority_classifier.fit(X, y)
        
    def analyze_message_priority(self, message_text: str, sender: str, channel: str) -> Dict:
        """Analyze message priority using AI models"""
        # Extract features
        features = self.vectorizer.transform([message_text])
        
        # Predict priority
        priority_num = self.priority_classifier.predict(features)[0]
        priority_proba = self.priority_classifier.predict_proba(features)[0]
        
        priority_labels = ['low', 'normal', 'important', 'critical']
        priority_level = priority_labels[priority_num]
        confidence = max(priority_proba)
        
        # Detect urgency keywords
        urgency_keywords = [
            'urgent', 'asap', 'immediately', 'emergency', 'critical', 
            'down', 'broken', 'failed', 'error', 'help', 'blocked'
        ]
        
        urgency_score = sum(1 for keyword in urgency_keywords 
                          if keyword in message_text.lower())
        
        # Determine if response is required
        response_indicators = ['?', 'please', 'can you', 'could you', 'need']
        requires_response = any(indicator in message_text.lower() 
                              for indicator in response_indicators)
        
        # Calculate final priority score (0-1)
        base_score = priority_num / 3.0
        urgency_boost = min(urgency_score * 0.2, 0.3)
        sender_boost = 0.1 if sender.lower() in ['ceo', 'manager', 'lead'] else 0
        channel_boost = 0.1 if 'urgent' in channel.lower() or 'critical' in channel.lower() else 0
        
        final_score = min(base_score + urgency_boost + sender_boost + channel_boost, 1.0)
        
        return {
            'priority_level': priority_level,
            'priority_score': final_score,
            'confidence': confidence,
            'urgency_score': urgency_score,
            'requires_response': requires_response
        }
        
    def summarize_thread(self, messages: List[str]) -> Dict:
        """Summarize a thread of messages"""
        if not messages:
            return {'summary': 'No messages to summarize', 'key_points': []}
            
        # Combine all messages
        combined_text = ' '.join(messages)
        
        # Extract key points using simple heuristics
        sentences = re.split(r'[.!?]+', combined_text)
        key_points = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and any(keyword in sentence.lower() 
                                        for keyword in ['decision', 'action', 'next', 'will', 'should']):
                key_points.append(sentence)
                
        # Use AI summarizer if available
        summary = combined_text[:200] + '...' if len(combined_text) > 200 else combined_text
        
        if self.summarizer and len(combined_text) > 100:
            try:
                summary_result = self.summarizer(combined_text)
                summary = summary_result[0]['summary_text']
            except Exception:
                pass  # Fall back to truncated text
                
        return {
            'summary': summary,
            'key_points': key_points[:5],  # Top 5 key points
            'message_count': len(messages)
        }
        
    def generate_sample_messages(self, count: int = 20) -> List[SlackMessage]:
        """Generate sample Slack messages for demo purposes"""
        sample_data = [
            ("Server is experiencing high latency, investigating now", "DevOps Team", "#alerts"),
            ("Great job on the presentation today!", "Sarah Manager", "#general"),
            ("Can someone review my PR? It's blocking the release", "John Developer", "#dev-team"),
            ("Coffee break anyone?", "Mike Designer", "#random"),
            ("URGENT: Customer reporting payment issues", "Support Team", "#customer-support"),
            ("Meeting moved to 3 PM today", "Alice PM", "#project-alpha"),
            ("New security vulnerability found, patch needed ASAP", "Security Team", "#security"),
            ("Anyone know where the office keys are?", "Receptionist", "#general"),
            ("Deployment scheduled for tonight at 11 PM", "DevOps Team", "#deployments"),
            ("Happy birthday to our amazing designer!", "HR Team", "#celebrations"),
            ("Database backup failed, need to investigate", "DBA Team", "#database"),
            ("Lunch and learn session tomorrow on AI", "Learning Team", "#learning"),
            ("Client wants to add new features to the project", "Sales Team", "#sales"),
            ("Code freeze starts Monday for release", "Release Team", "#releases"),
            ("Internet is slow today, anyone else experiencing this?", "IT Support", "#it-support"),
            ("Quarterly review meeting next week", "Management", "#management"),
            ("New employee starting Monday, please welcome them", "HR Team", "#general"),
            ("API rate limits exceeded, scaling up servers", "Backend Team", "#backend"),
            ("Design mockups ready for review", "Design Team", "#design"),
            ("Weekend hackathon results are in!", "Engineering", "#hackathon")
        ]
        
        messages = []
        for i, (text, sender, channel) in enumerate(sample_data[:count]):
            # Analyze message priority
            analysis = self.analyze_message_priority(text, sender, channel)
            
            message = SlackMessage(
                id=f"msg_{i+1}",
                text=text,
                sender=sender,
                channel=channel,
                timestamp=datetime.now() - timedelta(minutes=np.random.randint(1, 1440)),
                priority_score=analysis['priority_score'],
                urgency_level=analysis['priority_level'],
                requires_response=analysis['requires_response']
            )
            messages.append(message)
            
        return sorted(messages, key=lambda x: x.priority_score, reverse=True)

def main():
    """Main Streamlit application"""
    st.title("🎯 SlackFocus AI - Intelligent Message Prioritization")
    st.markdown("*Transform your Slack experience with AI-powered focus management*")
    
    # Initialize the AI system
    if 'slack_ai' not in st.session_state:
        with st.spinner("Initializing AI models..."):
            st.session_state.slack_ai = SlackFocusAI()
    
    slack_ai = st.session_state.slack_ai
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature",
        ["Priority Dashboard", "Focus Session Manager", "Thread Summarizer", "Analytics"]
    )
    
    if page == "Priority Dashboard":
        show_priority_dashboard(slack_ai)
    elif page == "Focus Session Manager":
        show_focus_manager(slack_ai)
    elif page == "Thread Summarizer":
        show_thread_summarizer(slack_ai)
    elif page == "Analytics":
        show_analytics(slack_ai)

def show_priority_dashboard(slack_ai):
    """Display the priority dashboard"""
    st.header("📊 Priority Dashboard")
    st.markdown("AI-filtered messages based on importance and urgency")
    
    # Generate sample messages
    messages = slack_ai.generate_sample_messages(20)
    
    # Priority filter
    col1, col2, col3 = st.columns(3)
    with col1:
        priority_filter = st.selectbox(
            "Minimum Priority Level",
            ["All", "Normal+", "Important+", "Critical Only"]
        )
    
    with col2:
        channel_filter = st.selectbox(
            "Channel Filter",
            ["All Channels"] + list(set(msg.channel for msg in messages))
        )
    
    with col3:
        response_filter = st.checkbox("Only messages requiring response")
    
    # Filter messages
    filtered_messages = messages
    
    if priority_filter == "Normal+":
        filtered_messages = [msg for msg in filtered_messages if msg.priority_score >= 0.33]
    elif priority_filter == "Important+":
        filtered_messages = [msg for msg in filtered_messages if msg.priority_score >= 0.66]
    elif priority_filter == "Critical Only":
        filtered_messages = [msg for msg in filtered_messages if msg.priority_score >= 0.8]
    
    if channel_filter != "All Channels":
        filtered_messages = [msg for msg in filtered_messages if msg.channel == channel_filter]
    
    if response_filter:
        filtered_messages = [msg for msg in filtered_messages if msg.requires_response]
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Messages", len(messages))
    with col2:
        st.metric("Filtered Messages", len(filtered_messages))
    with col3:
        critical_count = len([msg for msg in filtered_messages if msg.priority_score >= 0.8])
        st.metric("Critical Messages", critical_count)
    with col4:
        response_count = len([msg for msg in filtered_messages if msg.requires_response])
        st.metric("Need Response", response_count)
    
    # Display messages
    st.subheader("Messages")
    for msg in filtered_messages:
        priority_color = {
            'critical': '🔴',
            'important': '🟡', 
            'normal': '🟢',
            'low': '⚪'
        }.get(msg.urgency_level, '⚪')
        
        with st.expander(f"{priority_color} {msg.sender} in {msg.channel} - {msg.urgency_level.title()}"):
            st.write(f"**Message:** {msg.text}")
            st.write(f"**Priority Score:** {msg.priority_score:.2f}")
            st.write(f"**Timestamp:** {msg.timestamp.strftime('%Y-%m-%d %H:%M')}")
            if msg.requires_response:
                st.write("**⚠️ Response Required**")

def show_focus_manager(slack_ai):
    """Display the focus session manager"""
    st.header("🔕 Focus Session Manager")
    st.markdown("Manage your focus time with intelligent message filtering")
    
    # Focus session controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Start Focus Session")
        duration = st.slider("Duration (minutes)", 15, 240, 90)
        allow_critical = st.checkbox("Allow critical messages", value=True)
        
        if st.button("Start Focus Session", type="primary"):
            session_id = f"session_{int(time.time())}"
            session = FocusSession(
                id=session_id,
                user_id="demo_user",
                start_time=datetime.now(),
                duration_minutes=duration,
                allow_critical=allow_critical
            )
            st.session_state.current_session = session
            st.success(f"Focus session started! Duration: {duration} minutes")
    
    with col2:
        st.subheader("Session Status")
        if hasattr(st.session_state, 'current_session'):
            session = st.session_state.current_session
            elapsed = datetime.now() - session.start_time
            remaining = timedelta(minutes=session.duration_minutes) - elapsed
            
            if remaining.total_seconds() > 0:
                st.success(f"🎯 Focus session active")
                st.write(f"Time remaining: {str(remaining).split('.')[0]}")
                
                if st.button("End Session Early"):
                    del st.session_state.current_session
                    st.rerun()
            else:
                st.info("Focus session completed!")
                del st.session_state.current_session
        else:
            st.info("No active focus session")
    
    # Simulate message filtering during focus session
    if hasattr(st.session_state, 'current_session'):
        st.subheader("Messages During Focus Session")
        messages = slack_ai.generate_sample_messages(10)
        session = st.session_state.current_session
        
        allowed_messages = []
        filtered_messages = []
        
        for msg in messages:
            if session.allow_critical and msg.priority_score >= 0.8:
                allowed_messages.append(msg)
            else:
                filtered_messages.append(msg)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Allowed Messages ({len(allowed_messages)})**")
            for msg in allowed_messages:
                st.write(f"🔴 {msg.sender}: {msg.text[:50]}...")
        
        with col2:
            st.write(f"**Filtered Messages ({len(filtered_messages)})**")
            for msg in filtered_messages[:5]:
                st.write(f"⚪ {msg.sender}: {msg.text[:50]}...")

def show_thread_summarizer(slack_ai):
    """Display the thread summarizer"""
    st.header("🧠 Thread Summarizer")
    st.markdown("AI-powered summarization of long Slack conversations")
    
    # Sample thread messages
    sample_thread = [
        "Hey team, we need to discuss the new feature requirements for the mobile app.",
        "I think we should prioritize the user authentication flow first.",
        "Good point! We also need to consider the offline functionality.",
        "The design team has prepared some mockups. Should we review them in the next meeting?",
        "Yes, let's schedule a review session for tomorrow at 2 PM.",
        "I'll send out the calendar invite. We should also include the backend team.",
        "Agreed. The API endpoints need to be finalized before we start development.",
        "I'll prepare the technical specifications document by end of day.",
        "Perfect! This will help us stay on track for the Q2 release."
    ]
    
    # Input area for custom thread
    st.subheader("Thread Messages")
    use_sample = st.checkbox("Use sample thread", value=True)
    
    if use_sample:
        thread_messages = sample_thread
        st.write("**Sample Thread Messages:**")
        for i, msg in enumerate(thread_messages, 1):
            st.write(f"{i}. {msg}")
    else:
        thread_input = st.text_area(
            "Enter thread messages (one per line)",
            height=200,
            placeholder="Enter each message on a new line..."
        )
        thread_messages = [msg.strip() for msg in thread_input.split('\n') if msg.strip()]
    
    if st.button("Summarize Thread", type="primary") and thread_messages:
        with st.spinner("Analyzing thread..."):
            summary_result = slack_ai.summarize_thread(thread_messages)
        
        st.subheader("📝 Thread Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**AI Summary:**")
            st.write(summary_result['summary'])
            
        with col2:
            st.write("**Key Points:**")
            for point in summary_result['key_points']:
                st.write(f"• {point}")
        
        st.metric("Messages Analyzed", summary_result['message_count'])

def show_analytics(slack_ai):
    """Display analytics and insights"""
    st.header("📈 Productivity Analytics")
    st.markdown("Insights into your communication patterns and focus effectiveness")
    
    # Generate sample data for analytics
    messages = slack_ai.generate_sample_messages(50)
    
    # Priority distribution
    priority_counts = {}
    for msg in messages:
        level = msg.urgency_level
        priority_counts[level] = priority_counts.get(level, 0) + 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Message Priority Distribution")
        fig = px.pie(
            values=list(priority_counts.values()),
            names=list(priority_counts.keys()),
            color_discrete_map={
                'critical': '#ff4444',
                'important': '#ffaa00', 
                'normal': '#44ff44',
                'low': '#cccccc'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Messages by Channel")
        channel_counts = {}
        for msg in messages:
            channel = msg.channel
            channel_counts[channel] = channel_counts.get(channel, 0) + 1
        
        fig = px.bar(
            x=list(channel_counts.keys()),
            y=list(channel_counts.values()),
            labels={'x': 'Channel', 'y': 'Message Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Time-based analysis
    st.subheader("Message Timeline")
    df = pd.DataFrame([
        {
            'timestamp': msg.timestamp,
            'priority_score': msg.priority_score,
            'channel': msg.channel,
            'urgency_level': msg.urgency_level
        }
        for msg in messages
    ])
    
    fig = px.scatter(
        df,
        x='timestamp',
        y='priority_score',
        color='urgency_level',
        hover_data=['channel'],
        color_discrete_map={
            'critical': '#ff4444',
            'important': '#ffaa00', 
            'normal': '#44ff44',
            'low': '#cccccc'
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Productivity metrics
    st.subheader("Productivity Insights")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_priority = np.mean([msg.priority_score for msg in messages])
        st.metric("Avg Priority Score", f"{avg_priority:.2f}")
    
    with col2:
        response_rate = len([msg for msg in messages if msg.requires_response]) / len(messages)
        st.metric("Response Required", f"{response_rate:.1%}")
    
    with col3:
        critical_rate = len([msg for msg in messages if msg.priority_score >= 0.8]) / len(messages)
        st.metric("Critical Messages", f"{critical_rate:.1%}")
    
    with col4:
        focus_efficiency = 1 - critical_rate  # Simplified metric
        st.metric("Focus Efficiency", f"{focus_efficiency:.1%}")

if __name__ == "__main__":
    main()