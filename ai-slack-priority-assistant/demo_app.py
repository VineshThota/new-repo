#!/usr/bin/env python3
"""
AI Slack Priority Assistant - Streamlit Demo Application

Interactive demo interface for testing message analysis and priority scoring.
Provides a user-friendly way to test the AI capabilities without Slack integration.

Author: Vinesh Thota
Date: January 2026
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from typing import Dict, List

# Page configuration
st.set_page_config(
    page_title="AI Slack Priority Assistant Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.priority-high {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.priority-medium {
    background-color: #fff3e0;
    border-left: 4px solid #ff9800;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.priority-low {
    background-color: #e8f5e8;
    border-left: 4px solid #4caf50;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Sample messages for testing
SAMPLE_MESSAGES = [
    {
        "text": "URGENT: Production server is down! Need immediate attention from the engineering team. This is affecting all our customers.",
        "user_name": "John Smith (CTO)",
        "channel_name": "#critical-alerts",
        "user_id": "U123456",
        "channel_id": "C123456",
        "message_ts": "1704723600.123456"
    },
    {
        "text": "Hey team, just wanted to share some funny memes I found online 😄 Hope everyone is having a great day!",
        "user_name": "Sarah Johnson",
        "channel_name": "#random",
        "user_id": "U234567",
        "channel_id": "C234567",
        "message_ts": "1704723700.123456"
    },
    {
        "text": "Can someone please review the quarterly budget proposal? We need to submit it by end of day today. The client meeting is tomorrow morning.",
        "user_name": "Mike Davis (Director)",
        "channel_name": "#finance",
        "user_id": "U345678",
        "channel_id": "C345678",
        "message_ts": "1704723800.123456"
    },
    {
        "text": "Does anyone know where the coffee machine manual is? The new intern is trying to figure out how to use it.",
        "user_name": "Lisa Wong",
        "channel_name": "#general",
        "user_id": "U456789",
        "channel_id": "C456789",
        "message_ts": "1704723900.123456"
    },
    {
        "text": "Project Alpha milestone completed! Great work everyone. The client is very happy with our progress. Next phase starts Monday.",
        "user_name": "Alex Chen (Project Manager)",
        "channel_name": "#project-alpha",
        "user_id": "U567890",
        "channel_id": "C567890",
        "message_ts": "1704724000.123456"
    }
]

def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API call to the FastAPI backend"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return {}
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the FastAPI server is running on http://localhost:8000")
        return {}
    except requests.exceptions.Timeout:
        st.error("⏱️ API request timed out. The AI models might be loading.")
        return {}
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return {}

def analyze_message(message_data: Dict) -> Dict:
    """Analyze a single message using the API"""
    return call_api("/analyze-message", "POST", message_data)

def get_priority_color(score: float) -> str:
    """Get color based on priority score"""
    if score >= 7.0:
        return "#f44336"  # Red
    elif score >= 5.0:
        return "#ff9800"  # Orange
    elif score >= 3.0:
        return "#2196f3"  # Blue
    else:
        return "#4caf50"  # Green

def get_priority_label(score: float) -> str:
    """Get priority label based on score"""
    if score >= 7.0:
        return "🔴 High Priority"
    elif score >= 5.0:
        return "🟡 Medium Priority"
    elif score >= 3.0:
        return "🔵 Low Priority"
    else:
        return "🟢 Very Low Priority"

def display_message_analysis(analysis: Dict, message_data: Dict):
    """Display detailed message analysis"""
    if not analysis:
        return
    
    priority_score = analysis.get('priority_score', 0)
    priority_class = "priority-high" if priority_score >= 7 else "priority-medium" if priority_score >= 5 else "priority-low"
    
    st.markdown(f"""
    <div class="{priority_class}">
        <h4>{get_priority_label(priority_score)} (Score: {priority_score}/10)</h4>
        <p><strong>From:</strong> {message_data['user_name']} in {message_data['channel_name']}</p>
        <p><strong>Message:</strong> {message_data['text']}</p>
        <p><strong>AI Summary:</strong> {analysis.get('ai_summary', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed scores
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Urgency", f"{analysis.get('urgency_score', 0)}/3", 
                 help="How time-sensitive is this message?")
    
    with col2:
        st.metric("Importance", f"{analysis.get('importance_score', 0)}/2", 
                 help="How important is the sender/content?")
    
    with col3:
        st.metric("Relevance", f"{analysis.get('relevance_score', 0)}/3", 
                 help="How relevant is this to the user?")
    
    with col4:
        st.metric("Sentiment", f"{analysis.get('sentiment_score', 0):.2f}", 
                 help="Emotional tone (-1 to 1)")
    
    # Entities and Topics
    if analysis.get('entities'):
        st.write("**Entities Found:**", ", ".join(analysis['entities']))
    
    if analysis.get('topics'):
        st.write("**Topics:**", ", ".join(analysis['topics']))

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Slack Priority Assistant Demo</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This demo showcases how AI can intelligently analyze and prioritize Slack messages to reduce information overload.
    The system uses multiple AI techniques including NLP, sentiment analysis, and named entity recognition.
    """)
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # API Health Check
    st.sidebar.subheader("API Status")
    health_check = call_api("/health")
    if health_check:
        st.sidebar.success("✅ API Connected")
        st.sidebar.json(health_check)
    else:
        st.sidebar.error("❌ API Disconnected")
        st.sidebar.info("Start the FastAPI server: `uvicorn main:app --reload`")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Demo Mode",
        ["Single Message Analysis", "Batch Analysis", "Custom Message"]
    )
    
    if mode == "Single Message Analysis":
        st.header("📝 Single Message Analysis")
        
        # Message selection
        selected_idx = st.selectbox(
            "Choose a sample message to analyze:",
            range(len(SAMPLE_MESSAGES)),
            format_func=lambda x: f"Message {x+1}: {SAMPLE_MESSAGES[x]['text'][:50]}..."
        )
        
        selected_message = SAMPLE_MESSAGES[selected_idx]
        
        # Display original message
        st.subheader("Original Message")
        st.info(f"**From:** {selected_message['user_name']} in {selected_message['channel_name']}\n\n{selected_message['text']}")
        
        # Analyze button
        if st.button("🔍 Analyze Message", type="primary"):
            with st.spinner("Analyzing message with AI..."):
                analysis = analyze_message(selected_message)
                
                if analysis:
                    st.subheader("AI Analysis Results")
                    display_message_analysis(analysis, selected_message)
                    
                    # Visualization
                    st.subheader("Score Breakdown")
                    scores = {
                        'Urgency': analysis.get('urgency_score', 0),
                        'Importance': analysis.get('importance_score', 0),
                        'Relevance': analysis.get('relevance_score', 0),
                        'Sentiment': abs(analysis.get('sentiment_score', 0))
                    }
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(scores.keys()),
                            y=list(scores.values()),
                            marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
                        )
                    ])
                    fig.update_layout(
                        title="Message Analysis Scores",
                        yaxis_title="Score",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    elif mode == "Batch Analysis":
        st.header("📊 Batch Analysis")
        st.write("Analyze all sample messages and compare their priority scores.")
        
        if st.button("🚀 Analyze All Messages", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, message in enumerate(SAMPLE_MESSAGES):
                status_text.text(f"Analyzing message {i+1}/{len(SAMPLE_MESSAGES)}...")
                
                analysis = analyze_message(message)
                if analysis:
                    results.append({
                        'Message': message['text'][:100] + "...",
                        'User': message['user_name'],
                        'Channel': message['channel_name'],
                        'Priority Score': analysis.get('priority_score', 0),
                        'Urgency': analysis.get('urgency_score', 0),
                        'Importance': analysis.get('importance_score', 0),
                        'Relevance': analysis.get('relevance_score', 0),
                        'Is Important': analysis.get('is_important', False)
                    })
                
                progress_bar.progress((i + 1) / len(SAMPLE_MESSAGES))
                time.sleep(0.5)  # Small delay for demo effect
            
            status_text.text("Analysis complete!")
            
            if results:
                # Display results table
                df = pd.DataFrame(results)
                st.subheader("Analysis Results")
                st.dataframe(df, use_container_width=True)
                
                # Priority distribution chart
                st.subheader("Priority Score Distribution")
                fig = px.bar(
                    df, 
                    x='User', 
                    y='Priority Score',
                    color='Priority Score',
                    color_continuous_scale='RdYlGn_r',
                    title="Message Priority Scores by User"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Score comparison radar chart
                st.subheader("Score Comparison")
                fig = go.Figure()
                
                for i, result in enumerate(results):
                    fig.add_trace(go.Scatterpolar(
                        r=[result['Urgency'], result['Importance'], result['Relevance'], result['Priority Score']/3],
                        theta=['Urgency', 'Importance', 'Relevance', 'Priority'],
                        fill='toself',
                        name=f"Message {i+1}"
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 3]
                        )
                    ),
                    showlegend=True,
                    title="Multi-dimensional Score Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif mode == "Custom Message":
        st.header("✏️ Custom Message Analysis")
        st.write("Enter your own message to see how the AI analyzes it.")
        
        # Custom message input
        col1, col2 = st.columns(2)
        
        with col1:
            custom_text = st.text_area(
                "Message Text",
                placeholder="Enter the message content here...",
                height=150
            )
            
            user_name = st.text_input(
                "User Name",
                value="John Doe",
                help="Name of the message sender"
            )
        
        with col2:
            channel_name = st.text_input(
                "Channel Name",
                value="#general",
                help="Slack channel name"
            )
            
            # Advanced options
            st.subheader("Advanced Options")
            
            include_authority = st.checkbox(
                "Sender has authority (CEO, CTO, Manager, etc.)",
                help="This will increase the importance score"
            )
            
            if include_authority:
                authority_titles = ["CEO", "CTO", "Manager", "Director", "VP", "Lead"]
                selected_title = st.selectbox("Authority Title", authority_titles)
                user_name = f"{user_name} ({selected_title})"
        
        if st.button("🔍 Analyze Custom Message", type="primary") and custom_text:
            custom_message = {
                "text": custom_text,
                "user_name": user_name,
                "channel_name": channel_name,
                "user_id": "U999999",
                "channel_id": "C999999",
                "message_ts": str(time.time())
            }
            
            with st.spinner("Analyzing your custom message..."):
                analysis = analyze_message(custom_message)
                
                if analysis:
                    st.subheader("Analysis Results")
                    display_message_analysis(analysis, custom_message)
                    
                    # Explanation
                    st.subheader("Why this score?")
                    
                    explanations = []
                    
                    urgency = analysis.get('urgency_score', 0)
                    if urgency >= 2.5:
                        explanations.append("🔥 High urgency detected due to time-sensitive keywords")
                    elif urgency >= 1.5:
                        explanations.append("⚡ Medium urgency detected")
                    
                    importance = analysis.get('importance_score', 0)
                    if importance >= 1.0:
                        explanations.append("👑 High importance due to sender authority or business keywords")
                    
                    sentiment = analysis.get('sentiment_score', 0)
                    if abs(sentiment) > 0.5:
                        explanations.append(f"😊 Strong emotional tone detected ({sentiment:.2f})")
                    
                    if explanations:
                        for explanation in explanations:
                            st.info(explanation)
                    else:
                        st.info("📝 This appears to be a standard message with normal priority")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🎯 How It Works
    
    The AI Slack Priority Assistant uses multiple machine learning techniques:
    
    - **Natural Language Processing (NLP)**: Understands message content and context
    - **Named Entity Recognition**: Identifies important people, dates, and topics
    - **Sentiment Analysis**: Detects emotional tone and urgency
    - **Authority Detection**: Recognizes important senders (managers, executives)
    - **Keyword Analysis**: Identifies business-critical terms and time-sensitive language
    - **Multi-factor Scoring**: Combines all factors into a single priority score
    
    **Priority Scoring (0-10 scale):**
    - 🔴 **7-10**: High Priority (immediate attention needed)
    - 🟡 **5-7**: Medium Priority (important but not urgent)
    - 🔵 **3-5**: Low Priority (can be addressed later)
    - 🟢 **0-3**: Very Low Priority (informational/casual)
    """)
    
    # Technical details in expander
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **AI Models Used:**
        - Sentiment Analysis: `cardiffnlp/twitter-roberta-base-sentiment-latest`
        - Summarization: `facebook/bart-large-cnn`
        - Named Entity Recognition: `spaCy en_core_web_sm`
        - Text Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
        
        **Scoring Algorithm:**
        ```
        Priority Score = (
            Urgency × 0.30 +      # Time sensitivity
            Importance × 0.25 +   # Sender/content importance
            Relevance × 0.35 +    # Personal relevance
            |Sentiment| × 0.10    # Emotional intensity
        ) × scaling_factor
        ```
        
        **API Endpoints:**
        - `POST /analyze-message`: Analyze single message
        - `POST /priority-feed`: Get prioritized message feed
        - `POST /filter-messages`: Filter messages by criteria
        - `GET /health`: API health check
        """)

if __name__ == "__main__":
    main()