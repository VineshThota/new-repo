import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import re
from textblob import TextBlob
import random

# Import our AI modules
from slack_ai.thread_summarizer import ThreadSummarizer
from slack_ai.priority_classifier import PriorityClassifier
from slack_ai.context_search import ContextSearch
from slack_ai.sample_data import generate_sample_data

# Page configuration
st.set_page_config(
    page_title="AI Slack Thread Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
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

# Initialize session state
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = generate_sample_data()
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = ThreadSummarizer()
if 'classifier' not in st.session_state:
    st.session_state.classifier = PriorityClassifier()
if 'search' not in st.session_state:
    st.session_state.search = ContextSearch(st.session_state.sample_data)

def main():
    st.markdown('<h1 class="main-header">🧠 AI Slack Thread Intelligence Enhancement</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Solving Slack's Information Overload Problem with AI**
    
    This demo showcases how AI can transform Slack's biggest pain points:
    - 📊 **Thread Summarization**: Get instant summaries of long discussions
    - 🎯 **Priority Detection**: Never miss critical messages again
    - 🔍 **Smart Search**: Find context with natural language queries
    - 📈 **Analytics**: Understand communication patterns
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["Dashboard", "Thread Summarizer", "Priority Inbox", "Smart Search", "Analytics"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Thread Summarizer":
        show_thread_summarizer()
    elif page == "Priority Inbox":
        show_priority_inbox()
    elif page == "Smart Search":
        show_smart_search()
    elif page == "Analytics":
        show_analytics()

def show_dashboard():
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Messages",
            value=len(st.session_state.sample_data),
            delta="+23 today"
        )
    
    with col2:
        high_priority = sum(1 for msg in st.session_state.sample_data 
                           if st.session_state.classifier.classify_priority(msg['text'])['level'] == 'HIGH')
        st.metric(
            label="High Priority",
            value=high_priority,
            delta="-2 from yesterday"
        )
    
    with col3:
        active_threads = len(set(msg.get('thread_ts', msg['ts']) for msg in st.session_state.sample_data))
        st.metric(
            label="Active Threads",
            value=active_threads,
            delta="+5 this week"
        )
    
    with col4:
        avg_response_time = "2.3 hrs"
        st.metric(
            label="Avg Response Time",
            value=avg_response_time,
            delta="-0.5 hrs"
        )
    
    st.markdown("---")
    
    # Recent activity
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Message Volume Trend")
        
        # Generate sample trend data
        dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='D')
        volumes = [45, 52, 38, 61, 47, 33, 28]
        
        fig = px.line(
            x=dates, y=volumes,
            title="Daily Message Volume",
            labels={'x': 'Date', 'y': 'Messages'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Priority Distribution")
        
        priority_counts = {'HIGH': high_priority, 'MEDIUM': 15, 'LOW': 8}
        fig = px.pie(
            values=list(priority_counts.values()),
            names=list(priority_counts.keys()),
            color_discrete_map={'HIGH': '#f44336', 'MEDIUM': '#ff9800', 'LOW': '#4caf50'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent high-priority messages
    st.subheader("🚨 Recent High-Priority Messages")
    
    high_priority_messages = []
    for msg in st.session_state.sample_data[-10:]:
        priority = st.session_state.classifier.classify_priority(msg['text'])
        if priority['level'] == 'HIGH':
            high_priority_messages.append({
                'Channel': msg['channel'],
                'User': msg['user'],
                'Message': msg['text'][:100] + '...' if len(msg['text']) > 100 else msg['text'],
                'Time': msg['timestamp'],
                'Confidence': f"{priority['confidence']:.1%}"
            })
    
    if high_priority_messages:
        df = pd.DataFrame(high_priority_messages)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No high-priority messages in recent activity.")

def show_thread_summarizer():
    st.header("📝 Thread Summarizer")
    
    st.markdown("""
    **Problem**: Long Slack threads become unmanageable, causing users to miss key decisions and action items.
    
    **AI Solution**: Automatically extract summaries, decisions, and action items from any thread.
    """)
    
    # Thread selection
    threads = {}
    for msg in st.session_state.sample_data:
        thread_id = msg.get('thread_ts', msg['ts'])
        if thread_id not in threads:
            threads[thread_id] = []
        threads[thread_id].append(msg)
    
    # Filter threads with multiple messages
    multi_message_threads = {k: v for k, v in threads.items() if len(v) > 2}
    
    if multi_message_threads:
        thread_options = [f"Thread {i+1}: {v[0]['channel']} - {len(v)} messages" 
                         for i, (k, v) in enumerate(multi_message_threads.items())]
        
        selected_thread = st.selectbox("Select a thread to summarize:", thread_options)
        
        if selected_thread:
            thread_index = int(selected_thread.split(":")[0].split()[1]) - 1
            thread_data = list(multi_message_threads.values())[thread_index]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📄 Original Thread")
                for msg in thread_data:
                    st.markdown(f"**{msg['user']}** ({msg['timestamp']})")
                    st.markdown(f"> {msg['text']}")
                    st.markdown("---")
            
            with col2:
                st.subheader("🤖 AI Summary")
                
                if st.button("Generate Summary", type="primary"):
                    with st.spinner("Analyzing thread..."):
                        summary = st.session_state.summarizer.summarize_thread(thread_data)
                    
                    st.markdown("**📋 Summary:**")
                    st.info(summary['summary'])
                    
                    if summary['key_decisions']:
                        st.markdown("**🎯 Key Decisions:**")
                        for decision in summary['key_decisions']:
                            st.success(f"• {decision}")
                    
                    if summary['action_items']:
                        st.markdown("**✅ Action Items:**")
                        for item in summary['action_items']:
                            st.warning(f"• {item}")
                    
                    if summary['participants']:
                        st.markdown("**👥 Key Participants:**")
                        st.write(", ".join(summary['participants']))
    else:
        st.info("No multi-message threads found in sample data. In a real implementation, this would analyze your actual Slack threads.")

def show_priority_inbox():
    st.header("🎯 Priority Inbox")
    
    st.markdown("""
    **Problem**: Important messages get buried in the noise, causing missed deadlines and poor communication.
    
    **AI Solution**: Automatically classify message priority and surface what needs immediate attention.
    """)
    
    # Priority filter
    priority_filter = st.selectbox(
        "Filter by priority:",
        ["All", "HIGH", "MEDIUM", "LOW"]
    )
    
    # Analyze all messages for priority
    prioritized_messages = []
    for msg in st.session_state.sample_data:
        priority = st.session_state.classifier.classify_priority(msg['text'])
        prioritized_messages.append({
            'message': msg,
            'priority': priority
        })
    
    # Sort by priority and confidence
    priority_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    prioritized_messages.sort(
        key=lambda x: (priority_order[x['priority']['level']], x['priority']['confidence']),
        reverse=True
    )
    
    # Filter messages
    if priority_filter != "All":
        prioritized_messages = [msg for msg in prioritized_messages 
                              if msg['priority']['level'] == priority_filter]
    
    # Display messages
    for msg_data in prioritized_messages[:20]:  # Show top 20
        msg = msg_data['message']
        priority = msg_data['priority']
        
        priority_class = f"priority-{priority['level'].lower()}"
        
        st.markdown(f"""
        <div class="{priority_class}">
            <strong>🔥 {priority['level']} PRIORITY</strong> (Confidence: {priority['confidence']:.1%})<br>
            <strong>Channel:</strong> #{msg['channel']} | <strong>User:</strong> {msg['user']} | <strong>Time:</strong> {msg['timestamp']}<br>
            <strong>Message:</strong> {msg['text']}
        </div>
        """, unsafe_allow_html=True)

def show_smart_search():
    st.header("🔍 Smart Search")
    
    st.markdown("""
    **Problem**: Finding specific information in Slack requires remembering exact keywords, channels, and timeframes.
    
    **AI Solution**: Natural language search that understands context and intent, not just keywords.
    """)
    
    # Search interface
    query = st.text_input(
        "Ask a question about your Slack conversations:",
        placeholder="e.g., 'What did the team decide about the pricing model?' or 'Any production issues this week?'"
    )
    
    # Sample queries
    st.markdown("**Try these example queries:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💰 Pricing decisions"):
            query = "What decisions were made about pricing?"
    
    with col2:
        if st.button("🚨 Production issues"):
            query = "Any production or server issues?"
    
    with col3:
        if st.button("📅 Meeting schedules"):
            query = "When are the upcoming meetings?"
    
    if query:
        with st.spinner("Searching through conversations..."):
            results = st.session_state.search.query(query)
        
        st.subheader("🎯 Search Results")
        
        if results:
            for i, result in enumerate(results[:5]):
                st.markdown(f"**Result {i+1}** (Relevance: {result['score']:.1%})")
                st.markdown(f"**Channel:** #{result['channel']} | **User:** {result['user']} | **Time:** {result['timestamp']}")
                st.markdown(f"**Message:** {result['text']}")
                st.markdown("---")
        else:
            st.info("No relevant results found. Try rephrasing your query.")

def show_analytics():
    st.header("📈 Communication Analytics")
    
    st.markdown("""
    **Insights**: Understanding communication patterns helps optimize team productivity.
    """)
    
    # Channel activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Channel Activity")
        
        channel_counts = {}
        for msg in st.session_state.sample_data:
            channel = msg['channel']
            channel_counts[channel] = channel_counts.get(channel, 0) + 1
        
        fig = px.bar(
            x=list(channel_counts.keys()),
            y=list(channel_counts.values()),
            title="Messages per Channel"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("👥 User Activity")
        
        user_counts = {}
        for msg in st.session_state.sample_data:
            user = msg['user']
            user_counts[user] = user_counts.get(user, 0) + 1
        
        # Top 10 users
        top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        fig = px.bar(
            x=[user[1] for user in top_users],
            y=[user[0] for user in top_users],
            orientation='h',
            title="Top 10 Most Active Users"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment analysis
    st.subheader("😊 Sentiment Analysis")
    
    sentiments = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for msg in st.session_state.sample_data[:100]:  # Sample for demo
        blob = TextBlob(msg['text'])
        if blob.sentiment.polarity > 0.1:
            sentiments['Positive'] += 1
        elif blob.sentiment.polarity < -0.1:
            sentiments['Negative'] += 1
        else:
            sentiments['Neutral'] += 1
    
    fig = px.pie(
        values=list(sentiments.values()),
        names=list(sentiments.keys()),
        title="Overall Sentiment Distribution",
        color_discrete_map={'Positive': '#4caf50', 'Neutral': '#2196f3', 'Negative': '#f44336'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Time-based analysis
    st.subheader("⏰ Activity Timeline")
    
    # Generate hourly activity data
    hours = list(range(24))
    activity = [random.randint(5, 25) for _ in hours]
    
    fig = px.line(
        x=hours,
        y=activity,
        title="Message Activity by Hour of Day",
        labels={'x': 'Hour (24h format)', 'y': 'Messages'}
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()