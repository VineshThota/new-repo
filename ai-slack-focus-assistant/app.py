import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import re
from typing import List, Dict, Tuple
import time

# Import our custom modules
from slack_ai_assistant import (
    MessageClassifier,
    ThreadSummarizer,
    FocusTimeAnalyzer,
    DailyDigestGenerator,
    generate_sample_messages
)

# Page configuration
st.set_page_config(
    page_title="AI Slack Focus Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
.priority-urgent {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
    padding: 0.5rem;
    margin: 0.25rem 0;
    border-radius: 0.25rem;
}
.priority-important {
    background-color: #fff3e0;
    border-left: 4px solid #ff9800;
    padding: 0.5rem;
    margin: 0.25rem 0;
    border-radius: 0.25rem;
}
.priority-fyi {
    background-color: #e8f5e8;
    border-left: 4px solid #4caf50;
    padding: 0.5rem;
    margin: 0.25rem 0;
    border-radius: 0.25rem;
}
.priority-noise {
    background-color: #f5f5f5;
    border-left: 4px solid #9e9e9e;
    padding: 0.5rem;
    margin: 0.25rem 0;
    border-radius: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 AI Slack Focus Assistant</h1>', unsafe_allow_html=True)
    st.markdown("""
    **Intelligent Message Prioritization & Summarization**
    
    Transform your Slack experience from overwhelming noise to focused productivity.
    This AI-powered assistant helps you:
    - 📊 Classify messages by priority and urgency
    - 📝 Summarize long threads into actionable insights
    - ⏰ Optimize your focus time and reduce interruptions
    - 📈 Analyze communication patterns for better productivity
    """)
    
    # Sidebar navigation
    st.sidebar.title("🚀 Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        [
            "📊 Message Priority Dashboard",
            "📝 Thread Summarizer",
            "⏰ Focus Time Analytics",
            "📈 Daily Digest Generator",
            "🔬 Live Demo Simulator"
        ]
    )
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = generate_sample_messages(50)
    if 'classifier' not in st.session_state:
        st.session_state.classifier = MessageClassifier()
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = ThreadSummarizer()
    if 'focus_analyzer' not in st.session_state:
        st.session_state.focus_analyzer = FocusTimeAnalyzer()
    if 'digest_generator' not in st.session_state:
        st.session_state.digest_generator = DailyDigestGenerator()
    
    # Route to different pages
    if page == "📊 Message Priority Dashboard":
        show_priority_dashboard()
    elif page == "📝 Thread Summarizer":
        show_thread_summarizer()
    elif page == "⏰ Focus Time Analytics":
        show_focus_analytics()
    elif page == "📈 Daily Digest Generator":
        show_daily_digest()
    elif page == "🔬 Live Demo Simulator":
        show_live_demo()

def show_priority_dashboard():
    st.header("📊 Message Priority Dashboard")
    st.write("Analyze and classify your Slack messages by priority and urgency.")
    
    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        num_messages = st.slider("Number of messages to analyze", 10, 100, 30)
    
    with col2:
        if st.button("🔄 Generate New Messages"):
            st.session_state.messages = generate_sample_messages(num_messages)
            st.rerun()
    
    with col3:
        auto_classify = st.checkbox("Auto-classify", value=True)
    
    # Get messages and classify them
    messages = st.session_state.messages[:num_messages]
    
    if auto_classify:
        classified_messages = []
        progress_bar = st.progress(0)
        
        for i, msg in enumerate(messages):
            classification = st.session_state.classifier.classify_priority(msg['content'])
            msg['priority'] = classification['level']
            msg['confidence'] = classification['confidence']
            msg['reasoning'] = classification['reasoning']
            classified_messages.append(msg)
            progress_bar.progress((i + 1) / len(messages))
        
        progress_bar.empty()
        
        # Priority distribution
        priority_counts = pd.DataFrame(classified_messages)['priority'].value_counts()
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            urgent_count = priority_counts.get('URGENT', 0)
            st.metric("🚨 Urgent", urgent_count, f"{urgent_count/len(messages)*100:.1f}%")
        
        with col2:
            important_count = priority_counts.get('IMPORTANT', 0)
            st.metric("⚠️ Important", important_count, f"{important_count/len(messages)*100:.1f}%")
        
        with col3:
            fyi_count = priority_counts.get('FYI', 0)
            st.metric("ℹ️ FYI", fyi_count, f"{fyi_count/len(messages)*100:.1f}%")
        
        with col4:
            noise_count = priority_counts.get('NOISE', 0)
            st.metric("🔇 Noise", noise_count, f"{noise_count/len(messages)*100:.1f}%")
        
        # Priority distribution chart
        fig = px.pie(
            values=priority_counts.values,
            names=priority_counts.index,
            title="Message Priority Distribution",
            color_discrete_map={
                'URGENT': '#f44336',
                'IMPORTANT': '#ff9800',
                'FYI': '#4caf50',
                'NOISE': '#9e9e9e'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Message list with priorities
        st.subheader("📋 Classified Messages")
        
        # Filter options
        priority_filter = st.multiselect(
            "Filter by priority:",
            ['URGENT', 'IMPORTANT', 'FYI', 'NOISE'],
            default=['URGENT', 'IMPORTANT']
        )
        
        filtered_messages = [msg for msg in classified_messages if msg['priority'] in priority_filter]
        
        for msg in filtered_messages:
            priority_class = f"priority-{msg['priority'].lower()}"
            
            st.markdown(f"""
            <div class="{priority_class}">
                <strong>{msg['priority']}</strong> (Confidence: {msg['confidence']:.2f}) - 
                <em>{msg['sender']}</em> in #{msg['channel']}<br>
                <strong>Message:</strong> {msg['content']}<br>
                <small><strong>Reasoning:</strong> {msg['reasoning']}</small>
            </div>
            """, unsafe_allow_html=True)

def show_thread_summarizer():
    st.header("📝 Thread Summarizer")
    st.write("Transform long Slack conversations into concise, actionable summaries.")
    
    # Sample thread or custom input
    option = st.radio(
        "Choose input method:",
        ["📋 Use Sample Thread", "✏️ Enter Custom Thread"]
    )
    
    if option == "📋 Use Sample Thread":
        # Generate a sample thread
        sample_threads = {
            "Q4 Planning Discussion": [
                "Hey team, we need to start planning for Q4. What are our priorities?",
                "I think we should focus on the mobile app redesign. It's been requested by many users.",
                "Agreed on mobile, but we also need to address the performance issues in the backend.",
                "Good point. Should we allocate 60% to mobile and 40% to performance?",
                "That sounds reasonable. We'll also need to hire 2 more developers.",
                "I can start the hiring process next week. Let's schedule a detailed planning meeting.",
                "How about Thursday at 2 PM? I'll send out calendar invites.",
                "Perfect. I'll prepare the technical requirements document by then.",
                "Great! Also, don't forget we need to update the stakeholders on our progress.",
                "I'll handle the stakeholder communication. Meeting confirmed for Thursday 2 PM."
            ],
            "Bug Report Discussion": [
                "URGENT: Users are reporting login issues on the mobile app.",
                "I'm seeing the same reports. It seems to be affecting iOS users specifically.",
                "Looking into it now. The authentication service logs show some errors.",
                "Found the issue - it's related to the recent SSL certificate update.",
                "Can you fix it quickly? This is affecting about 30% of our mobile users.",
                "Working on it. I'll need about 2 hours to implement and test the fix.",
                "Keep me posted. I'll prepare a communication for affected users.",
                "Fix is ready for deployment. Running final tests now.",
                "Tests passed. Deploying the fix to production.",
                "Deployment successful. Login issues should be resolved now.",
                "Confirmed - users are reporting that login is working again. Great job team!"
            ],
            "Marketing Campaign Planning": [
                "We need to plan the launch campaign for our new feature.",
                "What's our target audience for this campaign?",
                "Primarily existing users who haven't used the premium features yet.",
                "I suggest we focus on email marketing and in-app notifications.",
                "Good idea. We should also consider social media promotion.",
                "I can handle the social media content. When do we want to launch?",
                "Ideally next month, after the feature is fully tested.",
                "That gives us 3 weeks to prepare all materials.",
                "I'll create a project timeline and share it by tomorrow.",
                "Perfect. Let's also budget for some paid advertising.",
                "I'll get quotes from our usual ad platforms."
            ]
        }
        
        selected_thread = st.selectbox("Select a sample thread:", list(sample_threads.keys()))
        thread_messages = sample_threads[selected_thread]
        
    else:
        st.write("Enter your thread messages (one per line):")
        thread_input = st.text_area(
            "Thread messages:",
            height=200,
            placeholder="Enter each message on a new line...\nMessage 1\nMessage 2\nMessage 3"
        )
        thread_messages = [msg.strip() for msg in thread_input.split('\n') if msg.strip()]
    
    if thread_messages and len(thread_messages) > 1:
        st.subheader("📄 Original Thread")
        
        # Display original thread
        with st.expander(f"View full thread ({len(thread_messages)} messages)", expanded=False):
            for i, msg in enumerate(thread_messages, 1):
                st.write(f"**Message {i}:** {msg}")
        
        # Summarize button
        if st.button("🔄 Generate Summary", type="primary"):
            with st.spinner("Analyzing thread and generating summary..."):
                summary = st.session_state.summarizer.summarize_thread(thread_messages)
            
            # Display summary
            st.subheader("📊 Thread Summary")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**📝 Main Summary:**")
                st.info(summary['summary'])
                
                if summary['key_decisions']:
                    st.markdown("**🎯 Key Decisions:**")
                    for decision in summary['key_decisions']:
                        st.success(f"• {decision}")
                
                if summary['action_items']:
                    st.markdown("**✅ Action Items:**")
                    for item in summary['action_items']:
                        st.warning(f"• {item}")
            
            with col2:
                # Summary metrics
                st.metric("Original Messages", len(thread_messages))
                st.metric("Key Decisions", len(summary['key_decisions']))
                st.metric("Action Items", len(summary['action_items']))
                
                # Estimated time saved
                original_read_time = len(thread_messages) * 0.5  # 30 seconds per message
                summary_read_time = 1  # 1 minute to read summary
                time_saved = max(0, original_read_time - summary_read_time)
                st.metric("Time Saved", f"{time_saved:.1f} min")
    
    else:
        st.info("Please select a sample thread or enter at least 2 messages to generate a summary.")

def show_focus_analytics():
    st.header("⏰ Focus Time Analytics")
    st.write("Analyze your communication patterns and optimize your focus time.")
    
    # Generate sample user activity data
    if 'user_activity' not in st.session_state:
        st.session_state.user_activity = generate_sample_activity_data()
    
    activity_data = st.session_state.user_activity
    
    # Focus time recommendations
    recommendations = st.session_state.focus_analyzer.suggest_focus_blocks(activity_data)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_interruptions = np.mean([hour['interruptions'] for hour in activity_data])
        st.metric("Avg Interruptions/Hour", f"{avg_interruptions:.1f}")
    
    with col2:
        total_focus_time = sum([rec['duration'] for rec in recommendations])
        st.metric("Recommended Focus Time", f"{total_focus_time} min")
    
    with col3:
        peak_productivity_hour = max(activity_data, key=lambda x: x['productivity_score'])['hour']
        st.metric("Peak Productivity Hour", f"{peak_productivity_hour}:00")
    
    with col4:
        total_messages = sum([hour['message_count'] for hour in activity_data])
        st.metric("Daily Messages", total_messages)
    
    # Activity heatmap
    st.subheader("📊 Daily Activity Pattern")
    
    # Create heatmap data
    hours = [data['hour'] for data in activity_data]
    interruptions = [data['interruptions'] for data in activity_data]
    productivity = [data['productivity_score'] for data in activity_data]
    
    fig = go.Figure()
    
    # Add interruptions bar
    fig.add_trace(go.Bar(
        x=hours,
        y=interruptions,
        name='Interruptions',
        marker_color='rgba(255, 99, 132, 0.7)',
        yaxis='y'
    ))
    
    # Add productivity line
    fig.add_trace(go.Scatter(
        x=hours,
        y=productivity,
        name='Productivity Score',
        line=dict(color='rgba(54, 162, 235, 1)', width=3),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Hourly Interruptions vs Productivity',
        xaxis_title='Hour of Day',
        yaxis=dict(title='Interruptions', side='left'),
        yaxis2=dict(title='Productivity Score', side='right', overlaying='y'),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Focus time recommendations
    st.subheader("🎯 Recommended Focus Blocks")
    
    if recommendations:
        for i, block in enumerate(recommendations, 1):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Block {i}:** {block['start']} - {block['end']}")
            
            with col2:
                st.write(f"Duration: {block['duration']} min")
            
            with col3:
                st.write(f"Quality: {block['quality']}")
    
    # Tips and insights
    st.subheader("💡 Productivity Insights")
    
    insights = [
        "🌅 Your productivity peaks in the morning hours (9-11 AM)",
        "📱 Consider batching message checks to 3-4 times per day",
        "🔕 Use 'Do Not Disturb' during your recommended focus blocks",
        "⏰ Schedule important work during your high-productivity hours",
        "📊 Your interruption rate is 23% below average - great job!"
    ]
    
    for insight in insights:
        st.info(insight)

def show_daily_digest():
    st.header("📈 Daily Digest Generator")
    st.write("Get a comprehensive summary of your day's Slack activity.")
    
    # Date selector
    selected_date = st.date_input("Select date for digest:", datetime.now().date())
    
    # Generate digest button
    if st.button("📊 Generate Daily Digest", type="primary"):
        with st.spinner("Analyzing daily activity and generating digest..."):
            # Simulate processing time
            time.sleep(2)
            
            digest = st.session_state.digest_generator.generate_digest(
                st.session_state.messages,
                selected_date
            )
        
        # Display digest
        st.subheader(f"📋 Daily Digest - {selected_date.strftime('%B %d, %Y')}")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", digest['total_messages'])
        
        with col2:
            st.metric("Channels Active", digest['active_channels'])
        
        with col3:
            st.metric("Action Items", len(digest['action_items']))
        
        with col4:
            st.metric("Key Decisions", len(digest['key_decisions']))
        
        # Top conversations
        st.subheader("🔥 Most Active Conversations")
        for conv in digest['top_conversations']:
            st.write(f"**#{conv['channel']}** - {conv['message_count']} messages")
            st.write(f"*{conv['summary']}*")
            st.write("---")
        
        # Action items
        if digest['action_items']:
            st.subheader("✅ Action Items")
            for item in digest['action_items']:
                st.warning(f"• {item}")
        
        # Key decisions
        if digest['key_decisions']:
            st.subheader("🎯 Key Decisions")
            for decision in digest['key_decisions']:
                st.success(f"• {decision}")
        
        # Trending topics
        st.subheader("📈 Trending Topics")
        topics_df = pd.DataFrame(digest['trending_topics'])
        fig = px.bar(
            topics_df,
            x='mentions',
            y='topic',
            orientation='h',
            title='Most Mentioned Topics'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_live_demo():
    st.header("🔬 Live Demo Simulator")
    st.write("Experience real-time message classification and processing.")
    
    # Demo controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        demo_speed = st.slider("Demo Speed (messages/sec)", 0.5, 5.0, 2.0, 0.5)
    
    with col2:
        message_types = st.multiselect(
            "Message Types",
            ['urgent', 'important', 'fyi', 'noise'],
            default=['urgent', 'important', 'fyi']
        )
    
    with col3:
        auto_start = st.checkbox("Auto-start demo", value=False)
    
    # Demo state
    if 'demo_running' not in st.session_state:
        st.session_state.demo_running = False
    if 'demo_messages' not in st.session_state:
        st.session_state.demo_messages = []
    
    # Demo controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Demo") or auto_start:
            st.session_state.demo_running = True
    
    with col2:
        if st.button("⏸️ Pause Demo"):
            st.session_state.demo_running = False
    
    with col3:
        if st.button("🔄 Reset Demo"):
            st.session_state.demo_running = False
            st.session_state.demo_messages = []
    
    # Live message feed
    if st.session_state.demo_running:
        message_container = st.container()
        
        # Simulate incoming messages
        if len(st.session_state.demo_messages) < 20:  # Limit demo length
            new_message = generate_demo_message(message_types)
            classification = st.session_state.classifier.classify_priority(new_message['content'])
            new_message.update(classification)
            
            st.session_state.demo_messages.append(new_message)
            
            # Display latest messages
            with message_container:
                st.subheader("📱 Live Message Feed")
                
                for msg in reversed(st.session_state.demo_messages[-10:]):  # Show last 10
                    priority_class = f"priority-{msg['level'].lower()}"
                    
                    st.markdown(f"""
                    <div class="{priority_class}">
                        <strong>{msg['level']}</strong> (Confidence: {msg['confidence']:.2f}) - 
                        <em>{msg['sender']}</em> in #{msg['channel']}<br>
                        <strong>Message:</strong> {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Auto-refresh
            time.sleep(1 / demo_speed)
            st.rerun()
        else:
            st.session_state.demo_running = False
            st.success("Demo completed! Click 'Reset Demo' to run again.")
    
    # Demo statistics
    if st.session_state.demo_messages:
        st.subheader("📊 Demo Statistics")
        
        demo_df = pd.DataFrame(st.session_state.demo_messages)
        priority_counts = demo_df['level'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=priority_counts.values,
                names=priority_counts.index,
                title="Message Priority Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_confidence = demo_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.2f}")
            st.metric("Total Messages Processed", len(st.session_state.demo_messages))
            st.metric("Processing Speed", f"{demo_speed} msg/sec")

def generate_sample_activity_data():
    """Generate sample user activity data for focus analytics"""
    activity_data = []
    
    for hour in range(9, 18):  # 9 AM to 6 PM
        # Simulate realistic patterns
        if hour in [9, 10, 14, 15]:  # Peak hours
            interruptions = random.randint(8, 15)
            productivity = random.uniform(0.7, 0.9)
        elif hour in [12, 13]:  # Lunch time
            interruptions = random.randint(2, 5)
            productivity = random.uniform(0.3, 0.5)
        else:  # Regular hours
            interruptions = random.randint(5, 10)
            productivity = random.uniform(0.5, 0.8)
        
        activity_data.append({
            'hour': hour,
            'interruptions': interruptions,
            'productivity_score': productivity,
            'message_count': random.randint(5, 25),
            'focus_time': random.randint(10, 45)
        })
    
    return activity_data

def generate_demo_message(message_types):
    """Generate a demo message for live simulation"""
    templates = {
        'urgent': [
            "URGENT: Production server is down!",
            "Critical bug found in payment system",
            "Client meeting moved to NOW - need presentation",
            "Security breach detected - immediate action required"
        ],
        'important': [
            "Please review the Q4 budget proposal by EOD",
            "New feature requirements from stakeholders",
            "Team meeting scheduled for tomorrow 2 PM",
            "Code review needed for release candidate"
        ],
        'fyi': [
            "FYI: New team member starting next week",
            "Office will be closed on Friday for maintenance",
            "Updated company policies in the handbook",
            "Lunch and learn session next Tuesday"
        ],
        'noise': [
            "Anyone know a good coffee shop nearby?",
            "Happy birthday to Sarah! 🎉",
            "Weather is great today!",
            "Did anyone watch the game last night?"
        ]
    }
    
    msg_type = random.choice(message_types)
    content = random.choice(templates[msg_type])
    
    return {
        'content': content,
        'sender': f"user_{random.randint(1, 20)}",
        'channel': random.choice(['general', 'dev-team', 'marketing', 'support', 'random']),
        'timestamp': datetime.now()
    }

if __name__ == "__main__":
    main()