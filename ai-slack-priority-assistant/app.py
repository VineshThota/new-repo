import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass
from slack_priority_ai import MessagePrioritizer, ThreadSummarizer
import time

# Page configuration
st.set_page_config(
    page_title="AI Slack Priority Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.priority-high {
    background-color: #ffebee;
    border-left: 5px solid #f44336;
    padding: 10px;
    margin: 5px 0;
    border-radius: 5px;
}
.priority-medium {
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
    padding: 10px;
    margin: 5px 0;
    border-radius: 5px;
}
.priority-low {
    background-color: #e8f5e8;
    border-left: 5px solid #4caf50;
    padding: 10px;
    margin: 5px 0;
    border-radius: 5px;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e9ecef;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

@dataclass
class SlackMessage:
    """Data class for Slack message structure"""
    id: str
    text: str
    user: str
    channel: str
    timestamp: datetime
    thread_ts: str = None
    reactions: List[Dict] = None
    mentions: List[str] = None
    priority_score: float = 0.0
    urgency_keywords: List[str] = None
    action_items: List[str] = None

class SlackPriorityApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.prioritizer = MessagePrioritizer()
        self.summarizer = ThreadSummarizer()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = self.generate_sample_messages()
        if 'processed_messages' not in st.session_state:
            st.session_state.processed_messages = []
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'min_priority': 50,
                'notification_threshold': 75,
                'preferred_channels': ['general', 'engineering', 'product']
            }
    
    def generate_sample_messages(self) -> List[SlackMessage]:
        """Generate realistic sample Slack messages for demo"""
        sample_messages = [
            SlackMessage(
                id="msg_001",
                text="🚨 URGENT: Production server is down! Need immediate attention from DevOps team. Customer complaints are coming in.",
                user="john.doe",
                channel="alerts",
                timestamp=datetime.now() - timedelta(minutes=5),
                reactions=[{"name": "fire", "count": 3}, {"name": "eyes", "count": 5}],
                mentions=["@devops", "@oncall"]
            ),
            SlackMessage(
                id="msg_002",
                text="Great job on the quarterly presentation! The client loved our new features. Let's celebrate at happy hour.",
                user="sarah.manager",
                channel="general",
                timestamp=datetime.now() - timedelta(hours=2),
                reactions=[{"name": "tada", "count": 8}, {"name": "clap", "count": 12}]
            ),
            SlackMessage(
                id="msg_003",
                text="@channel Please review the new security policy document by EOD Friday. This is mandatory for compliance.",
                user="security.team",
                channel="announcements",
                timestamp=datetime.now() - timedelta(hours=1),
                mentions=["@channel"],
                urgency_keywords=["mandatory", "EOD", "compliance"]
            ),
            SlackMessage(
                id="msg_004",
                text="Anyone know where the coffee machine manual is? The new one is acting up.",
                user="office.manager",
                channel="random",
                timestamp=datetime.now() - timedelta(minutes=30)
            ),
            SlackMessage(
                id="msg_005",
                text="Sprint planning meeting moved to 3 PM today. @product-team please prepare your user stories.",
                user="scrum.master",
                channel="engineering",
                timestamp=datetime.now() - timedelta(minutes=45),
                mentions=["@product-team"],
                action_items=["Prepare user stories for sprint planning"]
            ),
            SlackMessage(
                id="msg_006",
                text="Bug report: Login page crashes on Safari. Steps to reproduce attached. Priority: High",
                user="qa.tester",
                channel="bugs",
                timestamp=datetime.now() - timedelta(minutes=20),
                urgency_keywords=["bug", "crashes", "Priority: High"]
            ),
            SlackMessage(
                id="msg_007",
                text="Lunch recommendations for the team outing next week? Thinking Italian or Mexican.",
                user="team.lead",
                channel="general",
                timestamp=datetime.now() - timedelta(hours=3)
            ),
            SlackMessage(
                id="msg_008",
                text="Database migration scheduled for tonight 11 PM - 2 AM. Expect brief downtime. @engineering @devops",
                user="database.admin",
                channel="infrastructure",
                timestamp=datetime.now() - timedelta(minutes=15),
                mentions=["@engineering", "@devops"],
                urgency_keywords=["migration", "downtime", "tonight"]
            )
        ]
        
        # Calculate priority scores for sample messages
        for message in sample_messages:
            message.priority_score = self.prioritizer.calculate_priority(message)
        
        return sample_messages
    
    def render_sidebar(self):
        """Render the sidebar with controls and settings"""
        st.sidebar.title("🎯 Priority Controls")
        
        # Priority threshold slider
        min_priority = st.sidebar.slider(
            "Minimum Priority Score",
            min_value=0,
            max_value=100,
            value=st.session_state.user_preferences['min_priority'],
            help="Only show messages above this priority score"
        )
        st.session_state.user_preferences['min_priority'] = min_priority
        
        # Notification threshold
        notification_threshold = st.sidebar.slider(
            "Notification Threshold",
            min_value=50,
            max_value=100,
            value=st.session_state.user_preferences['notification_threshold'],
            help="Send notifications for messages above this score"
        )
        st.session_state.user_preferences['notification_threshold'] = notification_threshold
        
        # Channel filter
        st.sidebar.subheader("📺 Channel Filters")
        all_channels = list(set([msg.channel for msg in st.session_state.messages]))
        selected_channels = st.sidebar.multiselect(
            "Select Channels",
            options=all_channels,
            default=all_channels,
            help="Filter messages by channel"
        )
        
        # Time filter
        st.sidebar.subheader("⏰ Time Filter")
        time_filter = st.sidebar.selectbox(
            "Show messages from",
            options=["Last hour", "Last 4 hours", "Last 24 hours", "All time"],
            index=2
        )
        
        # AI Settings
        st.sidebar.subheader("🤖 AI Settings")
        enable_smart_notifications = st.sidebar.checkbox(
            "Smart Notifications",
            value=True,
            help="Use AI to determine notification importance"
        )
        
        enable_thread_summarization = st.sidebar.checkbox(
            "Thread Summarization",
            value=True,
            help="Auto-summarize long thread discussions"
        )
        
        return selected_channels, time_filter, enable_smart_notifications, enable_thread_summarization
    
    def filter_messages(self, messages: List[SlackMessage], channels: List[str], time_filter: str, min_priority: int) -> List[SlackMessage]:
        """Filter messages based on user preferences"""
        filtered = []
        
        # Time filtering
        now = datetime.now()
        time_thresholds = {
            "Last hour": now - timedelta(hours=1),
            "Last 4 hours": now - timedelta(hours=4),
            "Last 24 hours": now - timedelta(hours=24),
            "All time": datetime.min
        }
        time_threshold = time_thresholds[time_filter]
        
        for message in messages:
            # Apply filters
            if (message.channel in channels and 
                message.timestamp >= time_threshold and 
                message.priority_score >= min_priority):
                filtered.append(message)
        
        # Sort by priority score (descending)
        return sorted(filtered, key=lambda x: x.priority_score, reverse=True)
    
    def render_message_card(self, message: SlackMessage):
        """Render a single message card with priority styling"""
        # Determine priority level and styling
        if message.priority_score >= 75:
            priority_class = "priority-high"
            priority_label = "🔴 HIGH"
            priority_color = "#f44336"
        elif message.priority_score >= 50:
            priority_class = "priority-medium"
            priority_label = "🟡 MEDIUM"
            priority_color = "#ff9800"
        else:
            priority_class = "priority-low"
            priority_label = "🟢 LOW"
            priority_color = "#4caf50"
        
        # Create message card
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="{priority_class}">
                    <strong>#{message.channel}</strong> • <em>{message.user}</em> • {message.timestamp.strftime('%H:%M')}<br>
                    {message.text}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    label="Priority",
                    value=f"{message.priority_score:.0f}",
                    delta=priority_label
                )
            
            with col3:
                # Show reactions if any
                if message.reactions:
                    reaction_text = " ".join([f"{r['name']} {r['count']}" for r in message.reactions])
                    st.text(f"Reactions: {reaction_text}")
                
                # Show action items if any
                if message.action_items:
                    st.text("📋 Action items detected")
    
    def render_analytics_dashboard(self):
        """Render analytics and insights dashboard"""
        st.subheader("📊 Communication Analytics")
        
        messages = st.session_state.messages
        
        # Create metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_messages = len(messages)
            st.metric("Total Messages", total_messages)
        
        with col2:
            high_priority = len([m for m in messages if m.priority_score >= 75])
            st.metric("High Priority", high_priority, delta=f"{high_priority/total_messages*100:.1f}%")
        
        with col3:
            avg_priority = np.mean([m.priority_score for m in messages])
            st.metric("Avg Priority", f"{avg_priority:.1f}")
        
        with col4:
            notifications = len([m for m in messages if m.priority_score >= st.session_state.user_preferences['notification_threshold']])
            st.metric("Notifications", notifications)
        
        # Priority distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Priority Distribution")
            priority_scores = [m.priority_score for m in messages]
            fig = px.histogram(
                x=priority_scores,
                nbins=20,
                title="Message Priority Distribution",
                labels={'x': 'Priority Score', 'y': 'Count'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Channel Activity")
            channel_counts = pd.Series([m.channel for m in messages]).value_counts()
            fig = px.pie(
                values=channel_counts.values,
                names=channel_counts.index,
                title="Messages by Channel"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Timeline chart
        st.subheader("Message Timeline")
        df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'priority': m.priority_score,
                'channel': m.channel,
                'text': m.text[:50] + '...' if len(m.text) > 50 else m.text
            }
            for m in messages
        ])
        
        fig = px.scatter(
            df,
            x='timestamp',
            y='priority',
            color='channel',
            hover_data=['text'],
            title="Message Priority Over Time"
        )
        fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="High Priority Threshold")
        fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Medium Priority Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_smart_notifications(self):
        """Render smart notifications panel"""
        st.subheader("🔔 Smart Notifications")
        
        threshold = st.session_state.user_preferences['notification_threshold']
        high_priority_messages = [
            m for m in st.session_state.messages 
            if m.priority_score >= threshold
        ]
        
        if high_priority_messages:
            st.info(f"You have {len(high_priority_messages)} high-priority notifications")
            
            for message in high_priority_messages[:3]:  # Show top 3
                with st.expander(f"🚨 {message.channel} - Priority {message.priority_score:.0f}"):
                    st.write(f"**From:** {message.user}")
                    st.write(f"**Time:** {message.timestamp.strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Message:** {message.text}")
                    
                    if message.urgency_keywords:
                        st.write(f"**Urgency Keywords:** {', '.join(message.urgency_keywords)}")
                    
                    if message.action_items:
                        st.write(f"**Action Items:** {', '.join(message.action_items)}")
        else:
            st.success("No high-priority notifications at the moment! 🎉")
    
    def run(self):
        """Main application runner"""
        # Header
        st.title("🎯 AI Slack Priority Assistant")
        st.markdown("*Intelligent message prioritization to reduce information overload*")
        
        # Sidebar controls
        selected_channels, time_filter, enable_smart_notifications, enable_thread_summarization = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📬 Messages", "📊 Analytics", "🔔 Notifications", "⚙️ Settings"])
        
        with tab1:
            st.subheader("Prioritized Messages")
            
            # Filter messages
            filtered_messages = self.filter_messages(
                st.session_state.messages,
                selected_channels,
                time_filter,
                st.session_state.user_preferences['min_priority']
            )
            
            if filtered_messages:
                st.info(f"Showing {len(filtered_messages)} messages (filtered from {len(st.session_state.messages)} total)")
                
                # Render message cards
                for message in filtered_messages:
                    self.render_message_card(message)
                    st.markdown("---")
            else:
                st.warning("No messages match your current filters. Try adjusting the priority threshold or channel selection.")
        
        with tab2:
            self.render_analytics_dashboard()
        
        with tab3:
            if enable_smart_notifications:
                self.render_smart_notifications()
            else:
                st.info("Smart notifications are disabled. Enable them in the sidebar to see priority alerts.")
        
        with tab4:
            st.subheader("⚙️ System Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### AI Model Configuration")
                st.selectbox("Priority Model", ["BERT-base", "RoBERTa-large", "Custom-trained"], index=0)
                st.slider("Model Confidence Threshold", 0.0, 1.0, 0.7)
                st.checkbox("Enable Learning Mode", value=True, help="Allow the model to learn from your feedback")
            
            with col2:
                st.markdown("### Performance Metrics")
                st.metric("Processing Speed", "47 msg/sec")
                st.metric("Accuracy", "87.3%")
                st.metric("Noise Reduction", "65%")
            
            # Simulate processing button
            if st.button("🔄 Refresh Messages", type="primary"):
                with st.spinner("Processing new messages..."):
                    time.sleep(2)  # Simulate processing
                    st.session_state.messages = self.generate_sample_messages()
                st.success("Messages refreshed successfully!")
                st.rerun()

# Run the application
if __name__ == "__main__":
    app = SlackPriorityApp()
    app.run()