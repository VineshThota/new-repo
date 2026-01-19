import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import time

# Configure Streamlit page
st.set_page_config(
    page_title="WhatsApp Business AI Conversation Manager",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ConversationStatus(Enum):
    ACTIVE = "Active (Within 24h)"
    EXPIRING_SOON = "Expiring Soon (<2h)"
    EXPIRED = "Expired (Template Only)"
    TEMPLATE_SENT = "Template Sent - Awaiting Response"

@dataclass
class Customer:
    id: str
    name: str
    phone: str
    last_message_time: datetime
    conversation_status: ConversationStatus
    engagement_score: float
    total_messages: int
    avg_response_time: float  # in minutes
    preferred_contact_time: str
    conversation_history: List[Dict]
    tags: List[str]
    
class WhatsAppConversationManager:
    def __init__(self):
        self.customers = self._load_sample_data()
        self.message_templates = self._load_message_templates()
        
    def _load_sample_data(self) -> List[Customer]:
        """Load sample customer data for demonstration"""
        sample_customers = [
            {
                "id": str(uuid.uuid4()),
                "name": "John Smith",
                "phone": "+1234567890",
                "last_message_time": datetime.now() - timedelta(hours=2),
                "conversation_status": ConversationStatus.EXPIRING_SOON,
                "engagement_score": 8.5,
                "total_messages": 45,
                "avg_response_time": 15.5,
                "preferred_contact_time": "9:00-17:00",
                "conversation_history": [
                    {"timestamp": datetime.now() - timedelta(hours=2), "message": "Hi, I need help with my order", "sender": "customer"},
                    {"timestamp": datetime.now() - timedelta(hours=2, minutes=-5), "message": "Sure! Let me check that for you", "sender": "business"}
                ],
                "tags": ["VIP", "Support"]
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Sarah Johnson",
                "phone": "+1234567891",
                "last_message_time": datetime.now() - timedelta(hours=25),
                "conversation_status": ConversationStatus.EXPIRED,
                "engagement_score": 6.2,
                "total_messages": 12,
                "avg_response_time": 45.0,
                "preferred_contact_time": "18:00-21:00",
                "conversation_history": [
                    {"timestamp": datetime.now() - timedelta(hours=25), "message": "What are your business hours?", "sender": "customer"}
                ],
                "tags": ["New Customer"]
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Mike Chen",
                "phone": "+1234567892",
                "last_message_time": datetime.now() - timedelta(minutes=30),
                "conversation_status": ConversationStatus.ACTIVE,
                "engagement_score": 9.1,
                "total_messages": 78,
                "avg_response_time": 8.2,
                "preferred_contact_time": "10:00-16:00",
                "conversation_history": [
                    {"timestamp": datetime.now() - timedelta(minutes=30), "message": "Thanks for the quick delivery!", "sender": "customer"},
                    {"timestamp": datetime.now() - timedelta(minutes=25), "message": "You're welcome! How was everything?", "sender": "business"}
                ],
                "tags": ["VIP", "Repeat Customer"]
            }
        ]
        
        return [Customer(**customer) for customer in sample_customers]
    
    def _load_message_templates(self) -> Dict[str, List[str]]:
        """Load WhatsApp Business message templates by category"""
        return {
            "re_engagement": [
                "Hi {{customer_name}}, we noticed you had a question earlier. We're here to help! How can we assist you today?",
                "Hello {{customer_name}}! Just following up on our previous conversation. Is there anything else we can help you with?",
                "Hi {{customer_name}}, we want to make sure all your questions were answered. Feel free to reach out anytime!"
            ],
            "promotional": [
                "Hi {{customer_name}}! We have a special offer just for you. Check out our latest deals: {{offer_link}}",
                "Hello {{customer_name}}! New products are here. Get 20% off your next purchase with code SAVE20",
                "Hi {{customer_name}}! Don't miss our flash sale - 24 hours only! Shop now: {{shop_link}}"
            ],
            "support_follow_up": [
                "Hi {{customer_name}}, we wanted to follow up on your recent support request. Was everything resolved to your satisfaction?",
                "Hello {{customer_name}}! Just checking in - how is everything working after our recent assistance?",
                "Hi {{customer_name}}, we hope our support team was able to help you. Please let us know if you need anything else!"
            ],
            "order_updates": [
                "Hi {{customer_name}}, your order #{{order_id}} has been shipped! Track it here: {{tracking_link}}",
                "Hello {{customer_name}}! Your order #{{order_id}} is being prepared and will ship soon.",
                "Hi {{customer_name}}, your order #{{order_id}} has been delivered! We hope you love it!"
            ]
        }
    
    def calculate_time_remaining(self, last_message_time: datetime) -> timedelta:
        """Calculate time remaining in 24-hour window"""
        window_end = last_message_time + timedelta(hours=24)
        return window_end - datetime.now()
    
    def get_conversation_status(self, customer: Customer) -> ConversationStatus:
        """Determine current conversation status based on timing"""
        time_remaining = self.calculate_time_remaining(customer.last_message_time)
        
        if time_remaining.total_seconds() <= 0:
            return ConversationStatus.EXPIRED
        elif time_remaining.total_seconds() <= 7200:  # 2 hours
            return ConversationStatus.EXPIRING_SOON
        else:
            return ConversationStatus.ACTIVE
    
    def calculate_engagement_score(self, customer: Customer) -> float:
        """Calculate customer engagement score based on various factors"""
        # Base score from response time (faster = higher score)
        response_score = max(0, 10 - (customer.avg_response_time / 10))
        
        # Message frequency score
        frequency_score = min(10, customer.total_messages / 10)
        
        # Recency score (more recent = higher score)
        hours_since_last = (datetime.now() - customer.last_message_time).total_seconds() / 3600
        recency_score = max(0, 10 - (hours_since_last / 24))
        
        # VIP bonus
        vip_bonus = 2 if "VIP" in customer.tags else 0
        
        total_score = (response_score + frequency_score + recency_score + vip_bonus) / 3
        return min(10, max(0, total_score))
    
    def suggest_optimal_template(self, customer: Customer) -> Dict[str, str]:
        """Suggest the best message template based on customer context"""
        # Analyze customer context
        if customer.conversation_status == ConversationStatus.EXPIRED:
            if "Support" in customer.tags:
                category = "support_follow_up"
                reason = "Customer had a support inquiry that expired"
            elif customer.engagement_score > 7:
                category = "re_engagement"
                reason = "High-engagement customer needs re-engagement"
            else:
                category = "promotional"
                reason = "Low-engagement customer might respond to offers"
        elif customer.conversation_status == ConversationStatus.EXPIRING_SOON:
            category = "re_engagement"
            reason = "Conversation expiring soon - re-engage while possible"
        else:
            category = "promotional"
            reason = "Active conversation - good time for promotional content"
        
        # Select best template from category
        templates = self.message_templates[category]
        selected_template = templates[0]  # In real implementation, use ML to select best
        
        return {
            "template": selected_template,
            "category": category,
            "reason": reason,
            "personalized": selected_template.replace("{{customer_name}}", customer.name)
        }
    
    def get_priority_customers(self) -> List[Customer]:
        """Get customers prioritized by urgency and engagement"""
        # Update conversation statuses
        for customer in self.customers:
            customer.conversation_status = self.get_conversation_status(customer)
            customer.engagement_score = self.calculate_engagement_score(customer)
        
        # Sort by priority: expiring soon > high engagement > VIP status
        def priority_score(customer):
            status_priority = {
                ConversationStatus.EXPIRING_SOON: 100,
                ConversationStatus.ACTIVE: 50,
                ConversationStatus.EXPIRED: 20,
                ConversationStatus.TEMPLATE_SENT: 10
            }
            
            vip_bonus = 25 if "VIP" in customer.tags else 0
            return status_priority[customer.conversation_status] + customer.engagement_score + vip_bonus
        
        return sorted(self.customers, key=priority_score, reverse=True)

# Initialize the conversation manager
if 'conversation_manager' not in st.session_state:
    st.session_state.conversation_manager = WhatsAppConversationManager()

manager = st.session_state.conversation_manager

# Main UI
st.title("🚀 WhatsApp Business AI Conversation Manager")
st.markdown("""
**Solve the 24-Hour Window Limitation with AI-Powered Conversation Management**

This tool helps businesses maximize WhatsApp Business API effectiveness by:
- 🕐 Tracking conversation windows in real-time
- 🤖 AI-powered template suggestions
- 📊 Customer engagement scoring
- ⚡ Proactive conversation management
""")

# Sidebar for controls
with st.sidebar:
    st.header("🎛️ Controls")
    
    # Refresh data
    if st.button("🔄 Refresh Data"):
        st.session_state.conversation_manager = WhatsAppConversationManager()
        st.rerun()
    
    # Filter options
    st.subheader("Filters")
    status_filter = st.multiselect(
        "Conversation Status",
        [status.value for status in ConversationStatus],
        default=[status.value for status in ConversationStatus]
    )
    
    engagement_threshold = st.slider(
        "Min Engagement Score",
        0.0, 10.0, 0.0, 0.1
    )

# Main dashboard tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "👥 Customer Management", "📝 Template Suggestions", "📈 Analytics"])

with tab1:
    st.header("📊 Real-Time Dashboard")
    
    # Get priority customers
    priority_customers = manager.get_priority_customers()
    
    # Filter customers based on sidebar selections
    filtered_customers = [
        customer for customer in priority_customers
        if customer.conversation_status.value in status_filter
        and customer.engagement_score >= engagement_threshold
    ]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        active_count = len([c for c in filtered_customers if c.conversation_status == ConversationStatus.ACTIVE])
        st.metric("Active Conversations", active_count, delta=None)
    
    with col2:
        expiring_count = len([c for c in filtered_customers if c.conversation_status == ConversationStatus.EXPIRING_SOON])
        st.metric("Expiring Soon", expiring_count, delta="-2h" if expiring_count > 0 else None)
    
    with col3:
        expired_count = len([c for c in filtered_customers if c.conversation_status == ConversationStatus.EXPIRED])
        st.metric("Expired (Template Only)", expired_count)
    
    with col4:
        avg_engagement = np.mean([c.engagement_score for c in filtered_customers]) if filtered_customers else 0
        st.metric("Avg Engagement Score", f"{avg_engagement:.1f}/10")
    
    # Priority alerts
    st.subheader("🚨 Priority Alerts")
    
    urgent_customers = [c for c in filtered_customers if c.conversation_status == ConversationStatus.EXPIRING_SOON]
    
    if urgent_customers:
        for customer in urgent_customers[:3]:  # Show top 3 urgent
            time_remaining = manager.calculate_time_remaining(customer.last_message_time)
            hours, remainder = divmod(int(time_remaining.total_seconds()), 3600)
            minutes, _ = divmod(remainder, 60)
            
            st.warning(f"⏰ **{customer.name}** - Conversation expires in {hours}h {minutes}m | Engagement: {customer.engagement_score:.1f}/10")
    else:
        st.success("✅ No urgent conversations requiring immediate attention")
    
    # Customer overview table
    st.subheader("👥 Customer Overview")
    
    if filtered_customers:
        # Prepare data for display
        display_data = []
        for customer in filtered_customers:
            time_remaining = manager.calculate_time_remaining(customer.last_message_time)
            
            if time_remaining.total_seconds() > 0:
                hours, remainder = divmod(int(time_remaining.total_seconds()), 3600)
                minutes, _ = divmod(remainder, 60)
                time_str = f"{hours}h {minutes}m"
            else:
                time_str = "Expired"
            
            display_data.append({
                "Name": customer.name,
                "Phone": customer.phone,
                "Status": customer.conversation_status.value,
                "Time Remaining": time_str,
                "Engagement": f"{customer.engagement_score:.1f}/10",
                "Messages": customer.total_messages,
                "Avg Response": f"{customer.avg_response_time:.1f}min",
                "Tags": ", ".join(customer.tags)
            })
        
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No customers match the current filters")

with tab2:
    st.header("👥 Customer Management")
    
    if filtered_customers:
        # Customer selection
        customer_names = [f"{c.name} ({c.phone})" for c in filtered_customers]
        selected_customer_idx = st.selectbox(
            "Select Customer",
            range(len(customer_names)),
            format_func=lambda x: customer_names[x]
        )
        
        selected_customer = filtered_customers[selected_customer_idx]
        
        # Customer details
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📋 {selected_customer.name} Details")
            
            # Customer info
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.write(f"**Phone:** {selected_customer.phone}")
                st.write(f"**Status:** {selected_customer.conversation_status.value}")
                st.write(f"**Total Messages:** {selected_customer.total_messages}")
            
            with info_col2:
                st.write(f"**Engagement Score:** {selected_customer.engagement_score:.1f}/10")
                st.write(f"**Avg Response Time:** {selected_customer.avg_response_time:.1f} minutes")
                st.write(f"**Preferred Contact:** {selected_customer.preferred_contact_time}")
            
            st.write(f"**Tags:** {', '.join(selected_customer.tags)}")
            
            # Conversation history
            st.subheader("💬 Recent Conversation")
            
            for msg in selected_customer.conversation_history[-5:]:  # Show last 5 messages
                sender_icon = "👤" if msg["sender"] == "customer" else "🏢"
                timestamp = msg["timestamp"].strftime("%Y-%m-%d %H:%M")
                st.write(f"{sender_icon} **{msg['sender'].title()}** ({timestamp}): {msg['message']}")
        
        with col2:
            # Time remaining visualization
            time_remaining = manager.calculate_time_remaining(selected_customer.last_message_time)
            
            if time_remaining.total_seconds() > 0:
                hours_remaining = time_remaining.total_seconds() / 3600
                progress = max(0, min(1, hours_remaining / 24))
                
                st.subheader("⏰ Window Status")
                st.progress(progress)
                
                hours, remainder = divmod(int(time_remaining.total_seconds()), 3600)
                minutes, _ = divmod(remainder, 60)
                st.write(f"**Time Remaining:** {hours}h {minutes}m")
                
                if hours < 2:
                    st.error("⚠️ Conversation expiring soon!")
                elif hours < 6:
                    st.warning("⚡ Consider engaging soon")
                else:
                    st.success("✅ Window still active")
            else:
                st.error("❌ 24-hour window expired")
                st.write("Only template messages allowed")
            
            # Engagement score visualization
            st.subheader("📊 Engagement Score")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = selected_customer.engagement_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Engagement"},
                gauge = {
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgray"},
                        {'range': [5, 8], 'color': "yellow"},
                        {'range': [8, 10], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 9
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("📝 AI-Powered Template Suggestions")
    
    if filtered_customers:
        st.subheader("🤖 Smart Template Recommendations")
        
        for customer in filtered_customers[:5]:  # Show top 5 customers
            with st.expander(f"📱 {customer.name} - {customer.conversation_status.value}"):
                suggestion = manager.suggest_optimal_template(customer)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Recommended Template Category:** {suggestion['category'].replace('_', ' ').title()}")
                    st.write(f"**Reasoning:** {suggestion['reason']}")
                    
                    st.subheader("📄 Template Preview")
                    st.code(suggestion['template'], language="text")
                    
                    st.subheader("✨ Personalized Version")
                    st.success(suggestion['personalized'])
                    
                    # Action buttons
                    button_col1, button_col2, button_col3 = st.columns(3)
                    
                    with button_col1:
                        if st.button(f"📤 Send Template", key=f"send_{customer.id}"):
                            st.success(f"✅ Template sent to {customer.name}!")
                    
                    with button_col2:
                        if st.button(f"✏️ Customize", key=f"edit_{customer.id}"):
                            st.info("Template customization would open here")
                    
                    with button_col3:
                        if st.button(f"⏰ Schedule", key=f"schedule_{customer.id}"):
                            st.info("Scheduling options would open here")
                
                with col2:
                    # Customer quick stats
                    st.metric("Engagement", f"{customer.engagement_score:.1f}/10")
                    st.metric("Total Messages", customer.total_messages)
                    
                    time_remaining = manager.calculate_time_remaining(customer.last_message_time)
                    if time_remaining.total_seconds() > 0:
                        hours = int(time_remaining.total_seconds() / 3600)
                        st.metric("Hours Left", f"{hours}h")
                    else:
                        st.metric("Status", "Expired")
    
    # Template management
    st.subheader("📚 Template Library")
    
    template_category = st.selectbox(
        "Browse Templates by Category",
        list(manager.message_templates.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    st.write(f"**{template_category.replace('_', ' ').title()} Templates:**")
    
    for i, template in enumerate(manager.message_templates[template_category], 1):
        st.write(f"{i}. {template}")

with tab4:
    st.header("📈 Analytics & Insights")
    
    # Conversation status distribution
    st.subheader("📊 Conversation Status Distribution")
    
    status_counts = {}
    for status in ConversationStatus:
        status_counts[status.value] = len([c for c in filtered_customers if c.conversation_status == status])
    
    if any(status_counts.values()):
        fig_pie = px.pie(
            values=list(status_counts.values()),
            names=list(status_counts.keys()),
            title="Current Conversation Status"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Engagement score distribution
    st.subheader("🎯 Customer Engagement Distribution")
    
    if filtered_customers:
        engagement_scores = [c.engagement_score for c in filtered_customers]
        
        fig_hist = px.histogram(
            x=engagement_scores,
            nbins=20,
            title="Customer Engagement Score Distribution",
            labels={'x': 'Engagement Score', 'y': 'Number of Customers'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Response time analysis
        st.subheader("⚡ Response Time Analysis")
        
        response_times = [c.avg_response_time for c in filtered_customers]
        customer_names = [c.name for c in filtered_customers]
        
        fig_bar = px.bar(
            x=customer_names,
            y=response_times,
            title="Average Response Time by Customer",
            labels={'x': 'Customer', 'y': 'Response Time (minutes)'}
        )
        fig_bar.update_xaxis(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Key insights
        st.subheader("🔍 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Engagement Insights:**
            - Average engagement score: {np.mean(engagement_scores):.1f}/10
            - Highest engagement: {max(engagement_scores):.1f}/10
            - Customers with high engagement (>8): {len([s for s in engagement_scores if s > 8])}
            """)
        
        with col2:
            st.info(f"""
            **Response Time Insights:**
            - Average response time: {np.mean(response_times):.1f} minutes
            - Fastest responder: {min(response_times):.1f} minutes
            - Customers responding <15min: {len([t for t in response_times if t < 15])}
            """)
        
        # Recommendations
        st.subheader("💡 AI Recommendations")
        
        recommendations = []
        
        # Check for expiring conversations
        expiring_soon = [c for c in filtered_customers if c.conversation_status == ConversationStatus.EXPIRING_SOON]
        if expiring_soon:
            recommendations.append(f"🚨 **Urgent:** {len(expiring_soon)} conversations expiring soon - send re-engagement templates immediately")
        
        # Check for high-engagement expired customers
        high_engagement_expired = [c for c in filtered_customers if c.conversation_status == ConversationStatus.EXPIRED and c.engagement_score > 7]
        if high_engagement_expired:
            recommendations.append(f"⭐ **Opportunity:** {len(high_engagement_expired)} high-engagement customers need template messages to re-open conversations")
        
        # Check for slow responders
        slow_responders = [c for c in filtered_customers if c.avg_response_time > 60]
        if slow_responders:
            recommendations.append(f"⏰ **Attention:** {len(slow_responders)} customers have slow response times - consider adjusting engagement strategy")
        
        # Check for VIP customers
        vip_customers = [c for c in filtered_customers if "VIP" in c.tags]
        if vip_customers:
            recommendations.append(f"👑 **VIP Focus:** {len(vip_customers)} VIP customers require priority attention and personalized templates")
        
        if recommendations:
            for rec in recommendations:
                st.warning(rec)
        else:
            st.success("✅ All conversations are being managed optimally!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 <strong>WhatsApp Business AI Conversation Manager</strong> - Solving the 24-Hour Window Challenge</p>
    <p>Built with ❤️ using Streamlit, Plotly, and AI-powered insights</p>
</div>
""", unsafe_allow_html=True)
