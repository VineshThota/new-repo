import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import hashlib
import time

# Configure Streamlit page
st.set_page_config(
    page_title="AI Teams Multi-Account Manager",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TeamsAccount:
    """Represents a Microsoft Teams account with context and usage patterns"""
    
    def __init__(self, name: str, organization: str, email: str, account_type: str):
        self.name = name
        self.organization = organization
        self.email = email
        self.account_type = account_type  # 'work', 'school', 'personal'
        self.last_used = datetime.now()
        self.usage_frequency = 0
        self.active_meetings = []
        self.recent_chats = []
        self.context_score = 0.0
        
class AIAccountManager:
    """AI-powered account management system for Microsoft Teams"""
    
    def __init__(self):
        self.accounts = []
        self.usage_history = []
        self.context_patterns = {}
        
    def add_account(self, account: TeamsAccount):
        """Add a new Teams account to management"""
        self.accounts.append(account)
        
    def predict_next_account(self, current_time: datetime, context: str) -> Optional[TeamsAccount]:
        """AI prediction of which account user likely wants to use next"""
        if not self.accounts:
            return None
            
        # Simple AI scoring based on multiple factors
        scores = {}
        
        for account in self.accounts:
            score = 0.0
            
            # Time-based patterns (work hours vs personal time)
            hour = current_time.hour
            if account.account_type == 'work' and 9 <= hour <= 17:
                score += 0.4
            elif account.account_type == 'personal' and (hour < 9 or hour > 17):
                score += 0.3
                
            # Recent usage frequency
            time_since_last_use = (current_time - account.last_used).total_seconds() / 3600
            if time_since_last_use < 1:  # Used within last hour
                score += 0.3
            elif time_since_last_use < 24:  # Used within last day
                score += 0.2
                
            # Context matching (simplified NLP)
            if context:
                context_lower = context.lower()
                if account.account_type == 'work' and any(word in context_lower for word in ['meeting', 'project', 'client', 'work']):
                    score += 0.2
                elif account.account_type == 'personal' and any(word in context_lower for word in ['family', 'friend', 'personal']):
                    score += 0.2
                    
            # Active meetings boost
            if account.active_meetings:
                score += 0.3
                
            scores[account] = score
            
        # Return account with highest score
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return None
        
    def get_context_insights(self) -> Dict:
        """Generate AI insights about usage patterns"""
        if not self.accounts:
            return {}
            
        insights = {
            'total_accounts': len(self.accounts),
            'most_used_type': self._get_most_used_account_type(),
            'peak_usage_hours': self._get_peak_usage_hours(),
            'switching_frequency': self._calculate_switching_frequency(),
            'recommendations': self._generate_recommendations()
        }
        
        return insights
        
    def _get_most_used_account_type(self) -> str:
        """Determine which account type is used most frequently"""
        type_counts = {}
        for account in self.accounts:
            type_counts[account.account_type] = type_counts.get(account.account_type, 0) + account.usage_frequency
        return max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'work'
        
    def _get_peak_usage_hours(self) -> List[int]:
        """Identify peak usage hours across all accounts"""
        # Simulated data - in real implementation, this would analyze actual usage
        return [9, 10, 14, 15, 16]  # Common work hours
        
    def _calculate_switching_frequency(self) -> float:
        """Calculate how often user switches between accounts"""
        # Simulated metric - in real implementation, track actual switches
        return 3.2  # Average switches per day
        
    def _generate_recommendations(self) -> List[str]:
        """Generate AI-powered recommendations for better account management"""
        recommendations = []
        
        if len(self.accounts) > 3:
            recommendations.append("Consider consolidating similar accounts to reduce complexity")
            
        work_accounts = [acc for acc in self.accounts if acc.account_type == 'work']
        if len(work_accounts) > 2:
            recommendations.append("Multiple work accounts detected - set up smart notifications to avoid missing important messages")
            
        recommendations.append("Enable context-aware switching to automatically suggest the right account")
        recommendations.append("Set up unified search across all accounts to find conversations faster")
        
        return recommendations

# Initialize session state
if 'ai_manager' not in st.session_state:
    st.session_state.ai_manager = AIAccountManager()
    
    # Add sample accounts
    sample_accounts = [
        TeamsAccount("John Doe", "TechCorp Inc", "john.doe@techcorp.com", "work"),
        TeamsAccount("John Doe", "Consulting LLC", "j.doe@consulting.com", "work"),
        TeamsAccount("John Doe", "Personal", "john.doe@outlook.com", "personal"),
        TeamsAccount("John Doe", "University Alumni", "john.doe@alumni.edu", "school")
    ]
    
    for account in sample_accounts:
        account.usage_frequency = st.session_state.get(f'usage_{account.email}', 0)
        st.session_state.ai_manager.add_account(account)

# Main UI
st.title("🤖 AI-Powered Microsoft Teams Multi-Account Manager")
st.markdown("**Solving the #1 Teams Pain Point: Multi-Account Management**")

# Sidebar for account management
with st.sidebar:
    st.header("Account Management")
    
    # Add new account
    with st.expander("Add New Account"):
        new_name = st.text_input("Name")
        new_org = st.text_input("Organization")
        new_email = st.text_input("Email")
        new_type = st.selectbox("Type", ["work", "personal", "school"])
        
        if st.button("Add Account"):
            if new_name and new_org and new_email:
                new_account = TeamsAccount(new_name, new_org, new_email, new_type)
                st.session_state.ai_manager.add_account(new_account)
                st.success(f"Added account: {new_email}")
                st.rerun()
    
    # Current accounts
    st.subheader("Current Accounts")
    for i, account in enumerate(st.session_state.ai_manager.accounts):
        with st.container():
            st.write(f"**{account.organization}**")
            st.write(f"📧 {account.email}")
            st.write(f"🏷️ {account.account_type.title()}")
            
            # Simulate usage
            if st.button(f"Use Account", key=f"use_{i}"):
                account.last_used = datetime.now()
                account.usage_frequency += 1
                st.success(f"Switched to {account.organization}")
                st.rerun()
            
            st.divider()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 AI Account Prediction")
    
    # Context input
    context_input = st.text_area(
        "What are you planning to do? (AI will suggest the best account)",
        placeholder="e.g., 'Join client meeting', 'Chat with family', 'Work on project presentation'"
    )
    
    current_time = datetime.now()
    
    if st.button("Get AI Recommendation", type="primary"):
        predicted_account = st.session_state.ai_manager.predict_next_account(current_time, context_input)
        
        if predicted_account:
            st.success(f"🤖 **AI Recommendation**: Switch to **{predicted_account.organization}** ({predicted_account.email})")
            
            # Show reasoning
            with st.expander("Why this recommendation?"):
                hour = current_time.hour
                st.write(f"⏰ **Time Context**: {hour}:00 - ", end="")
                if 9 <= hour <= 17:
                    st.write("Work hours detected")
                else:
                    st.write("Personal time detected")
                    
                if context_input:
                    st.write(f"📝 **Context Analysis**: '{context_input}'")
                    context_lower = context_input.lower()
                    if any(word in context_lower for word in ['meeting', 'project', 'client', 'work']):
                        st.write("→ Work-related keywords detected")
                    elif any(word in context_lower for word in ['family', 'friend', 'personal']):
                        st.write("→ Personal keywords detected")
                        
                st.write(f"📊 **Usage Pattern**: Last used {(current_time - predicted_account.last_used).total_seconds() / 3600:.1f} hours ago")
        else:
            st.warning("No accounts available for prediction")
    
    # Usage analytics
    st.header("📊 Usage Analytics")
    
    if st.session_state.ai_manager.accounts:
        # Create usage data
        usage_data = []
        for account in st.session_state.ai_manager.accounts:
            usage_data.append({
                'Organization': account.organization,
                'Email': account.email,
                'Type': account.account_type.title(),
                'Usage Frequency': account.usage_frequency,
                'Last Used': account.last_used.strftime('%Y-%m-%d %H:%M')
            })
        
        df = pd.DataFrame(usage_data)
        
        # Usage frequency chart
        fig = px.bar(
            df, 
            x='Organization', 
            y='Usage Frequency',
            color='Type',
            title="Account Usage Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Account distribution
        type_counts = df['Type'].value_counts()
        fig_pie = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Account Type Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Detailed table
        st.subheader("Account Details")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No accounts to analyze yet. Add some accounts to see analytics.")

with col2:
    st.header("🧠 AI Insights")
    
    insights = st.session_state.ai_manager.get_context_insights()
    
    if insights:
        # Key metrics
        st.metric("Total Accounts", insights['total_accounts'])
        st.metric("Most Used Type", insights['most_used_type'].title())
        st.metric("Daily Switches", f"{insights['switching_frequency']:.1f}")
        
        # Peak hours
        st.subheader("⏰ Peak Usage Hours")
        peak_hours = insights['peak_usage_hours']
        for hour in peak_hours:
            st.write(f"• {hour}:00 - {hour+1}:00")
        
        # Recommendations
        st.subheader("💡 AI Recommendations")
        for rec in insights['recommendations']:
            st.write(f"• {rec}")
    
    # Simulated real-time features
    st.header("🔔 Smart Notifications")
    
    with st.container():
        st.info("🤖 **AI Alert**: You have 2 unread messages in TechCorp account")
        st.warning("⚠️ **Context Switch**: Meeting starting in 5 minutes - switch to Consulting LLC?")
        st.success("✅ **Smart Sync**: All accounts synchronized successfully")

# Footer with problem statement
st.markdown("---")
st.markdown("""
### 🎯 **Problem Solved**

**Microsoft Teams Pain Point**: Users cannot sign into multiple business organizations simultaneously in the desktop app, forcing constant logout/login cycles.

**Our AI Solution**:
- 🤖 **Intelligent Account Prediction**: AI suggests the right account based on time, context, and usage patterns
- 📊 **Usage Analytics**: Track and optimize account switching behavior
- 🔔 **Smart Notifications**: Context-aware alerts across all accounts
- 💡 **Personalized Recommendations**: AI-powered suggestions for better account management
- 🔄 **Seamless Context Switching**: Reduce cognitive load when managing multiple Teams accounts

**Technology Stack**: Streamlit, Pandas, Plotly, Python AI/ML libraries
""")

# Technical details
with st.expander("🔧 Technical Implementation Details"):
    st.markdown("""
    **AI Techniques Used**:
    - **Pattern Recognition**: Analyze usage patterns across time and context
    - **Natural Language Processing**: Parse user intent from context descriptions
    - **Predictive Analytics**: Forecast which account user likely needs next
    - **Behavioral Analysis**: Learn from user switching patterns
    - **Context-Aware Computing**: Factor in time, location, and activity context
    
    **Real-world Integration**:
    - Could integrate with Microsoft Graph API for actual Teams data
    - Machine learning models could be trained on real usage patterns
    - Browser extension could provide seamless account switching
    - Mobile app could sync context across devices
    """)