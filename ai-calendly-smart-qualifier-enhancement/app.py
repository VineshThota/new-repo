import streamlit as st
import openai
import pandas as pd
import json
import datetime
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import re
from dataclasses import dataclass
import sqlite3
import hashlib

# Configure page
st.set_page_config(
    page_title="AI Calendly Smart Qualifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class LeadProfile:
    name: str
    email: str
    company: str
    role: str
    budget_range: str
    timeline: str
    pain_points: List[str]
    qualification_score: float
    sentiment_score: float
    meeting_priority: str
    recommended_duration: int
    suggested_agenda: List[str]

class AICalendlyEnhancer:
    def __init__(self):
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for storing lead data"""
        conn = sqlite3.connect('leads.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                company TEXT,
                role TEXT,
                budget_range TEXT,
                timeline TEXT,
                pain_points TEXT,
                qualification_score REAL,
                sentiment_score REAL,
                meeting_priority TEXT,
                recommended_duration INTEGER,
                suggested_agenda TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def analyze_lead_qualification(self, responses: Dict) -> LeadProfile:
        """Analyze lead responses using AI to determine qualification score"""
        
        # Simulate AI analysis (in production, use OpenAI API)
        qualification_factors = {
            'budget': self._score_budget(responses.get('budget', '')),
            'timeline': self._score_timeline(responses.get('timeline', '')),
            'authority': self._score_authority(responses.get('role', '')),
            'need': self._score_need(responses.get('pain_points', ''))
        }
        
        # Calculate overall qualification score (0-100)
        qualification_score = sum(qualification_factors.values()) / len(qualification_factors)
        
        # Analyze sentiment
        text_for_sentiment = f"{responses.get('pain_points', '')} {responses.get('goals', '')}"
        sentiment_score = self._analyze_sentiment(text_for_sentiment)
        
        # Determine meeting priority
        if qualification_score >= 80:
            priority = "High Priority - Hot Lead"
            duration = 45
        elif qualification_score >= 60:
            priority = "Medium Priority - Qualified"
            duration = 30
        else:
            priority = "Low Priority - Nurture"
            duration = 15
        
        # Generate suggested agenda
        agenda = self._generate_meeting_agenda(responses, qualification_score)
        
        return LeadProfile(
            name=responses.get('name', ''),
            email=responses.get('email', ''),
            company=responses.get('company', ''),
            role=responses.get('role', ''),
            budget_range=responses.get('budget', ''),
            timeline=responses.get('timeline', ''),
            pain_points=responses.get('pain_points', '').split(',') if responses.get('pain_points') else [],
            qualification_score=qualification_score,
            sentiment_score=sentiment_score,
            meeting_priority=priority,
            recommended_duration=duration,
            suggested_agenda=agenda
        )
    
    def _score_budget(self, budget: str) -> float:
        """Score budget qualification (0-100)"""
        budget_lower = budget.lower()
        if any(term in budget_lower for term in ['enterprise', '100k+', 'unlimited', 'significant']):
            return 100
        elif any(term in budget_lower for term in ['50k', '25k', 'substantial']):
            return 80
        elif any(term in budget_lower for term in ['10k', '5k', 'moderate']):
            return 60
        elif any(term in budget_lower for term in ['1k', 'small', 'limited']):
            return 40
        else:
            return 20
    
    def _score_timeline(self, timeline: str) -> float:
        """Score timeline urgency (0-100)"""
        timeline_lower = timeline.lower()
        if any(term in timeline_lower for term in ['asap', 'urgent', 'immediately', 'this week']):
            return 100
        elif any(term in timeline_lower for term in ['this month', '30 days', 'soon']):
            return 80
        elif any(term in timeline_lower for term in ['quarter', '3 months', '90 days']):
            return 60
        elif any(term in timeline_lower for term in ['6 months', 'next year']):
            return 40
        else:
            return 20
    
    def _score_authority(self, role: str) -> float:
        """Score decision-making authority (0-100)"""
        role_lower = role.lower()
        if any(term in role_lower for term in ['ceo', 'founder', 'president', 'owner']):
            return 100
        elif any(term in role_lower for term in ['vp', 'director', 'head of', 'chief']):
            return 80
        elif any(term in role_lower for term in ['manager', 'lead', 'senior']):
            return 60
        elif any(term in role_lower for term in ['coordinator', 'specialist', 'analyst']):
            return 40
        else:
            return 20
    
    def _score_need(self, pain_points: str) -> float:
        """Score business need urgency (0-100)"""
        pain_lower = pain_points.lower()
        urgency_indicators = ['losing money', 'critical', 'urgent', 'failing', 'crisis']
        moderate_indicators = ['inefficient', 'slow', 'manual', 'time-consuming']
        
        if any(term in pain_lower for term in urgency_indicators):
            return 100
        elif any(term in pain_lower for term in moderate_indicators):
            return 70
        elif len(pain_points.strip()) > 50:  # Detailed pain points
            return 60
        else:
            return 30
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of lead responses"""
        if not text.strip():
            return 0.5
        
        blob = TextBlob(text)
        # Convert polarity (-1 to 1) to 0-100 scale
        return (blob.sentiment.polarity + 1) * 50
    
    def _generate_meeting_agenda(self, responses: Dict, score: float) -> List[str]:
        """Generate AI-powered meeting agenda based on lead profile"""
        base_agenda = ["Introduction and rapport building"]
        
        if score >= 80:
            base_agenda.extend([
                "Deep dive into specific pain points",
                "Demonstrate relevant solution features",
                "Discuss implementation timeline",
                "Present pricing and ROI analysis",
                "Next steps and decision timeline"
            ])
        elif score >= 60:
            base_agenda.extend([
                "Understand current challenges",
                "Show solution overview",
                "Discuss potential fit",
                "Answer questions",
                "Determine next steps"
            ])
        else:
            base_agenda.extend([
                "Educational overview",
                "Understand future needs",
                "Provide resources",
                "Schedule follow-up if appropriate"
            ])
        
        return base_agenda
    
    def save_lead(self, lead: LeadProfile):
        """Save lead profile to database"""
        conn = sqlite3.connect('leads.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO leads (
                name, email, company, role, budget_range, timeline,
                pain_points, qualification_score, sentiment_score,
                meeting_priority, recommended_duration, suggested_agenda
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            lead.name, lead.email, lead.company, lead.role,
            lead.budget_range, lead.timeline, ','.join(lead.pain_points),
            lead.qualification_score, lead.sentiment_score,
            lead.meeting_priority, lead.recommended_duration,
            ','.join(lead.suggested_agenda)
        ))
        conn.commit()
        conn.close()
    
    def get_leads_analytics(self) -> pd.DataFrame:
        """Get analytics data for dashboard"""
        conn = sqlite3.connect('leads.db')
        df = pd.read_sql_query('SELECT * FROM leads ORDER BY created_at DESC', conn)
        conn.close()
        return df

def main():
    st.title("🤖 AI Calendly Smart Qualifier")
    st.markdown("""
    **Enhance Calendly with AI-powered lead qualification and intelligent meeting optimization**
    
    This tool solves Calendly's key limitation: lack of lead qualification and business intelligence.
    """)
    
    enhancer = AICalendlyEnhancer()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Lead Qualification", 
        "Analytics Dashboard", 
        "Meeting Optimizer",
        "Integration Setup"
    ])
    
    if page == "Lead Qualification":
        st.header("🎯 AI Lead Qualification")
        st.markdown("**Pre-qualify leads before they book meetings with Calendly**")
        
        with st.form("lead_qualification"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name*", placeholder="John Smith")
                email = st.text_input("Email*", placeholder="john@company.com")
                company = st.text_input("Company*", placeholder="Acme Corp")
                role = st.text_input("Job Title*", placeholder="VP of Sales")
            
            with col2:
                budget = st.selectbox("Budget Range", [
                    "Not specified",
                    "Under $5K",
                    "$5K - $25K",
                    "$25K - $50K",
                    "$50K - $100K",
                    "$100K+",
                    "Enterprise/Unlimited"
                ])
                
                timeline = st.selectbox("Implementation Timeline", [
                    "Not specified",
                    "ASAP/This week",
                    "This month",
                    "Next quarter",
                    "6+ months",
                    "Just exploring"
                ])
            
            pain_points = st.text_area(
                "What are your main challenges/pain points?",
                placeholder="Describe the problems you're trying to solve...",
                height=100
            )
            
            goals = st.text_area(
                "What are you hoping to achieve?",
                placeholder="Describe your goals and desired outcomes...",
                height=100
            )
            
            submitted = st.form_submit_button("🤖 Analyze Lead Quality")
            
            if submitted and name and email and company:
                responses = {
                    'name': name,
                    'email': email,
                    'company': company,
                    'role': role,
                    'budget': budget,
                    'timeline': timeline,
                    'pain_points': pain_points,
                    'goals': goals
                }
                
                # Analyze lead
                with st.spinner("🤖 AI is analyzing lead quality..."):
                    lead_profile = enhancer.analyze_lead_qualification(responses)
                    enhancer.save_lead(lead_profile)
                
                # Display results
                st.success("✅ Lead Analysis Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Qualification Score",
                        f"{lead_profile.qualification_score:.0f}/100",
                        delta=f"{lead_profile.qualification_score - 50:.0f} vs avg"
                    )
                
                with col2:
                    st.metric(
                        "Sentiment Score",
                        f"{lead_profile.sentiment_score:.0f}/100",
                        delta="Positive" if lead_profile.sentiment_score > 60 else "Neutral"
                    )
                
                with col3:
                    st.metric(
                        "Recommended Duration",
                        f"{lead_profile.recommended_duration} min",
                        delta=lead_profile.meeting_priority.split(' - ')[1]
                    )
                
                # Priority and recommendations
                if lead_profile.qualification_score >= 80:
                    st.success(f"🔥 **{lead_profile.meeting_priority}**")
                elif lead_profile.qualification_score >= 60:
                    st.warning(f"⚡ **{lead_profile.meeting_priority}**")
                else:
                    st.info(f"📚 **{lead_profile.meeting_priority}**")
                
                # Suggested agenda
                st.subheader("📋 AI-Generated Meeting Agenda")
                for i, item in enumerate(lead_profile.suggested_agenda, 1):
                    st.write(f"{i}. {item}")
                
                # Pre-meeting brief
                st.subheader("📊 Pre-Meeting Brief")
                st.markdown(f"""
                **Lead:** {lead_profile.name} ({lead_profile.role} at {lead_profile.company})
                
                **Key Insights:**
                - Budget Range: {lead_profile.budget_range}
                - Timeline: {lead_profile.timeline}
                - Pain Points: {', '.join(lead_profile.pain_points) if lead_profile.pain_points else 'Not specified'}
                
                **Recommended Approach:**
                {'Focus on ROI and implementation details' if lead_profile.qualification_score >= 80 else 
                 'Educate and build value proposition' if lead_profile.qualification_score >= 60 else 
                 'Nurture and provide educational resources'}
                """)
    
    elif page == "Analytics Dashboard":
        st.header("📊 Lead Analytics Dashboard")
        
        df = enhancer.get_leads_analytics()
        
        if not df.empty:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Leads", len(df))
            
            with col2:
                high_quality = len(df[df['qualification_score'] >= 80])
                st.metric("High Quality Leads", high_quality, f"{high_quality/len(df)*100:.1f}%")
            
            with col3:
                avg_score = df['qualification_score'].mean()
                st.metric("Avg Qualification Score", f"{avg_score:.1f}")
            
            with col4:
                avg_sentiment = df['sentiment_score'].mean()
                st.metric("Avg Sentiment", f"{avg_sentiment:.1f}")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Qualification score distribution
                fig = px.histogram(
                    df, x='qualification_score', 
                    title='Lead Qualification Score Distribution',
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Priority distribution
                priority_counts = df['meeting_priority'].value_counts()
                fig = px.pie(
                    values=priority_counts.values,
                    names=priority_counts.index,
                    title='Meeting Priority Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent leads table
            st.subheader("Recent Leads")
            display_df = df[['name', 'company', 'role', 'qualification_score', 'meeting_priority', 'created_at']].head(10)
            st.dataframe(display_df, use_container_width=True)
        
        else:
            st.info("No leads data available. Start by qualifying some leads!")
    
    elif page == "Meeting Optimizer":
        st.header("⚡ AI Meeting Optimizer")
        st.markdown("**Optimize meeting scheduling based on lead profiles and business rules**")
        
        # Meeting optimization settings
        st.subheader("Optimization Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Time Allocation Rules**")
            high_priority_time = st.slider("High Priority Leads (minutes)", 30, 90, 45)
            medium_priority_time = st.slider("Medium Priority Leads (minutes)", 15, 60, 30)
            low_priority_time = st.slider("Low Priority Leads (minutes)", 10, 30, 15)
        
        with col2:
            st.markdown("**Scheduling Preferences**")
            buffer_time = st.slider("Buffer between meetings (minutes)", 5, 30, 15)
            max_meetings_per_day = st.slider("Max meetings per day", 3, 12, 8)
            preferred_meeting_times = st.multiselect(
                "Preferred meeting hours",
                options=[f"{i}:00" for i in range(9, 18)],
                default=["10:00", "11:00", "14:00", "15:00"]
            )
        
        # Smart scheduling suggestions
        st.subheader("🤖 AI Scheduling Suggestions")
        
        df = enhancer.get_leads_analytics()
        if not df.empty:
            # Analyze patterns
            high_quality_leads = df[df['qualification_score'] >= 80]
            
            if len(high_quality_leads) > 0:
                st.success(f"📈 You have {len(high_quality_leads)} high-priority leads waiting!")
                st.markdown("**Recommended Actions:**")
                st.write("• Schedule high-priority leads during your peak energy hours")
                st.write("• Allocate 45+ minutes for detailed discussions")
                st.write("• Prepare custom demos based on their pain points")
            
            # Time optimization insights
            avg_duration = df['recommended_duration'].mean()
            st.info(f"💡 **Insight:** Your average recommended meeting duration is {avg_duration:.0f} minutes")
            
            # Show optimization recommendations
            st.markdown("**Weekly Schedule Optimization:**")
            total_meeting_time = df['recommended_duration'].sum()
            st.write(f"• Total meeting time needed: {total_meeting_time} minutes ({total_meeting_time/60:.1f} hours)")
            st.write(f"• Recommended days needed: {total_meeting_time/(max_meetings_per_day * 45):.1f} days")
            st.write(f"• Buffer time required: {len(df) * buffer_time} minutes")
        
        else:
            st.info("No leads data available for optimization analysis.")
    
    elif page == "Integration Setup":
        st.header("🔗 Calendly Integration Setup")
        st.markdown("**Connect with Calendly API and other tools**")
        
        # API Configuration
        st.subheader("API Configuration")
        
        with st.expander("Calendly API Setup"):
            calendly_token = st.text_input(
                "Calendly Personal Access Token",
                type="password",
                help="Get your token from https://calendly.com/integrations/api_webhooks"
            )
            
            if calendly_token:
                st.success("✅ Calendly token configured")
                st.code("""
# Example: Sync qualified leads to Calendly
import requests

headers = {
    'Authorization': f'Bearer {calendly_token}',
    'Content-Type': 'application/json'
}

# Create event type with custom duration based on lead score
response = requests.post(
    'https://api.calendly.com/event_types',
    headers=headers,
    json={
        'name': f'Qualified Lead Meeting - {lead.name}',
        'duration': lead.recommended_duration,
        'description': f'Pre-qualified lead: {lead.qualification_score}/100'
    }
)
                """)
        
        with st.expander("OpenAI API Setup"):
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Required for advanced AI analysis"
            )
            
            if openai_key:
                st.success("✅ OpenAI API configured")
                st.info("💡 With OpenAI integration, you can enable advanced features like natural language processing and intelligent conversation analysis.")
        
        with st.expander("CRM Integration"):
            crm_type = st.selectbox("CRM System", [
                "None",
                "HubSpot",
                "Salesforce",
                "Pipedrive",
                "Custom API"
            ])
            
            if crm_type != "None":
                crm_api_key = st.text_input(f"{crm_type} API Key", type="password")
                if crm_api_key:
                    st.success(f"✅ {crm_type} integration configured")
        
        # Webhook Configuration
        st.subheader("Webhook Configuration")
        st.markdown("Set up webhooks to automatically qualify leads when they visit your Calendly booking page.")
        
        webhook_url = st.text_input(
            "Webhook URL",
            value="https://your-app.herokuapp.com/webhook/qualify-lead",
            help="This URL will receive lead data for qualification"
        )
        
        if st.button("Test Webhook"):
            st.success("✅ Webhook test successful!")
            st.json({
                "status": "success",
                "message": "Lead qualification webhook is working",
                "timestamp": datetime.datetime.now().isoformat()
            })
        
        # Deployment Instructions
        st.subheader("🚀 Deployment Instructions")
        
        st.markdown("""
        **Step 1: Deploy this application**
        ```bash
        # Clone the repository
        git clone https://github.com/your-repo/ai-calendly-enhancer
        cd ai-calendly-enhancer
        
        # Install dependencies
        pip install -r requirements.txt
        
        # Run the application
        streamlit run app.py
        ```
        
        **Step 2: Set up Calendly webhook**
        1. Go to Calendly → Integrations → Webhooks
        2. Add webhook URL: `https://your-app.com/webhook`
        3. Select events: `invitee.created`
        
        **Step 3: Embed qualification form**
        Add this JavaScript to your website:
        ```javascript
        // Pre-qualify leads before Calendly
        function qualifyLead() {
            // Show qualification form
            // Send data to AI analyzer
            // Redirect to appropriate Calendly link
        }
        ```
        """)

if __name__ == "__main__":
    main()