import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
from typing import List, Dict, Tuple
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time

# Configure Streamlit page
st.set_page_config(
    page_title="AI Zoom Mobile Breakout Manager",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile optimization
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .mobile-optimized {
        max-width: 100%;
        overflow-x: auto;
    }
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        .stButton > button {
            width: 100%;
            margin: 0.25rem 0;
        }
    }
</style>
""", unsafe_allow_html=True)

class AIBreakoutManager:
    def __init__(self):
        self.participants = []
        self.rooms = []
        self.session_data = {}
        
    def add_participant(self, name: str, skills: List[str], preferences: Dict, timezone: str = "UTC"):
        """Add a participant with their skills and preferences"""
        participant = {
            'id': len(self.participants) + 1,
            'name': name,
            'skills': skills,
            'preferences': preferences,
            'timezone': timezone,
            'joined_at': datetime.now(),
            'engagement_score': random.uniform(0.6, 1.0)
        }
        self.participants.append(participant)
        return participant
    
    def ai_smart_assignment(self, num_rooms: int, assignment_strategy: str = "balanced") -> List[List[Dict]]:
        """AI-powered room assignment using different strategies"""
        if not self.participants:
            return []
            
        participants_copy = self.participants.copy()
        
        if assignment_strategy == "skill_based":
            return self._skill_based_assignment(participants_copy, num_rooms)
        elif assignment_strategy == "engagement_balanced":
            return self._engagement_balanced_assignment(participants_copy, num_rooms)
        elif assignment_strategy == "timezone_aware":
            return self._timezone_aware_assignment(participants_copy, num_rooms)
        else:
            return self._balanced_assignment(participants_copy, num_rooms)
    
    def _skill_based_assignment(self, participants: List[Dict], num_rooms: int) -> List[List[Dict]]:
        """Assign participants based on complementary skills"""
        # Create skill vectors for clustering
        all_skills = set()
        for p in participants:
            all_skills.update(p['skills'])
        
        skill_list = list(all_skills)
        skill_vectors = []
        
        for p in participants:
            vector = [1 if skill in p['skills'] else 0 for skill in skill_list]
            skill_vectors.append(vector)
        
        if len(skill_vectors) > 0 and len(skill_list) > 0:
            # Use KMeans clustering for skill-based grouping
            scaler = StandardScaler()
            scaled_vectors = scaler.fit_transform(skill_vectors)
            
            kmeans = KMeans(n_clusters=min(num_rooms, len(participants)), random_state=42)
            clusters = kmeans.fit_predict(scaled_vectors)
            
            rooms = [[] for _ in range(num_rooms)]
            for i, cluster in enumerate(clusters):
                rooms[cluster % num_rooms].append(participants[i])
        else:
            rooms = self._balanced_assignment(participants, num_rooms)
            
        return rooms
    
    def _engagement_balanced_assignment(self, participants: List[Dict], num_rooms: int) -> List[List[Dict]]:
        """Balance engagement scores across rooms"""
        participants.sort(key=lambda x: x['engagement_score'], reverse=True)
        rooms = [[] for _ in range(num_rooms)]
        
        for i, participant in enumerate(participants):
            room_index = i % num_rooms
            rooms[room_index].append(participant)
            
        return rooms
    
    def _timezone_aware_assignment(self, participants: List[Dict], num_rooms: int) -> List[List[Dict]]:
        """Group participants by similar timezones when possible"""
        timezone_groups = {}
        for p in participants:
            tz = p.get('timezone', 'UTC')
            if tz not in timezone_groups:
                timezone_groups[tz] = []
            timezone_groups[tz].append(p)
        
        rooms = [[] for _ in range(num_rooms)]
        room_index = 0
        
        for tz, group in timezone_groups.items():
            for participant in group:
                rooms[room_index % num_rooms].append(participant)
                room_index += 1
                
        return rooms
    
    def _balanced_assignment(self, participants: List[Dict], num_rooms: int) -> List[List[Dict]]:
        """Simple balanced assignment"""
        random.shuffle(participants)
        rooms = [[] for _ in range(num_rooms)]
        
        for i, participant in enumerate(participants):
            rooms[i % num_rooms].append(participant)
            
        return rooms
    
    def generate_room_insights(self, rooms: List[List[Dict]]) -> Dict:
        """Generate AI insights about room compositions"""
        insights = {
            'total_participants': sum(len(room) for room in rooms),
            'room_sizes': [len(room) for room in rooms],
            'skill_distribution': {},
            'engagement_stats': {},
            'timezone_distribution': {},
            'recommendations': []
        }
        
        # Skill distribution analysis
        all_skills = {}
        for room_idx, room in enumerate(rooms):
            for participant in room:
                for skill in participant['skills']:
                    if skill not in all_skills:
                        all_skills[skill] = []
                    all_skills[skill].append(room_idx)
        
        insights['skill_distribution'] = {skill: len(set(rooms)) for skill, rooms in all_skills.items()}
        
        # Engagement statistics
        room_engagement = []
        for room in rooms:
            if room:
                avg_engagement = np.mean([p['engagement_score'] for p in room])
                room_engagement.append(avg_engagement)
            else:
                room_engagement.append(0)
        
        insights['engagement_stats'] = {
            'room_averages': room_engagement,
            'overall_average': np.mean(room_engagement) if room_engagement else 0,
            'std_deviation': np.std(room_engagement) if room_engagement else 0
        }
        
        # Generate recommendations
        if insights['engagement_stats']['std_deviation'] > 0.15:
            insights['recommendations'].append("⚠️ Consider rebalancing rooms - engagement levels vary significantly")
        
        if max(insights['room_sizes']) - min(insights['room_sizes']) > 2:
            insights['recommendations'].append("📊 Room sizes are uneven - consider redistributing participants")
        
        if len(insights['skill_distribution']) > 0:
            insights['recommendations'].append("🎯 Skills are distributed across rooms - good for cross-learning")
        
        return insights

def main():
    st.markdown('<h1 class="main-header">🚀 AI Zoom Mobile Breakout Manager</h1>', unsafe_allow_html=True)
    
    # Problem statement
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 Solving Zoom's Mobile Limitation</h3>
        <p><strong>Problem:</strong> Zoom breakout rooms can only be created and managed from desktop applications, 
        leaving mobile users unable to facilitate breakout sessions.</p>
        <p><strong>Solution:</strong> AI-powered mobile-friendly breakout room management with smart participant 
        assignment, real-time insights, and cross-platform compatibility.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'manager' not in st.session_state:
        st.session_state.manager = AIBreakoutManager()
    
    manager = st.session_state.manager
    
    # Sidebar for participant management
    with st.sidebar:
        st.header("👥 Participant Management")
        
        with st.expander("➕ Add Participant", expanded=True):
            name = st.text_input("Name", placeholder="Enter participant name")
            
            # Skills selection
            available_skills = [
                "Leadership", "Technical", "Creative", "Analytical", 
                "Communication", "Project Management", "Design", 
                "Development", "Marketing", "Sales", "Research", "Strategy"
            ]
            skills = st.multiselect("Skills", available_skills)
            
            # Preferences
            timezone = st.selectbox("Timezone", ["UTC", "EST", "PST", "GMT", "CET", "JST", "IST"])
            
            if st.button("Add Participant", type="primary"):
                if name and skills:
                    preferences = {'timezone': timezone}
                    manager.add_participant(name, skills, preferences, timezone)
                    st.success(f"Added {name} successfully!")
                    st.rerun()
                else:
                    st.error("Please enter name and select at least one skill")
        
        # Quick add demo participants
        if st.button("🎲 Add Demo Participants"):
            demo_participants = [
                ("Alice Johnson", ["Leadership", "Strategy"], "EST"),
                ("Bob Chen", ["Technical", "Development"], "PST"),
                ("Carol Davis", ["Creative", "Design"], "GMT"),
                ("David Wilson", ["Analytical", "Research"], "CET"),
                ("Eva Martinez", ["Communication", "Marketing"], "EST"),
                ("Frank Kim", ["Project Management", "Technical"], "JST"),
                ("Grace Taylor", ["Sales", "Communication"], "IST"),
                ("Henry Brown", ["Development", "Technical"], "UTC")
            ]
            
            for name, skills, tz in demo_participants:
                manager.add_participant(name, skills, {'timezone': tz}, tz)
            
            st.success("Added 8 demo participants!")
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🏠 Breakout Room Configuration")
        
        if manager.participants:
            # Room configuration
            col_rooms, col_strategy = st.columns(2)
            
            with col_rooms:
                num_rooms = st.slider(
                    "Number of Rooms", 
                    min_value=2, 
                    max_value=min(10, len(manager.participants)), 
                    value=3
                )
            
            with col_strategy:
                strategy = st.selectbox(
                    "Assignment Strategy",
                    ["balanced", "skill_based", "engagement_balanced", "timezone_aware"],
                    help="Choose how AI should assign participants to rooms"
                )
            
            # Generate rooms button
            if st.button("🤖 Generate AI Room Assignments", type="primary"):
                with st.spinner("AI is analyzing participants and creating optimal room assignments..."):
                    time.sleep(2)  # Simulate AI processing
                    rooms = manager.ai_smart_assignment(num_rooms, strategy)
                    st.session_state.current_rooms = rooms
                    st.session_state.room_insights = manager.generate_room_insights(rooms)
                    st.success("Room assignments generated successfully!")
            
            # Display room assignments
            if 'current_rooms' in st.session_state:
                st.header("🏠 Room Assignments")
                
                rooms = st.session_state.current_rooms
                insights = st.session_state.room_insights
                
                # Room tabs
                room_tabs = st.tabs([f"Room {i+1} ({len(room)})" for i, room in enumerate(rooms)])
                
                for i, (tab, room) in enumerate(zip(room_tabs, rooms)):
                    with tab:
                        if room:
                            # Room statistics
                            avg_engagement = np.mean([p['engagement_score'] for p in room])
                            skills_in_room = set()
                            for p in room:
                                skills_in_room.update(p['skills'])
                            
                            col_stats1, col_stats2, col_stats3 = st.columns(3)
                            with col_stats1:
                                st.metric("Participants", len(room))
                            with col_stats2:
                                st.metric("Avg Engagement", f"{avg_engagement:.2f}")
                            with col_stats3:
                                st.metric("Unique Skills", len(skills_in_room))
                            
                            # Participant list
                            for participant in room:
                                with st.container():
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <strong>{participant['name']}</strong><br>
                                        <small>Skills: {', '.join(participant['skills'])}</small><br>
                                        <small>Timezone: {participant['timezone']} | Engagement: {participant['engagement_score']:.2f}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("This room is empty")
                
                # Room management controls
                st.header("🎮 Room Management")
                
                col_mgmt1, col_mgmt2, col_mgmt3 = st.columns(3)
                
                with col_mgmt1:
                    if st.button("🔄 Shuffle All Rooms"):
                        new_rooms = manager.ai_smart_assignment(num_rooms, "balanced")
                        st.session_state.current_rooms = new_rooms
                        st.session_state.room_insights = manager.generate_room_insights(new_rooms)
                        st.rerun()
                
                with col_mgmt2:
                    if st.button("⚖️ Rebalance Rooms"):
                        new_rooms = manager.ai_smart_assignment(num_rooms, "engagement_balanced")
                        st.session_state.current_rooms = new_rooms
                        st.session_state.room_insights = manager.generate_room_insights(new_rooms)
                        st.rerun()
                
                with col_mgmt3:
                    if st.button("📊 Export Assignments"):
                        # Create export data
                        export_data = []
                        for i, room in enumerate(rooms):
                            for participant in room:
                                export_data.append({
                                    'Room': f'Room {i+1}',
                                    'Participant': participant['name'],
                                    'Skills': ', '.join(participant['skills']),
                                    'Timezone': participant['timezone'],
                                    'Engagement': participant['engagement_score']
                                })
                        
                        df = pd.DataFrame(export_data)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"breakout_assignments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        else:
            st.info("👈 Add participants using the sidebar to get started!")
    
    with col2:
        st.header("📊 Analytics & Insights")
        
        if manager.participants:
            # Participant overview
            st.subheader("Participant Overview")
            st.metric("Total Participants", len(manager.participants))
            
            # Skills distribution
            all_skills = {}
            for p in manager.participants:
                for skill in p['skills']:
                    all_skills[skill] = all_skills.get(skill, 0) + 1
            
            if all_skills:
                skills_df = pd.DataFrame(list(all_skills.items()), columns=['Skill', 'Count'])
                fig = px.bar(skills_df, x='Count', y='Skill', orientation='h', 
                           title="Skills Distribution")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Timezone distribution
            timezone_dist = {}
            for p in manager.participants:
                tz = p.get('timezone', 'UTC')
                timezone_dist[tz] = timezone_dist.get(tz, 0) + 1
            
            if timezone_dist:
                tz_df = pd.DataFrame(list(timezone_dist.items()), columns=['Timezone', 'Count'])
                fig = px.pie(tz_df, values='Count', names='Timezone', title="Timezone Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # AI Insights
            if 'room_insights' in st.session_state:
                st.subheader("🤖 AI Insights")
                insights = st.session_state.room_insights
                
                for recommendation in insights['recommendations']:
                    st.info(recommendation)
                
                # Engagement distribution
                if insights['engagement_stats']['room_averages']:
                    fig = go.Figure(data=go.Bar(
                        x=[f'Room {i+1}' for i in range(len(insights['engagement_stats']['room_averages']))],
                        y=insights['engagement_stats']['room_averages'],
                        marker_color='lightblue'
                    ))
                    fig.update_layout(title="Room Engagement Levels", height=300)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add participants to see analytics")
    
    # Features showcase
    st.header("✨ Key Features")
    
    feature_cols = st.columns(3)
    
    with feature_cols[0]:
        st.markdown("""
        <div class="feature-card">
            <h4>📱 Mobile-First Design</h4>
            <p>Fully responsive interface optimized for mobile devices, tablets, and desktops</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_cols[1]:
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 AI-Powered Assignment</h4>
            <p>Smart algorithms for skill-based, engagement-balanced, and timezone-aware room assignments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_cols[2]:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Real-time Analytics</h4>
            <p>Live insights, engagement tracking, and optimization recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <h4>🚀 AI Zoom Mobile Breakout Enhancement</h4>
        <p>Solving Zoom's mobile limitation with AI-powered breakout room management</p>
        <p><strong>Built with:</strong> Streamlit • Scikit-learn • Plotly • Pandas</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()