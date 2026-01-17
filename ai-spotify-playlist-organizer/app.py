#!/usr/bin/env python3
"""
Streamlit Web Application for AI Spotify Playlist Organizer
Provides an intuitive web interface for organizing Spotify libraries using AI.

Author: Vinesh Thota
Date: January 2026
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from typing import List, Dict

# Import our custom organizer
from spotify_organizer import SpotifyOrganizer, PlaylistSuggestion

# Page configuration
st.set_page_config(
    page_title="AI Spotify Playlist Organizer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .playlist-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1DB954;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'organizer' not in st.session_state:
    st.session_state.organizer = None
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'tracks_loaded' not in st.session_state:
    st.session_state.tracks_loaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'playlist_suggestions' not in st.session_state:
    st.session_state.playlist_suggestions = []

def load_credentials():
    """Load Spotify API credentials from environment or user input"""
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        st.sidebar.header("🔑 Spotify API Credentials")
        st.sidebar.markdown("""
        To use this app, you need Spotify API credentials:
        1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
        2. Create a new app
        3. Copy your Client ID and Client Secret
        4. Add `http://localhost:8501` to Redirect URIs
        """)
        
        client_id = st.sidebar.text_input("Client ID", type="password")
        client_secret = st.sidebar.text_input("Client Secret", type="password")
        
        if not client_id or not client_secret:
            st.warning("Please enter your Spotify API credentials in the sidebar.")
            return None, None
    
    return client_id, client_secret

def authenticate_spotify(client_id: str, client_secret: str):
    """Authenticate with Spotify API"""
    try:
        organizer = SpotifyOrganizer(client_id, client_secret, "http://localhost:8501")
        
        if organizer.authenticate():
            st.session_state.organizer = organizer
            st.session_state.authenticated = True
            return True
        else:
            st.error("Authentication failed. Please check your credentials.")
            return False
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False

def display_playlist_suggestions(suggestions: List[PlaylistSuggestion]):
    """Display AI-generated playlist suggestions"""
    st.header("🎵 AI-Generated Playlist Suggestions")
    
    if not suggestions:
        st.warning("No playlist suggestions available. Please run the analysis first.")
        return
    
    # Create columns for playlist cards
    cols = st.columns(2)
    
    for i, suggestion in enumerate(suggestions):
        col = cols[i % 2]
        
        with col:
            with st.container():
                st.markdown(f"""
                <div class="playlist-card">
                    <h3>{suggestion.name}</h3>
                    <p><strong>Tracks:</strong> {len(suggestion.tracks)}</p>
                    <p><strong>Mood:</strong> {suggestion.mood}</p>
                    <p><strong>Energy Level:</strong> {suggestion.energy_level}</p>
                    <p><strong>Avg Tempo:</strong> {suggestion.characteristics['avg_tempo']:.0f} BPM</p>
                    <p><strong>Avg Energy:</strong> {suggestion.characteristics['avg_energy']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show top tracks in this playlist
                with st.expander(f"View tracks in {suggestion.name}"):
                    track_data = []
                    for track in suggestion.tracks[:10]:  # Show first 10 tracks
                        track_data.append({
                            'Track': track.name,
                            'Artist': track.artist,
                            'Popularity': track.popularity,
                            'Mood': track.mood
                        })
                    
                    if track_data:
                        df = pd.DataFrame(track_data)
                        st.dataframe(df, use_container_width=True)

def display_music_insights(insights: Dict):
    """Display music library insights"""
    st.header("📊 Your Music Insights")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Library Size", f"{insights['library_size']:,} tracks")
    
    with col2:
        st.metric("Energy Level", insights['energy_level'])
    
    with col3:
        st.metric("Avg Tempo", f"{insights['avg_tempo']:.0f} BPM")
    
    with col4:
        st.metric("Avg Danceability", f"{insights['avg_danceability']:.2f}")
    
    # Energy distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Energy Distribution")
        energy_data = insights['energy_distribution']
        fig_energy = px.pie(
            values=list(energy_data.values()),
            names=list(energy_data.keys()),
            title="Energy Levels in Your Library",
            color_discrete_map={
                'high': '#ff6b6b',
                'medium': '#feca57', 
                'low': '#48dbfb'
            }
        )
        st.plotly_chart(fig_energy, use_container_width=True)
    
    with col2:
        st.subheader("Mood Distribution")
        if insights['mood_distribution']:
            mood_data = insights['mood_distribution']
            fig_mood = px.bar(
                x=list(mood_data.keys()),
                y=list(mood_data.values()),
                title="Mood Distribution in Your Library",
                color=list(mood_data.values()),
                color_continuous_scale='viridis'
            )
            fig_mood.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_mood, use_container_width=True)
        else:
            st.info("Mood analysis not available yet.")
    
    # Most popular tracks
    if insights['most_popular_tracks']:
        st.subheader("🔥 Your Most Popular Tracks")
        popular_df = pd.DataFrame(insights['most_popular_tracks'])
        st.dataframe(popular_df, use_container_width=True)

def display_clustering_visualization(organizer: SpotifyOrganizer):
    """Display clustering visualization"""
    st.header("🎨 Music Clustering Visualization")
    
    try:
        fig = organizer.visualize_clusters()
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **How to read this chart:**
        - Each dot represents a song in your library
        - Colors represent different AI-identified clusters
        - Similar songs are grouped together
        - Hover over dots to see song details
        """)
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">🎵 AI Spotify Playlist Organizer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Transform your chaotic Spotify library into organized, AI-curated playlists! 
    This tool uses machine learning to analyze your music and create smart playlists based on mood, energy, and musical characteristics.
    """)
    
    # Sidebar for controls
    st.sidebar.title("🎛️ Controls")
    
    # Load credentials
    client_id, client_secret = load_credentials()
    if not client_id or not client_secret:
        return
    
    # Authentication
    if not st.session_state.authenticated:
        st.sidebar.header("Step 1: Connect to Spotify")
        if st.sidebar.button("🔗 Connect to Spotify", type="primary"):
            with st.spinner("Connecting to Spotify..."):
                if authenticate_spotify(client_id, client_secret):
                    st.success("✅ Successfully connected to Spotify!")
                    st.rerun()
        return
    
    # Show authenticated user info
    st.sidebar.success("✅ Connected to Spotify")
    
    # Step 2: Load library
    st.sidebar.header("Step 2: Load Your Library")
    
    if not st.session_state.tracks_loaded:
        track_limit = st.sidebar.slider("Number of tracks to analyze", 100, 5000, 1000, 100)
        
        if st.sidebar.button("📚 Load My Library", type="primary"):
            with st.spinner(f"Loading {track_limit} tracks from your library..."):
                try:
                    tracks = st.session_state.organizer.fetch_user_library(limit=track_limit)
                    st.session_state.tracks_loaded = True
                    st.success(f"✅ Loaded {len(tracks)} tracks from your library!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading library: {str(e)}")
        return
    
    st.sidebar.success(f"✅ Library loaded ({len(st.session_state.organizer.tracks)} tracks)")
    
    # Step 3: Analyze library
    st.sidebar.header("Step 3: AI Analysis")
    
    if not st.session_state.analysis_complete:
        if st.sidebar.button("🧠 Analyze with AI", type="primary"):
            with st.spinner("Performing AI analysis..."):
                try:
                    # Analyze audio features
                    st.session_state.organizer.analyze_audio_features()
                    
                    # Perform clustering
                    st.session_state.organizer.perform_clustering()
                    
                    # Create playlist suggestions
                    suggestions = st.session_state.organizer.create_playlist_suggestions()
                    st.session_state.playlist_suggestions = suggestions
                    st.session_state.analysis_complete = True
                    
                    st.success(f"✅ AI analysis complete! Generated {len(suggestions)} playlist suggestions.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        return
    
    st.sidebar.success("✅ AI analysis complete")
    
    # Step 4: Create playlists
    st.sidebar.header("Step 4: Create Playlists")
    
    if st.sidebar.button("🎵 Create Playlists in Spotify", type="primary"):
        with st.spinner("Creating playlists in your Spotify account..."):
            try:
                playlist_urls = st.session_state.organizer.create_spotify_playlists()
                st.success(f"✅ Successfully created {len(playlist_urls)} playlists in your Spotify account!")
                
                st.markdown("**Created Playlists:**")
                for i, url in enumerate(playlist_urls, 1):
                    st.markdown(f"{i}. [Open Playlist {i}]({url})")
                    
            except Exception as e:
                st.error(f"Error creating playlists: {str(e)}")
    
    # Main content area
    if st.session_state.analysis_complete:
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Insights", "🎵 Playlists", "🎨 Visualization", "📁 Export"])
        
        with tab1:
            insights = st.session_state.organizer.get_music_insights()
            display_music_insights(insights)
        
        with tab2:
            display_playlist_suggestions(st.session_state.playlist_suggestions)
        
        with tab3:
            display_clustering_visualization(st.session_state.organizer)
        
        with tab4:
            st.header("📁 Export Analysis")
            st.markdown("Download your music analysis results as a JSON file.")
            
            if st.button("📥 Export Analysis"):
                try:
                    # Create export data
                    export_data = {
                        'timestamp': datetime.now().isoformat(),
                        'library_size': len(st.session_state.organizer.tracks),
                        'insights': st.session_state.organizer.get_music_insights(),
                        'playlist_suggestions': [
                            {
                                'name': suggestion.name,
                                'description': suggestion.description,
                                'track_count': len(suggestion.tracks),
                                'characteristics': suggestion.characteristics,
                                'mood': suggestion.mood,
                                'energy_level': suggestion.energy_level
                            }
                            for suggestion in st.session_state.playlist_suggestions
                        ]
                    }
                    
                    # Convert to JSON string
                    json_str = json.dumps(export_data, indent=2)
                    
                    # Create download button
                    st.download_button(
                        label="📥 Download Analysis",
                        data=json_str,
                        file_name=f"spotify_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                    st.success("✅ Analysis ready for download!")
                    
                except Exception as e:
                    st.error(f"Error exporting analysis: {str(e)}")
    
    else:
        # Show progress and instructions
        st.info("""
        👆 **Follow the steps in the sidebar to get started:**
        
        1. **Connect to Spotify** - Authenticate with your Spotify account
        2. **Load Your Library** - Import your saved tracks for analysis
        3. **AI Analysis** - Let our AI analyze your music and create smart clusters
        4. **Create Playlists** - Generate organized playlists in your Spotify account
        
        The entire process takes just a few minutes and will transform how you organize your music!
        """)
        
        # Show some example results
        st.header("🎯 What You'll Get")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎵 Smart Playlists:**
            - 🔥 High Energy Bangers
            - 😌 Chill Vibes  
            - 🎸 Rock Anthems
            - 💃 Dance Floor
            - 🌙 Midnight Thoughts
            """)
        
        with col2:
            st.markdown("""
            **📊 Music Insights:**
            - Energy distribution analysis
            - Mood classification
            - Tempo and genre patterns
            - Most popular tracks
            - Listening preferences
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        Built with ❤️ to solve real Spotify user frustrations | 
        <a href="https://github.com/VineshThota/new-repo" target="_blank">View on GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
