#!/usr/bin/env python3
"""
AI Spotify Playlist Organizer
Automatically organize large Spotify libraries using machine learning clustering.

Author: Vinesh Thota
Date: January 2026
"""

import os
import json
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Track:
    """Data class for track information"""
    id: str
    name: str
    artist: str
    album: str
    popularity: int
    duration_ms: int
    audio_features: Dict
    genres: List[str] = None
    mood: str = None
    cluster: int = None

@dataclass
class PlaylistSuggestion:
    """Data class for AI-generated playlist suggestions"""
    name: str
    description: str
    tracks: List[Track]
    characteristics: Dict
    mood: str
    energy_level: str

class SpotifyOrganizer:
    """AI-powered Spotify playlist organizer"""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str = "http://localhost:8080/callback"):
        """Initialize Spotify organizer with API credentials"""
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        
        # Spotify API scopes needed
        self.scope = "user-library-read playlist-modify-public playlist-modify-private user-top-read"
        
        # Initialize Spotify client
        self.sp = None
        self.user_id = None
        
        # Data storage
        self.tracks = []
        self.audio_features_df = None
        self.clusters = None
        self.playlist_suggestions = []
        
        # ML models
        self.scaler = StandardScaler()
        self.kmeans = None
        self.optimal_clusters = 8
        
        logger.info("SpotifyOrganizer initialized")
    
    def authenticate(self) -> bool:
        """Authenticate with Spotify API"""
        try:
            auth_manager = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope=self.scope,
                cache_path=".spotify_cache"
            )
            
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            
            # Get user info
            user_info = self.sp.current_user()
            self.user_id = user_info['id']
            
            logger.info(f"Successfully authenticated as {user_info['display_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def fetch_user_library(self, limit: int = 5000) -> List[Track]:
        """Fetch user's saved tracks from Spotify"""
        if not self.sp:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        logger.info("Fetching user library...")
        tracks = []
        offset = 0
        batch_size = 50
        
        while len(tracks) < limit:
            try:
                # Get saved tracks
                results = self.sp.current_user_saved_tracks(
                    limit=min(batch_size, limit - len(tracks)),
                    offset=offset
                )
                
                if not results['items']:
                    break
                
                # Process tracks
                track_ids = []
                for item in results['items']:
                    track = item['track']
                    if track and track['id']:
                        track_obj = Track(
                            id=track['id'],
                            name=track['name'],
                            artist=', '.join([artist['name'] for artist in track['artists']]),
                            album=track['album']['name'],
                            popularity=track['popularity'],
                            duration_ms=track['duration_ms'],
                            audio_features={}
                        )
                        tracks.append(track_obj)
                        track_ids.append(track['id'])
                
                # Get audio features in batches
                if track_ids:
                    audio_features = self.sp.audio_features(track_ids)
                    for i, features in enumerate(audio_features):
                        if features:
                            tracks[len(tracks) - len(track_ids) + i].audio_features = features
                
                offset += batch_size
                logger.info(f"Fetched {len(tracks)} tracks...")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching tracks: {e}")
                break
        
        self.tracks = tracks
        logger.info(f"Successfully fetched {len(tracks)} tracks")
        return tracks
    
    def analyze_audio_features(self) -> pd.DataFrame:
        """Analyze audio features of all tracks"""
        if not self.tracks:
            raise Exception("No tracks loaded. Call fetch_user_library() first.")
        
        logger.info("Analyzing audio features...")
        
        # Extract audio features
        features_data = []
        for track in self.tracks:
            if track.audio_features:
                features = track.audio_features.copy()
                features['track_id'] = track.id
                features['track_name'] = track.name
                features['artist'] = track.artist
                features['popularity'] = track.popularity
                features['duration_ms'] = track.duration_ms
                features_data.append(features)
        
        # Create DataFrame
        self.audio_features_df = pd.DataFrame(features_data)
        
        # Remove tracks without audio features
        self.audio_features_df = self.audio_features_df.dropna(subset=[
            'danceability', 'energy', 'valence', 'tempo'
        ])
        
        logger.info(f"Analyzed {len(self.audio_features_df)} tracks with complete audio features")
        return self.audio_features_df
    
    def classify_mood(self, valence: float, energy: float) -> str:
        """Classify mood based on valence and energy"""
        if valence > 0.6 and energy > 0.6:
            return "Happy & Energetic"
        elif valence > 0.6 and energy < 0.4:
            return "Happy & Chill"
        elif valence < 0.4 and energy > 0.6:
            return "Intense & Energetic"
        elif valence < 0.4 and energy < 0.4:
            return "Sad & Mellow"
        else:
            return "Neutral"
    
    def find_optimal_clusters(self, features: np.ndarray, max_k: int = 15) -> int:
        """Find optimal number of clusters using elbow method and silhouette score"""
        logger.info("Finding optimal number of clusters...")
        
        silhouette_scores = []
        inertias = []
        k_range = range(3, min(max_k + 1, len(features) // 10))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            silhouette_avg = silhouette_score(features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)
        
        # Find optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def perform_clustering(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """Perform K-means clustering on audio features"""
        if self.audio_features_df is None:
            raise Exception("Audio features not analyzed. Call analyze_audio_features() first.")
        
        logger.info("Performing clustering analysis...")
        
        # Select features for clustering
        feature_columns = [
            'danceability', 'energy', 'valence', 'tempo',
            'acousticness', 'instrumentalness', 'speechiness', 'loudness'
        ]
        
        features = self.audio_features_df[feature_columns].values
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Find optimal clusters if not specified
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(features_scaled)
        
        self.optimal_clusters = n_clusters
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = self.kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to dataframe
        self.audio_features_df['cluster'] = self.clusters
        
        # Add mood classification
        self.audio_features_df['mood'] = self.audio_features_df.apply(
            lambda row: self.classify_mood(row['valence'], row['energy']), axis=1
        )
        
        logger.info(f"Clustering completed with {n_clusters} clusters")
        return self.clusters
    
    def generate_playlist_names(self, cluster_characteristics: Dict) -> str:
        """Generate creative playlist names based on cluster characteristics"""
        avg_energy = cluster_characteristics['avg_energy']
        avg_valence = cluster_characteristics['avg_valence']
        avg_danceability = cluster_characteristics['avg_danceability']
        avg_tempo = cluster_characteristics['avg_tempo']
        dominant_mood = cluster_characteristics['dominant_mood']
        
        # Energy-based names
        if avg_energy > 0.8:
            energy_names = ["🔥 High Energy Bangers", "⚡ Power Surge", "🚀 Adrenaline Rush"]
        elif avg_energy > 0.6:
            energy_names = ["🎸 Rock Solid", "💪 Pump It Up", "🔊 Turn It Loud"]
        elif avg_energy > 0.4:
            energy_names = ["🎵 Steady Groove", "🎶 Mid-Tempo Mix", "🎤 Balanced Beats"]
        else:
            energy_names = ["😌 Chill Vibes", "🌙 Mellow Moods", "☁️ Soft Sounds"]
        
        # Mood-based names
        mood_names = {
            "Happy & Energetic": ["☀️ Feel Good Hits", "🎉 Party Starters", "😄 Good Vibes Only"],
            "Happy & Chill": ["🌸 Happy Place", "🦋 Light & Breezy", "🌻 Sunshine Melodies"],
            "Intense & Energetic": ["⚡ Intense Moments", "🔥 Raw Energy", "💥 Power Anthems"],
            "Sad & Mellow": ["🌧️ Rainy Day Blues", "💭 Contemplative", "🌙 Midnight Thoughts"],
            "Neutral": ["🎵 Everyday Mix", "📻 Background Beats", "🎧 Study Session"]
        }
        
        # Dance-based names
        if avg_danceability > 0.8:
            dance_names = ["💃 Dance Floor", "🕺 Move Your Body", "🎪 Party Mix"]
        
        # Tempo-based names
        if avg_tempo > 140:
            tempo_names = ["🏃 Fast Lane", "⚡ Speed Demon", "🚀 High Velocity"]
        elif avg_tempo < 80:
            tempo_names = ["🐌 Slow Motion", "🌙 Late Night", "☁️ Dreamy"]
        
        # Select name based on dominant characteristic
        if dominant_mood in mood_names:
            return np.random.choice(mood_names[dominant_mood])
        else:
            return np.random.choice(energy_names)
    
    def create_playlist_suggestions(self) -> List[PlaylistSuggestion]:
        """Create AI-generated playlist suggestions based on clusters"""
        if self.clusters is None:
            raise Exception("Clustering not performed. Call perform_clustering() first.")
        
        logger.info("Creating playlist suggestions...")
        
        suggestions = []
        
        for cluster_id in range(self.optimal_clusters):
            # Get tracks in this cluster
            cluster_tracks = self.audio_features_df[self.audio_features_df['cluster'] == cluster_id]
            
            if len(cluster_tracks) < 5:  # Skip small clusters
                continue
            
            # Calculate cluster characteristics
            characteristics = {
                'avg_energy': cluster_tracks['energy'].mean(),
                'avg_valence': cluster_tracks['valence'].mean(),
                'avg_danceability': cluster_tracks['danceability'].mean(),
                'avg_tempo': cluster_tracks['tempo'].mean(),
                'avg_acousticness': cluster_tracks['acousticness'].mean(),
                'avg_loudness': cluster_tracks['loudness'].mean(),
                'dominant_mood': cluster_tracks['mood'].mode().iloc[0] if not cluster_tracks['mood'].mode().empty else 'Neutral',
                'track_count': len(cluster_tracks)
            }
            
            # Generate playlist name
            playlist_name = self.generate_playlist_names(characteristics)
            
            # Create description
            description = f"AI-curated playlist with {len(cluster_tracks)} tracks. "
            description += f"Energy: {characteristics['avg_energy']:.2f}, "
            description += f"Mood: {characteristics['dominant_mood']}, "
            description += f"Avg Tempo: {characteristics['avg_tempo']:.0f} BPM"
            
            # Determine energy level
            if characteristics['avg_energy'] > 0.7:
                energy_level = "High"
            elif characteristics['avg_energy'] > 0.4:
                energy_level = "Medium"
            else:
                energy_level = "Low"
            
            # Create Track objects
            track_objects = []
            for _, track_row in cluster_tracks.iterrows():
                track_obj = Track(
                    id=track_row['track_id'],
                    name=track_row['track_name'],
                    artist=track_row['artist'],
                    album='',  # Not stored in features df
                    popularity=track_row['popularity'],
                    duration_ms=track_row['duration_ms'],
                    audio_features=track_row.to_dict(),
                    mood=track_row['mood'],
                    cluster=cluster_id
                )
                track_objects.append(track_obj)
            
            # Create playlist suggestion
            suggestion = PlaylistSuggestion(
                name=playlist_name,
                description=description,
                tracks=track_objects,
                characteristics=characteristics,
                mood=characteristics['dominant_mood'],
                energy_level=energy_level
            )
            
            suggestions.append(suggestion)
        
        self.playlist_suggestions = suggestions
        logger.info(f"Created {len(suggestions)} playlist suggestions")
        return suggestions
    
    def create_spotify_playlists(self, suggestions: List[PlaylistSuggestion] = None) -> List[str]:
        """Create actual Spotify playlists from suggestions"""
        if not self.sp:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        if suggestions is None:
            suggestions = self.playlist_suggestions
        
        if not suggestions:
            raise Exception("No playlist suggestions. Call create_playlist_suggestions() first.")
        
        logger.info(f"Creating {len(suggestions)} Spotify playlists...")
        
        created_playlists = []
        
        for suggestion in suggestions:
            try:
                # Create playlist
                playlist = self.sp.user_playlist_create(
                    user=self.user_id,
                    name=suggestion.name,
                    description=suggestion.description,
                    public=False
                )
                
                # Add tracks to playlist
                track_ids = [track.id for track in suggestion.tracks]
                
                # Add tracks in batches of 100 (Spotify API limit)
                for i in range(0, len(track_ids), 100):
                    batch = track_ids[i:i+100]
                    self.sp.playlist_add_items(playlist['id'], batch)
                
                created_playlists.append(playlist['external_urls']['spotify'])
                logger.info(f"Created playlist: {suggestion.name} ({len(suggestion.tracks)} tracks)")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error creating playlist {suggestion.name}: {e}")
        
        logger.info(f"Successfully created {len(created_playlists)} playlists")
        return created_playlists
    
    def get_music_insights(self) -> Dict:
        """Generate insights about user's music taste"""
        if self.audio_features_df is None:
            raise Exception("Audio features not analyzed. Call analyze_audio_features() first.")
        
        df = self.audio_features_df
        
        insights = {
            'library_size': len(df),
            'avg_energy': df['energy'].mean(),
            'avg_valence': df['valence'].mean(),
            'avg_danceability': df['danceability'].mean(),
            'avg_tempo': df['tempo'].mean(),
            'energy_distribution': {
                'high': len(df[df['energy'] > 0.7]) / len(df) * 100,
                'medium': len(df[(df['energy'] >= 0.4) & (df['energy'] <= 0.7)]) / len(df) * 100,
                'low': len(df[df['energy'] < 0.4]) / len(df) * 100
            },
            'mood_distribution': df['mood'].value_counts().to_dict() if 'mood' in df.columns else {},
            'most_popular_tracks': df.nlargest(10, 'popularity')[['track_name', 'artist', 'popularity']].to_dict('records'),
            'tempo_range': {'min': df['tempo'].min(), 'max': df['tempo'].max()},
            'processing_time': datetime.now().isoformat()
        }
        
        # Determine overall energy level
        if insights['avg_energy'] > 0.7:
            insights['energy_level'] = 'High'
        elif insights['avg_energy'] > 0.4:
            insights['energy_level'] = 'Medium'
        else:
            insights['energy_level'] = 'Low'
        
        return insights
    
    def visualize_clusters(self, save_path: str = None) -> go.Figure:
        """Create interactive visualization of music clusters"""
        if self.audio_features_df is None or self.clusters is None:
            raise Exception("Clustering not performed. Call perform_clustering() first.")
        
        # Perform PCA for 2D visualization
        feature_columns = [
            'danceability', 'energy', 'valence', 'tempo',
            'acousticness', 'instrumentalness', 'speechiness', 'loudness'
        ]
        
        features = self.audio_features_df[feature_columns].values
        features_scaled = self.scaler.transform(features)
        
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        
        # Create scatter plot
        fig = px.scatter(
            x=features_2d[:, 0],
            y=features_2d[:, 1],
            color=self.clusters.astype(str),
            hover_data={
                'Track': self.audio_features_df['track_name'],
                'Artist': self.audio_features_df['artist'],
                'Energy': self.audio_features_df['energy'].round(2),
                'Valence': self.audio_features_df['valence'].round(2),
                'Mood': self.audio_features_df['mood']
            },
            title="AI Music Clustering Visualization",
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                   'color': 'Cluster'}
        )
        
        fig.update_layout(
            width=800,
            height=600,
            title_font_size=16
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Visualization saved to {save_path}")
        
        return fig
    
    def export_analysis(self, filepath: str) -> None:
        """Export analysis results to JSON file"""
        if not self.playlist_suggestions:
            raise Exception("No analysis to export. Run full analysis first.")
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'library_size': len(self.tracks),
            'analyzed_tracks': len(self.audio_features_df) if self.audio_features_df is not None else 0,
            'num_clusters': self.optimal_clusters,
            'insights': self.get_music_insights(),
            'playlist_suggestions': [
                {
                    'name': suggestion.name,
                    'description': suggestion.description,
                    'track_count': len(suggestion.tracks),
                    'characteristics': suggestion.characteristics,
                    'mood': suggestion.mood,
                    'energy_level': suggestion.energy_level
                }
                for suggestion in self.playlist_suggestions
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Analysis exported to {filepath}")

# Example usage
if __name__ == "__main__":
    # Load credentials from environment variables
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables")
        exit(1)
    
    # Initialize organizer
    organizer = SpotifyOrganizer(client_id, client_secret)
    
    # Authenticate
    if not organizer.authenticate():
        print("Authentication failed")
        exit(1)
    
    # Analyze library
    print("Fetching your music library...")
    tracks = organizer.fetch_user_library(limit=1000)
    
    print("Analyzing audio features...")
    organizer.analyze_audio_features()
    
    print("Performing AI clustering...")
    organizer.perform_clustering()
    
    print("Creating playlist suggestions...")
    suggestions = organizer.create_playlist_suggestions()
    
    # Display results
    print(f"\n🎵 Generated {len(suggestions)} smart playlists:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion.name} ({len(suggestion.tracks)} tracks)")
        print(f"   Mood: {suggestion.mood}, Energy: {suggestion.energy_level}")
        print(f"   {suggestion.description}\n")
    
    # Get insights
    insights = organizer.get_music_insights()
    print(f"📊 Your Music Insights:")
    print(f"Library Size: {insights['library_size']} tracks")
    print(f"Overall Energy Level: {insights['energy_level']}")
    print(f"Average Tempo: {insights['avg_tempo']:.0f} BPM")
    
    # Export analysis
    organizer.export_analysis('music_analysis.json')
    print("\n💾 Analysis exported to music_analysis.json")
