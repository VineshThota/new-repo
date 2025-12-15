import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple
import base64
from io import BytesIO
from PIL import Image

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Indoor Navigation Assistant",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

class IndoorNavigationRAG:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []
        self.building_data = {}
        self.initialize_sample_data()
        
    def initialize_sample_data(self):
        """Initialize with sample building and navigation data"""
        self.documents = [
            "The main entrance is located on the ground floor facing the parking lot. Security desk is immediately to the right.",
            "Elevators are located in the central atrium. There are 4 elevators serving floors 1-10.",
            "Emergency exits are marked with red signs and located at both ends of each floor corridor.",
            "The cafeteria is on the 2nd floor, accessible via the main elevators or stairs near the east wing.",
            "Conference rooms A-D are on the 3rd floor. Room A has video conferencing capabilities.",
            "IT department is located on the 4th floor, room 401. Help desk hours are 9 AM to 5 PM.",
            "Restrooms are located near the elevator banks on each floor, marked with blue signs.",
            "The library and quiet study areas are on the 5th floor with 24/7 access for employees.",
            "Parking garage entrance is on the basement level, accessible via the south stairwell.",
            "Fire assembly point is in the main parking lot, 50 meters from the building entrance."
        ]
        
        # Create embeddings and FAISS index
        embeddings = self.model.encode(self.documents)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        
        # Sample building layout data
        self.building_data = {
            "floors": {
                "Ground": {"rooms": ["Lobby", "Security", "Reception", "Main Entrance"], "coordinates": [(0, 0), (10, 0), (20, 0), (30, 0)]},
                "Floor 2": {"rooms": ["Cafeteria", "Kitchen", "Dining Area", "Vending"], "coordinates": [(0, 10), (10, 10), (20, 10), (30, 10)]},
                "Floor 3": {"rooms": ["Conference A", "Conference B", "Conference C", "Conference D"], "coordinates": [(0, 20), (10, 20), (20, 20), (30, 20)]},
                "Floor 4": {"rooms": ["IT Dept", "Server Room", "Help Desk", "Storage"], "coordinates": [(0, 30), (10, 30), (20, 30), (30, 30)]},
                "Floor 5": {"rooms": ["Library", "Study Area 1", "Study Area 2", "Reading Room"], "coordinates": [(0, 40), (10, 40), (20, 40), (30, 40)]}
            }
        }
    
    def search_navigation_info(self, query: str, top_k: int = 3) -> List[str]:
        """Search for relevant navigation information using RAG"""
        query_embedding = self.model.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    "content": self.documents[idx],
                    "score": float(scores[0][i])
                })
        return results
    
    def generate_navigation_response(self, query: str, context: List[str]) -> str:
        """Generate contextual navigation response"""
        context_text = "\n".join([item["content"] for item in context])
        
        # Simple rule-based response generation (in production, use LLM)
        response = f"Based on the building information, here's what I found for '{query}':\n\n"
        
        for i, item in enumerate(context, 1):
            response += f"{i}. {item['content']}\n"
        
        # Add specific guidance based on query keywords
        query_lower = query.lower()
        if "elevator" in query_lower:
            response += "\n💡 Tip: Elevators are fastest during off-peak hours (10-11 AM, 2-3 PM)"
        elif "emergency" in query_lower:
            response += "\n🚨 Important: In case of emergency, use stairs, not elevators"
        elif "parking" in query_lower:
            response += "\n🚗 Note: Parking garage has height restriction of 2.1 meters"
        elif "cafeteria" in query_lower or "food" in query_lower:
            response += "\n🍽️ Tip: Cafeteria is busiest between 12-1 PM. Consider visiting at 11:30 AM or 1:30 PM"
        
        return response
    
    def calculate_route(self, start: str, destination: str) -> Dict:
        """Calculate optimal route between two points"""
        # Simplified route calculation
        route_info = {
            "start": start,
            "destination": destination,
            "estimated_time": np.random.randint(2, 15),  # Random time in minutes
            "distance": np.random.randint(50, 500),  # Random distance in meters
            "steps": [
                f"Start from {start}",
                "Head towards the main corridor",
                "Follow signs to elevator/stairs",
                f"Navigate to {destination}"
            ]
        }
        return route_info

def create_floor_plan_visualization(floor_data: Dict) -> go.Figure:
    """Create interactive floor plan visualization"""
    fig = go.Figure()
    
    rooms = floor_data["rooms"]
    coordinates = floor_data["coordinates"]
    
    # Add room markers
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]
    
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers+text',
        marker=dict(size=20, color='lightblue', line=dict(width=2, color='darkblue')),
        text=rooms,
        textposition="middle center",
        name="Rooms"
    ))
    
    # Add connecting lines to show corridors
    for i in range(len(coordinates) - 1):
        fig.add_trace(go.Scatter(
            x=[coordinates[i][0], coordinates[i+1][0]],
            y=[coordinates[i][1], coordinates[i+1][1]],
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            showlegend=False
        ))
    
    fig.update_layout(
        title="Interactive Floor Plan",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        showlegend=True,
        height=400,
        hovermode='closest'
    )
    
    return fig

def main():
    st.title("🧭 Smart Indoor Navigation Assistant")
    st.markdown("### AI-Powered Indoor Navigation with RAG Technology")
    
    # Initialize the RAG system
    if 'nav_rag' not in st.session_state:
        with st.spinner("Initializing AI Navigation System..."):
            st.session_state.nav_rag = IndoorNavigationRAG()
    
    nav_rag = st.session_state.nav_rag
    
    # Sidebar for navigation options
    st.sidebar.header("Navigation Options")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Ask Navigation Question", "Route Planning", "Floor Plan Explorer", "Emergency Info"]
    )
    
    if mode == "Ask Navigation Question":
        st.header("🤖 Ask the Navigation Assistant")
        
        # Query input
        query = st.text_input(
            "What would you like to know about the building?",
            placeholder="e.g., Where is the cafeteria? How do I get to the IT department?"
        )
        
        if query:
            with st.spinner("Searching building information..."):
                # Get relevant context using RAG
                context = nav_rag.search_navigation_info(query)
                
                # Generate response
                response = nav_rag.generate_navigation_response(query, context)
                
                # Display response
                st.markdown("### 📍 Navigation Information")
                st.markdown(response)
                
                # Show retrieved context
                with st.expander("📚 Retrieved Information Sources"):
                    for i, item in enumerate(context, 1):
                        st.markdown(f"**Source {i}** (Relevance: {item['score']:.3f})")
                        st.markdown(f"*{item['content']}*")
                        st.markdown("---")
    
    elif mode == "Route Planning":
        st.header("🗺️ Route Planning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_location = st.selectbox(
                "Starting Point",
                ["Main Entrance", "Lobby", "Cafeteria", "IT Department", "Library", "Parking Garage"]
            )
        
        with col2:
            destination = st.selectbox(
                "Destination",
                ["Conference Room A", "Cafeteria", "IT Department", "Library", "Emergency Exit", "Restroom"]
            )
        
        if st.button("Calculate Route"):
            route_info = nav_rag.calculate_route(start_location, destination)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Estimated Time", f"{route_info['estimated_time']} min")
            
            with col2:
                st.metric("Distance", f"{route_info['distance']} m")
            
            with col3:
                st.metric("Difficulty", "Easy")
            
            st.markdown("### 📋 Step-by-Step Directions")
            for i, step in enumerate(route_info['steps'], 1):
                st.markdown(f"{i}. {step}")
    
    elif mode == "Floor Plan Explorer":
        st.header("🏢 Interactive Floor Plan Explorer")
        
        # Floor selection
        selected_floor = st.selectbox(
            "Select Floor",
            list(nav_rag.building_data["floors"].keys())
        )
        
        # Display floor plan
        floor_data = nav_rag.building_data["floors"][selected_floor]
        fig = create_floor_plan_visualization(floor_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Room information
        st.markdown("### 🏠 Rooms on this Floor")
        rooms_df = pd.DataFrame({
            "Room": floor_data["rooms"],
            "Coordinates": [f"({x}, {y})" for x, y in floor_data["coordinates"]],
            "Status": ["Available" if np.random.random() > 0.3 else "Occupied" for _ in floor_data["rooms"]]
        })
        st.dataframe(rooms_df, use_container_width=True)
    
    elif mode == "Emergency Info":
        st.header("🚨 Emergency Information")
        
        # Emergency procedures
        st.markdown("### Emergency Procedures")
        
        emergency_info = {
            "Fire Emergency": {
                "icon": "🔥",
                "steps": [
                    "Remain calm and alert others",
                    "Use nearest emergency exit (NOT elevators)",
                    "Proceed to fire assembly point in parking lot",
                    "Wait for further instructions from emergency personnel"
                ]
            },
            "Medical Emergency": {
                "icon": "🏥",
                "steps": [
                    "Call emergency services (911)",
                    "Contact building security",
                    "Provide first aid if trained",
                    "Guide emergency responders to location"
                ]
            },
            "Evacuation": {
                "icon": "🚪",
                "steps": [
                    "Follow evacuation route signs",
                    "Use stairs, not elevators",
                    "Assist others if possible",
                    "Report to designated assembly area"
                ]
            }
        }
        
        for emergency_type, info in emergency_info.items():
            with st.expander(f"{info['icon']} {emergency_type}"):
                for i, step in enumerate(info['steps'], 1):
                    st.markdown(f"{i}. {step}")
        
        # Emergency contacts
        st.markdown("### 📞 Emergency Contacts")
        contacts_df = pd.DataFrame({
            "Service": ["Emergency Services", "Building Security", "Facilities Management", "IT Help Desk"],
            "Phone": ["911", "(555) 123-4567", "(555) 234-5678", "(555) 345-6789"],
            "Available": ["24/7", "24/7", "8 AM - 6 PM", "9 AM - 5 PM"]
        })
        st.dataframe(contacts_df, use_container_width=True)
    
    # Footer with app information
    st.markdown("---")
    st.markdown(
        "**Smart Indoor Navigation Assistant** | "
        "Powered by RAG Technology and AI | "
        f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

if __name__ == "__main__":
    main()