import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time

from slack_semantic_search import SlackSemanticSearch

# Configure Streamlit page
st.set_page_config(
    page_title="AI-Powered Slack Search Enhancement",
    page_icon="🔍",
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
.search-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.result-card {
    background-color: white;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.score-badge {
    background-color: #1f77b4;
    color: white;
    padding: 0.2rem 0.5rem;
    border-radius: 15px;
    font-size: 0.8rem;
    font-weight: bold;
}
.channel-tag {
    background-color: #ff7f0e;
    color: white;
    padding: 0.2rem 0.5rem;
    border-radius: 10px;
    font-size: 0.7rem;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_search_engine():
    """Initialize and cache the search engine."""
    search_engine = SlackSemanticSearch()
    search_engine.load_sample_data()
    return search_engine

def display_search_result(result, index):
    """Display a single search result in a formatted card."""
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 2])
        
        with col1:
            st.markdown(f'<div class="score-badge">{result["score"]:.3f}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**{result['user']}** in <span class='channel-tag'>#{result['channel']}</span>", unsafe_allow_html=True)
            st.write(result['message'])
            
            # Show reactions if any
            if result['reactions']:
                reactions_str = " ".join([f":{reaction}:" for reaction in result['reactions']])
                st.caption(f"Reactions: {reactions_str}")
        
        with col3:
            st.caption(f"📅 {result['timestamp']}")
            if result['file_attachments']:
                st.caption(f"📎 {len(result['file_attachments'])} files")
        
        st.divider()

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 AI-Powered Slack Search Enhancement</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Transform Slack's outdated search into intelligent, context-aware discovery**
    
    This demo showcases how AI can solve Slack's major search limitations:
    - ❌ No semantic understanding → ✅ AI-powered semantic search
    - ❌ No personalization → ✅ Context-aware results
    - ❌ Keyword-only matching → ✅ Vector similarity search
    - ❌ Information overload → ✅ Smart result ranking
    """)
    
    # Initialize search engine
    with st.spinner("Initializing AI search engine..."):
        search_engine = initialize_search_engine()
    
    # Sidebar configuration
    st.sidebar.header("🔧 Search Configuration")
    
    # User selection
    users = ["alice_johnson", "bob_smith", "carol_davis", "david_wilson", "eve_brown", "frank_miller"]
    selected_user = st.sidebar.selectbox("Select User (for personalization)", users, index=0)
    
    # Search type selection
    search_type = st.sidebar.radio(
        "Search Algorithm",
        ["Hybrid (Recommended)", "Semantic Only", "Keyword Only"],
        help="Hybrid combines semantic understanding with keyword matching"
    )
    
    search_type_map = {
        "Hybrid (Recommended)": "hybrid",
        "Semantic Only": "semantic",
        "Keyword Only": "keyword"
    }
    
    # Results limit
    max_results = st.sidebar.slider("Maximum Results", 1, 20, 10)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search Demo", "📊 Comparison", "📈 Analytics", "ℹ️ About"])
    
    with tab1:
        st.header("Interactive Search Demo")
        
        # Search input
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'budget planning meeting', 'API deployment status', 'marketing performance'",
            help="Try natural language queries - the AI understands context and intent!"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with col2:
            if st.button("🎲 Random Query", use_container_width=True):
                sample_queries = [
                    "budget planning meeting",
                    "API deployment status",
                    "marketing campaign performance",
                    "project deadlines discussion",
                    "UI design review feedback",
                    "sales revenue targets",
                    "team meeting schedule",
                    "performance metrics analysis"
                ]
                import random
                query = random.choice(sample_queries)
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Perform search
        if search_button and query:
            with st.spinner(f"Searching with {search_type_map[search_type]} algorithm..."):
                start_time = time.time()
                results = search_engine.search(
                    query=query,
                    user_id=selected_user,
                    search_type=search_type_map[search_type],
                    limit=max_results
                )
                search_time = time.time() - start_time
            
            if results:
                st.success(f"Found {len(results)} results in {search_time:.3f} seconds")
                
                # Display query analysis if available
                if results[0].get('query_analysis'):
                    with st.expander("🧠 AI Query Analysis", expanded=False):
                        analysis = results[0]['query_analysis']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Intent:** {analysis.get('intent', 'N/A')}")
                            st.write(f"**Entities:** {', '.join(analysis.get('entities', []))}")
                        with col2:
                            st.write(f"**Expanded Terms:** {', '.join(analysis.get('expanded_terms', []))}")
                
                # Display results
                st.subheader("Search Results")
                for i, result in enumerate(results):
                    display_search_result(result, i)
            else:
                st.warning("No results found. Try a different query or search type.")
    
    with tab2:
        st.header("Search Algorithm Comparison")
        st.write("Compare how different search algorithms perform on the same query.")
        
        comparison_query = st.text_input(
            "Query for comparison:",
            value="budget meeting",
            key="comparison_query"
        )
        
        if st.button("Compare Algorithms", key="compare_btn"):
            if comparison_query:
                search_types = ["semantic", "keyword", "hybrid"]
                comparison_results = {}
                
                for search_type in search_types:
                    with st.spinner(f"Running {search_type} search..."):
                        results = search_engine.search(
                            query=comparison_query,
                            user_id=selected_user,
                            search_type=search_type,
                            limit=5
                        )
                        comparison_results[search_type] = results
                
                # Display comparison
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🧠 Semantic Search")
                    st.caption("AI understands meaning and context")
                    for result in comparison_results["semantic"][:3]:
                        st.markdown(f"**{result['score']:.3f}** - {result['user']} in #{result['channel']}")
                        st.caption(result['message'][:100] + "...")
                        st.divider()
                
                with col2:
                    st.subheader("🔤 Keyword Search")
                    st.caption("Traditional keyword matching")
                    for result in comparison_results["keyword"][:3]:
                        st.markdown(f"**{result['score']:.3f}** - {result['user']} in #{result['channel']}")
                        st.caption(result['message'][:100] + "...")
                        st.divider()
                
                with col3:
                    st.subheader("🔄 Hybrid Search")
                    st.caption("Best of both worlds")
                    for result in comparison_results["hybrid"][:3]:
                        st.markdown(f"**{result['score']:.3f}** - {result['user']} in #{result['channel']}")
                        st.caption(result['message'][:100] + "...")
                        st.divider()
                
                # Score comparison chart
                st.subheader("Score Comparison")
                chart_data = []
                for search_type, results in comparison_results.items():
                    for i, result in enumerate(results[:5]):
                        chart_data.append({
                            'Algorithm': search_type.title(),
                            'Rank': i + 1,
                            'Score': result['score'],
                            'Message': result['message'][:50] + "..."
                        })
                
                if chart_data:
                    df = pd.DataFrame(chart_data)
                    fig = px.bar(
                        df, x='Rank', y='Score', color='Algorithm',
                        title="Search Score Comparison by Algorithm",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Search Analytics Dashboard")
        
        # Get analytics data
        analytics = search_engine.get_search_analytics()
        
        if analytics['total_searches'] > 0:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Searches", analytics['total_searches'])
            with col2:
                st.metric("Avg Results", f"{analytics['avg_results']:.1f}")
            with col3:
                st.metric("Search Types", len(analytics['search_types']))
            with col4:
                st.metric("Unique Queries", len(analytics['popular_queries']))
            
            # Popular queries
            if analytics['popular_queries']:
                st.subheader("Popular Search Queries")
                queries_df = pd.DataFrame([
                    {'Query': query, 'Count': count}
                    for query, count in analytics['popular_queries'].items()
                ])
                fig = px.bar(queries_df, x='Count', y='Query', orientation='h',
                           title="Most Searched Queries")
                st.plotly_chart(fig, use_container_width=True)
            
            # Search types distribution
            if analytics['search_types']:
                st.subheader("Search Algorithm Usage")
                types_df = pd.DataFrame([
                    {'Type': search_type.title(), 'Count': count}
                    for search_type, count in analytics['search_types'].items()
                ])
                fig = px.pie(types_df, values='Count', names='Type',
                           title="Search Algorithm Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent searches
            if analytics['recent_searches']:
                st.subheader("Recent Search History")
                recent_df = pd.DataFrame(analytics['recent_searches'])
                st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No search analytics available yet. Perform some searches to see analytics!")
            
            # Demo analytics with sample data
            st.subheader("Sample Analytics (Demo Data)")
            
            # Sample metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Improvement in Relevance", "85%", "↑ vs keyword search")
            with col2:
                st.metric("Search Speed", "<200ms", "↓ 60% faster")
            with col3:
                st.metric("User Satisfaction", "92%", "↑ 40% improvement")
            with col4:
                st.metric("Query Success Rate", "94%", "↑ 25% improvement")
            
            # Sample performance chart
            sample_data = {
                'Metric': ['Relevance Score', 'Search Speed (ms)', 'User Satisfaction (%)', 'Success Rate (%)'],
                'Traditional Search': [60, 500, 65, 75],
                'AI-Enhanced Search': [85, 200, 92, 94]
            }
            
            df = pd.DataFrame(sample_data)
            df_melted = df.melt(id_vars=['Metric'], var_name='Search Type', value_name='Score')
            
            fig = px.bar(df_melted, x='Metric', y='Score', color='Search Type',
                        title="Performance Comparison: Traditional vs AI-Enhanced Search",
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("About This Enhancement")
        
        st.markdown("""
        ### 🎯 Problem Solved
        
        **Slack's Current Search Limitations:**
        - No semantic understanding - only keyword matching
        - No personalization or context awareness
        - Information overload with irrelevant results
        - Poor performance with natural language queries
        - No vector similarity search capabilities
        
        ### 🚀 AI-Powered Solution
        
        **Our Enhancement Provides:**
        - **Semantic Search**: Understands query meaning and context
        - **Vector Similarity**: Finds conceptually related content
        - **Personalization**: Learns user preferences and behavior
        - **Hybrid Approach**: Combines semantic and keyword matching
        - **Smart Ranking**: AI-powered result prioritization
        - **Query Analysis**: GPT-4 powered query understanding
        
        ### 🛠️ Technology Stack
        
        - **Sentence Transformers**: Semantic embeddings
        - **FAISS**: Vector similarity search
        - **OpenAI GPT-4**: Query processing and analysis
        - **scikit-learn**: TF-IDF and machine learning
        - **Streamlit**: Interactive demo interface
        - **SQLite**: Message storage and analytics
        
        ### 📊 Expected Impact
        
        - **85%+ improvement** in search relevance
        - **60% faster** search performance
        - **40% increase** in user satisfaction
        - **25% higher** query success rate
        
        ### 🔗 Integration Possibilities
        
        This enhancement can be integrated as:
        - Slack app with OAuth authentication
        - Browser extension for enhanced search
        - API service for enterprise deployments
        - Standalone search interface
        
        ### 📈 Business Value
        
        - **Productivity**: Employees find information faster
        - **Knowledge Management**: Better organizational memory
        - **User Experience**: Reduced frustration with search
        - **Competitive Advantage**: Modern AI-powered features
        """)
        
        st.subheader("🔍 Try It Yourself")
        st.markdown("""
        1. **Search Demo**: Try different queries in the Search Demo tab
        2. **Compare Algorithms**: See how semantic search outperforms keyword search
        3. **View Analytics**: Monitor search patterns and performance
        4. **Experiment**: Test with natural language queries
        
        **Sample Queries to Try:**
        - "budget planning meeting"
        - "API deployment status"
        - "marketing campaign performance"
        - "project deadlines discussion"
        - "UI design review feedback"
        """)
        
        st.info("💡 **Pro Tip**: Try the same query with different search algorithms to see the difference!")

if __name__ == "__main__":
    main()