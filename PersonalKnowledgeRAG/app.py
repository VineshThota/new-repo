import streamlit as st
import openai
import chromadb
import pandas as pd
from datetime import datetime
import json
import hashlib
from typing import List, Dict, Any

# PersonalKnowledgeRAG - AI-Powered Personalized Knowledge Assistant
# Combines trending LinkedIn topic (AI-powered personalization) with RAG applications
# Solves knowledge base integration and document processing challenges

class PersonalKnowledgeRAG:
    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="knowledge_base")
        self.user_profiles = {}
        self.interaction_history = []
        
    def add_document(self, content: str, metadata: Dict[str, Any]):
        """Add document to knowledge base with metadata"""
        doc_id = hashlib.md5(content.encode()).hexdigest()
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return doc_id
    
    def create_user_profile(self, user_id: str, preferences: Dict[str, Any]):
        """Create personalized user profile"""
        self.user_profiles[user_id] = {
            'preferences': preferences,
            'interaction_count': 0,
            'topics_of_interest': [],
            'response_style': preferences.get('style', 'professional'),
            'expertise_level': preferences.get('level', 'intermediate')
        }
    
    def update_user_interaction(self, user_id: str, query: str, response: str):
        """Track user interactions for personalization"""
        if user_id in self.user_profiles:
            self.user_profiles[user_id]['interaction_count'] += 1
            
        interaction = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response
        }
        self.interaction_history.append(interaction)
    
    def personalized_retrieval(self, query: str, user_id: str, n_results: int = 3):
        """Retrieve documents with personalization based on user profile"""
        # Get user preferences
        user_profile = self.user_profiles.get(user_id, {})
        expertise_level = user_profile.get('expertise_level', 'intermediate')
        
        # Modify query based on user expertise
        if expertise_level == 'beginner':
            enhanced_query = f"basic introduction {query} simple explanation"
        elif expertise_level == 'expert':
            enhanced_query = f"advanced technical {query} detailed analysis"
        else:
            enhanced_query = query
            
        # Retrieve relevant documents
        results = self.collection.query(
            query_texts=[enhanced_query],
            n_results=n_results
        )
        
        return results
    
    def generate_personalized_response(self, query: str, user_id: str, retrieved_docs: List[str]):
        """Generate personalized response using retrieved documents"""
        user_profile = self.user_profiles.get(user_id, {})
        response_style = user_profile.get('response_style', 'professional')
        expertise_level = user_profile.get('expertise_level', 'intermediate')
        
        # Create context from retrieved documents
        context = "\n\n".join(retrieved_docs)
        
        # Personalized prompt based on user profile
        style_instructions = {
            'professional': "Provide a professional, structured response",
            'casual': "Use a friendly, conversational tone",
            'technical': "Focus on technical details and implementation",
            'creative': "Use creative examples and analogies"
        }
        
        level_instructions = {
            'beginner': "Explain concepts simply with basic examples",
            'intermediate': "Provide balanced detail with practical examples",
            'expert': "Include advanced concepts and technical depth"
        }
        
        prompt = f"""
        Based on the following context, answer the user's question.
        
        Context: {context}
        
        User Question: {query}
        
        Instructions:
        - {style_instructions.get(response_style, 'Provide a clear response')}
        - {level_instructions.get(expertise_level, 'Use appropriate detail level')}
        - Make the response personally relevant and actionable
        
        Response:
        """
        
        # Simulate AI response (in real implementation, use OpenAI API)
        response = f"Based on your {expertise_level} level and {response_style} preference, here's a personalized answer to '{query}': [Generated response would appear here using the retrieved context and user preferences]"
        
        return response
    
    def get_personalized_answer(self, query: str, user_id: str):
        """Main method to get personalized answer using RAG"""
        # Retrieve relevant documents
        retrieval_results = self.personalized_retrieval(query, user_id)
        retrieved_docs = retrieval_results['documents'][0] if retrieval_results['documents'] else []
        
        # Generate personalized response
        response = self.generate_personalized_response(query, user_id, retrieved_docs)
        
        # Update interaction history
        self.update_user_interaction(user_id, query, response)
        
        return {
            'response': response,
            'retrieved_docs': len(retrieved_docs),
            'personalization_applied': True,
            'user_profile': self.user_profiles.get(user_id, {})
        }

# Streamlit Web Interface
def main():
    st.set_page_config(
        page_title="PersonalKnowledgeRAG",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 PersonalKnowledgeRAG")
    st.subtitle("AI-Powered Personalized Knowledge Assistant")
    
    # Initialize the RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = PersonalKnowledgeRAG()
        
        # Add sample documents
        sample_docs = [
            {
                'content': "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence.",
                'metadata': {'topic': 'AI basics', 'difficulty': 'beginner'}
            },
            {
                'content': "Machine Learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning approaches, each with specific use cases and implementation strategies.",
                'metadata': {'topic': 'Machine Learning', 'difficulty': 'intermediate'}
            },
            {
                'content': "Advanced neural network architectures like Transformers have revolutionized natural language processing through attention mechanisms and parallel processing capabilities.",
                'metadata': {'topic': 'Deep Learning', 'difficulty': 'expert'}
            }
        ]
        
        for doc in sample_docs:
            st.session_state.rag_system.add_document(doc['content'], doc['metadata'])
    
    # Sidebar for user profile setup
    st.sidebar.header("User Profile")
    user_id = st.sidebar.text_input("User ID", value="user_001")
    
    expertise_level = st.sidebar.selectbox(
        "Expertise Level",
        ["beginner", "intermediate", "expert"]
    )
    
    response_style = st.sidebar.selectbox(
        "Response Style",
        ["professional", "casual", "technical", "creative"]
    )
    
    if st.sidebar.button("Create/Update Profile"):
        st.session_state.rag_system.create_user_profile(
            user_id,
            {
                'level': expertise_level,
                'style': response_style
            }
        )
        st.sidebar.success("Profile updated!")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Ask Your Question")
        query = st.text_area(
            "Enter your question:",
            placeholder="Ask anything about AI, Machine Learning, or related topics...",
            height=100
        )
        
        if st.button("Get Personalized Answer", type="primary"):
            if query and user_id:
                with st.spinner("Generating personalized response..."):
                    result = st.session_state.rag_system.get_personalized_answer(query, user_id)
                    
                    st.success("Response Generated!")
                    st.write("**Personalized Answer:**")
                    st.write(result['response'])
                    
                    st.info(f"Retrieved {result['retrieved_docs']} relevant documents")
            else:
                st.error("Please enter a question and user ID")
    
    with col2:
        st.header("System Stats")
        if user_id in st.session_state.rag_system.user_profiles:
            profile = st.session_state.rag_system.user_profiles[user_id]
            st.metric("Interactions", profile['interaction_count'])
            st.write(f"**Expertise:** {profile.get('expertise_level', 'Not set')}")
            st.write(f"**Style:** {profile.get('response_style', 'Not set')}")
        
        st.write(f"**Total Users:** {len(st.session_state.rag_system.user_profiles)}")
        st.write(f"**Total Interactions:** {len(st.session_state.rag_system.interaction_history)}")
    
    # Document management
    st.header("Knowledge Base Management")
    with st.expander("Add New Document"):
        new_content = st.text_area("Document Content")
        topic = st.text_input("Topic")
        difficulty = st.selectbox("Difficulty", ["beginner", "intermediate", "expert"])
        
        if st.button("Add Document"):
            if new_content and topic:
                doc_id = st.session_state.rag_system.add_document(
                    new_content,
                    {'topic': topic, 'difficulty': difficulty}
                )
                st.success(f"Document added with ID: {doc_id[:8]}...")
            else:
                st.error("Please provide content and topic")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **PersonalKnowledgeRAG** - Combining AI-powered personalization with RAG technology
        
        🎯 **Focus Area:** RAG Applications  
        📈 **LinkedIn Trend:** AI-powered personalization  
        💡 **Problem Solved:** Knowledge base integration and personalized information retrieval
        """
    )

if __name__ == "__main__":
    main()