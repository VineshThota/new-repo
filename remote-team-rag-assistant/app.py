import streamlit as st
import os
from pathlib import Path
import tempfile
import hashlib
from datetime import datetime
import json

# Vector database and embeddings
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("Please install required packages: pip install chromadb sentence-transformers")
    st.stop()

# Document processing
try:
    import PyPDF2
    import docx
except ImportError:
    st.error("Please install document processing packages: pip install PyPDF2 python-docx")
    st.stop()

class RemoteTeamRAGAssistant:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="team_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
    def extract_text_from_file(self, uploaded_file):
        """Extract text from uploaded documents"""
        text = ""
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Extract text from PDF
                with open(tmp_file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
            elif file_extension == '.docx':
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Extract text from DOCX
                doc = docx.Document(tmp_file_path)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
            elif file_extension == '.txt':
                text = str(uploaded_file.getvalue(), "utf-8")
            
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return None
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None
            
        return text.strip()
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
                
        return chunks
    
    def add_document(self, text, filename, team_member, document_type):
        """Add document to the vector database"""
        chunks = self.chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            # Create unique ID for each chunk
            chunk_id = hashlib.md5(f"{filename}_{i}_{chunk[:50]}".encode()).hexdigest()
            
            # Generate embedding
            embedding = self.model.encode(chunk).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "filename": filename,
                    "chunk_id": i,
                    "team_member": team_member,
                    "document_type": document_type,
                    "upload_date": datetime.now().isoformat(),
                    "chunk_length": len(chunk)
                }],
                ids=[chunk_id]
            )
    
    def search_documents(self, query, n_results=5, team_filter=None, doc_type_filter=None):
        """Search for relevant documents"""
        query_embedding = self.model.encode(query).tolist()
        
        # Build where clause for filtering
        where_clause = {}
        if team_filter and team_filter != "All":
            where_clause["team_member"] = team_filter
        if doc_type_filter and doc_type_filter != "All":
            where_clause["document_type"] = doc_type_filter
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        return results
    
    def get_document_stats(self):
        """Get statistics about stored documents"""
        try:
            count = self.collection.count()
            return {"total_chunks": count}
        except:
            return {"total_chunks": 0}

def main():
    st.set_page_config(
        page_title="Remote Team RAG Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Remote Team RAG Assistant")
    st.markdown("""
    **AI-Powered Document Processing for Distributed Teams**
    
    Upload team documents, search through knowledge base, and get instant answers using Retrieval-Augmented Generation (RAG).
    Perfect for remote teams managing shared documentation and knowledge.
    """)
    
    # Initialize RAG assistant
    if 'rag_assistant' not in st.session_state:
        with st.spinner("Initializing RAG Assistant..."):
            st.session_state.rag_assistant = RemoteTeamRAGAssistant()
    
    rag_assistant = st.session_state.rag_assistant
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Document upload
        st.subheader("Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        if uploaded_file:
            team_member = st.text_input("Team Member Name", value="Anonymous")
            document_type = st.selectbox(
                "Document Type",
                ["Meeting Notes", "Project Documentation", "Policy", "Training Material", "Other"]
            )
            
            if st.button("📤 Upload & Process"):
                with st.spinner("Processing document..."):
                    text = rag_assistant.extract_text_from_file(uploaded_file)
                    
                    if text:
                        rag_assistant.add_document(
                            text, uploaded_file.name, team_member, document_type
                        )
                        st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to process document")
        
        # Document statistics
        st.subheader("📊 Knowledge Base Stats")
        stats = rag_assistant.get_document_stats()
        st.metric("Total Document Chunks", stats["total_chunks"])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Search Knowledge Base")
        
        # Search interface
        query = st.text_input(
            "Ask a question or search for information:",
            placeholder="e.g., What was discussed in the last team meeting?"
        )
        
        # Search filters
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            team_filter = st.selectbox("Filter by Team Member", ["All"] + ["Anonymous"])
        with col_filter2:
            doc_type_filter = st.selectbox(
                "Filter by Document Type",
                ["All", "Meeting Notes", "Project Documentation", "Policy", "Training Material", "Other"]
            )
        
        if st.button("🔍 Search") and query:
            with st.spinner("Searching knowledge base..."):
                results = rag_assistant.search_documents(
                    query, n_results=5, team_filter=team_filter, doc_type_filter=doc_type_filter
                )
                
                if results['documents'][0]:
                    st.subheader("📋 Search Results")
                    
                    for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                    )):
                        with st.expander(f"Result {i+1}: {metadata['filename']} (Relevance: {1-distance:.2f})"):
                            st.write(f"**Document:** {metadata['filename']}")
                            st.write(f"**Type:** {metadata['document_type']}")
                            st.write(f"**Uploaded by:** {metadata['team_member']}")
                            st.write(f"**Date:** {metadata['upload_date'][:10]}")
                            st.write("**Content:**")
                            st.write(doc)
                else:
                    st.info("No relevant documents found. Try different keywords or upload more documents.")
    
    with col2:
        st.header("💡 Usage Tips")
        st.markdown("""
        **For Best Results:**
        - Upload various document types (meeting notes, policies, etc.)
        - Use specific keywords in searches
        - Tag documents with team member names
        - Categorize documents by type
        
        **Supported Features:**
        - PDF, DOCX, TXT file processing
        - Semantic search across all documents
        - Team member and document type filtering
        - Chunk-based retrieval for precise results
        
        **Perfect for Remote Teams:**
        - Centralized knowledge management
        - Quick information retrieval
        - Meeting notes searchability
        - Policy and procedure lookup
        """)
        
        st.header("🚀 Quick Actions")
        if st.button("📊 View All Documents"):
            st.info("Feature coming soon: Document browser")
        
        if st.button("🔄 Refresh Stats"):
            st.rerun()

if __name__ == "__main__":
    main()