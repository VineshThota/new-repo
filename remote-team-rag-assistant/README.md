# 📚 Remote Team RAG Assistant

**AI-Powered Document Processing for Distributed Teams**

A Retrieval-Augmented Generation (RAG) application built with Python and Streamlit that helps remote teams efficiently manage, search, and retrieve information from their shared documents.

## 🌟 Features

### Document Processing
- **Multi-format Support**: Upload PDF, DOCX, and TXT files
- **Intelligent Text Extraction**: Automatic text extraction from various document formats
- **Smart Chunking**: Documents are split into overlapping chunks for optimal retrieval
- **Metadata Tracking**: Track document type, uploader, and upload date

### AI-Powered Search
- **Semantic Search**: Find relevant information using natural language queries
- **Vector Embeddings**: Uses sentence-transformers for high-quality embeddings
- **Contextual Retrieval**: RAG implementation for accurate and relevant results
- **Advanced Filtering**: Filter by team member and document type

### Team Collaboration
- **Multi-user Support**: Track contributions from different team members
- **Document Categorization**: Organize by meeting notes, policies, training materials, etc.
- **Knowledge Base Statistics**: Monitor the growth of your team's knowledge base
- **Real-time Updates**: Instant search results as new documents are added

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/VineshThota/new-repo.git
   cd new-repo/remote-team-rag-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

## 📖 Usage Guide

### Uploading Documents
1. Use the sidebar to upload documents (PDF, DOCX, or TXT)
2. Enter the team member name who's uploading
3. Select the document type (Meeting Notes, Project Documentation, etc.)
4. Click "Upload & Process" to add to the knowledge base

### Searching Documents
1. Enter your question or search query in natural language
2. Optionally filter by team member or document type
3. Click "Search" to find relevant information
4. Review results with relevance scores and source metadata

### Example Queries
- "What was discussed in the last team meeting?"
- "Show me the onboarding process for new employees"
- "Find information about our security policies"
- "What are the project deadlines mentioned in recent documents?"

## 🛠️ Technical Architecture

### Core Components

**Vector Database**: ChromaDB for persistent storage of document embeddings
- Cosine similarity for semantic search
- Persistent storage across sessions
- Efficient querying and filtering

**Embeddings Model**: Sentence-Transformers (all-MiniLM-L6-v2)
- Lightweight and fast
- High-quality semantic embeddings
- Optimized for search and retrieval tasks

**Document Processing Pipeline**:
1. File upload and validation
2. Text extraction (PDF/DOCX/TXT)
3. Text chunking with overlap
4. Embedding generation
5. Vector storage with metadata

**RAG Implementation**:
1. Query embedding generation
2. Semantic similarity search
3. Context retrieval with metadata
4. Ranked result presentation

### Technology Stack
- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python with ChromaDB
- **ML/AI**: Sentence-Transformers, PyTorch
- **Document Processing**: PyPDF2, python-docx
- **Vector Database**: ChromaDB

## 📁 Project Structure

```
remote-team-rag-assistant/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── chroma_db/            # Vector database storage (created at runtime)
```

## 🔧 Configuration

### Customizable Parameters
- **Chunk Size**: Default 500 words (adjustable in code)
- **Chunk Overlap**: Default 50 words for context preservation
- **Search Results**: Default 5 results per query
- **Embedding Model**: Configurable sentence-transformer model

### Environment Variables
No environment variables required for basic setup. All configuration is handled in the code.

## 🎯 Use Cases

### Perfect for Remote Teams
- **Meeting Notes Management**: Quickly find information from past meetings
- **Policy and Procedure Lookup**: Instant access to company policies
- **Project Documentation**: Search through technical documentation
- **Training Material Access**: Find relevant training resources
- **Knowledge Sharing**: Centralized repository for team knowledge

### Industry Applications
- **Software Development Teams**: API documentation, code reviews, meeting notes
- **Consulting Firms**: Client documents, project reports, best practices
- **Educational Institutions**: Course materials, research papers, administrative docs
- **Healthcare Organizations**: Protocols, training materials, compliance documents

## 🔒 Privacy and Security

- **Local Storage**: All data stored locally in ChromaDB
- **No External APIs**: Embeddings generated locally
- **File Security**: Temporary files cleaned up after processing
- **Data Isolation**: Each deployment maintains separate knowledge base

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Docker Deployment
```dockerfile
# Dockerfile example
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment
- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: With buildpack for Python
- **AWS/GCP/Azure**: Container-based deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support, please open an issue on GitHub or contact the development team.

## 🔮 Future Enhancements

- **Multi-language Support**: Support for documents in different languages
- **Advanced Analytics**: Usage statistics and search analytics
- **Integration APIs**: REST API for external integrations
- **Advanced Filters**: Date range, file size, and content type filters
- **Collaborative Features**: Comments, annotations, and document ratings
- **Export Functionality**: Export search results and summaries

---

**Built with ❤️ for Remote Teams**

This application addresses the growing need for efficient knowledge management in distributed work environments, combining the power of AI with user-friendly interfaces to make team collaboration more effective.