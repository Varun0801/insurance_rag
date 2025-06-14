# 🤖 Advanced RAG Customer Support Chatbot

*Powered by Pinecone Vector Database & Large Language Models*

## 📋 Project Overview

This project implements an advanced **Retrieval-Augmented Generation (RAG)** chatbot specifically designed for customer support operations. The system combines the power of **Pinecone's serverless vector database** with state-of-the-art language models to provide accurate, contextual responses based on your organization's support documentation.

### 🎯 Key Features

- **🌐 Multi-Source Document Processing**: Website scraping, PDF, and DOCX file support
- **🔍 Pinecone Vector Database**: Serverless, high-performance semantic search
- **🧠 Advanced Embeddings**: Multilingual-e5-large model for superior semantic understanding
- **💬 Intelligent Chat Interface**: Context-aware responses with source attribution
- **📊 Real-time Analytics**: Document indexing statistics and system monitoring
- **🔧 Flexible Configuration**: Multiple LLM providers through OpenRouter
- **⚡ Production Ready**: Scalable architecture with proper error handling

## 🏗️ Technical Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Processing     │    │   Vector Store  │
│                 │    │                  │    │                 │
│ • Website       │───▶│ • Text Chunking  │───▶│ • Pinecone DB   │
│ • PDF Files     │    │ • Embeddings     │    │ • Semantic      │
│ • DOCX Files    │    │ • Metadata       │    │   Search        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐           │
│   User Query    │───▶│   RAG Pipeline   │◀──────────┘
│                 │    │                  │
│ • Natural       │    │ • Query          │    ┌─────────────────┐
│   Language      │    │   Embedding      │───▶│   LLM Response  │
│ • Chat          │    │ • Context        │    │                 │
│   Interface     │    │   Retrieval      │    │ • Claude/GPT    │
└─────────────────┘    └──────────────────┘    │ • Contextual    │
                                               │ • Source Cited  │
                                               └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Pinecone API Key ([Get here](https://pinecone.io/))
- OpenRouter API Key ([Get here](https://openrouter.ai/))

### 1. Clone & Setup

```bash
# Clone the repository
git clone <repository-url>
cd insurance_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
# API Keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Optional: Pinecone Configuration
PINECONE_INDEX_NAME=support-docs
PINECONE_ENVIRONMENT=us-east-1
```

### 3. Launch Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

## 📚 Dependencies

```
streamlit>=1.28.0
requests>=2.31.0
python-dotenv>=1.0.0
pinecone>=7.0.1
PyPDF2>=3.0.1
python-docx>=0.8.11
beautifulsoup4>=4.12.2
pandas>=2.0.3
numpy>=1.24.3
```

## 🔧 Configuration Guide

### API Keys Setup

#### Option 1: Environment Variables (Recommended)
```bash
export OPENROUTER_API_KEY="your_key_here"
export PINECONE_API_KEY="your_key_here"
```

#### Option 2: .env File
```env
OPENROUTER_API_KEY=your_openrouter_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

#### Option 3: Streamlit Interface
Enter keys directly in the sidebar when running the application.

### Model Selection

Available models through OpenRouter:
- **meta-llama/llama-3.3-8b-instruct:free**
- **qwen/qwen3-8b:free**
- **google/gemma-3-4b-it:free**
- **mistralai/mistral-small-3.1-24b-instruct:free**

## 📖 Usage Guide

### 1. Document Loading

#### Website Scraping
- Enter your support website URL (e.g., `https://company.com/support`)
- Adjust max pages to scrape (5-100 pages)
- System automatically discovers linked pages

#### File Upload
- Upload PDF or DOCX files containing support documentation
- Multiple files supported
- Automatic text extraction and processing

### 2. Chat Interface

Once documents are loaded:
1. Type your question in the chat input
2. System searches Pinecone vector database
3. Retrieves relevant document chunks
4. Generates contextual response with sources
5. View source documents in expandable sections

### 3. Index Management

- **Use Existing Index**: Leverage previously indexed documents
- **Re-index Documents**: Replace existing documents with new ones
- **Clear Index**: Remove all documents from vector database
- **View Statistics**: Monitor document count and index status

## 💼 Business Value

### For Customer Support Teams
- **⏱️ Faster Response Times**: Instant access to relevant answers
- **📈 Improved Accuracy**: AI-powered context understanding
- **🔄 24/7 Availability**: Always-on customer support capability
- **📊 Consistent Responses**: Standardized information delivery

### For Organizations
- **💰 Cost Reduction**: Decreased support ticket volume
- **📈 Scalability**: Can handle increased query volume without proportional staff increase
- **🎯 Better Customer Experience**: Accurate, immediate responses
- **📚 Knowledge Management**: Centralized, searchable documentation

## 🔧 Advanced Configuration

### Embedding Model Settings
```python
embedding_model = "multilingual-e5-large"  # 1024 dimensions
chunk_size = 500  # Words per chunk
chunk_overlap = 50  # Overlap between chunks
```

### Pinecone Configuration
```python
index_name = "support-docs"
metric = "cosine"
dimension = 1024
cloud = "aws"
region = "us-east-1"
```

### Search Parameters
```python
similarity_threshold = 0.3  # Minimum similarity score
max_results = 5  # Maximum retrieved documents
```

## 📊 Monitoring & Analytics

### System Metrics
- **Vector Count**: Total documents in Pinecone index
- **Index Fullness**: Storage utilization percentage
- **Query Performance**: Average response times
- **Source Attribution**: Document usage statistics

### Logging
- Comprehensive logging for debugging and monitoring
- Error tracking and performance metrics
- User interaction analytics

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🛠️ Troubleshooting

### Common Issues

#### Pinecone Connection Errors
```bash
# Check API key validity
curl -H "Api-Key: YOUR_API_KEY" https://api.pinecone.io/indexes

# Verify index exists
curl -H "Api-Key: YOUR_API_KEY" https://api.pinecone.io/indexes/support-docs
```

#### Embedding Generation Failures
- **Rate Limits**: Implement exponential backoff
- **Batch Size**: Reduce from 50 to 25 documents
- **API Quotas**: Check Pinecone usage limits

#### Memory Issues
- **Large Documents**: Increase chunk overlap
- **File Processing**: Process files individually
- **Vector Storage**: Monitor index size


## 📈 Performance Optimization (Also Includes Future Suggestions)

### Document Processing
- **Optimal Chunk Size**: 500 words with 50-word overlap
- **Batch Processing**: Process multiple documents simultaneously
- **Caching**: Store processed embeddings

### Vector Search
- **Index Optimization**: Regular index maintenance
- **Query Caching**: Cache frequent queries
- **Parallel Processing**: Concurrent embedding generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For technical support or questions:
- Create an issue in the repository
- Contact the development team
- Check the troubleshooting section

## 🗺️ Roadmap

### Phase 1 (Current)
- ✅ Basic RAG functionality
- ✅ Pinecone integration
- ✅ Multi-format document support
- ✅ Streamlit interface

### Phase 2 (Planned)
- 🔄 Advanced filters and search
- 🔄 User authentication
- 🔄 Analytics dashboard
- 🔄 API endpoints

### Phase 3 (Future)
- 📅 Multi-language support
- 📅 Voice interface
- 📅 Advanced LLM models
- 📅 Enterprise features

---

## 🏢 For Stakeholders

### Executive Summary
This RAG chatbot solution transforms customer support operations by providing instant, accurate responses based on your organization's documentation. Built with enterprise-grade vector database technology (Pinecone) and state-of-the-art AI models, it delivers measurable improvements in response time, accuracy, and customer satisfaction.

### Technical Highlights
- **Serverless Architecture**: Auto-scaling, pay-per-use model
- **Semantic Search**: Advanced understanding beyond keyword matching
- **Multi-modal Input**: Handles various document types and sources
- **Real-time Processing**: Immediate response generation
- **Source Attribution**: Transparent information sourcing

### Implementation Benefits
- **Rapid Deployment**: Setup in under 30 minutes
- **Cost Effective**: Reduces support costs
- **Scalable Solution**: Grows with your organization
- **Future-Proof**: Built on cutting-edge AI technology

### ROI Expectations
- **Month 1**: Basic functionality, initial time savings
- **Month 3**: Measurable reduction in support ticket volume
- **Month 6**: Significant improvement in customer satisfaction scores
- **Month 12**: Full ROI realization through operational efficiency

---

*Built using Pinecone, Streamlit, and OpenRouter*
