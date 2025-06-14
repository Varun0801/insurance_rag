import streamlit as st
import requests
import json
import hashlib
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import re
import os
from pathlib import Path
import logging
from datetime import datetime
import uuid

# Document processing imports
import PyPDF2
import docx
from bs4 import BeautifulSoup
import pandas as pd

# Pinecone and embeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document data structure"""
    content: str
    source: str
    title: str
    chunk_id: str
    metadata: Dict[str, Any]

class DocumentProcessor:
    """Handles processing of various document types"""
    
    def __init__(self):
        self.chunk_size = 500
        self.chunk_overlap = 50
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF files"""
        documents = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text() + "\n"
                
                chunks = self._chunk_text(full_text)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        content=chunk,
                        source=file_path,
                        title=f"PDF Document - {Path(file_path).name}",
                        chunk_id=f"pdf_{Path(file_path).stem}_{i}",
                        metadata={"type": "pdf", "page_count": len(pdf_reader.pages)}
                    )
                    documents.append(doc)
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
        
        return documents
    
    def process_docx(self, file_path: str) -> List[Document]:
        """Process DOCX files"""
        documents = []
        try:
            doc = docx.Document(file_path)
            full_text = ""
            for paragraph in doc.paragraphs:
                full_text += paragraph.text + "\n"
            
            chunks = self._chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                document = Document(
                    content=chunk,
                    source=file_path,
                    title=f"DOCX Document - {Path(file_path).name}",
                    chunk_id=f"docx_{Path(file_path).stem}_{i}",
                    metadata={"type": "docx"}
                )
                documents.append(document)
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
        
        return documents
    
    def process_website(self, base_url: str, max_pages: int = 50) -> List[Document]:
        """Scrape and process website content"""
        documents = []
        visited_urls = set()
        urls_to_visit = [base_url]
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        while urls_to_visit and len(visited_urls) < max_pages:
            url = urls_to_visit.pop(0)
            if url in visited_urls:
                continue
            
            try:
                response = session.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract text content
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    text = soup.get_text()
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    if len(text) > 100:  # Only process pages with substantial content
                        chunks = self._chunk_text(text)
                        for i, chunk in enumerate(chunks):
                            doc = Document(
                                content=chunk,
                                source=url,
                                title=soup.title.string if soup.title else url,
                                chunk_id=f"web_{hashlib.md5(url.encode()).hexdigest()}_{i}",
                                metadata={"type": "website", "url": url}
                            )
                            documents.append(doc)
                    
                    # Find more URLs to scrape (only within same domain)
                    if len(visited_urls) < max_pages:
                        links = soup.find_all('a', href=True)
                        for link in links[:10]:  # Limit links per page
                            href = link['href']
                            full_url = urljoin(url, href)
                            if (urlparse(full_url).netloc == urlparse(base_url).netloc and 
                                full_url not in visited_urls and 
                                full_url not in urls_to_visit):
                                urls_to_visit.append(full_url)
                
                visited_urls.add(url)
                time.sleep(1)  # Be respectful to the server
                
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
                visited_urls.add(url)
        
        return documents
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            if len(chunk.strip()) > 50:  # Only keep substantial chunks
                chunks.append(chunk.strip())
        
        return chunks

class PineconeVectorStore:
    """Pinecone vector database manager with Pinecone embeddings"""
    
    def __init__(self, api_key: str, index_name: str = "support-docs"):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.embedding_model = "multilingual-e5-large"
        self.dimension = 1024  # multilingual-e5-large dimension
        self.index = None
        self._setup_index()
    
    def _setup_index(self):
        """Setup Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                # Create new index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
                time.sleep(10)  # Wait for index to be ready
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {e}")
            raise
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Pinecone's embedding service"""
        try:
            response = self.pc.inference.embed(
                model=self.embedding_model,
                inputs=texts,
                parameters={"input_type": "passage", "truncate": "END"}
            )
            return [embedding['values'] for embedding in response['data']]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Fallback to dummy embeddings for testing
            return [[0.0] * self.dimension for _ in texts]
    
    def add_documents(self, documents: List[Document]):
        """Add documents to Pinecone vector store"""
        if not self.index:
            logger.error("Pinecone index not initialized")
            return
        
        # Prepare texts for embedding
        texts = [doc.content for doc in documents]
        
        # Generate embeddings in batches
        batch_size = 100
        vectors_to_upsert = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            
            # Get embeddings for this batch
            embeddings = self._get_embeddings(batch_texts)
            
            # Prepare vectors for upsert
            for doc, embedding in zip(batch_docs, embeddings):
                vector_id = str(uuid.uuid4())  # Generate unique ID
                vector = {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "content": doc.content,
                        "source": doc.source,
                        "title": doc.title,
                        "chunk_id": doc.chunk_id,
                        "doc_type": doc.metadata.get("type", "unknown"),
                        "url": doc.metadata.get("url", ""),
                    }
                }
                vectors_to_upsert.append(vector)
        
        # Upsert vectors in batches
        upsert_batch_size = 100
        for i in range(0, len(vectors_to_upsert), upsert_batch_size):
            batch = vectors_to_upsert[i:i + upsert_batch_size]
            try:
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//upsert_batch_size + 1}")
            except Exception as e:
                logger.error(f"Error upserting batch: {e}")
        
        logger.info(f"Added {len(documents)} documents to Pinecone")
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        if not self.index:
            logger.error("Pinecone index not initialized")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._get_embeddings([query])[0]
            
            # Search in Pinecone
            search_result = self.index.query(
                vector=query_embedding,
                top_k=limit,
                include_metadata=True,
                filter=None  # Add filters if needed
            )
            
            results = []
            for match in search_result['matches']:
                if match['score'] >= 0.3:  # Minimum similarity threshold
                    results.append({
                        "content": match['metadata']['content'],
                        "source": match['metadata']['source'],
                        "title": match['metadata']['title'],
                        "score": match['score'],
                        "metadata": {
                            "type": match['metadata']['doc_type'],
                            "url": match['metadata'].get('url', ''),
                            "chunk_id": match['metadata']['chunk_id']
                        }
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return []
    
    def get_index_stats(self) -> Dict:
        """Get index statistics"""
        if not self.index:
            return {"error": "Index not initialized"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "index_fullness": stats.get('index_fullness', 0)
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}

class RAGChatbot:
    """Main RAG chatbot class with Pinecone integration"""
    
    def __init__(self, openrouter_api_key: str, pinecone_api_key: str, model: str = "anthropic/claude-3-haiku"):
        self.openrouter_api_key = openrouter_api_key
        self.pinecone_api_key = pinecone_api_key
        self.model = model
        self.vector_store = PineconeVectorStore(pinecone_api_key)
        self.doc_processor = DocumentProcessor()
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def load_documents(self, sources: Dict[str, Any]):
        """Load documents from various sources"""
        all_documents = []
        
        # Process website
        if "website" in sources and sources["website"]:
            st.info("🌐 Scraping website content...")
            website_docs = self.doc_processor.process_website(
                sources["website"], 
                max_pages=sources.get("max_pages", 20)
            )
            all_documents.extend(website_docs)
            st.success(f"✅ Processed {len(website_docs)} website chunks")
        
        # Process uploaded files
        if "files" in sources:
            for uploaded_file in sources["files"]:
                file_path = f"temp_{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if uploaded_file.name.endswith('.pdf'):
                    st.info(f"📄 Processing PDF: {uploaded_file.name}")
                    pdf_docs = self.doc_processor.process_pdf(file_path)
                    all_documents.extend(pdf_docs)
                    st.success(f"✅ Processed {len(pdf_docs)} PDF chunks")
                
                elif uploaded_file.name.endswith('.docx'):
                    st.info(f"📝 Processing DOCX: {uploaded_file.name}")
                    docx_docs = self.doc_processor.process_docx(file_path)
                    all_documents.extend(docx_docs)
                    st.success(f"✅ Processed {len(docx_docs)} DOCX chunks")
                
                # Clean up temp file
                os.remove(file_path)
        
        # Add to Pinecone vector store
        if all_documents:
            st.info("🔍 Creating embeddings with Pinecone and storing in vector database...")
            self.vector_store.add_documents(all_documents)
            st.success(f"✅ Successfully loaded {len(all_documents)} document chunks into Pinecone!")
        
        return len(all_documents)
    
    def generate_response(self, query: str, relevant_docs: List[Dict]) -> str:
        """Generate response using OpenRouter API"""
        if not relevant_docs:
            return "I don't know. I couldn't find relevant information in the support documentation to answer your question."
        
        # Prepare context from relevant documents
        context = "\n\n".join([
            f"Source: {doc['title']}\nContent: {doc['content']}"
            for doc in relevant_docs
        ])
        
        system_prompt = """You are a helpful customer support assistant. You can ONLY answer questions based on the provided support documentation context. 

IMPORTANT RULES:
1. ONLY use information from the provided context to answer questions
2. If the answer is not in the context, respond with "I don't know"
3. Be helpful and provide detailed answers when information is available
4. Always be polite and professional
5. If you reference information, mention the source when helpful

Context from support documentation:
{context}
"""
        
        messages = [
            {"role": "system", "content": system_prompt.format(context=context)},
            {"role": "user", "content": query}
        ]
        
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            response = requests.post(self.base_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return "I apologize, but I'm experiencing technical difficulties. Please try again later."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Main chat function"""
        # Search for relevant documents in Pinecone
        relevant_docs = self.vector_store.search(query, limit=5)
        
        # Generate response
        response = self.generate_response(query, relevant_docs)
        
        return {
            "response": response,
            "sources": relevant_docs,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return self.vector_store.get_index_stats()

def main():
    """Streamlit app main function"""
    st.set_page_config(
        page_title="🤖 Pinecone RAG Customer Support Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Advanced RAG Customer Support Chatbot")
    st.markdown("*Powered by Pinecone Vector Database & Embeddings*")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys input
        st.subheader("🔑 API Keys")
        openrouter_api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Get your API key from https://openrouter.ai/"
        )
        
        pinecone_api_key = st.text_input(
            "Pinecone API Key",
            type="password",
            help="Get your API key from https://pinecone.io/"
        )
        
        # Model selection
        model = st.selectbox(
            "Select Model",
            [
                "anthropic/claude-3-haiku",
                "anthropic/claude-3-sonnet",
                "openai/gpt-3.5-turbo",
                "openai/gpt-4o-mini"
            ]
        )
        
        st.markdown("---")
        st.header("📚 Document Sources")
        
        # Website URL
        website_url = st.text_input(
            "Website URL",
            value="https://www.angelone.in/support",
            help="Base URL to scrape for support documentation"
        )
        
        max_pages = st.slider("Max pages to scrape", 5, 100, 20)
        
        # File uploads
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF or DOCX files containing support documentation"
        )
        
        # Load documents button
        load_docs = st.button("🔄 Load Documents", type="primary")
    
    # Check API keys
    if not openrouter_api_key or not pinecone_api_key:
        st.warning("⚠️ Please enter both OpenRouter and Pinecone API keys in the sidebar to continue.")
        st.info("📖 **Getting Started:**")
        st.markdown("""
        1. **OpenRouter API Key**: Sign up at [https://openrouter.ai/](https://openrouter.ai/)
        2. **Pinecone API Key**: Sign up at [https://pinecone.io/](https://pinecone.io/)
        3. Enter both keys in the sidebar
        4. Load your support documents
        5. Start chatting!
        """)
        st.stop()
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        try:
            with st.spinner("Initializing Pinecone connection..."):
                st.session_state.chatbot = RAGChatbot(openrouter_api_key, pinecone_api_key, model)
            st.success("✅ Connected to Pinecone!")
        except Exception as e:
            st.error(f"❌ Error initializing Pinecone: {e}")
            st.stop()
    
    # Update chatbot settings if changed
    if 'chatbot' in st.session_state:
        st.session_state.chatbot.model = model
        st.session_state.chatbot.openrouter_api_key = openrouter_api_key
    
    # Load documents
    if load_docs:
        sources = {
            "website": website_url if website_url else None,
            "max_pages": max_pages,
            "files": uploaded_files if uploaded_files else []
        }
        
        if sources["website"] or sources["files"]:
            with st.spinner("Loading and processing documents..."):
                try:
                    doc_count = st.session_state.chatbot.load_documents(sources)
                    st.session_state.docs_loaded = True
                    st.session_state.doc_count = doc_count
                except Exception as e:
                    st.error(f"Error loading documents: {e}")
        else:
            st.warning("⚠️ Please provide at least a website URL or upload some files.")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Check if documents are loaded
        if 'docs_loaded' not in st.session_state:
            st.info("👆 Please load your support documents using the sidebar first.")
        else:
            st.success(f"✅ {st.session_state.doc_count} document chunks loaded and ready!")
            
            # Initialize chat history
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me anything about the support documentation..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Searching Pinecone and generating response..."):
                        chat_result = st.session_state.chatbot.chat(prompt)
                        response = chat_result["response"]
                        sources = chat_result["sources"]
                    
                    st.markdown(response)
                    
                    # Show sources if available
                    if sources and not response.startswith("I don't know"):
                        with st.expander("📚 Sources", expanded=False):
                            for i, source in enumerate(sources[:3]):
                                st.markdown(f"**Source {i+1}:** {source['title']}")
                                st.markdown(f"*Similarity: {source['score']:.3f}*")
                                st.markdown(f"*Type: {source['metadata']['type']}*")
                                st.markdown(f"```\n{source['content'][:200]}...\n```")
                                st.markdown("---")
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.header("📊 System Info")
        
        # Pinecone stats
        if 'chatbot' in st.session_state:
            try:
                stats = st.session_state.chatbot.get_stats()
                if 'error' not in stats:
                    st.metric("Vectors in Pinecone", stats.get('total_vectors', 0))
                    st.metric("Vector Dimension", stats.get('dimension', 0))
                    st.metric("Index Fullness", f"{stats.get('index_fullness', 0)*100:.1f}%")
                else:
                    st.error(f"Stats error: {stats['error']}")
            except Exception as e:
                st.error(f"Error fetching stats: {e}")
        
        if 'docs_loaded' in st.session_state:
            st.metric("Documents Loaded", st.session_state.doc_count)
            st.metric("Chat Messages", len(st.session_state.messages) if 'messages' in st.session_state else 0)
        
        st.markdown("---")
        st.header("🌲 Pinecone Features")
        st.markdown("""
        - ✅ **Serverless Architecture**
        - ✅ **Pinecone Embeddings** (multilingual-e5-large)
        - ✅ **Auto-scaling**
        - ✅ **High Performance**
        - ✅ **Real-time Updates**
        - ✅ **Metadata Filtering**
        """)
        
        st.markdown("---")
        st.header("ℹ️ How it works")
        st.markdown("""
        1. **Document Processing**: Website scraping and file processing
        2. **Chunking**: Text split with overlap
        3. **Pinecone Embeddings**: Generate high-quality vectors
        4. **Vector Storage**: Store in Pinecone serverless index
        5. **Semantic Search**: Find relevant content
        6. **LLM Generation**: Generate contextual responses
        """)

if __name__ == "__main__":
    main()