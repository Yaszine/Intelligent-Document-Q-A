import streamlit as st
import openai
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import io
import time
from datetime import datetime
import hashlib
import pickle
import os
import re
import json
from pathlib import Path
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PDF processing libraries
try:
    import pypdf
    PDF_LIBRARY = "pypdf"
except ImportError:
    try:
        import pdfplumber
        PDF_LIBRARY = "pdfplumber"
    except ImportError:
        st.error("Please install either 'pypdf' or 'pdfplumber' to process PDF files.")
        st.stop()

# Optional chat UI
try:
    from streamlit_chat import message
    CHAT_UI_AVAILABLE = True
except ImportError:
    CHAT_UI_AVAILABLE = False

# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 5
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"
MAX_FILE_SIZE_MB = 10
CACHE_DIR = Path("./.cache")

# Create cache directory
CACHE_DIR.mkdir(exist_ok=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        "messages": [],
        "embeddings_cache": {},
        "vector_store": None,
        "document_chunks": [],
        "processed_files": set(),
        "current_files": {},
        "processing_status": "idle",
        "error_message": None,
        "chat_history": [],
        "api_key_valid": False
    }
    
    for key, default_value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def setup_custom_css():
    """Setup custom CSS for better styling"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sidebar-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .status-success {
        padding: 0.5rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-warning {
        padding: 0.5rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    .status-error {
        padding: 0.5rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    .document-card {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
    }
    .stButton button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

def clean_text(text: str) -> str:
    """Clean text by removing problematic Unicode characters and normalizing"""
    if not text:
        return ""
    
    # Remove emojis and symbols
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642" 
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub(r'', text)
    
    # Normalize whitespace and clean up
    text = re.sub(r'\s+', ' ', text)
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    return text.strip()

def validate_pdf_file(file) -> bool:
    """Validate PDF file before processing"""
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File {file.name} is too large. Max size: {MAX_FILE_SIZE_MB}MB")
        return False
    
    if file.type != "application/pdf":
        st.error(f"File {file.name} is not a PDF")
        return False
    
    return True

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file using available library"""
    try:
        # Create a temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        
        text = ""
        if PDF_LIBRARY == "pypdf":
            reader = pypdf.PdfReader(tmp_file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += clean_text(page_text) + "\n"
        else:  # pdfplumber
            with pdfplumber.open(tmp_file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += clean_text(page_text) + "\n"
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return clean_text(text)
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        st.error(f"Error processing {pdf_file.name}: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks with improved chunking logic"""
    if not text.strip():
        return []
    
    text = clean_text(text)
    words = text.split()
    chunks = []
    
    # Improved chunking with sentence awareness
    if len(words) > chunk_size * 2:
        # Use sentence splitting for better chunks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep overlap from the end of current chunk
                overlap_words = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk = overlap_words + sentence_words
                current_length = len(current_chunk)
            else:
                current_chunk.extend(sentence_words)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    else:
        # Fallback to simple word-based chunking
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(clean_text(chunk))
            if i + chunk_size >= len(words):
                break
    
    return chunks

def get_cached_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding from cache if available"""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{text_hash}.npy"
    
    if cache_file.exists():
        try:
            return np.load(cache_file)
        except:
            pass
    return None

def cache_embedding(text: str, embedding: np.ndarray):
    """Cache embedding for future use"""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{text_hash}.npy"
    np.save(cache_file, embedding)

def get_embeddings(texts: List[str], client) -> np.ndarray:
    """Generate embeddings for a list of texts with caching"""
    try:
        # Clean and validate texts
        cleaned_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            cleaned_text = clean_text(text)
            if cleaned_text and len(cleaned_text.strip()) > 0:
                if len(cleaned_text) > 8000:
                    cleaned_text = cleaned_text[:8000] + "..."
                cleaned_texts.append(cleaned_text)
                valid_indices.append(i)
        
        if not cleaned_texts:
            return np.array([], dtype=np.float32)
        
        # Check cache first
        embeddings = np.zeros((len(texts), 1536), dtype=np.float32)  # ada-002 has 1536 dimensions
        texts_to_process = []
        process_indices = []
        
        for i, text in enumerate(cleaned_texts):
            cached_embedding = get_cached_embedding(text)
            if cached_embedding is not None:
                embeddings[valid_indices[i]] = cached_embedding
            else:
                texts_to_process.append(text)
                process_indices.append(valid_indices[i])
        
        # Process remaining texts with OpenAI
        if texts_to_process:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts_to_process
            )
            
            for idx, embedding_data in zip(process_indices, response.data):
                embedding_array = np.array(embedding_data.embedding, dtype=np.float32)
                embeddings[idx] = embedding_array
                # Cache the embedding
                cache_embedding(texts_to_process[process_indices.index(idx)], embedding_array)
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        st.error(f"Embedding generation failed: {str(e)}")
        return np.array([], dtype=np.float32)

def create_vector_store(embeddings: np.ndarray) -> Optional[faiss.IndexFlatIP]:
    """Create FAISS vector store from embeddings"""
    if embeddings.size == 0:
        return None
    
    try:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings_normalized = embeddings / norms
        
        index.add(embeddings_normalized)
        return index
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None

def retrieve_relevant_chunks(query: str, vector_store, chunks: List[str], client, k: int = TOP_K_RETRIEVAL) -> List[Tuple[str, float]]:
    """Retrieve top-k most relevant chunks for a query"""
    try:
        # Get query embedding
        query_response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[clean_text(query)]
        )
        query_embedding = np.array([query_response.data[0].embedding], dtype=np.float32)
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search vector store
        scores, indices = vector_store.search(query_embedding, min(k, vector_store.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(chunks):
                results.append((chunks[idx], float(score)))
        
        return results
    except Exception as e:
        logger.error(f"Error during retrieval: {str(e)}")
        return []

def generate_answer(query: str, relevant_chunks: List[Tuple[str, float]], client) -> str:
    """Generate answer using RAG approach with improved prompt engineering"""
    if not relevant_chunks:
        return "I couldn't find relevant information in the documents to answer your question."
    
    # Construct context with source information
    context_parts = []
    for i, (chunk, score) in enumerate(relevant_chunks, 1):
        context_parts.append(f"[Source {i}, Relevance: {score:.3f}]\n{chunk}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are an expert AI assistant that answers questions based strictly on the provided document context.

**Context Information:**
{context}

**Question:** {query}

**Instructions:**
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say "The documents don't contain enough information to answer this question completely."
3. Be precise and factual
4. If relevant, mention which source(s) you used for your answer
5. If the question is unclear or cannot be answered with the context, politely indicate this

**Answer:**"""

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful, precise assistant that answers questions based on provided document context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800,
            timeout=30  # Add timeout
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"I encountered an error while generating the answer: {str(e)}"

def validate_api_key(api_key: str, client) -> bool:
    """Validate OpenAI API key"""
    try:
        # Simple validation by making a small embedding request
        client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=["test"]
        )
        return True
    except:
        return False

def process_uploaded_files(uploaded_files, client):
    """Process uploaded PDF files and create vector store"""
    if not uploaded_files:
        return
    
    # Validate files first
    valid_files = []
    for file in uploaded_files:
        if validate_pdf_file(file):
            valid_files.append(file)
    
    if not valid_files:
        return
    
    # Check if files have changed
    current_files = {f.name: hashlib.md5(f.getvalue()).hexdigest() for f in valid_files}
    
    if current_files == st.session_state.get("current_files", {}):
        return  # No changes, skip processing
    
    st.session_state.current_files = current_files
    st.session_state.processing_status = "processing"
    
    try:
        with st.spinner("📄 Processing documents..."):
            progress_bar = st.progress(0)
            all_chunks = []
            total_files = len(valid_files)
            
            for i, uploaded_file in enumerate(valid_files):
                progress_bar.progress(i / total_files, text=f"Processing {uploaded_file.name}...")
                
                # Extract text
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    # Chunk text
                    chunks = chunk_text(text)
                    all_chunks.extend([(chunk, uploaded_file.name) for chunk in chunks])
            
            progress_bar.progress(1.0, text="Generating embeddings...")
            
            if not all_chunks:
                st.error("No text could be extracted from the uploaded files.")
                st.session_state.processing_status = "error"
                return
            
            # Generate embeddings
            chunk_texts = [chunk for chunk, filename in all_chunks]
            embeddings = get_embeddings(chunk_texts, client)
            
            if embeddings.size > 0:
                # Create vector store
                vector_store = create_vector_store(embeddings)
                
                # Update session state
                st.session_state.vector_store = vector_store
                st.session_state.document_chunks = all_chunks
                st.session_state.processing_status = "success"
                
                st.success(f"✅ Processed {len(valid_files)} files with {len(chunk_texts)} text chunks")
            else:
                st.error("Failed to generate embeddings for the documents.")
                st.session_state.processing_status = "error"
                
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        st.error(f"Error processing files: {str(e)}")
        st.session_state.processing_status = "error"
    finally:
        progress_bar.empty()

def display_chat_message(message_type: str, content: str, timestamp: str = None):
    """Display a chat message with consistent formatting"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    if CHAT_UI_AVAILABLE:
        is_user = message_type == "user"
        message(content, is_user=is_user, key=f"{message_type}_{len(st.session_state.messages)}_{timestamp}")
    else:
        avatar = "👤" if message_type == "user" else "🤖"
        color = "#e1f5fe" if message_type == "user" else "#f3e5f5"
        
        st.markdown(f"""
        <div style="background-color: {color}; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;">
            <strong>{avatar} {message_type.capitalize()} ({timestamp}):</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

def display_retrieved_chunks(chunks: List[Tuple[str, float]]):
    """Display retrieved document chunks in an expandable section"""
    if chunks:
        with st.expander(f"🔍 Retrieved Document Snippets (Top {len(chunks)})", expanded=False):
            for i, (chunk, score) in enumerate(chunks, 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Snippet {i}**")
                with col2:
                    st.markdown(f"*Relevance: `{score:.3f}`*")
                
                st.markdown(f"```\n{chunk[:400]}{'...' if len(chunk) > 400 else ''}\n```")
                st.markdown("---")

def display_document_stats():
    """Display document statistics"""
    if st.session_state.document_chunks:
        files = set(filename for _, filename in st.session_state.document_chunks)
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Document Statistics")
        st.sidebar.markdown(f"**Files processed:** {len(files)}")
        st.sidebar.markdown(f"**Text chunks:** {len(st.session_state.document_chunks)}")
        
        # Show file list
        with st.sidebar.expander("View File List"):
            for file in sorted(files):
                st.markdown(f"• {file}")

def save_chat_history():
    """Save chat history to file"""
    try:
        chat_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages,
            "documents": list(set(filename for _, filename in st.session_state.document_chunks))
        }
        
        filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(CACHE_DIR / filename, 'w') as f:
            json.dump(chat_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")

def main():
    st.set_page_config(
        page_title="Intelligent Document Q&A",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Setup custom CSS
    setup_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📚 Document Q&A</div>', unsafe_allow_html=True)
        st.markdown("Upload PDFs and ask questions about their content using AI-powered retrieval.")
        
        st.markdown("---")
        
        # OpenAI API Key input
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use the service",
            placeholder="sk-...",
            key="api_key_input"
        )
        
        if openai_api_key:
            try:
                client = openai.OpenAI(api_key=openai_api_key)
                # Validate API key
                if validate_api_key(openai_api_key, client):
                    st.session_state.api_key_valid = True
                else:
                    st.error("Invalid API key. Please check and try again.")
                    st.stop()
            except Exception as e:
                st.error(f"API key validation failed: {str(e)}")
                st.stop()
        else:
            st.warning("Please enter your OpenAI API key to continue.")
            st.stop()
        
        st.markdown("---")
        
        # File upload
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Upload one or more PDF documents (max {MAX_FILE_SIZE_MB}MB each)"
        )
        
        # Process files
        if uploaded_files and st.session_state.api_key_valid:
            process_uploaded_files(uploaded_files, client)
        
        # Display document statistics
        display_document_stats()
        
        st.markdown("---")
        
        # Chat management
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear chat history"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("💾 Save Chat", use_container_width=True, help="Save chat history"):
                save_chat_history()
                st.success("Chat history saved!")
        
        st.markdown("---")
        
        # Status information
        st.subheader("📈 Status")
        if st.session_state.vector_store is not None:
            st.markdown('<div class="status-success">✅ Documents indexed and ready</div>', unsafe_allow_html=True)
        elif st.session_state.processing_status == "processing":
            st.markdown('<div class="status-warning">⏳ Processing documents...</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-info">📥 Upload documents to start</div>', unsafe_allow_html=True)
    
    # Main area
    st.markdown('<div class="main-header">💬 Intelligent Document Assistant</div>', unsafe_allow_html=True)
    
    # Display chat history in a container with scroll
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            display_chat_message(
                message["type"], 
                message["content"], 
                message.get("timestamp", "")
            )
            
            # Show retrieved chunks if available
            if message["type"] == "assistant" and "chunks" in message:
                display_retrieved_chunks(message["chunks"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Query input
    st.markdown("---")
    
    # Create columns for input layout
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_query = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main findings in the research?",
            key="user_input",
            label_visibility="collapsed",
            disabled=st.session_state.vector_store is None
        )
    
    with col2:
        ask_button = st.button("🚀 Ask", 
                             use_container_width=True, 
                             type="primary",
                             disabled=st.session_state.vector_store is None or not user_query.strip())
    
    # Process query
    if ask_button and user_query.strip() and st.session_state.vector_store is not None:
        # Add user message
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({
            "type": "user",
            "content": user_query,
            "timestamp": timestamp
        })
        
        # Show processing
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                # Retrieve relevant chunks
                chunk_texts = [chunk for chunk, filename in st.session_state.document_chunks]
                relevant_chunks = retrieve_relevant_chunks(
                    user_query, 
                    st.session_state.vector_store, 
                    chunk_texts, 
                    client
                )
                
                # Generate answer
                answer = generate_answer(user_query, relevant_chunks, client)
                
                # Add assistant message
                st.session_state.messages.append({
                    "type": "assistant",
                    "content": answer,
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "chunks": relevant_chunks
                })
                
                # Save to chat history
                st.session_state.chat_history = st.session_state.messages.copy()
                
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                st.error(f"Error processing your question: {str(e)}")
        
        # Rerun to show new messages
        st.rerun()

if __name__ == "__main__":
    main()