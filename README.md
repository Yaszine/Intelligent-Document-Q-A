# Intelligent-Document-QA


LINK: https://yaszinedocumentintelligent.streamlit.app/

**Intelligent-Document-QA** is an AI-powered agent that answers questions from company documents such as PDFs, policies, contracts, and research papers.

---

## How It Works

1. **Document Indexing:** Uses FAISS embeddings to index documents.  
2. **Context Retrieval:** Implements Retrieval-Augmented Generation (RAG) to fetch relevant context before prompting the LLM.  
3. **Few-Shot Prompting:** Guides the AI to answer concisely and accurately.  

---

## Key Enhancements

### Visual Improvements
- Custom CSS styling with improved colors, spacing, and hierarchy  
- Enhanced chat interface with avatars and colored message bubbles  
- Color-coded status indicators  
- Document statistics displayed in the sidebar  
- Progress bars for file processing  
- Responsive layout with better use of columns  

### Practical Improvements
- File validation (type and size checks)  
- Embedding caching to reduce API calls and costs  
- Improved chunking algorithm with sentence awareness  
- Robust error handling and logging  
- API key validation before processing  
- Temporary file handling for PDFs  
- Chat history saving  
- Timeout handling for API calls  
- Enhanced prompt engineering for more accurate answers  
- Memory management and proper cleanup  
- Configurable settings via constants at the top  
- Better document source tracking  

### Performance Enhancements
- Reduced API calls through caching  
- Faster processing with optimized chunking  
- Efficient memory usage  
- Improved reliability with comprehensive error handling  

### User Experience
- Clear status indicators throughout the workflow  
- Feedback during file processing  
- Enhanced document insights with statistics  
- Improved answer quality via better prompting  
- Chat management (clear/save)  

---

This enhanced version makes **Intelligent-Document-QA** more production-ready, user-friendly, and robust while retaining all the original functionality.
