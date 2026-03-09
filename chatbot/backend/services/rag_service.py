"""
RAG Service Layer.

This module is responsible for:
- Loading documents from various formats (txt, pdf, docx)
- Chunking text into smaller segments
- Generating embeddings and storing in vector database
- Retrieving relevant context for queries
"""

import io
import logging
import os
import uuid
from typing import List, Dict, Tuple, Optional

import config

logger = logging.getLogger(__name__)

_chroma_client = None
_collection = None


def get_chroma_client():
    """
    Get or initialize the ChromaDB client.
    
    Returns:
        ChromaDB client instance
    """
    global _chroma_client
    
    if _chroma_client is None:
        try:
            import chromadb
            from chromadb.config import Settings
            logger.info(f"Initializing ChromaDB at: {config.CHROMA_PERSIST_DIR}")
            _chroma_client = chromadb.Client(Settings(
                persist_directory=config.CHROMA_PERSIST_DIR,
                anonymized_telemetry=False
            ))
        except ImportError:
            logger.error("chromadb not installed")
            raise ImportError("Please install chromadb: pip install chromadb")
    
    return _chroma_client


def get_collection():
    """
    Get or create the documents collection.
    
    Returns:
        ChromaDB collection
    """
    global _collection
    
    client = get_chroma_client()
    
    if _collection is None:
        try:
            _collection = client.get_or_create_collection(name="documents")
            logger.info("Documents collection initialized")
        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}")
            raise
    
    return _collection


def load_text_file(file_content: bytes, filename: str) -> str:
    """
    Load text from a plain text file.
    
    Args:
        file_content: Raw file content
        filename: Original filename
        
    Returns:
        Extracted text content
    """
    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        return file_content.decode('latin-1')


def load_pdf_file(file_content: bytes, filename: str) -> str:
    """
    Load text from a PDF file.
    
    Args:
        file_content: Raw file content
        filename: Original filename
        
    Returns:
        Extracted text content
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("Please install PyPDF2: pip install PyPDF2")
    
    pdf_file = io.BytesIO(file_content)
    reader = PdfReader(pdf_file)
    
    text_parts = []
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
    
    return "\n\n".join(text_parts)


def load_docx_file(file_content: bytes, filename: str) -> str:
    """
    Load text from a DOCX file.
    
    Args:
        file_content: Raw file content
        filename: Original filename
        
    Returns:
        Extracted text content
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Please install python-docx: pip install python-docx")
    
    doc_file = io.BytesIO(file_content)
    doc = Document(doc_file)
    
    text_parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
    
    return "\n\n".join(text_parts)


def load_document(file_content: bytes, filename: str) -> str:
    """
    Load text from a document based on file extension.
    
    Args:
        file_content: Raw file content
        filename: Original filename
        
    Returns:
        Extracted text content
        
    Raises:
        ValueError: If file format is not supported
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.txt' or ext == '.md':
        return load_text_file(file_content, filename)
    elif ext == '.pdf':
        return load_pdf_file(file_content, filename)
    elif ext == '.docx':
        return load_docx_file(file_content, filename)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def chunk_text(text: str) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + config.RAG_CHUNK_SIZE
        
        if end < text_length:
            newline_pos = text.rfind('\n', start, end)
            if newline_pos > start:
                end = newline_pos
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - config.RAG_CHUNK_OVERLAP if end < text_length else text_length
    
    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks


def add_document(file_content: bytes, filename: str) -> Dict:
    """
    Add a document to the vector store.
    
    Args:
        file_content: Raw file content
        filename: Original filename
        
    Returns:
        Dictionary with document info
    """
    from services.embedding_service import generate_embeddings
    
    logger.info(f"Adding document: {filename}")
    
    text = load_document(file_content, filename)
    
    if not text.strip():
        raise ValueError("Document is empty or contains no extractable text")
    
    chunks = chunk_text(text)
    
    doc_id = str(uuid.uuid4())
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    
    embeddings = generate_embeddings(chunks)
    
    collection = get_collection()
    
    metadatas = [
        {
            "document_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        for i in range(len(chunks))
    ]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    
    logger.info(f"Document added successfully with ID: {doc_id}")
    
    return {
        "document_id": doc_id,
        "filename": filename,
        "chunks": len(chunks)
    }


def retrieve_context(query: str, top_k: Optional[int] = None) -> List[Dict]:
    """
    Retrieve relevant context for a query.
    
    Args:
        query: The search query
        top_k: Number of results to retrieve (default from config)
        
    Returns:
        List of relevant document chunks with metadata
    """
    from services.embedding_service import generate_query_embedding
    
    if top_k is None:
        top_k = config.RAG_TOP_K
    
    logger.info(f"Retrieving context for query: {query[:50]}...")
    
    query_embedding = generate_query_embedding(query)
    
    collection = get_collection()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    context_docs = []
    
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            context_docs.append({
                "text": doc,
                "filename": results['metadatas'][0][i].get("filename", "unknown"),
                "document_id": results['metadatas'][0][i].get("document_id", ""),
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
    
    logger.info(f"Retrieved {len(context_docs)} relevant chunks")
    
    return context_docs


def get_all_documents() -> List[Dict]:
    """
    Get list of all indexed documents.
    
    Returns:
        List of document metadata
    """
    collection = get_collection()
    
    try:
        results = collection.get()
    except Exception as e:
        logger.error(f"Failed to get documents: {e}")
        return []
    
    if not results['ids']:
        return []
    
    document_ids = set()
    documents = []
    
    for i, metadata in enumerate(results.get('metadatas', [])):
        doc_id = metadata.get('document_id')
        if doc_id and doc_id not in document_ids:
            document_ids.add(doc_id)
            documents.append({
                "document_id": doc_id,
                "filename": metadata.get('filename', 'unknown'),
                "total_chunks": metadata.get('total_chunks', 0)
            })
    
    return documents


def delete_document(document_id: str) -> bool:
    """
    Delete a document and its chunks from the vector store.
    
    Args:
        document_id: The document ID to delete
        
    Returns:
        True if deleted successfully
    """
    collection = get_collection()
    
    try:
        results = collection.get()
        
        ids_to_delete = []
        for i, metadata in enumerate(results.get('metadatas', [])):
            if metadata.get('document_id') == document_id:
                ids_to_delete.append(results['ids'][i])
        
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted document: {document_id}")
            return True
        else:
            logger.warning(f"Document not found: {document_id}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        return False
