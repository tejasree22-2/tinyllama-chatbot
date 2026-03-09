"""
Embedding Service Layer.

This module is responsible for generating embeddings from text using
sentence-transformers. It provides a clean interface for the RAG service.
"""

import logging
import numpy as np
from typing import List

import config

logger = logging.getLogger(__name__)

_embedding_model = None


def get_embedding_model():
    """
    Get or initialize the sentence-transformers model.
    
    Returns:
        The sentence-transformers model instance
    """
    global _embedding_model
    
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
            _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    
    return _embedding_model


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    model = get_embedding_model()
    
    logger.info(f"Generating embeddings for {len(texts)} text chunks")
    
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    return embeddings.tolist()


def generate_query_embedding(query: str) -> List[float]:
    """
    Generate embedding for a query string.
    
    Args:
        query: The query string to embed
        
    Returns:
        Embedding vector
    """
    model = get_embedding_model()
    
    logger.info(f"Generating embedding for query: {query[:50]}...")
    
    embedding = model.encode([query], convert_to_numpy=True)[0]
    
    return embedding.tolist()
