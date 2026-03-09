"""
RAG Controller Layer.

This module handles HTTP request/response processing for RAG endpoints.
It provides endpoints for document management and context retrieval.

Endpoints:
- POST /rag/document - Upload a document
- GET /rag/documents - List all documents
- DELETE /rag/document/<id> - Delete a document
- POST /rag/retrieve - Retrieve context for a query
"""

import logging
from flask import Blueprint, request, jsonify

from services import rag_service

logger = logging.getLogger(__name__)

rag_bp = Blueprint('rag', __name__)


@rag_bp.route('/rag/document', methods=['POST'])
def upload_document():
    """
    Handle document upload.
    
    Expected form data: file (multipart/form-data)
    Returns JSON response with document info
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        file_content = file.read()
        result = rag_service.add_document(file_content, file.filename)
        logger.info(f"Document uploaded: {result['filename']}")
        return jsonify(result), 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except ImportError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        return jsonify({'error': f'Failed to process document: {str(e)}'}), 500


@rag_bp.route('/rag/documents', methods=['GET'])
def list_documents():
    """
    List all indexed documents.
    
    Returns JSON response with list of documents
    """
    try:
        documents = rag_service.get_all_documents()
        return jsonify({'documents': documents})
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        return jsonify({'error': str(e)}), 500


@rag_bp.route('/rag/document/<document_id>', methods=['DELETE'])
def delete_document(document_id):
    """
    Delete a document by ID.
    
    Args:
        document_id: The document ID to delete
        
    Returns JSON response with success status
    """
    try:
        success = rag_service.delete_document(document_id)
        if success:
            return jsonify({'message': 'Document deleted successfully'})
        else:
            return jsonify({'error': 'Document not found'}), 404
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        return jsonify({'error': str(e)}), 500


@rag_bp.route('/rag/retrieve', methods=['POST'])
def retrieve_context():
    """
    Retrieve relevant context for a query.
    
    Expected JSON input: {"query": "search query", "top_k": 5}
    Returns JSON response with relevant document chunks
    """
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing "query" field in request'}), 400
    
    query = data['query']
    top_k = data.get('top_k')
    
    try:
        results = rag_service.retrieve_context(query, top_k)
        return jsonify({'context': results})
    except Exception as e:
        logger.error(f"Failed to retrieve context: {e}")
        return jsonify({'error': str(e)}), 500
