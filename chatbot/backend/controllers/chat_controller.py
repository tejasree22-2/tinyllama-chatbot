"""
Chat Controller Layer.

This module handles HTTP request/response processing for the chat endpoint.
It acts as an intermediary between the Flask routing layer and the service layer.

Responsibilities:
- Validate incoming request data
- Call the appropriate service to process business logic
- Format and return HTTP responses (streaming or non-streaming)
- Handle errors and return appropriate status codes

Request/Response Format:
- Input (non-streaming):  {"message": "user message here"}
- Output (non-streaming): {"reply": "response message here"}
- Output (streaming):    Server-Sent Events (SSE) with "data: {chunk}\n\n"
"""

import logging
from flask import Blueprint, request, jsonify, Response

from services.ollama_service import (
    generate_response,
    generate_streaming_response,
    generate_response_with_rag,
    generate_streaming_response_with_rag,
    OllamaAPIError,
    OllamaConnectionError
)

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)


# ==============================================================================
# Streaming Helper Functions
# ==============================================================================

def generate_sse_stream(prompt: str):
    """
    Generate a Server-Sent Events (SSE) stream from the streaming response.
    
    Args:
        prompt: The user's input message
        
    Yields:
        Formatted SSE messages ready to send to the client
    """
    try:
        for chunk in generate_streaming_response(prompt):
            yield f"data: {chunk}\n\n"
    except OllamaConnectionError as e:
        logger.error(f"Ollama connection error: {e}")
        yield "data: [ERROR] Unable to connect to Ollama server. Make sure it is running.\n\n"
    except OllamaAPIError as e:
        logger.error(f"Ollama API error: {e}")
        yield f"data: [ERROR] Ollama API error: {str(e)}\n\n"


def generate_rag_sse_stream(prompt: str, use_rag: bool = True, top_k: int = None):
    """
    Generate a Server-Sent Events (SSE) stream from the RAG streaming response.
    
    Args:
        prompt: The user's input message
        use_rag: Whether to use RAG context
        top_k: Number of context chunks to retrieve
        
    Yields:
        Formatted SSE messages ready to send to the client
    """
    try:
        for chunk in generate_streaming_response_with_rag(prompt, use_rag=use_rag, top_k=top_k):
            yield f"data: {chunk}\n\n"
    except OllamaConnectionError as e:
        logger.error(f"Ollama connection error: {e}")
        yield "data: [ERROR] Unable to connect to Ollama server. Make sure it is running.\n\n"
    except OllamaAPIError as e:
        logger.error(f"Ollama API error: {e}")
        yield f"data: [ERROR] Ollama API error: {str(e)}\n\n"


# ==============================================================================
# Route Handlers
# ==============================================================================

@chat_bp.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat POST requests.
    
    Expected JSON input: {"message": "user message here"}
    Returns JSON response: {"reply": "response message here"}
    
    Error Responses:
        400: Missing "message" field in request
        503: Ollama server unavailable
        500: Internal server error
    """
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field in request'}), 400
    
    user_message = data['message']
    logger.info(f"Chat request received: {user_message[:50]}...")
    
    try:
        reply = generate_response(user_message)
        logger.info(f"Chat response sent ({len(reply)} chars)")
    except OllamaConnectionError as e:
        logger.error(f"Ollama connection error: {e}")
        return jsonify({
            'error': 'Unable to connect to Ollama server. Make sure it is running.'
        }), 503
    except OllamaAPIError as e:
        logger.error(f"Ollama API error: {e}")
        return jsonify({'error': f'Ollama API error: {str(e)}'}), 500
    
    return jsonify({'reply': reply})


@chat_bp.route('/chat/stream', methods=['POST'])
def chat_stream():
    """
    Handle streaming chat POST requests.
    
    This endpoint provides real-time streaming responses using Server-Sent Events (SSE).
    
    Expected JSON input: {"message": "user message here"}
    Returns: Server-Sent Events (SSE) stream with response chunks
    
    Error Responses:
        400: Missing "message" field in request
    """
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field in request'}), 400
    
    user_message = data['message']
    logger.info(f"Streaming chat request received: {user_message[:50]}...")
    
    return Response(
        generate_sse_stream(user_message),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'disable'
        }
    )


@chat_bp.route('/chat/rag', methods=['POST'])
def chat_rag():
    """
    Handle RAG-enabled chat POST requests.
    
    Retrieves relevant context from uploaded documents before generating response.
    
    Expected JSON input: {"message": "user message here", "use_rag": true, "top_k": 5}
    Returns JSON response: {"reply": "response message here"}
    
    Error Responses:
        400: Missing "message" field in request
        503: Ollama server unavailable
        500: Internal server error
    """
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field in request'}), 400
    
    user_message = data['message']
    use_rag = data.get('use_rag', True)
    top_k = data.get('top_k')
    logger.info(f"RAG chat request received: {user_message[:50]}... (use_rag={use_rag})")
    
    try:
        reply = generate_response_with_rag(user_message, use_rag=use_rag, top_k=top_k)
        logger.info(f"RAG chat response sent ({len(reply)} chars)")
    except OllamaConnectionError as e:
        logger.error(f"Ollama connection error: {e}")
        return jsonify({
            'error': 'Unable to connect to Ollama server. Make sure it is running.'
        }), 503
    except OllamaAPIError as e:
        logger.error(f"Ollama API error: {e}")
        return jsonify({'error': f'Ollama API error: {str(e)}'}), 500
    
    return jsonify({'reply': reply})


@chat_bp.route('/chat/rag/stream', methods=['POST'])
def chat_rag_stream():
    """
    Handle streaming RAG-enabled chat POST requests.
    
    Retrieves relevant context from uploaded documents before generating streaming response.
    
    Expected JSON input: {"message": "user message here", "use_rag": true, "top_k": 5}
    Returns: Server-Sent Events (SSE) stream with response chunks
    """
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field in request'}), 400
    
    user_message = data['message']
    use_rag = data.get('use_rag', True)
    top_k = data.get('top_k')
    logger.info(f"Streaming RAG chat request received: {user_message[:50]}... (use_rag={use_rag})")
    
    return Response(
        generate_rag_sse_stream(user_message, use_rag, top_k),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'disable'
        }
    )
