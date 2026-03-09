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

from flask import Blueprint, request, jsonify, Response
import requests

from services.ollama_service import generate_response, generate_streaming_response

# ==============================================================================
# Controller Setup
# ==============================================================================

# Create a Blueprint for the chat controller
# This allows modular route registration with app.register_blueprint()
chat_bp = Blueprint('chat', __name__)


# ==============================================================================
# Streaming Helper Functions
# ==============================================================================

def generate_sse_stream(prompt: str):
    """
    Generate a Server-Sent Events (SSE) stream from the streaming response.
    
    This function wraps the streaming response generator and formats each chunk
    as an SSE message. SSE is a standard protocol for streaming data over HTTP
    where each message is prefixed with "data: " and ends with double newline.
    
    SSE Format:
        data: chunk1
    
        data: chunk2
    
    The "data: " prefix tells the client this is an SSE message.
    The double newline (\n\n) signals end of the message.
    
    Args:
        prompt: The user's input message
        
    Yields:
        Formatted SSE messages ready to send to the client
    """
    try:
        # Iterate over chunks from the streaming response generator
        # Each chunk is a piece of the assistant's response
        for chunk in generate_streaming_response(prompt):
            # Format the chunk as an SSE message
            # This follows the SSE protocol specification
            yield f"data: {chunk}\n\n"
    except requests.exceptions.ConnectionError:
        yield "data: [ERROR] Unable to connect to Ollama server. Make sure it is running on localhost:11434\n\n"
    except requests.exceptions.RequestException as e:
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
    # -------------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------------
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field in request'}), 400
    
    user_message = data['message']
    
    # -------------------------------------------------------------------------
    # Business Logic (delegate to service layer)
    # -------------------------------------------------------------------------
    try:
        reply = generate_response(user_message)
    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Unable to connect to Ollama server. Make sure it is running on localhost:11434'
        }), 503
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Ollama API error: {str(e)}'}), 500
    
    # -------------------------------------------------------------------------
    # Response
    # -------------------------------------------------------------------------
    return jsonify({'reply': reply})


@chat_bp.route('/chat/stream', methods=['POST'])
def chat_stream():
    """
    Handle streaming chat POST requests.
    
    This endpoint provides real-time streaming responses using Server-Sent Events (SSE).
    The client receives the assistant's response incrementally as tokens are generated,
    rather than waiting for the complete response.
    
    Expected JSON input: {"message": "user message here"}
    Returns: Server-Sent Events (SSE) stream with response chunks
    
    SSE Response Format:
        data: chunk1
    
        data: chunk2
    
    Error Responses:
        400: Missing "message" field in request
    
    Client Implementation Notes:
        - Use EventSource API in JavaScript to consume this stream
        - Listen for 'message' events to receive chunks
        - The stream ends when the server closes the connection
    """
    # -------------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------------
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field in request'}), 400
    
    user_message = data['message']
    
    # -------------------------------------------------------------------------
    # Streaming Response
    # -------------------------------------------------------------------------
    # Create an SSE response with our streaming generator
    # 
    # Content-Type: text/event-stream - Required for SSE
    # Cache-Control: no-cache - Prevents caching of streaming response
    # Connection: keep-alive - Keeps the connection open for streaming
    #
    # The Response object from Flask handles the streaming internally;
    # it iterates over the generator and sends each yield to the client
    return Response(
        generate_sse_stream(user_message),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'disable'  # Disables nginx buffering if present
        }
    )
