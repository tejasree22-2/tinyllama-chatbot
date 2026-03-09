"""
Chat Controller Layer.

This module handles HTTP request/response processing for the chat endpoint.
It acts as an intermediary between the Flask routing layer and the service layer.

Responsibilities:
- Validate incoming request data
- Call the appropriate service to process business logic
- Format and return HTTP responses
- Handle errors and return appropriate status codes

Request/Response Format:
- Input:  {"message": "user message here"}
- Output: {"reply": "response message here"}
"""

from flask import Blueprint, request, jsonify
import requests

from services.ollama_service import generate_response

# ==============================================================================
# Controller Setup
# ==============================================================================

# Create a Blueprint for the chat controller
# This allows modular route registration with app.register_blueprint()
chat_bp = Blueprint('chat', __name__)


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
