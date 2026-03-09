"""
Flask backend server for the TinyLlama chatbot.
Provides a simple chat endpoint for processing user messages.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# Initialize Flask application
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing) to allow frontend to communicate with backend
# This allows requests from different origins (domains, ports, or protocols)
CORS(app)

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "tinyllama"


@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat POST requests.
    
    Expected JSON input: {"message": "user message here"}
    Returns JSON response: {"reply": "response message here"}
    """
    # Parse JSON data from the request body
    data = request.get_json()
    
    # Validate that 'message' field exists in the request
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field in request'}), 400
    
    # Extract the user's message to use as prompt for TinyLlama
    user_message = data['message']
    
    # Prepare payload for Ollama API request
    # - model: specifies which LLM model to use (tinyllama)
    # - prompt: the user's message that the model will respond to
    # - stream: set to False to disable streaming and get complete response at once
    payload = {
        "model": MODEL_NAME,
        "prompt": user_message,
        "stream": False
    }
    
    try:
        # Send POST request to Ollama API
        # This communicates with the local Ollama server running on port 11434
        response = requests.post(OLLAMA_API_URL, json=payload)
        
        # Check if the request was successful (status code 200)
        response.raise_for_status()
        
        # Parse the JSON response from Ollama
        ollama_response = response.json()
        
        # Extract the generated text from the response
        # Ollama returns the model output in the "response" field
        reply = ollama_response.get("response", "No response from model")
        
    except requests.exceptions.ConnectionError:
        # Handle case when Ollama server is not running or not accessible
        return jsonify({'error': 'Unable to connect to Ollama server. Make sure it is running on localhost:11434'}), 503
    except requests.exceptions.RequestException as e:
        # Handle other request errors (timeout, invalid response, etc.)
        return jsonify({'error': f'Ollama API error: {str(e)}'}), 500
    
    # Return the model's response as JSON
    return jsonify({'reply': reply})


if __name__ == '__main__':
    # Run the Flask development server
    # debug=True enables auto-reload and detailed error messages
    app.run(host='0.0.0.0', port=5000, debug=True)
