"""
Flask backend server for the TinyLlama chatbot.
Provides a simple chat endpoint for processing user messages.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing) to allow frontend to communicate with backend
# This allows requests from different origins (domains, ports, or protocols)
CORS(app)


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
    
    # Extract the user's message
    user_message = data['message']
    
    # Temporary response - replace with actual chatbot logic later
    reply = f"You said: {user_message}. This is a temporary response."
    
    # Return the response as JSON
    return jsonify({'reply': reply})


if __name__ == '__main__':
    # Run the Flask development server
    # debug=True enables auto-reload and detailed error messages
    app.run(host='0.0.0.0', port=5000, debug=True)
