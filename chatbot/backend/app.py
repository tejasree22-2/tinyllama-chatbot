"""
Flask Application Entry Point.

This module initializes the Flask application and registers all blueprints.
It serves as the composition root where infrastructure and routes are wired together.

Application Structure (Clean Architecture):
    1. app.py         - Flask initialization and route registration
    2. controllers/   - HTTP request/response handling
    3. services/      - Business logic and external API integrations
"""

from flask import Flask
from flask_cors import CORS
from controllers.chat_controller import chat_bp


# ==============================================================================
# Application Factory
# ==============================================================================

def create_app():
    """
    Create and configure the Flask application.
    
    This factory pattern allows for better testability and configuration management.
    
    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)
    
    # Enable CORS (Cross-Origin Resource Sharing) to allow frontend to communicate with backend
    # This allows requests from different origins (domains, ports, or protocols)
    CORS(app)
    
    # Register blueprints (controllers)
    # Each blueprint handles a specific set of routes
    app.register_blueprint(chat_bp)
    
    return app


# ==============================================================================
# Application Instance
# ==============================================================================

# Create the Flask application instance
app = create_app()


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == '__main__':
    # Run the Flask development server
    # debug=True enables auto-reload and detailed error messages
    app.run(host='0.0.0.0', port=5000, debug=True)
