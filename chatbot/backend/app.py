"""
Flask Application Entry Point.

This module initializes the Flask application and registers all blueprints.
It serves as the composition root where infrastructure and routes are wired together.

Application Structure (Clean Architecture):
    1. app.py         - Flask initialization and route registration
    2. controllers/   - HTTP request/response handling
    3. services/      - Business logic and external API integrations
"""

import logging
from flask import Flask
from flask_cors import CORS
from controllers.chat_controller import chat_bp
import config


# ==============================================================================
# Application Factory
# ==============================================================================

logger = logging.getLogger(__name__)

def create_app():
    """
    Create and configure the Flask application.
    
    This factory pattern allows for better testability and configuration management.
    
    Returns:
        Configured Flask application instance.
    """
    logger.info("Starting Flask application")
    
    app = Flask(__name__)
    
    CORS(app)
    
    app.register_blueprint(chat_bp)
    
    logger.info("Flask application initialized successfully")
    
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
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )
