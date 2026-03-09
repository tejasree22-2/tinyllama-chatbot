"""
Ollama Service Layer.

This module is responsible for all interactions with the Ollama LLM API.
It abstracts the external API calls away from the business logic, providing
a clean interface for the controller layer.

Responsibilities:
- Communicate with Ollama API (http://localhost:11434/api/generate)
- Handle API configuration (model selection)
- Manage connection errors and API failures
- Return parsed responses to the controller
"""

import requests

# ==============================================================================
# Configuration
# ==============================================================================

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "tinyllama"


# ==============================================================================
# Service Functions
# ==============================================================================

def generate_response(prompt: str) -> str:
    """
    Send a prompt to the Ollama API and get the generated response.

    Args:
        prompt: The user message/prompt to send to the LLM.

    Returns:
        The generated text response from the model.

    Raises:
        ConnectionError: If unable to connect to Ollama server.
        RequestException: If the API request fails.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_API_URL, json=payload)
    response.raise_for_status()

    ollama_response = response.json()
    return ollama_response.get("response", "No response from model")
