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
MAX_HISTORY = 10

conversation_history = []


def generate_response(prompt: str) -> str:
    global conversation_history

    conversation_history.append(f"User: {prompt}")
    
    context = "\n".join(conversation_history[-MAX_HISTORY:])
    full_prompt = f"{context}\nAssistant:"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_API_URL, json=payload)
    response.raise_for_status()

    ollama_response = response.json()
    assistant_reply = ollama_response.get("response", "No response from model")
    
    conversation_history.append(f"Assistant: {assistant_reply}")
    
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]
    
    return assistant_reply
