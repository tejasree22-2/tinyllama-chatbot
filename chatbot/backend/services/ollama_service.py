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
- Support streaming responses for real-time output
"""

import json
import requests

# ==============================================================================
# Configuration
# ==============================================================================

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "tinyllama"
MAX_HISTORY = 10

conversation_history = []


def generate_response(prompt: str) -> str:
    """
    Generate a non-streaming response from Ollama.
    
    Args:
        prompt: The user's input message
        
    Returns:
        The complete assistant response as a string
    """
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


def generate_streaming_response(prompt: str):
    """
    Generate a streaming response from Ollama using Server-Sent Events (SSE).
    
    This function enables real-time streaming of the LLM's output token by token,
    providing a better user experience as users can see the response as it's
    being generated rather than waiting for the complete response.
    
    How Streaming Works:
    1. Set stream=True in the API request to enable chunked transfer encoding
    2. Use requests with stream=True to handle the response incrementally
    3. Parse each line of the response as it's received (Ollama uses newline-delimited JSON)
    4. Extract the "response" field from each chunk and yield it to the caller
    5. Accumulate the full response for conversation history
    
    Args:
        prompt: The user's input message
        
    Yields:
        String chunks of the assistant's response as they are generated
        
    Note:
        - The final chunk will have "done": true in the JSON payload
        - We accumulate the full response to maintain conversation history
        - Each chunk may contain multiple tokens, not just single characters
    """
    global conversation_history

    # Add user message to conversation history
    conversation_history.append(f"User: {prompt}")
    
    # Build context from recent conversation history (limited to MAX_HISTORY)
    context = "\n".join(conversation_history[-MAX_HISTORY:])
    full_prompt = f"{context}\nAssistant:"
    
    # Configure the payload for streaming
    # stream=True tells Ollama to use chunked transfer encoding
    # This sends the response in small pieces as they're generated
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": True  # Enable streaming mode
    }

    # Make the API request with stream=True
    # This keeps the connection open and allows reading response chunks incrementally
    # Without stream=True, requests would wait for the entire response before returning
    response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
    response.raise_for_status()
    
    # Initialize accumulator for the complete response
    # We need this to maintain conversation history after streaming completes
    full_response = ""
    
    # Process the streaming response line by line
    # Ollama returns each chunk as a separate JSON object on its own line
    # Format: {"response": "partial text", "done": false}\n{"response": "more text", "done": false}\n...
    for line in response.iter_lines():
        if line:
            # Decode the line from bytes to string
            line_text = line.decode('utf-8')
            
            # Parse the JSON chunk
            # Each chunk contains a "response" field with incremental text
            # and a "done" field indicating if generation is complete
            chunk = json.loads(line_text)
            
            # Extract the response text from this chunk
            chunk_text = chunk.get("response", "")
            
            # Yield the chunk to the caller for real-time streaming
            # This allows the client to receive tokens as they're generated
            if chunk_text:
                full_response += chunk_text
                yield chunk_text
            
            # Check if the stream is complete
            # done=True signals the final chunk - we can stop reading
            if chunk.get("done", False):
                break
    
    # Add the complete assistant response to conversation history
    # This maintains context for future conversations
    conversation_history.append(f"Assistant: {full_response}")
    
    # Trim conversation history to prevent unbounded growth
    # Keep only the most recent MAX_HISTORY messages
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]


def clear_conversation_history():
    """
    Clear the conversation history.
    
    Useful when starting a new conversation or session.
    """
    global conversation_history
    conversation_history = []
