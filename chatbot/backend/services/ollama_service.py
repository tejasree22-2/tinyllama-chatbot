"""
Ollama Service Layer.

This module is responsible for all interactions with the Ollama LLM API.
It abstracts the external API calls away from the business logic, providing
a clean interface for the controller layer.

Responsibilities:
- Communicate with Ollama API
- Handle API configuration (model selection)
- Manage connection errors and API failures
- Return parsed responses to the controller
- Support streaming responses for real-time output
"""

import json
import logging
import requests
from typing import Optional

import config

logger = logging.getLogger(__name__)


class OllamaAPIError(Exception):
    """Custom exception for Ollama API errors."""
    pass


class OllamaConnectionError(Exception):
    """Custom exception for Ollama connection errors."""
    pass


conversation_history = []


def generate_response(prompt: str) -> str:
    """
    Generate a non-streaming response from Ollama.
    
    Args:
        prompt: The user's input message
        
    Returns:
        The complete assistant response as a string
        
    Raises:
        OllamaConnectionError: If unable to connect to Ollama server
        OllamaAPIError: If the API returns an error
    """
    global conversation_history

    logger.info(f"Generating response for prompt: {prompt[:50]}...")
    
    conversation_history.append(f"User: {prompt}")
    
    context = "\n".join(conversation_history[-config.MAX_HISTORY:])
    full_prompt = f"{context}\nAssistant:"
    
    payload = {
        "model": config.MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    }

    try:
        response = requests.post(config.OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to Ollama server: {e}")
        raise OllamaConnectionError(f"Unable to connect to Ollama server at {config.OLLAMA_API_URL}") from e
    except requests.exceptions.Timeout as e:
        logger.error(f"Request to Ollama timed out: {e}")
        raise OllamaAPIError("Request timed out") from e
    except requests.exceptions.HTTPError as e:
        logger.error(f"Ollama API returned HTTP error: {e}")
        raise OllamaAPIError(f"API error: {e}") from e
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to Ollama failed: {e}")
        raise OllamaAPIError(f"Request failed: {e}") from e

    try:
        ollama_response = response.json()
        assistant_reply = ollama_response.get("response", "No response from model")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Ollama response: {e}")
        raise OllamaAPIError("Invalid response format from Ollama") from e
    
    logger.info(f"Response generated successfully ({len(assistant_reply)} chars)")
    
    conversation_history.append(f"Assistant: {assistant_reply}")
    
    if len(conversation_history) > config.MAX_HISTORY:
        conversation_history = conversation_history[-config.MAX_HISTORY:]
    
    return assistant_reply


def generate_streaming_response(prompt: str):
    """
    Generate a streaming response from Ollama using Server-Sent Events (SSE).
    
    This function enables real-time streaming of the LLM's output token by token,
    providing a better user experience as users can see the response as it's
    being generated rather than waiting for the complete response.
    
    Args:
        prompt: The user's input message
        
    Yields:
        String chunks of the assistant's response as they are generated
        
    Raises:
        OllamaConnectionError: If unable to connect to Ollama server
        OllamaAPIError: If the API returns an error
    """
    global conversation_history

    logger.info(f"Generating streaming response for prompt: {prompt[:50]}...")
    
    conversation_history.append(f"User: {prompt}")
    
    context = "\n".join(conversation_history[-config.MAX_HISTORY:])
    full_prompt = f"{context}\nAssistant:"
    
    payload = {
        "model": config.MODEL_NAME,
        "prompt": full_prompt,
        "stream": True
    }

    try:
        response = requests.post(config.OLLAMA_API_URL, json=payload, stream=True, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to Ollama server: {e}")
        raise OllamaConnectionError(f"Unable to connect to Ollama server at {config.OLLAMA_API_URL}") from e
    except requests.exceptions.Timeout as e:
        logger.error(f"Request to Ollama timed out: {e}")
        raise OllamaAPIError("Request timed out") from e
    except requests.exceptions.HTTPError as e:
        logger.error(f"Ollama API returned HTTP error: {e}")
        raise OllamaAPIError(f"API error: {e}") from e
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to Ollama failed: {e}")
        raise OllamaAPIError(f"Request failed: {e}") from e
    
    full_response = ""
    
    try:
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                chunk = json.loads(line_text)
                chunk_text = chunk.get("response", "")
                
                if chunk_text:
                    full_response += chunk_text
                    yield chunk_text
                
                if chunk.get("done", False):
                    break
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Ollama streaming response: {e}")
        raise OllamaAPIError("Invalid response format from Ollama") from e
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during streaming: {e}")
        raise OllamaAPIError(f"Streaming error: {e}") from e
    
    logger.info(f"Streaming response completed ({len(full_response)} chars)")
    
    conversation_history.append(f"Assistant: {full_response}")
    
    if len(conversation_history) > config.MAX_HISTORY:
        conversation_history = conversation_history[-config.MAX_HISTORY:]


def clear_conversation_history():
    """
    Clear the conversation history.
    
    Useful when starting a new conversation or session.
    """
    global conversation_history
    logger.info("Clearing conversation history")
    conversation_history = []


def generate_response_with_rag(prompt: str, use_rag: bool = True, top_k: Optional[int] = None) -> str:
    """
    Generate a response from Ollama with optional RAG context.
    
    Args:
        prompt: The user's input message
        use_rag: Whether to use RAG for context retrieval
        top_k: Number of context chunks to retrieve
        
    Returns:
        The complete assistant response as a string
    """
    global conversation_history

    logger.info(f"Generating RAG-enabled response for prompt: {prompt[:50]}...")
    
    context_parts = []
    
    if use_rag:
        try:
            from services import rag_service
            context_docs = rag_service.retrieve_context(prompt, top_k)
            if context_docs:
                context_parts.append("Relevant context from documents:")
                for i, doc in enumerate(context_docs, 1):
                    context_parts.append(f"[{i}] {doc['text']}")
                context_parts.append("")
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}. Proceeding without context.")
    
    conversation_history.append(f"User: {prompt}")
    
    context = "\n".join(conversation_history[-config.MAX_HISTORY:])
    
    full_prompt_parts = []
    if context_parts:
        full_prompt_parts.append("\n".join(context_parts))
    if context:
        full_prompt_parts.append(f"Conversation history:\n{context}")
    full_prompt_parts.append("User question: " + prompt)
    full_prompt_parts.append("Assistant:")
    
    full_prompt = "\n\n".join(full_prompt_parts)
    
    payload = {
        "model": config.MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    }

    try:
        response = requests.post(config.OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to Ollama server: {e}")
        raise OllamaConnectionError(f"Unable to connect to Ollama server at {config.OLLAMA_API_URL}") from e
    except requests.exceptions.Timeout as e:
        logger.error(f"Request to Ollama timed out: {e}")
        raise OllamaAPIError("Request timed out") from e
    except requests.exceptions.HTTPError as e:
        logger.error(f"Ollama API returned HTTP error: {e}")
        raise OllamaAPIError(f"API error: {e}") from e
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to Ollama failed: {e}")
        raise OllamaAPIError(f"Request failed: {e}") from e

    try:
        ollama_response = response.json()
        assistant_reply = ollama_response.get("response", "No response from model")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Ollama response: {e}")
        raise OllamaAPIError("Invalid response format from Ollama") from e
    
    logger.info(f"RAG response generated successfully ({len(assistant_reply)} chars)")
    
    conversation_history.append(f"Assistant: {assistant_reply}")
    
    if len(conversation_history) > config.MAX_HISTORY:
        conversation_history = conversation_history[-config.MAX_HISTORY:]
    
    return assistant_reply


def generate_streaming_response_with_rag(prompt: str, use_rag: bool = True, top_k: Optional[int] = None):
    """
    Generate a streaming response from Ollama with optional RAG context.
    
    Args:
        prompt: The user's input message
        use_rag: Whether to use RAG for context retrieval
        top_k: Number of context chunks to retrieve
        
    Yields:
        String chunks of the assistant's response as they are generated
    """
    global conversation_history

    logger.info(f"Generating streaming RAG response for prompt: {prompt[:50]}...")
    
    context_parts = []
    
    if use_rag:
        try:
            from services import rag_service
            context_docs = rag_service.retrieve_context(prompt, top_k)
            if context_docs:
                context_parts.append("Relevant context from documents:")
                for i, doc in enumerate(context_docs, 1):
                    context_parts.append(f"[{i}] {doc['text']}")
                context_parts.append("")
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}. Proceeding without context.")
    
    conversation_history.append(f"User: {prompt}")
    
    context = "\n".join(conversation_history[-config.MAX_HISTORY:])
    
    full_prompt_parts = []
    if context_parts:
        full_prompt_parts.append("\n".join(context_parts))
    if context:
        full_prompt_parts.append(f"Conversation history:\n{context}")
    full_prompt_parts.append("User question: " + prompt)
    full_prompt_parts.append("Assistant:")
    
    full_prompt = "\n\n".join(full_prompt_parts)
    
    payload = {
        "model": config.MODEL_NAME,
        "prompt": full_prompt,
        "stream": True
    }

    try:
        response = requests.post(config.OLLAMA_API_URL, json=payload, stream=True, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to Ollama server: {e}")
        raise OllamaConnectionError(f"Unable to connect to Ollama server at {config.OLLAMA_API_URL}") from e
    except requests.exceptions.Timeout as e:
        logger.error(f"Request to Ollama timed out: {e}")
        raise OllamaAPIError("Request timed out") from e
    except requests.exceptions.HTTPError as e:
        logger.error(f"Ollama API returned HTTP error: {e}")
        raise OllamaAPIError(f"API error: {e}") from e
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to Ollama failed: {e}")
        raise OllamaAPIError(f"Request failed: {e}") from e
    
    full_response = ""
    
    try:
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                chunk = json.loads(line_text)
                chunk_text = chunk.get("response", "")
                
                if chunk_text:
                    full_response += chunk_text
                    yield chunk_text
                
                if chunk.get("done", False):
                    break
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Ollama streaming response: {e}")
        raise OllamaAPIError("Invalid response format from Ollama") from e
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during streaming: {e}")
        raise OllamaAPIError(f"Streaming error: {e}") from e
    
    logger.info(f"Streaming RAG response completed ({len(full_response)} chars)")
    
    conversation_history.append(f"Assistant: {full_response}")
    
    if len(conversation_history) > config.MAX_HISTORY:
        conversation_history = conversation_history[-config.MAX_HISTORY:]
