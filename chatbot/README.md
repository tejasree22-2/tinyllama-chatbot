# TinyLlama Local Chatbot

A lightweight, privacy-focused chatbot that runs entirely on your local machine. Powered by TinyLlama through Ollama, with a Flask backend and simple HTML interface.

## Overview

TinyLlama Local Chatbot is a beginner-friendly project that demonstrates how to build a local AI chatbot. It processes all conversations locally—no data leaves your computer.

### Key Features

- **Privacy First**: All AI processing happens locally; no external API calls
- **Lightweight Model**: TinyLlama is a compact 1.1B parameter model
- **Simple Interface**: Clean, minimal HTML chat UI
- **Easy to Extend**: Modular Flask architecture

## Architecture Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   HTML Chat     │────▶│   Flask API      │────▶│   Ollama        │
│   Interface     │     │   (Python)       │     │   (TinyLlama)  │
│                 │◀────│                  │◀────│                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

1. User sends a message via the web interface
2. Flask backend receives the request
3. Backend forwards the message to Ollama running TinyLlama
4. Ollama generates a response
5. Response is returned to the frontend and displayed in the chat

## Project Structure

```
tinyllama-chatbot/
├── README.md                 # This file
├── app.py                    # Flask application (API routes)
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Chat interface
└── static/
    └── style.css            # Styling (optional)
```

### File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Flask server handling API endpoints and Ollama communication |
| `requirements.txt` | Python packages needed (Flask, requests) |
| `templates/index.html` | User-facing chat interface |
| `static/style.css` | CSS styling for the interface |

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai) installed and running
- TinyLlama model pulled locally

## Installation Steps

### 1. Install Ollama

Follow instructions at [https://ollama.ai](https://ollama.ai) for your operating system.

### 2. Pull TinyLlama Model

```bash
ollama pull tinyllama
```

### 3. Start Ollama Service

```bash
ollama serve
```

Keep this terminal window open (or run in background).

### 4. Clone and Setup Project

```bash
# Navigate to project directory
cd tinyllama-chatbot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

### 6. Open in Browser

Navigate to: `http://localhost:5000`

You should see the chat interface. Start typing messages and press Enter or click Send to chat with TinyLlama!

## Usage

1. Ensure Ollama is running (`ollama serve`)
2. Start the Flask app: `python app.py`
3. Open `http://localhost:5000` in your browser
4. Type your message and press Enter
5. Wait for TinyLlama to generate a response

## API Endpoint

The Flask backend exposes a single endpoint:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send a message and receive AI response |

**Request:**
```json
{
  "message": "Hello, how are you?"
}
```

**Response:**
```json
{
  "response": "Hello! I'm doing well, thank you for asking..."
}
```

## Configuration

In `app.py`, you can modify:

- `OLLAMA_BASE_URL`: Default `http://localhost:11434`
- `MODEL_NAME`: Default `tinyllama`
- `FLASK_PORT`: Default `5000`

## Troubleshooting

### Ollama not running
```
Error: Connection refused to localhost:11434
```
**Solution**: Ensure `ollama serve` is running in a terminal.

### Model not found
```
Error: model 'tinyllama' not found
```
**Solution**: Run `ollama pull tinyllama` to download the model.

### Port already in use
```
Error: Port 5000 is already in use
```
**Solution**: Change the port in `app.py` or stop the conflicting process.

## Future Improvements

- [ ] Add streaming responses for real-time chat feel
- [ ] Implement chat history persistence
- [ ] Add multiple model selection (Llama2, Mistral, etc.)
- [ ] Add system prompt customization
- [ ] Include streaming token display
- [ ] Add loading indicators during generation
- [ ] Implement conversation export/import
- [ ] Add voice input support
- [ ] Mobile-responsive design improvements

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| AI Model | TinyLlama via Ollama |
| Runtime | Ollama |

## License

MIT License - Feel free to use and modify for your projects.

## Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [TinyLlama Model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
