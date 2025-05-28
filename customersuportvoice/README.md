# Voice-to-Voice AI Assistant

🎤 Speak to your AI assistant and hear it talk back!

## Features
- Whisper for STT
- OpenAI GPT (or fallback to OpenRouter) for NLP
- Coqui TTS for natural responses
- Streamlit for UI

## Setup
```bash
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
streamlit run app.py
```

> Add your OpenAI and/or OpenRouter key in `.env`
