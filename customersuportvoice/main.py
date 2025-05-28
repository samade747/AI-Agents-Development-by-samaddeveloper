import openai
import whisper
from TTS.api import TTS
from dotenv import load_dotenv
import os
import requests

load_dotenv()

# Load models
whisper_model = whisper.load_model("base")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

openai.api_key = os.getenv("OPENAI_API_KEY")
openrouter_key = os.getenv("OPENROUTER_API_KEY")

def transcribe(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result['text']

def generate_response(prompt):
    try:
        if openai.api_key:
            return call_openai(prompt)
        elif openrouter_key:
            return call_openrouter(prompt)
        else:
            return local_fallback(prompt)
    except Exception as e:
        print(f"Primary and fallback error: {e}")
        return local_fallback(prompt)

def call_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()

def call_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    }
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

def local_fallback(prompt):
    if "your name" in prompt.lower():
        return "I'm your voice assistant, always online!"
    if "joke" in prompt.lower():
        return "Why don’t scientists trust atoms? Because they make up everything!"
    return "Sorry, I'm having trouble responding now, but I'm still listening!"


def speak_text(text):
    tts.tts_to_file(text=text, file_path="response.wav")
