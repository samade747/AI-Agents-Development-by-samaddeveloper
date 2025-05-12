import tempfile
from io import BytesIO

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import openai
from gtts import gTTS

from config import OPENAI_API_KEY
from fallback_router import openrouter_chat

# Step 0: Configure OpenAI key
openai.api_key = OPENAI_API_KEY

# Step 4: Initialize in-memory session history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Step 1: UI – title and instructions
st.title("🔊 AI Voice Agent")
st.markdown("Speak to the agent and receive a spoken reply. Falls back to OpenRouter if OpenAI fails.")

# Step 2: Audio Capture via WebRTC
webrtc_ctx = webrtc_streamer(
    key="voice_agent",
    mode="SENDRECV",
    media_stream_constraints={"audio": True, "video": False},
    audio_receiver_size=1024
)

if webrtc_ctx.audio_receiver:
    frames = webrtc_ctx.audio_receiver.get_frames(timeout=5)
    if frames:
        pcm_bytes = frames[0].to_ndarray().tobytes()

        # Step 2: ASR via Whisper
        def transcribe(audio_bytes: bytes) -> str:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            result = openai.Audio.transcribe(model="whisper-1", file=open(tmp_path, "rb"))
            return result.get("text", "[untranscribable]")

        transcript = transcribe(pcm_bytes)
        st.write("**You said:**", transcript)
        # Append to session history
        st.session_state["history"].append({"role": "user", "content": transcript})

        # Step 3 & 4: NLP & Dialogue Management
        def get_response(user_text: str) -> str:
            messages = [{"role": "system", "content": "You are a helpful voice assistant."}]
            messages += st.session_state["history"]
            messages.append({"role": "user", "content": user_text})
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
                assistant_text = resp.choices[0].message.content
            except Exception as e:
                st.warning(f"OpenAI request failed ({e}), using OpenRouter fallback.")
                assistant_text = openrouter_chat(messages)
            # Append assistant reply to history
            st.session_state["history"].append({"role": "assistant", "content": assistant_text})
            return assistant_text

        reply = get_response(transcript)
        st.write("**Assistant:**", reply)

        # Step 5: Text-to-Speech via gTTS
        def synthesize(text: str) -> bytes:
            tts = gTTS(text)
            buf = BytesIO()
            tts.write_to_fp(buf)
            return buf.getvalue()

        audio_data = synthesize(reply)

        # Step 7: Playback of synthesized speech
        st.audio(audio_data, format="audio/mp3")