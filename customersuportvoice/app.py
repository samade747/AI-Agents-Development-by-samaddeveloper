import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
from main import transcribe, generate_response, speak_text
from fpdf import FPDF

st.set_page_config(page_title="Voice-to-Voice AI", layout="centered")
st.title("🗣️ Voice-to-Voice AI Assistant")
st.markdown("Talk to your AI assistant using your microphone.")

qa_pairs = []

duration = st.slider("Recording Duration (seconds)", 2, 10, 5)
start = st.button("🎤 Start Recording")

if start:
    st.info("Recording...")
    fs = 44100
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        write(f.name, fs, recording)
        audio_path = f.name

    st.success("Recording complete.")

    with st.spinner("Transcribing..."):
        text = transcribe(audio_path)
        st.text_area("You said:", text)

    with st.spinner("Thinking..."):
        response = generate_response(text)
        st.text_area("AI says:", response)

    qa_pairs.append((text, response))

    with st.spinner("Speaking..."):
        speak_text(response)
        st.audio("response.wav", format="audio/wav")

if qa_pairs:
    def save_to_pdf(pairs):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for i, (q, a) in enumerate(pairs, 1):
            pdf.multi_cell(0, 10, f"Q{i}: {q}
A{i}: {a}

")
        pdf_path = os.path.join(tempfile.gettempdir(), "conversation.pdf")
        pdf.output(pdf_path)
        return pdf_path

    if st.button("📄 Export Q&A to PDF"):
        pdf_file = save_to_pdf(qa_pairs)
        with open(pdf_file, "rb") as f:
            st.download_button("Download PDF", f, "conversation.pdf", "application/pdf")
