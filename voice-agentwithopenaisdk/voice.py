import os
import tempfile
from datetime import datetime
from typing import List

import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from fastembed import TextEmbedding
from openai import AsyncOpenAI
from fpdf import FPDF
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "voice-rag-agent"

# Streamlit setup
st.set_page_config(page_title="Voice RAG Agent", layout="wide")
st.title("🧠 Voice RAG Agent")

# Session state
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "qdrant_client" not in st.session_state:
    st.session_state.qdrant_client = None

# Initialize Qdrant client
def init_qdrant():
    if not st.session_state.qdrant_client:
        st.session_state.qdrant_client = QdrantClient(path=".qdrant")
    try:
        st.session_state.qdrant_client.get_collection(COLLECTION_NAME)
    except:
        st.session_state.qdrant_client.create_collection(
            COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

# Load and split PDF
def load_and_split_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file.flush()
        loader = PyPDFLoader(tmp_file.name)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)

# Embed and store in Qdrant
def embed_and_store(docs):
    embed_model = TextEmbedding()
    client = st.session_state.qdrant_client
    vectors = embed_model.embed([doc.page_content for doc in docs])
    points = [
        PointStruct(
            id=i,
            vector=vector,
            payload={"content": docs[i].page_content, "file_name": docs[i].metadata.get("source", "")}
        )
        for i, vector in enumerate(vectors)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

# Search relevant documents
def search_docs(query):
    embed_model = TextEmbedding()
    query_vector = embed_model.embed_query(query)
    results = st.session_state.qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5,
        with_payload=True
    )
    return results

# Transcribe audio using Whisper
def transcribe_audio(file, openai_api_key: str) -> str:
    try:
        async_openai = AsyncOpenAI(api_key=openai_api_key)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(file.read())
            tmp.flush()
            audio_file = open(tmp.name, "rb")
            transcript = asyncio.run(
                async_openai.audio.transcriptions.create(file=audio_file, model="whisper-1")
            )
            return transcript.text
    except Exception as e:
        st.error(f"❌ Whisper transcription error: {str(e)}")
        return ""

# Generate PDF from Q&A
def generate_pdf(question: str, answer: str, sources: List[str]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"🧠 Question:\n{question}\n")
    pdf.multi_cell(0, 10, f"💬 Answer:\n{answer}\n")
    pdf.multi_cell(0, 10, "📁 Sources:\n" + "\n".join(sources))
    pdf_bytes = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(pdf_bytes.name)
    with open(pdf_bytes.name, "rb") as f:
        return f.read()

# Main logic
init_qdrant()

st.sidebar.header("🔐 API Keys")
openai_api_key_input = st.sidebar.text_input("OpenAI API Key", type="password")
if openai_api_key_input:
    st.session_state.openai_api_key = openai_api_key_input

st.markdown("## 📤 Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file and st.session_state.openai_api_key:
    docs = load_and_split_pdf(uploaded_file)
    embed_and_store(docs)
    st.success("✅ Document embedded and stored in vector DB")

st.markdown("## ❓ Ask a Question")
query = st.text_input("Type your question here:")

st.markdown("### 🎙️ Or ask with your voice")
audio_file = st.file_uploader("Upload your question (MP3 only)", type=["mp3"])
if audio_file:
    query = transcribe_audio(audio_file, st.session_state.openai_api_key)
    st.success(f"🎤 Transcribed: {query}")

if query and st.session_state.openai_api_key:
    search_results = search_docs(query)
    context = "\n\n".join([doc.payload.get("content", "") for doc in search_results if doc.payload])

    st.markdown("### 🔍 Top Matching Documents (with confidence):")
    for i, result in enumerate(search_results, 1):
        payload = result.payload
        if not payload:
            continue
        content = payload.get("content", "")[:300]
        score = result.score if hasattr(result, 'score') else "N/A"
        source = payload.get('file_name', 'Unknown Source')
        st.markdown(f"**{i}. {source}** — Score: `{score:.2f}`")
        st.caption(content + "...")

    # Generate response
    openai_client = AsyncOpenAI(api_key=st.session_state.openai_api_key)
    response = asyncio.run(
        openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            ]
        )
    )
    answer = response.choices[0].message.content
    st.markdown("### 💬 Answer:")
    st.write(answer)

    # Export as PDF
    pdf_bytes = generate_pdf(query, answer, [doc.payload.get("file_name", "") for doc in search_results])
    st.download_button(
        label="📄 Download Q&A as PDF",
        data=pdf_bytes,
        file_name="response.pdf",
        mime="application/pdf"
    )
