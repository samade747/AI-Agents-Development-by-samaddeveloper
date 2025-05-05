# app.py

from typing import List, Dict, Optional, Tuple
import os
import tempfile
from datetime import datetime
import uuid
import asyncio

import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from fastembed import TextEmbedding
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
from agents import Agent, Runner

load_dotenv()

COLLECTION_NAME = "voice-rag-agent"

def init_session_state():
    defaults = {
        "initialized": False,
        "qdrant_url": os.getenv("QDRANT_URL", ""),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY", ""),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "setup_complete": False,
        "client": None,
        "embedding_model": None,
        "processor_agent": None,
        "tts_agent": None,
        "selected_voice": "coral",
        "processed_documents": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def setup_sidebar():
    with st.sidebar:
        st.title("🔧 Configuration")
        st.markdown("---")

        st.session_state.qdrant_url = st.text_input("Qdrant URL", value=st.session_state.qdrant_url, type="password")
        st.session_state.qdrant_api_key = st.text_input("Qdrant API Key", value=st.session_state.qdrant_api_key, type="password")
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")

        st.markdown("---")
        st.markdown("### 🎤 Voice Settings")
        voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]
        st.session_state.selected_voice = st.selectbox("Select Voice", options=voices, index=voices.index(st.session_state.selected_voice))

def setup_qdrant() -> Tuple[QdrantClient, TextEmbedding]:
    if not all([st.session_state.qdrant_url, st.session_state.qdrant_api_key]):
        raise ValueError("Qdrant credentials not provided")

    client = QdrantClient(url=st.session_state.qdrant_url, api_key=st.session_state.qdrant_api_key)
    embedding_model = TextEmbedding()
    embedding_dim = len(list(embedding_model.embed(["test"]))[0])

    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise e

    return client, embedding_model

def process_pdf(file) -> List:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            return splitter.split_documents(documents)
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return []

def store_embeddings(client, model, documents, collection_name):
    for doc in documents:
        embedding = list(model.embed([doc.page_content]))[0]
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={"content": doc.page_content, **doc.metadata}
                )
            ]
        )

def setup_agents(openai_api_key):
    os.environ["OPENAI_API_KEY"] =  openai_api_key

    processor = Agent(
        name="Processor",
        instructions="""You are a helpful documentation assistant...""",
        model="gpt-4o"
    )

    tts = Agent(
        name="TTS",
        instructions="""You are a text-to-speech agent...""",
        model="gpt-4o"
    )

    return processor, tts

async def process_query(query, client, model, collection_name, api_key, voice):
    try:
        st.info("🔍 Generating embedding and retrieving relevant documents...")
        embedding = list(model.embed([query]))[0]
        results = client.query_points(
            collection_name=collection_name,
            query=embedding.tolist(),
            limit=3,
            with_payload=True
        )

        if not results.points:
            raise Exception("No documents found.")

        context = "\n\n".join(
            f"From {res.payload.get('file_name', 'Unknown')}:\n{res.payload.get('content', '')}"
            for res in results.points
        )
        context += f"\n\nUser Query: {query}"

        if not st.session_state.processor_agent:
            st.session_state.processor_agent, st.session_state.tts_agent = setup_agents(api_key)

        response = await Runner.run(st.session_state.processor_agent, context)
        answer = response.final_output

        tts_response = await Runner.run(st.session_state.tts_agent, answer)
        voice_instruction = tts_response.final_output

        async_openai = AsyncOpenAI(api_key=api_key)

        async with async_openai.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=answer,
            instructions=voice_instruction,
            response_format="pcm"
        ) as stream:
            await LocalAudioPlayer().play(stream)

        mp3 = await async_openai.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=answer,
            instructions=voice_instruction,
            response_format="mp3"
        )

        audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
        with open(audio_path, "wb") as f:
            f.write(mp3.content)

        return {"status": "success", "text_response": answer, "audio_path": audio_path, "sources": [r.payload.get("file_name", "Unknown") for r in results.points]}

    except Exception as e:
        st.error(f"Query processing failed: {str(e)}")
        return {"status": "error", "error": str(e)}

def main():
    st.set_page_config(page_title="🎙️ Voice RAG Agent", layout="wide")
    init_session_state()
    setup_sidebar()

    st.title("🎙️ Voice RAG Agent")
    st.markdown("Upload PDF docs and ask voice-powered questions.")

    file = st.file_uploader("📄 Upload a PDF", type="pdf")
    if file and file.name not in st.session_state.processed_documents:
        with st.spinner("Processing PDF..."):
            if not st.session_state.client:
                client, model = setup_qdrant()
                st.session_state.client = client
                st.session_state.embedding_model = model

            docs = process_pdf(file)
            if docs:
                store_embeddings(st.session_state.client, st.session_state.embedding_model, docs, COLLECTION_NAME)
                st.session_state.processed_documents.append(file.name)
                st.success(f"Uploaded: {file.name}")
                st.session_state.setup_complete = True

    if st.session_state.processed_documents:
        st.sidebar.subheader("📂 Documents")
        for doc in st.session_state.processed_documents:
            st.sidebar.write(f"📄 {doc}")

    query = st.text_input("Ask a question:", placeholder="e.g., How to authenticate API?")
    if query and st.session_state.setup_complete:
        with st.status("Processing...", expanded=True):
            result = asyncio.run(process_query(query, st.session_state.client, st.session_state.embedding_model, COLLECTION_NAME, st.session_state.openai_api_key, st.session_state.selected_voice))
            if result["status"] == "success":
                st.subheader("📝 Answer:")
                st.write(result["text_response"])

                if result.get("audio_path"):
                    st.subheader("🔊 Listen")
                    st.audio(result["audio_path"])
                    with open(result["audio_path"], "rb") as audio_file:
                        st.download_button("Download MP3", data=audio_file, file_name="response.mp3", mime="audio/mp3")

                st.subheader("📚 Sources")
                for src in result["sources"]:
                    st.markdown(f"- {src}")

if __name__ == "__main__":
    main()



# import os
# import tempfile
# from datetime import datetime
# from typing import List

# import streamlit as st
# from dotenv import load_dotenv
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams, PointStruct
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from fastembed import TextEmbedding
# from openai import AsyncOpenAI
# from fpdf import FPDF
# from pydub import AudioSegment

# # Load environment variables
# load_dotenv()

# # Constants
# COLLECTION_NAME = "voice-rag-agent"

# # Streamlit setup
# st.set_page_config(page_title="Voice RAG Agent", layout="wide")
# st.title("🧠 Voice RAG Agent")

# # Session state
# if "openai_api_key" not in st.session_state:
#     st.session_state.openai_api_key = ""
# if "qdrant_client" not in st.session_state:
#     st.session_state.qdrant_client = None

# # Initialize Qdrant client
# def init_qdrant():
#     if not st.session_state.qdrant_client:
#         st.session_state.qdrant_client = QdrantClient(path=".qdrant")
#     try:
#         st.session_state.qdrant_client.get_collection(COLLECTION_NAME)
#     except:
#         st.session_state.qdrant_client.create_collection(
#             COLLECTION_NAME,
#             vectors_config=VectorParams(size=384, distance=Distance.COSINE)
#         )

# # Load and split PDF
# def load_and_split_pdf(uploaded_file):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         tmp_file.flush()
#         loader = PyPDFLoader(tmp_file.name)
#         documents = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         return text_splitter.split_documents(documents)

# # Embed and store in Qdrant
# def embed_and_store(docs):
#     embed_model = TextEmbedding()
#     client = st.session_state.qdrant_client
#     vectors = embed_model.embed([doc.page_content for doc in docs])
#     points = [
#         PointStruct(
#             id=i,
#             vector=vector,
#             payload={"content": docs[i].page_content, "file_name": docs[i].metadata.get("source", "")}
#         )
#         for i, vector in enumerate(vectors)
#     ]
#     client.upsert(collection_name=COLLECTION_NAME, points=points)

# # Search relevant documents
# def search_docs(query):
#     embed_model = TextEmbedding()
#     query_vector = embed_model.embed_query(query)
#     results = st.session_state.qdrant_client.search(
#         collection_name=COLLECTION_NAME,
#         query_vector=query_vector,
#         limit=5,
#         with_payload=True
#     )
#     return results

# # Transcribe audio using Whisper
# def transcribe_audio(file, openai_api_key: str) -> str:
#     try:
#         async_openai = AsyncOpenAI(api_key=openai_api_key)
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
#             tmp.write(file.read())
#             tmp.flush()
#             audio_file = open(tmp.name, "rb")
#             transcript = asyncio.run(
#                 async_openai.audio.transcriptions.create(file=audio_file, model="whisper-1")
#             )
#             return transcript.text
#     except Exception as e:
#         st.error(f"❌ Whisper transcription error: {str(e)}")
#         return ""

# # Generate PDF from Q&A
# def generate_pdf(question: str, answer: str, sources: List[str]) -> bytes:
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.multi_cell(0, 10, f"🧠 Question:\n{question}\n")
#     pdf.multi_cell(0, 10, f"💬 Answer:\n{answer}\n")
#     pdf.multi_cell(0, 10, "📁 Sources:\n" + "\n".join(sources))
#     pdf_bytes = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
#     pdf.output(pdf_bytes.name)
#     with open(pdf_bytes.name, "rb") as f:
#         return f.read()

# # Main logic
# init_qdrant()

# st.sidebar.header("🔐 API Keys")
# openai_api_key_input = st.sidebar.text_input("OpenAI API Key", type="password")
# if openai_api_key_input:
#     st.session_state.openai_api_key = openai_api_key_input

# st.markdown("## 📤 Upload a PDF")
# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
# if uploaded_file and st.session_state.openai_api_key:
#     docs = load_and_split_pdf(uploaded_file)
#     embed_and_store(docs)
#     st.success("✅ Document embedded and stored in vector DB")

# st.markdown("## ❓ Ask a Question")
# query = st.text_input("Type your question here:")

# st.markdown("### 🎙️ Or ask with your voice")
# audio_file = st.file_uploader("Upload your question (MP3 only)", type=["mp3"])
# if audio_file:
#     query = transcribe_audio(audio_file, st.session_state.openai_api_key)
#     st.success(f"🎤 Transcribed: {query}")

# if query and st.session_state.openai_api_key:
#     search_results = search_docs(query)
#     context = "\n\n".join([doc.payload.get("content", "") for doc in search_results if doc.payload])

#     st.markdown("### 🔍 Top Matching Documents (with confidence):")
#     for i, result in enumerate(search_results, 1):
#         payload = result.payload
#         if not payload:
#             continue
#         content = payload.get("content", "")[:300]
#         score = result.score if hasattr(result, 'score') else "N/A"
#         source = payload.get('file_name', 'Unknown Source')
#         st.markdown(f"**{i}. {source}** — Score: `{score:.2f}`")
#         st.caption(content + "...")

#     # Generate response
#     openai_client = AsyncOpenAI(api_key=st.session_state.openai_api_key)
#     response = asyncio.run(
#         openai_client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
#             ]
#         )
#     )
#     answer = response.choices[0].message.content
#     st.markdown("### 💬 Answer:")
#     st.write(answer)

#     # Export as PDF
#     pdf_bytes = generate_pdf(query, answer, [doc.payload.get("file_name", "") for doc in search_results])
#     st.download_button(
#         label="📄 Download Q&A as PDF",
#         data=pdf_bytes,
#         file_name="response.pdf",
#         mime="application/pdf"
#     )
