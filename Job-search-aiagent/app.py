## 8. `app.py`
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from agent import agent
from tools.resume_tool import parse_pd

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
async def search(
    profile_type: str = Form(...),
    file: UploadFile = File(None),
    link: str = Form(None),
    location: str = Form(None),
    job_type: str = Form("remote")
):
    # Step 1: Ingest text via appropriate tool
    if profile_type == "cv" and file:
        profile_text = parse_pdf(file)
    elif profile_type == "github" and link:
        profile_text = fetch_github(link)
    elif profile_type == "linkedin" and link:
        profile_text = fetch_linkedin(link)
    else:
        return {"error": "Provide a CV file or GitHub/LinkedIn link."}

    # Step 2: Build prompt
    prompt = (
        f"User Profile Text:\n{profile_text}\n"
        f"### Task: Extract top skills and find {job_type} jobs in {location or 'anywhere'}."
    )

    # Step 3: Let the agent orchestrate tools
    result = agent.run(prompt)

    return {"result": result}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000