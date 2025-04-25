## 7. `agent.py`

import os, json
from dotenv import load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool

# Load keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENROUTER_API_BASE")
openai.api_type = "open_router"

# Import tool funcs
from tools.resume_tool import parse_pdf
from tools.github_tool import fetch_github
from tools.linkedin_tool import fetch_linkedin
from tools.job_search_tool import search_jobs

# Wrap as LangChain Tool objects
tools = [
    Tool(
        name="parse_pdf",
        func=parse_pdf,
        description="Parse PDF resume and return plain text"
    ),
    Tool(
        name="fetch_github",
        func=fetch_github,
        description="Fetch GitHub profile bio and languages"
    ),
    Tool(
        name="fetch_linkedin",
        func=fetch_linkedin,
        description="Fetch LinkedIn summary and skills"
    ),
    Tool(
        name="search_jobs",
        func=search_jobs,
        description="Search jobs by skills, location, and job type"
    ),
]

# Initialize Chat Agent (zero-shot React description)
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    openai_api_base=os.getenv("OPENROUTER_API_BASE"),
    openai_api_type="open_router",
)

agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True
)

