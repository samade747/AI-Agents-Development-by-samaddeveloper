import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, initialize_agent

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not OPENAI_API_KEY or not SERPAPI_API_KEY:
    st.error("Please set OPENAI_API_KEY and SERPAPI_API_KEY in your environment or .env file")
    st.stop()

# Initialize Chat Model with OpenAI API key
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY
)

# Initialize SerpAPI Wrapper with API key
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)

# Define a tool that will perform web search
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for searching domain name availability and related info"
    )
]

# Initialize agent with tools and LLM
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Streamlit UI
st.title("Domain Name Search AI")

domain_input = st.text_input("Enter domain name or keyword to search:")

if st.button("Search"):
    if not domain_input.strip():
        st.warning("Please enter a domain or keyword to search.")
    else:
        with st.spinner("Searching..."):
            # Use the agent to search about domain availability and info
            query = f"Is the domain '{domain_input}' available? Provide details and suggestions."
            result = agent.run(query)
            st.write(result)
