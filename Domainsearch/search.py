import os
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import ChatOpenAI



# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Define search tool
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for answering questions about current events or the internet"
    )
]

# Initialize ReAct-style agent
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Streamlit app
st.title("🔎 AI Domain Search Agent")

query = st.text_input("Enter your domain-related question")

if query:
    with st.spinner("Thinking..."):
        response = agent.run(query)
        st.success(response)


