from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
from agno.models.ollama import Ollama
from textwrap import dedent

def create_researcher_agent(api_key):
    return Agent(
        name="Researcher",
        role="Travel destination researcher",
        model=Ollama(id="llama3"),  # ✅ Removed unsupported kwargs
        description=dedent("""\
            You are a world-class travel researcher. 
            Generate search queries and gather top activities and accommodations.
        """),
        instructions=[
            "Generate 3 travel-related search queries for the destination and trip duration.",
            "Use `search_google` to gather info.",
            "Return a summary of the top 10 results.",
        ],
        tools=[SerpApiTools(api_key=api_key)],
        add_datetime_to_instructions=True,
    )
