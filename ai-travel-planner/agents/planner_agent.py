from agno.agent import Agent
from agno.models.ollama import Ollama
from textwrap import dedent

def create_planner_agent():
    return Agent(
        name="Planner",
        role="Creates travel itineraries",
        model=Ollama(id="llama3"),  # ✅ Removed unsupported kwargs
        description=dedent("""\
            You are a travel planner. Create detailed itineraries based on research data.
        """),
        instructions=[
            "Take research input and days.",
            "Output a well-structured, engaging itinerary.",
        ],
        add_datetime_to_instructions=True,
    )
