from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
from textwrap import dedent

def create_researcher_agent(serp_api_key):
    return Agent(
        name="Researcher",
        role="Searches for travel destinations, activities, and accommodations based on user preferences",
        model="llama3.2",  # You can replace this with your desired model
        description=dedent(
            """\
            You are a world-class travel researcher. Given a travel destination and the number of days the user wants to travel for,
            generate a list of search terms for finding relevant travel activities and accommodations.
            Then search the web for each term, analyze the results, and return the 10 most relevant results.
            """
        ),
        instructions=[
            "Given a travel destination and the number of days the user wants to travel for, first generate a list of 3 search terms related to that destination and the number of days.",
            "For each search term, search Google and analyze the results.",
            "From the results of all searches, return the 10 most relevant results to the user's preferences.",
            "Remember: the quality of the results is important."
        ],
        tools=[SerpApiTools(api_key=serp_api_key)],
        add_datetime_to_instructions=True,
    )
