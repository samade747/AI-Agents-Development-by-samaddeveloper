import chainlit as cl
from agents.researcher_agent import create_researcher_agent
from agents.planner_agent import create_planner_agent
from tools.search_tool import search_destination_info
from tools.itinerary_tool import create_itinerary

# Define Chainlit user message handling
@cl.on_message
async def main(message: str):
    # Set up the SerpAPI key from the environment or input
    serp_api_key = "your_serpapi_key"  # You can make this dynamic as per your requirements

    # Create the researcher and planner agents
    researcher_agent = create_researcher_agent(serp_api_key)
    planner_agent = create_planner_agent()

    # Parse the user's message (destination and days)
    try:
        destination, num_days = message.split(",")  # Expecting format: "Paris, 7"
        num_days = int(num_days.strip())
    except ValueError:
        await cl.Message(content="Please provide the destination and number of days in the format 'Destination, Days' (e.g., Paris, 7).").send()
        return

    # Step 1: Use the Researcher agent to gather information
    research_data = researcher_agent.run(destination.strip())

    # Step 2: Use the Planner agent to create an itinerary based on the research data
    itinerary = planner_agent.run(research_data, num_days)

    # Send the itinerary back to the user
    await cl.Message(content=itinerary).send()

# Start the Chainlit server
if __name__ == "__main__":
    cl.run()
