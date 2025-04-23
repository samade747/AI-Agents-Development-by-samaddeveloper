import chainlit as cl
from agents.researcher_agent import create_researcher_agent
from agents.planner_agent import create_planner_agent

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content

    serp_api_key = "your_serpapi_key_here"  # Replace this or fetch from .env

    researcher_agent = create_researcher_agent(serp_api_key)
    planner_agent = create_planner_agent()

    try:
        destination, num_days = user_input.split(",")
        destination = destination.strip()
        num_days = int(num_days.strip())
    except ValueError:
        await cl.Message(content="❌ Format: `Destination, Days` (e.g., `Paris, 7`)").send()
        return

    await cl.Message(content=f"🔍 Researching travel options for **{destination}**...").send()
    
    try:
        research_data = researcher_agent.run(destination)
    except Exception as e:
        await cl.Message(content=f"❌ Researcher error: {str(e)}").send()
        return

    await cl.Message(content="🗺️ Creating your personalized itinerary...").send()
    
    try:
        itinerary = planner_agent.run(research_data, num_days)
    except Exception as e:
        await cl.Message(content=f"❌ Planner error: {str(e)}").send()
        return

    await cl.Message(content=f"✅ Here's your itinerary for {destination}:\n\n{itinerary}").send()
