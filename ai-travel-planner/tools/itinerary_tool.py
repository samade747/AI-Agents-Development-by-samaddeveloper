import openai
import os

def create_itinerary(research_data, days):
    """
    Generates a travel itinerary using OpenAI's API.

    Args:
        research_data (str): Information about the destination.
        days (int): Number of travel days.

    Returns:
        str: The generated itinerary.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    openai.api_key = api_key
    openai.api_base = "https://openrouter.ai/api/v1"

    prompt = (
        f"Based on the following information:\n{research_data}\n"
        f"Plan a detailed {days}-day travel itinerary."
    )

    response = openai.ChatCompletion.create(
        model="openai/gpt-4o",  # Replace with your preferred model
        messages=[
            {"role": "system", "content": "You are a helpful travel assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response['choices'][0]['message']['content']
