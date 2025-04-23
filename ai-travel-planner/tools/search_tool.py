import os
import requests

def search_destination_info(destination):
    """
    Fetches information about the given destination using SerpAPI.

    Args:
        destination (str): The travel destination.

    Returns:
        str: A summary of the top search result.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY environment variable not set.")

    params = {
        "q": f"{destination} travel guide",
        "api_key": api_key,
        "engine": "google",
        "num": 1
    }

    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code != 200:
        raise Exception(f"SerpAPI request failed with status code {response.status_code}")

    data = response.json()
    try:
        top_result = data["organic_results"][0]["snippet"]
    except (KeyError, IndexError):
        top_result = f"No information found for {destination}."

    return top_result
