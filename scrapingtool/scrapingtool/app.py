# scrapingtool/app.py

import streamlit as st
from scrapegraphai.graphs import SmartScraperGraph

def main():
    st.title("Web Scraping AI Agent 🕵️‍♂️")
    st.caption("This app allows you to scrape a website using OpenAI API")

    openai_access_token = st.text_input("OpenAI API Key", type="password")

    if openai_access_token:
        model = st.radio("Select the model", ["gpt-3.5-turbo", "gpt-4"], index=0)

        graph_config = {
            "llm": {
                "api_key": openai_access_token,
                "model": model,
            },
        }

        url = st.text_input("Enter the URL of the website you want to scrape")
        user_prompt = st.text_input("What do you want the AI agent to scrape from the website?")

        if st.button("Scrape"):
            if url and user_prompt:
                try:
                    smart_scraper_graph = SmartScraperGraph(
                        prompt=user_prompt,
                        source=url,
                        config=graph_config
                    )
                    result = smart_scraper_graph.run()
                    st.write(result)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter both the URL and the prompt.")
