# app.py
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
from tools import web_search, file_search

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web for up-to-date info",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": "Searches uploaded files for relevant content",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "vector_store_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "max_num_results": {"type": "integer", "default": 3},
                },
                "required": ["query", "vector_store_ids"],
            },
        },
    },
]

def run_agent(user_input):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}],
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        fn_name = tool_call.function.name
        args = eval(tool_call.function.arguments)

        if fn_name == "web_search":
            result = web_search(**args)
        elif fn_name == "file_search":
            result = file_search(**args)

        followup = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": user_input},
                {"role": "assistant", "tool_calls": [tool_call]},
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": result,
                },
            ],
        )

        return followup.choices[0].message.content
    else:
        return message.content

# Streamlit UI
st.set_page_config(page_title="Your websearch AI Assistant", page_icon="🤖")
st.title("🤖Your websearch AI Assistant")

user_input = st.text_input("Ask me anything... Status🤖 Api not connected(issues unpaid)", "")

if user_input:
    with st.spinner("Thinking..."):
        output = run_agent(user_input)
        st.success(output)
