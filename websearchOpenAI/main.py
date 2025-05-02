import streamlit as st
import openai
from dotenv import load_dotenv
import os
from tools import web_search, file_search  # Make sure tools.py is present

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI Setup
st.set_page_config(page_title="Your Research AI Assistant", page_icon="🤖")
st.sidebar.title("🔐 API Status")
st.sidebar.write("API Key Loaded:", "✅" if openai.api_key else "❌")

# Stop execution if API key is missing
if not openai.api_key:
    st.error("❌ API Key not loaded. Please check your `.env` file.")
    st.stop()

# Tool definitions (same as before)
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

# Function to run agent
def run_agent(user_input):
    response = openai.chat.completions.create(
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

        # Call appropriate function
        if fn_name == "web_search":
            result = web_search(**args)
        elif fn_name == "file_search":
            result = file_search(**args)

        # Follow-up with result
        followup = openai.chat.completions.create(
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

# Streamlit Interface
st.title("🤖 Your Research AI Assistant")

user_input = st.text_input("Ask me anything...")

if user_input:
    with st.spinner("Thinking..."):
        try:
            output = run_agent(user_input)
            st.success(output)
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")


# import streamlit as st
# from openai import OpenAI
# from dotenv import load_dotenv
# import os
# from tools import web_search, file_search

# # Load environment variables from .env file
# load_dotenv()

# # Get OpenAI API Key
# api_key = os.getenv("OPENAI_API_KEY")

# # Display API Key load status
# st.set_page_config(page_title="Your research AI Assistant", page_icon="🤖")
# st.sidebar.title("🔐 API Status")
# st.sidebar.write("API Key Loaded:", "✅" if api_key else "❌")

# # If API Key not loaded, show error and stop app
# if not api_key:
#     st.error("❌ API Key not loaded. Please check your `.env` file.")
#     st.stop()

# # Initialize OpenAI client
# client = OpenAI(api_key=api_key)

# # Define tools
# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "web_search",
#             "description": "Searches the web for up-to-date info",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "query": {"type": "string"},
#                 },
#                 "required": ["query"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "file_search",
#             "description": "Searches uploaded files for relevant content",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "query": {"type": "string"},
#                     "vector_store_ids": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                     },
#                     "max_num_results": {"type": "integer", "default": 3},
#                 },
#                 "required": ["query", "vector_store_ids"],
#             },
#         },
#     },
# ]

# # Run agent function
# def run_agent(user_input):
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": user_input}],
#         tools=tools,
#         tool_choice="auto",
#     )

#     message = response.choices[0].message

#     if message.tool_calls:
#         tool_call = message.tool_calls[0]
#         fn_name = tool_call.function.name
#         args = eval(tool_call.function.arguments)

#         if fn_name == "web_search":
#             result = web_search(**args)
#         elif fn_name == "file_search":
#             result = file_search(**args)

#         followup = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "user", "content": user_input},
#                 {"role": "assistant", "tool_calls": [tool_call]},
#                 {
#                     "role": "tool",
#                     "tool_call_id": tool_call.id,
#                     "name": fn_name,
#                     "content": result,
#                 },
#             ],
#         )

#         return followup.choices[0].message.content
#     else:
#         return message.content

# # Streamlit UI
# st.title("🤖 Your Research AI Assistant")

# user_input = st.text_input("Ask me anything...")

# if user_input:
#     with st.spinner("Thinking..."):
#         output = run_agent(user_input)
#         st.success(output)
