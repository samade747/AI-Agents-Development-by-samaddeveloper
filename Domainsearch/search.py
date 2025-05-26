import os
import streamlit as st
from dotenv import load_dotenv
from agno.agent import Agent
from composio_agno import ComposioToolSet, Action
from agno.models.openai import OpenAI  

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI DeepResearch Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for API keys
st.sidebar.header("⚙️ Configuration")

# API key inputs
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password"
    
)

composio_api_key = st.sidebar.text_input(
    "Composio API Key",
    value=os.getenv("COMPOSIO_API_KEY", ""),
    type="password"
   
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This AI DeepResearch Agent uses OpenAI's GPT-4 model and Composio tools to perform comprehensive research on any topic. "
    "It generates research questions, finds answers, and compiles a professional report."
)

st.sidebar.markdown("### Tools Used")
st.sidebar.markdown("- 🔍 Tavily Search")
st.sidebar.markdown("- 🧠 Perplexity AI")
st.sidebar.markdown("- 📄 Google Docs Integration")

# Initialize session state
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'question_answers' not in st.session_state:
    st.session_state.question_answers = []
if 'report_content' not in st.session_state:
    st.session_state.report_content = ""
if 'research_complete' not in st.session_state:
    st.session_state.research_complete = False

# Main content
st.title("🔍 AI DeepResearch Agent with OpenAI and Composio")

# Function to initialize the LLM and tools
def initialize_agents(openai_key, composio_key):
    llm = OpenAI(model="gpt-4-turbo", api_key=openai_key)

    toolset = ComposioToolSet(api_key=composio_key)
    composio_tools = toolset.get_tools(actions=[
        Action.COMPOSIO_SEARCH_TAVILY_SEARCH,
        Action.PERPLEXITYAI_PERPLEXITY_AI_SEARCH,
        Action.GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN
    ])

    return llm, composio_tools

# Function to create agents
def create_agents(llm, composio_tools):
    question_generator = Agent(
        name="Question Generator",
        model=llm,
        instructions="""
        You are an expert at breaking down research topics into specific questions.
        Generate exactly 5 specific yes/no research questions about the given topic in the specified domain.
        Respond ONLY with the text of the 5 questions formatted as a numbered list, and NOTHING ELSE.
        """
    )
    return question_generator

# Function to extract questions after think tag
def extract_questions_after_think(text):
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()

# Function to generate research questions
def generate_questions(llm, composio_tools, topic, domain):
    question_generator = create_agents(llm, composio_tools)

    with st.spinner("🤖 Generating research questions..."):
        questions_task = question_generator.run(
            f"Generate exactly 5 specific yes/no research questions about the topic '{topic}' in the domain '{domain}'."
        )
        questions_text = questions_task.content
        questions_only = extract_questions_after_think(questions_text)

        questions_list = [q.strip() for q in questions_only.split('\n') if q.strip()]
        st.session_state.questions = questions_list
        return questions_list

# Function to research a specific question
def research_question(llm, composio_tools, topic, domain, question):
    research_task = Agent(
        model=llm,
        tools=[composio_tools],
        instructions=f"You are a sophisticated research assistant. Answer the following research question about the topic '{topic}' in the domain '{domain}':\n\n{question}\n\nUse the PERPLEXITYAI_PERPLEXITY_AI_SEARCH and COMPOSIO_SEARCH_TAVILY_SEARCH tools to provide a concise, well-sourced answer."
    )
    research_result = research_task.run()
    return research_result.content

# Function to compile final report
def compile_report(llm, composio_tools, topic, domain, question_answers):
    with st.spinner("📝 Compiling final report and creating Google Doc..."):
        qa_sections = "\n".join(
            f"<h2>{idx+1}. {qa['question']}</h2>\n<p>{qa['answer']}</p>"
            for idx, qa in enumerate(question_answers)
        )

        compile_report_task = Agent(
            name="Report Compiler",
            model=llm,
            tools=[composio_tools],
            instructions=f"""
            You are a sophisticated research assistant. Compile the following research findings into a professional, McKinsey-style report. The report should be structured as follows:

            1. Executive Summary/Introduction: Briefly introduce the topic and domain, and summarize the key findings.
            2. Research Analysis: For each research question, create a section with a clear heading and provide a detailed, analytical answer. Do NOT use a Q&A format; instead, weave the answer into a narrative and analytical style.
            3. Conclusion/Implications: Summarize the overall insights and implications of the research.

            Use clear, structured HTML for the report.

            Topic: {topic}
            Domain: {domain}

            Research Questions and Findings (for your reference):
            {qa_sections}

            Use the GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN tool to create a Google Doc with the report. The text should be in HTML format. You have to create the google document with all the compiled info. You have to do it.
            """
        )

        compile_result = compile_report_task.run()
        st.session_state.report_content = compile_result.content
        st.session_state.research_complete = True
        return compile_result.content

# Main application flow
if openai_api_key and composio_api_key:
    llm, composio_tools = initialize_agents(openai_api_key, composio_api_key)

    st.header("Research Topic")

    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("What topic would you like to research?", placeholder="American Tariffs")
    with col2:
        domain = st.text_input("What domain is this topic in?", placeholder="Politics, Economics, Technology, etc.")

    if topic and domain and st.button("Generate Research Questions", key="generate_questions"):
        questions = generate_questions(llm, composio_tools, topic, domain)
        st.header("Research Questions")
        for i, question in enumerate(questions):
            st.markdown(f"**{i+1}. {question}**")

    if st.session_state.questions and st.button("Start Research", key="start_research"):
        st.header("Research Results")

        question_answers = []
        progress_bar = st.progress(0)

        for i, question in enumerate(st.session_state.questions):
            progress_bar.progress((i) / len(st.session_state.questions))

            with st.spinner(f"🔍 Researching question {i+1}..."):
                answer = research_question(llm, composio_tools, topic, domain, question)
                question_answers.append({"question": question, "answer": answer})

            st.subheader(f"Question {i+1}:")
            st.markdown(f"**{question}**")
            st.markdown(answer)

            progress_bar.progress((i + 1) / len(st.session_state.questions))

        st.session_state.question_answers = question_answers

        if st.button("Compile Final Report", key="compile_report"):
            report_content = compile_report(llm, composio_tools, topic, domain, question_answers)
            st.header("Final Report")
            st.success("Your report has been compiled and a Google Doc has been created.")
            with st.expander("View Full Report Content", expanded=True):
                st.markdown(report_content, unsafe_allow_html=True)

    if len(st.session_state.question_answers) > 0 and not st.session_state.research_complete:
        st.header("Previous Research Results")
        for i, qa in enumerate(st.session_state.question_answers):
            with st.expander(f"Question {i+1}: {qa['question']}"):
                st.markdown(qa['answer'])

    if st.session_state.research_complete and st.session_state.report_content:
        st.header("Final Report")
        st.success("Your report has been compiled and a Google Doc has been created.")
        with st.expander("View Full Report Content", expanded=True):
            st.markdown(st.session_state.report_content, unsafe_allow_html=True)

else:
    st.warning("⚠️ Please enter your OpenAI and Composio API keys in the sidebar to get started.")

    st.header("How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1️⃣ Define Topic")
        st.write("Enter your research topic and domain to begin the research process.")
    with col2:
        st.subheader("2️⃣ Generate Questions")
        st.write("The AI generates specific research questions to explore your topic in depth.")
    with col3:
        st.subheader("3️⃣ Compile Report")
        st.write("Research findings are compiled into a professional report and saved to Google Docs.")
