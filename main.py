import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq 

def init():
    # Load the environment variables
    load_dotenv()

    # Validate Groq API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("Error: GROQ_API_KEY is not set in the .env file.")
        st.stop()

    # Configure Streamlit
    st.set_page_config(
        page_title="Custom ChatGPT with Groq and Meta Llama",
        page_icon="🤖"
    )

def main():
    init()

    # Initialize ChatGroq
    groq_api_key = os.getenv("GROQ_API_KEY")
    chat= ChatGroq(api_key=groq_api_key, model_name="llama-3.1-8b-instant")

    # Message history initialization
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant using Meta Llama.")
        ]

    st.header("Custom ChatGPT with Groq 🤖")

    # Sidebar input for user interaction
    with st.sidebar:
        user_input = st.text_input("Your message:", key="user_input")

        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                # Use ChatGroq for response generation
                response = chat(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))

    # Display chat messages
    messages = st.session_state.messages
    for i, msg in enumerate(messages[1:]):  # Skip the initial system message
        if isinstance(msg, HumanMessage):
            message(msg.content, is_user=True, key=f"{i}_user")
        elif isinstance(msg, AIMessage):
            message(msg.content, is_user=False, key=f"{i}_ai")

if __name__ == '__main__':
    main()
