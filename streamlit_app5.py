import streamlit as st
from typing import Generator
from groq import Groq
import os
import logging
import json
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

# Directory to store chat histories
CHAT_HISTORY_DIR = "chat_histories"

if not os.path.exists(CHAT_HISTORY_DIR):
    os.makedirs(CHAT_HISTORY_DIR)

# Set Streamlit page configuration
st.set_page_config(page_icon="💬", layout="wide", page_title="Groq Goes Brrrrrrrr...")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7  # Default temperature
    if "top_p" not in st.session_state:
        st.session_state.top_p = 0.9  # Default top_p
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = None

def get_groq_client() -> Groq:
    """Initialize and return the Groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("API key not found. Please set the GROQ_API_KEY environment variable.")
        st.stop()
    return Groq(api_key=api_key)

def display_chat_messages():
    """Display chat messages from the session state."""
    for message in st.session_state.messages:
        avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def save_chat_history(name=None):
    """Save the current chat history to a file."""
    if not name:
        name = f"Chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    chat_data = {
        "name": name,
        "messages": st.session_state.messages,
        "system_prompt": st.session_state.system_prompt,
        "model": st.session_state.selected_model,
        "temperature": st.session_state.temperature,
        "top_p": st.session_state.top_p,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open(os.path.join(CHAT_HISTORY_DIR, f"{name}.json"), "w") as f:
        json.dump(chat_data, f, indent=4)
    st.session_state.chat_history.append(chat_data)
    st.session_state.current_chat = name

def load_chat_history(name):
    """Load a chat history from a file."""
    with open(os.path.join(CHAT_HISTORY_DIR, f"{name}.json"), "r") as f:
        chat_data = json.load(f)
    st.session_state.messages = chat_data["messages"]
    st.session_state.system_prompt = chat_data["system_prompt"]
    st.session_state.selected_model = chat_data["model"]
    st.session_state.temperature = chat_data["temperature"]
    st.session_state.top_p = chat_data["top_p"]
    st.session_state.current_chat = name

def delete_chat_history(name):
    """Delete a chat history file."""
    os.remove(os.path.join(CHAT_HISTORY_DIR, f"{name}.json"))
    st.session_state.chat_history = [chat for chat in st.session_state.chat_history if chat["name"] != name]
    if st.session_state.current_chat == name:
        st.session_state.messages = []
        st.session_state.current_chat = None

def rename_chat_history(old_name, new_name):
    """Rename a chat history file."""
    os.rename(os.path.join(CHAT_HISTORY_DIR, f"{old_name}.json"), os.path.join(CHAT_HISTORY_DIR, f"{new_name}.json"))
    for chat in st.session_state.chat_history:
        if chat["name"] == old_name:
            chat["name"] = new_name
    if st.session_state.current_chat == old_name:
        st.session_state.current_chat = new_name

def list_chat_histories():
    """List all saved chat histories."""
    chat_histories = []
    for file in os.listdir(CHAT_HISTORY_DIR):
        if file.endswith(".json"):
            with open(os.path.join(CHAT_HISTORY_DIR, file), "r") as f:
                chat_data = json.load(f)
                chat_histories.append(chat_data)
    return chat_histories

def main():
    """Main function to run the Streamlit app."""
    icon("🏎️")
    st.subheader("Groq Chat Streamlit App", divider="rainbow", anchor=False)

    client = get_groq_client()
    initialize_session_state()

    # Load existing chat histories
    st.session_state.chat_history = list_chat_histories()

    # Sidebar for chat history
    with st.sidebar:
        st.header("Chat History")
        for chat in st.session_state.chat_history:
            if st.button(chat["name"]):
                load_chat_history(chat["name"])
            with st.expander(chat["name"]):
                if st.button("Load", key=f"load_{chat['name']}"):
                    load_chat_history(chat["name"])
                if st.button("Rename", key=f"rename_{chat['name']}"):
                    new_name = st.text_input("New name:", value=chat["name"], key=f"new_name_{chat['name']}")
                    if new_name and new_name != chat["name"]:
                        rename_chat_history(chat["name"], new_name)
                if st.button("Delete", key=f"delete_{chat['name']}"):
                    delete_chat_history(chat["name"])

    # Define model details
    models = {
        "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
        "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
        "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
        "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
        "davinci-003": {"name": "text-davinci-003", "tokens": 4096, "developer": "OpenAI"},
        "davinci-004": {"name": "davinci-004", "tokens": 8192, "developer": "OpenAI"},
        "gpt-4": {"name": "gpt-4", "tokens": 8192, "developer": "OpenAI"},
    }

    # Model selection
    model_option = st.selectbox("Choose a model:", options=list(models.keys()), format_func=lambda key: f"{models[key]['name']} by {models[key]['developer']}")
    if model_option:
        st.session_state.selected_model = models[model_option]["name"]

    # System prompt
    system_prompt = st.text_area("System Prompt:", value=st.session_state.system_prompt)
    st.session_state.system_prompt = system_prompt

    # Display chat messages
    display_chat_messages()

    # Text input for user message
    if prompt := st.chat_input(placeholder="Ask something or start chatting..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👨‍💻"):
            st.markdown(prompt)
        full_response = ""
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            try:
                chat_completion = client.chat(
                    messages=st.session_state.messages,
                    model=models[model_option]["name"],
                    max_tokens=models[model_option]["tokens"],
                    stream=True,
                    temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p,
                    system_prompt=st.session_state.system_prompt
                )
                for chunk in generate_chat_responses(chat_completion):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                logging.error(f"Error fetching response: {e}")
                st.error("An error occurred while fetching the response. Please try again.")
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Button to clear chat
    if st.button("Clear Chat"):
        st.session_state.messages = []

    # Button to save chat
    if st.button("Save Chat"):
        save_chat_history()

if __name__ == "__main__":
    main()
