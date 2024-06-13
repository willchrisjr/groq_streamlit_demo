import streamlit as st
from typing import Generator
from groq import Groq
import os
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set Streamlit page configuration
st.set_page_config(page_icon="💬", layout="wide", page_title="Groq Goes Brrrrrrrr...")

CHAT_HISTORY_DIR = "chat_history"
CHAT_HISTORY_FILE = os.path.join(CHAT_HISTORY_DIR, "chat_history.json")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.markdown(
        f'<div style="text-align: center; font-size: 78px;">{emoji}</div>',
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
        st.session_state.chat_history = load_chat_history()

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

def save_chat_history():
    """Save the current chat history to a file."""
    if not os.path.exists(CHAT_HISTORY_DIR):
        os.makedirs(CHAT_HISTORY_DIR)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = st.session_state.selected_model
    filename = os.path.join(CHAT_HISTORY_DIR, f"chat_{timestamp}_{model_name}.json")
    with open(filename, "w") as f:
        json.dump(st.session_state.messages, f)
    st.session_state.chat_history[filename] = f"{timestamp} ({model_name})"
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(st.session_state.chat_history, f)

def load_chat_history():
    """Load chat history from a file."""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def load_chat(filename):
    """Load chat messages from a file."""
    with open(filename, "r") as f:
        st.session_state.messages = json.load(f)

def main():
    """Main function to run the Streamlit app."""
    icon("🏎️")
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Groq Chat Streamlit App</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #4B8BBE;'>", unsafe_allow_html=True)

    client = get_groq_client()
    initialize_session_state()

    # Define model details
    models = {
        "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
        "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
        "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
        "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
        "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
    }

    # Add an expander for advanced model settings
    with st.expander("Advanced Settings"):
        model_option = st.selectbox(
            "Choose a model:",
            options=list(models.keys()),
            format_func=lambda x: models[x]["name"],
            index=4  # Default to mixtral
        )

        # Detect model change and clear chat history if model has changed
        if st.session_state.selected_model != model_option:
            st.session_state.messages = []
            st.session_state.selected_model = model_option

        # Add a text input for the system prompt
        st.text_input(
            "System Prompt (optional):",
            value=st.session_state.system_prompt,
            on_change=lambda: st.session_state.update({"system_prompt": st.session_state.system_prompt}),
            key="system_prompt",
            help="Enter a system prompt to guide the model's behavior."
        )

        max_tokens_range = models[model_option]["tokens"]

        # Adjust max_tokens slider dynamically based on the selected model
        max_tokens = st.slider(
            "Max Tokens:",
            min_value=512,  # Minimum value to allow some flexibility
            max_value=max_tokens_range,
            # Default value or max allowed if less
            value=min(32768, max_tokens_range),
            step=512,
            help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
        )

        st.session_state.temperature = st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Controls the randomness of the model's output. Lower values make the output more deterministic."
        )

        st.session_state.top_p = st.slider(
            "Top P:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.top_p,
            step=0.1,
            help="Controls the diversity of the model's output. Lower values make the output more focused."
        )

    # Layout the main UI
    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("Clear Chat"):
            st.session_state.messages = []

        if st.button("Save Chat"):
            save_chat_history()

        # Display chat messages from history on app rerun
        display_chat_messages()

        if prompt := st.text_area("Enter your prompt here...", key="chat_input", height=100):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar='👨‍💻'):
                st.markdown(prompt)
            
            # Prepare messages for the API call
            messages = []
            if st.session_state.system_prompt:
                messages.append({"role": "system", "content": st.session_state.system_prompt})
            messages.extend(st.session_state.messages)

            # Fetch response from Groq API
            try:
                chat_completion = client.chat.completions.create(
                    model=model_option,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p,
                    stream=True
                )

                # Use the generator function with st.write_stream
                with st.chat_message("assistant", avatar="🤖"):
                    chat_responses_generator = generate_chat_responses(chat_completion)
                    full_response = st.write_stream(chat_responses_generator)
            except Exception as e:
                st.error(f"An error occurred: {e}", icon="🚨")
                logging.error(f"Error fetching response: {e}")

            # Append the full response to session_state.messages
            if isinstance(full_response, str):
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response})
            else:
                # Handle the case where full_response is not a string
                combined_response = "\n".join(str(item) for item in full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": combined_response})

    with col2:
        st.sidebar.header("Chat History")
        for filename, timestamp in st.session_state.chat_history.items():
            if st.sidebar.button(timestamp):
                load_chat(filename)
                st.experimental_rerun()

if __name__ == "__main__":
    main()
