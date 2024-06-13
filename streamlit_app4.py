import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Generator
from groq import Groq
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

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
    if "response_times" not in st.session_state:
        st.session_state.response_times = []  # List to store response times
    if "token_speeds" not in st.session_state:
        st.session_state.token_speeds = []  # List to store token speeds
    if "token_counts" not in st.session_state:
        st.session_state.token_counts = []  # List to store token counts
    if "inference_times" not in st.session_state:
        st.session_state.inference_times = []  # List to store inference times

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

def main():
    """Main function to run the Streamlit app."""
    icon("🏎️")
    st.subheader("Groq Chat Streamlit App", divider="rainbow", anchor=False)

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

    # Add a button to clear the chat
    if st.button("Clear Chat"):
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    display_chat_messages()

    if prompt := st.chat_input("Enter your prompt here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar='👨‍💻'):
            st.markdown(prompt)

        # Prepare messages for the API call
        messages = []
        if st.session_state.system_prompt:
            messages.append({"role": "system", "content": st.session_state.system_prompt})
        messages.extend(st.session_state.messages)

        # Initialize full_response
        full_response = ""

        # Measure the start time for response time calculation
        start_time = time.time()

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

            # Measure the end time for response time calculation
            end_time = time.time()
            response_time = end_time - start_time
            st.session_state.response_times.append(response_time)

            # Example metrics (replace with actual data)
            input_tokens = 152
            output_tokens = 210
            total_tokens = input_tokens + output_tokens
            input_speed = 664
            output_speed = 833
            total_speed = (input_speed + output_speed) / 2
            input_inference_time = 0.23
            output_inference_time = 0.25
            total_inference_time = input_inference_time + output_inference_time

            st.session_state.token_counts.append((input_tokens, output_tokens, total_tokens))
            st.session_state.token_speeds.append((input_speed, output_speed, total_speed))
            st.session_state.inference_times.append((input_inference_time, output_inference_time, total_inference_time))

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

    # Display metrics
    st.subheader("Performance Metrics")
    if st.session_state.response_times:
        avg_response_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
        st.write(f"Round trip time: {avg_response_time:.2f}s")

    if st.session_state.token_counts:
        token_counts_df = pd.DataFrame(st.session_state.token_counts, columns=['Input Tokens', 'Output Tokens', 'Total Tokens'])
        st.write("Token Counts")
        st.dataframe(token_counts_df)

    if st.session_state.token_speeds:
        token_speeds_df = pd.DataFrame(st.session_state.token_speeds, columns=['Input Speed (T/s)', 'Output Speed (T/s)', 'Total Speed (T/s)'])
        st.write("Token Speeds")
        st.dataframe(token_speeds_df)

    if st.session_state.inference_times:
        inference_times_df = pd.DataFrame(st.session_state.inference_times, columns=['Input Inference Time (s)', 'Output Inference Time (s)', 'Total Inference Time (s)'])
        st.write("Inference Times")
        st.dataframe(inference_times_df)

if __name__ == "__main__":
    main()