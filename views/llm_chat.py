import os
import json
import streamlit as st
from langchain_community.llms import Ollama
from openai import OpenAI
from groq import Groq
from settings_manager import load_settings, save_settings

# Load settings
if "llm_settings" not in st.session_state:
    st.session_state.llm_settings = load_settings()

# Universal LLM Wrapper
class LLMWrapper:
    def __init__(self, settings):
        self.settings = settings
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        llm_type = self.settings["llm_type"]

        if llm_type == "ollama":
            return Ollama(
                model=self.settings["model"],
                temperature=self.settings["temperature"]
            )
        elif llm_type == "groq":
            return Groq(
                api_key=self.settings["api_key"]
            )
        elif llm_type == "openai":
            return OpenAI(
                api_key=self.settings["api_key"],
                temperature=self.settings["temperature"],
                max_tokens=self.settings["max_tokens"]
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    def generate_response(self, prompt):
        if self.settings["llm_type"] == "groq":
            chat_completion = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.settings["model"],
            )
            return chat_completion.choices[0].message.content
        else:
            return self.llm(prompt)

# Path to the history file
history_file = "chat_history.json"

# Functions for working with chat history
def load_chat_history():
    if os.path.exists(history_file):
        with open(history_file, "r") as file:
            return json.load(file)
    return []

def save_chat_history(history):
    with open(history_file, "w") as file:
        json.dump(history, file)

def clear_chat_history():
    if os.path.exists(history_file):
        os.remove(history_file)
    st.session_state["chat_messages"] = []

# Load chat history on startup
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = load_chat_history()

# Initialize the LLM
llm_wrapper = LLMWrapper(st.session_state.llm_settings)

# Streamlit interface
st.title("Universal LLM Chat")
st.write("Chat with an AI assistant. Ask questions, seek advice, or discuss any topic you like.")

# Display chat history
for message in st.session_state["chat_messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Enter your message here:"):
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
    save_chat_history(st.session_state["chat_messages"])
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Provide the full chat history to LLM
    history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["chat_messages"]])
    full_prompt = f"Here is the chat history:\n{history}\n\nRespond to the latest message: {prompt}"

    # Generate response and display it
    response = llm_wrapper.generate_response(full_prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state["chat_messages"].append({"role": "assistant", "content": response})
    save_chat_history(st.session_state["chat_messages"])

# Button to clear chat history
if st.button("Clear History"):
    clear_chat_history()