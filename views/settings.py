import streamlit as st
from settings_manager import load_settings, save_settings

# Load settings from a file
if "llm_settings" not in st.session_state:
    st.session_state.llm_settings = load_settings()

st.title("LLM Settings")

# Input fields for settings
st.session_state.llm_settings["model"] = st.text_input("Model:", st.session_state.llm_settings["model"])
st.session_state.llm_settings["base_url"] = st.text_input("Base URL:", st.session_state.llm_settings["base_url"])
st.session_state.llm_settings["api_key"] = st.text_input("API Key:", st.session_state.llm_settings["api_key"], type="password")
st.session_state.llm_settings["max_tokens"] = st.number_input("Max Tokens:", min_value=1, max_value=4096, value=st.session_state.llm_settings["max_tokens"])
st.session_state.llm_settings["temperature"] = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=st.session_state.llm_settings["temperature"])
st.session_state.llm_settings["llm_type"] = st.selectbox("LLM Type:", ["ollama", "groq", "openai"], index=["ollama", "groq", "openai"].index(st.session_state.llm_settings["llm_type"]))

# Save settings to the file
if st.button("Save Settings"):
    save_settings(st.session_state.llm_settings)
    st.success("Settings have been updated and saved.")
