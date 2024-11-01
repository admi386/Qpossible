import os
import json
import logging
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import re
from openai import OpenAI
from langchain_community.llms import Ollama
from groq import Groq
from settings_manager import load_settings, save_settings

# Load LLM settings
if "llm_settings" not in st.session_state:
    st.session_state.llm_settings = load_settings()

# Initialize session state for uploaded data, generated code and plot
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

if "generated_code" not in st.session_state:
    st.session_state.generated_code = None

if "generated_plot" not in st.session_state:
    st.session_state.generated_plot = None  # To store the generated plot

# Universal LLM Wrapper
class LLMWrapper:
    def __init__(self, settings):
        self.settings = settings
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        llm_type = self.settings["llm_type"]
        if llm_type == "ollama":
            return Ollama(model=self.settings["model"], temperature=self.settings["temperature"])
        elif llm_type == "groq":
            return Groq(api_key=self.settings["api_key"])
        elif llm_type == "openai":
            return OpenAI(api_key=self.settings["api_key"], temperature=self.settings["temperature"], max_tokens=self.settings["max_tokens"])
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    def generate_response(self, prompt):
        try:
            if self.settings["llm_type"] == "groq":
                chat_completion = self.llm.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.settings["model"]
                )
                return chat_completion.choices[0].message.content
            else:
                return self.llm(prompt)
        except Exception as e:
            logging.error(f"LLM error: {str(e)}")
            return f"LLM error: {str(e)}"

# Initialize LLM
llm_wrapper = LLMWrapper(st.session_state.llm_settings)

# Function to send data and user request to LLM
def generate_code_with_llm(dataframe, user_request):
    # Convert full dataframe to JSON format for analysis
    data_json = dataframe.to_json(orient="records")
    
    # Prompt for generating Python code
    prompt = f"""
    I have the following data in JSON format: {data_json}. 
    Based on this data, write Python code to generate a {user_request} plot using matplotlib or any other relevant library.
    Use the dataframe `df` directly, do not use any file loading in the code.
    Make sure the code runs without errors.
    """
    
    # Get the Python code from LLM
    llm_response = llm_wrapper.generate_response(prompt)
    return llm_response

# Function to clean the generated Python code from LLM
def extract_clean_code(llm_response):
    # Use regex to find the code block within the ```python or ``` markers
    code_match = re.search(r"```(?:python)?(.*?)```", llm_response, re.DOTALL)
    
    # Extract the code if found, else return the full response as fallback
    if code_match:
        return code_match.group(1).strip()  # Return the cleaned code without ``` markers
    else:
        return llm_response.strip()

# Streamlit interface
st.title("Data Plotting with LLM")

# File upload
uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load the data into a Pandas DataFrame
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    # Store uploaded data in session state
    st.session_state.uploaded_data = df

# Check if data is already uploaded in session state
if st.session_state.uploaded_data is not None:
    # Display the first few rows of the data
    st.write("### Preview of the uploaded data")
    st.write(st.session_state.uploaded_data.head())

    # User input for the type of plot
    user_request = st.text_input("What type of plot would you like to generate?")

    if st.button("Generate Plot"):
        if user_request:
            # Send data and user request to LLM for code generation
            code_response = generate_code_with_llm(st.session_state.uploaded_data, user_request)
            
            # Extract clean Python code using regex
            clean_code = extract_clean_code(code_response)

            # Save generated code in session state to persist after page reload
            st.session_state.generated_code = clean_code

            # Reset plot state to re-render it correctly
            st.session_state.generated_plot = None

# Check and display the generated code and plot if available
if st.session_state.generated_code:
    st.write("### Generated Python Code")
    st.code(st.session_state.generated_code, language="python")

    # Try to execute the generated Python code and store the plot
    if st.session_state.generated_plot is None:
        try:
            # Execute the extracted clean code in the current namespace
            exec(st.session_state.generated_code, {'df': st.session_state.uploaded_data, 'plt': plt})

            # Store the current plot in session state
            st.session_state.generated_plot = plt.gcf()
        except Exception as e:
            st.error(f"Error executing the code: {str(e)}")

# Display the stored plot from session state
if st.session_state.generated_plot is not None:
    st.pyplot(st.session_state.generated_plot)