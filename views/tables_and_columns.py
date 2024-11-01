import asyncio
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import threading
import json
from settings_manager import load_settings, save_settings
from db_settings_manager import load_db_settings, save_db_settings

# Load database and LLM settings
if "llm_settings" not in st.session_state:
    st.session_state.llm_settings = load_settings()

if "db_settings" not in st.session_state:
    st.session_state.db_settings = load_db_settings()

# Initialize state to store the last LLM response
if "llm_response" not in st.session_state:
    st.session_state.llm_response = None  # Default to an empty response

# Universal wrapper for working with LLM
class LLMWrapper:
    def __init__(self, settings):
        self.settings = settings
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        # Initialize the LLM based on settings
        llm_type = self.settings["llm_type"]
        if llm_type == "ollama":
            from langchain_community.llms import Ollama
            return Ollama(
                model=self.settings["model"],
                temperature=self.settings["temperature"]
            )
        elif llm_type == "groq":
            from groq import Groq
            return Groq(api_key=self.settings["api_key"])
        elif llm_type == "openai":
            from openai import OpenAI
            return OpenAI(
                api_key=self.settings["api_key"],
                temperature=self.settings["temperature"],
                max_tokens=self.settings["max_tokens"]
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    async def generate_response_async(self, prompt):
        if self.settings["llm_type"] == "groq":
            chat_completion = await self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.settings["model"]
            )
            return chat_completion.choices[0].message.content if chat_completion.choices else None
        else:
            return await asyncio.to_thread(self.llm, prompt)

# Initialize LLM
llm_wrapper = LLMWrapper(st.session_state.llm_settings)

# Streamlit interface
st.title("Find Required Tables and Columns")

# Function to connect to the database
def connect_to_database():
    db_settings = st.session_state.db_settings
    db_type = db_settings["db_type"]
    try:
        if db_type == "PostgreSQL":
            engine = create_engine(f"postgresql://{db_settings['username']}:{db_settings['password']}@{db_settings['host']}:{db_settings['port']}/{db_settings['db_name']}")
        elif db_type == "MySQL":
            engine = create_engine(f"mysql+pymysql://{db_settings['username']}:{db_settings['password']}@{db_settings['host']}:{db_settings['port']}/{db_settings['db_name']}")
        elif db_type == "SQLite":
            engine = create_engine(f"sqlite:///{db_settings['db_name']}")
        elif db_type == "SQL Server":
            engine = create_engine(f"mssql+pyodbc://{db_settings['username']}:{db_settings['password']}@{db_settings['host']}/{db_settings['db_name']}?driver=SQL+Server")
        else:
            st.error("Unsupported database type")
            return None
        return engine.connect()
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

# Function to get database metadata
def get_database_metadata(connection):
    db_settings = st.session_state.db_settings
    db_type = db_settings["db_type"]
    metadata = {}
    try:
        if db_type == "SQLite":
            tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
            tables_result = pd.read_sql_query(tables_query, connection)
            table_column_name = 'name'  # for SQLite
        else:
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            tables_result = pd.read_sql_query(tables_query, connection)
            table_column_name = 'table_name'  # for PostgreSQL, MySQL, SQL Server, etc.

        for table in tables_result[table_column_name]:
            if db_type == "SQLite":
                columns_query = f"PRAGMA table_info({table});"
                columns_result = pd.read_sql_query(columns_query, connection)
                columns = [{'column_name': row['name'], 'data_type': row['type']} for _, row in columns_result.iterrows()]
            else:
                columns_query = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}';"
                columns_result = pd.read_sql_query(columns_query, connection)
                columns = columns_result.to_dict(orient='records')

            metadata[table] = {'columns': columns}
    
    except Exception as e:
        st.error(f"Error retrieving database metadata: {e}")
    return metadata

# Function to find required tables and columns
async def find_required_tables_and_columns(user_query, metadata):
    prompt = f"User query: '{user_query}'. Analyze the request and provide a list of relevant tables and columns from the database metadata: {json.dumps(metadata)}."
    response = await llm_wrapper.generate_response_async(prompt)
    
    if response:
        st.session_state.llm_response = response  # Store the response in session_state
    else:
        st.error("Error: LLM did not return a response.")

# Thread wrapper to execute asynchronous functions
class RunThread(threading.Thread):
    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        super().__init__()

    def run(self):
        self.result = asyncio.run(self.func(*self.args, **self.kwargs))

def run_async(func, *args, **kwargs):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        thread = RunThread(func, args, kwargs)
        thread.start()
        thread.join()
        return thread.result
    else:
        return asyncio.run(func(*args, **kwargs))

# User input interface
user_query = st.text_area("Enter your query in natural language:", "Make a complete list of columns needed to analyze...")

# Button to find tables and columns
if st.button("Find Tables and Columns"):
    connection = connect_to_database()
    if connection:
        metadata = get_database_metadata(connection)
        run_async(find_required_tables_and_columns, user_query, metadata)
        connection.close()

# Display the last response, if available
if st.session_state.llm_response:
    st.write("## LLM Response:")
    st.write(st.session_state.llm_response)