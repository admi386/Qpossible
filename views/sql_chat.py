import os
import json
import logging
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import re
from settings_manager import load_settings, save_settings
from db_settings_manager import load_db_settings, save_db_settings

# Load LLM and database settings
if "llm_settings" not in st.session_state:
    st.session_state.llm_settings = load_settings()

if "db_settings" not in st.session_state:
    st.session_state.db_settings = load_db_settings()

# Initialize variable for chat history
def load_chat_history_from_file():
    history_file = "chat_history_sql.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as file:
            return json.load(file)
    return []

if "sql_chat_messages" not in st.session_state:
    st.session_state["sql_chat_messages"] = load_chat_history_from_file()

# Save chat history to file
def save_chat_history():
    history_file = "chat_history_sql.json"
    with open(history_file, "w") as file:
        json.dump(st.session_state["sql_chat_messages"], file)

# Clear chat history
def clear_chat_history():
    history_file = "chat_history_sql.json"
    if os.path.exists(history_file):
        os.remove(history_file)
    st.session_state["sql_chat_messages"] = []

# Universal LLM Wrapper
class LLMWrapper:
    def __init__(self, settings):
        self.settings = settings
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        llm_type = self.settings["llm_type"]
        if llm_type == "ollama":
            from langchain_community.llms import Ollama
            return Ollama(model=self.settings["model"], temperature=self.settings["temperature"])
        elif llm_type == "groq":
            from groq import Groq
            return Groq(api_key=self.settings["api_key"])
        elif llm_type == "openai":
            from openai import OpenAI
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

# Database connection function
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
            st.error("Unsupported database type.")
            return None
        return engine.connect()
    except SQLAlchemyError as e:
        st.error(f"Database connection error: {str(e)}")
        return None

# Function to get database tables
def get_db_tables(connection):
    db_settings = st.session_state.db_settings
    db_type = db_settings["db_type"]
    try:
        if db_type == "PostgreSQL":
            tables_query = text("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
        elif db_type == "MySQL":
            tables_query = text("SHOW TABLES;")
        elif db_type == "SQLite":
            tables_query = text("SELECT name FROM sqlite_master WHERE type='table';")
        elif db_type == "SQL Server":
            tables_query = text("SELECT table_name FROM information_schema.tables;")
        else:
            return []

        result = connection.execute(tables_query)
        tables = [row[0] for row in result]
        logging.info(f"Found tables: {tables}")
        return tables
    except Exception as e:
        logging.error(f"Error fetching tables: {str(e)}")
        st.error(f"Error fetching tables: {str(e)}")
        return []

# Function to get table columns
def get_table_columns(connection, table_name):
    db_settings = st.session_state.db_settings
    db_type = db_settings["db_type"]
    try:
        if db_type == "SQLite":
            query = text(f"PRAGMA table_info({table_name});")
        else:
            query = text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}';")
        
        result = connection.execute(query)
        columns = [row[0] if db_type != "SQLite" else row[1] for row in result]
        logging.info(f"Columns for table {table_name}: {columns}")
        return columns
    except Exception as e:
        logging.error(f"Error fetching columns for table {table_name}: {str(e)}")
        return []

# Function to clean SQL query from unnecessary symbols
def clean_sql_query(sql_query):
    sql_query_cleaned = sql_query.strip().replace("\n", " ").replace("\t", " ")
    return sql_query_cleaned

# Function to extract SQL query using regex
def extract_sql_query(llm_response):
    sql_match = re.search(r"```sql(.*?)```", llm_response, re.DOTALL)
    if not sql_match:
        sql_match = re.search(r"```(.*?)```", llm_response, re.DOTALL)

    if sql_match:
        sql_query = sql_match.group(1).strip()
    else:
        sql_query = llm_response.strip()

    sql_query_cleaned = clean_sql_query(sql_query)
    return sql_query_cleaned

# Function to send data for analysis to LLM
def send_data_for_analysis(df, prompt):
    # Convert dataframe to JSON format for analysis
    data_json = df.to_json(orient="records")
    analysis_prompt = f"""
    Based on the data: {data_json}, and the question: '{prompt}', please analyze the data and provide a detailed response.
    """
    
    # Generate the analysis response from the LLM
    llm_response = llm_wrapper.generate_response(analysis_prompt)
    
    return llm_response

# Multi-agent system for SQL analysis and data interpretation
def multiagent_sql_analysis(prompt):
    connection = connect_to_database()
    if not connection:
        return "Error: Failed to connect to the database."

    tables = get_db_tables(connection)
    if not tables:
        return "Error: No tables found in the database."

    table_columns = {table: get_table_columns(connection, table) for table in tables}
    logging.info(f"Database structure: {table_columns}")

    db_structure = "\n".join([f"Table: {table}, Columns: {', '.join(columns)}" for table, columns in table_columns.items()])
    analysis_prompt = f"""
    Analyze the database structure and generate a valid SQL query based on the user's request '{prompt}'.
    Here is the database structure:
    
    {db_structure}
    """

    llm_response = llm_wrapper.generate_response(analysis_prompt)
    logging.info(f"Generated SQL query: {llm_response}")

    sql_query_cleaned = extract_sql_query(llm_response)

    if "from" in sql_query_cleaned.lower():
        table_name = sql_query_cleaned.lower().split("from")[1].split()[0].replace(";", "").strip()
        if table_name.lower() not in [t.lower() for t in tables]:
            return f"Error: Table '{table_name}' not found in the database."

    try:
        result = connection.execute(text(sql_query_cleaned))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        # Send the retrieved data to LLM for analysis and generate a final report
        final_report = send_data_for_analysis(df, prompt)
        
        return final_report
    except Exception as e:
        return f"SQL query execution error: {str(e)}"

# Streamlit interface
st.title("SQL Database Chat with Data Analysis")
st.write("Ask a question about the data, and I will not only return the results but also analyze them.")

# Display chat history
for message in st.session_state["sql_chat_messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Enter your question about the data:"):
    st.session_state["sql_chat_messages"].append({"role": "user", "content": prompt})
    save_chat_history()

    with st.chat_message("user"):
        st.markdown(prompt)

    result = multiagent_sql_analysis(prompt)

    with st.chat_message("assistant"):
        st.markdown(result)

    st.session_state["sql_chat_messages"].append({"role": "assistant", "content": result})
    save_chat_history()

# Button to clear chat history
if st.button("Clear History"):
    clear_chat_history()
