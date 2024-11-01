import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from db_settings_manager import load_db_settings

# Load database settings
if "db_settings" not in st.session_state:
    st.session_state.db_settings = load_db_settings()

# Initialize session state for SQL query and result
if "sql_query" not in st.session_state:
    st.session_state.sql_query = ""

if "query_result" not in st.session_state:
    st.session_state.query_result = None

if "executed" not in st.session_state:
    st.session_state.executed = False  # Track if the query was executed

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

# Streamlit interface
st.title("SQL Query Executor")

# User input interface for SQL query
user_sql_query = st.text_area("Enter your SQL query:", st.session_state.sql_query)

# Button to execute the SQL query
if st.button("Run SQL Query") and user_sql_query:
    # Save the SQL query to session state
    st.session_state.sql_query = user_sql_query

    connection = connect_to_database()
    if connection:
        try:
            # Execute the query and fetch the result
            result = pd.read_sql_query(st.session_state.sql_query, connection)
            
            # Store the result in session state
            st.session_state.query_result = result

            # Mark query as executed
            st.session_state.executed = True

        except Exception as e:
            st.error(f"Error executing query: {e}")
        finally:
            connection.close()

# Display the result only if the query was executed and result is stored
if st.session_state.executed and st.session_state.query_result is not None:
    st.write("## Query Result:")
    st.dataframe(st.session_state.query_result)  # Using dataframe instead of write for better format
