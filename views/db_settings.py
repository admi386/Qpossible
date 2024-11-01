import streamlit as st
from db_settings_manager import load_db_settings, save_db_settings

# Load settings from a file
if "db_settings" not in st.session_state:
    st.session_state.db_settings = load_db_settings()

st.title("Database Settings")

# Warning message
st.warning("Please be aware that database data might be modified or lost during interaction. Ensure that you have read-only permissions or make a backup of the data before proceeding.")

# Input fields for database settings
st.session_state.db_settings["db_type"] = st.selectbox("Database Type:", ["PostgreSQL", "MySQL", "SQLite", "SQL Server"], 
                                                       index=["PostgreSQL", "MySQL", "SQLite", "SQL Server"].index(st.session_state.db_settings["db_type"]))

if st.session_state.db_settings["db_type"] == "SQLite":
    st.session_state.db_settings["db_name"] = st.text_input("SQLite Database Path:", st.session_state.db_settings["db_name"])
else:
    st.session_state.db_settings["host"] = st.text_input("Host:", st.session_state.db_settings["host"])
    st.session_state.db_settings["port"] = st.text_input("Port:", st.session_state.db_settings["port"])
    st.session_state.db_settings["db_name"] = st.text_input("Database Name:", st.session_state.db_settings["db_name"])
    st.session_state.db_settings["username"] = st.text_input("Username:", st.session_state.db_settings["username"])
    st.session_state.db_settings["password"] = st.text_input("Password:", st.session_state.db_settings["password"], type="password")

# Save the settings to a file
if st.button("Save Database Settings"):
    save_db_settings(st.session_state.db_settings)
    st.success("Database settings have been updated and saved.")