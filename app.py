import streamlit as st

about_page = st.Page(
    "views/about.py",
    title="About",
    icon=":material/info:",
)

settings_page = st.Page(
    "views/settings.py",
    title="LLM",
    icon=":material/settings:",
)

db_settings_page = st.Page(
    "views/db_settings.py",
    title="Database",
    icon=":material/settings:",
)

database_analysis_page = st.Page(
    "views/database_analysis.py",
    title="Database Analysis",
    icon=":material/analytics:",
    default=True,
)

tables_and_columns_page = st.Page(
    "views/tables_and_columns.py",
    title="Tables and Columns",
    icon=":material/table:",
)

sql_query_generator_page = st.Page(
    "views/sql_query_generator.py",
    title="SQL Query Generator",
    icon=":material/article_shortcut:",
)

sql_query_executor_page = st.Page(
    "views/sql_query_executor.py",
    title="SQL Query Executor",
    icon=":material/data_array:",
)

data_plotting_page = st.Page(
    "views/data_plotting.py",
    title="Data Plotting",
    icon=":material/query_stats:",
)

sql_chat_page = st.Page(
    "views/sql_chat.py",
    title="SQL Chat",
    icon=":material/smart_toy:",
)

llm_chat_page = st.Page(
    "views/llm_chat.py",
    title="LLM Chat",
    icon=":material/smart_toy:",
)

pg = st.navigation(
    {
        "Menu": [database_analysis_page, tables_and_columns_page, sql_query_generator_page, sql_query_executor_page, data_plotting_page, sql_chat_page, llm_chat_page],
        "Settings": [settings_page, db_settings_page, about_page],
    }
)

st.logo(
    "assets/Qpossible_logo.png",
)

pg.run()