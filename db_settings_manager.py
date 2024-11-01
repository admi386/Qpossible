import json
import os

DB_SETTINGS_FILE = "db_settings.json"

def load_db_settings():
    """Load database connection settings from a file."""
    if os.path.exists(DB_SETTINGS_FILE):
        with open(DB_SETTINGS_FILE, "r") as file:
            return json.load(file)
    else:
        # Default values
        return {
            "db_type": "PostgreSQL",
            "host": "localhost",
            "port": "5432",
            "db_name": "",
            "username": "",
            "password": ""
        }

def save_db_settings(settings):
    """Save database connection settings to a file."""
    with open(DB_SETTINGS_FILE, "w") as file:
        json.dump(settings, file, indent=4)