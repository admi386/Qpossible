import json
import os

SETTINGS_FILE = "llm_settings.json"

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as file:
            return json.load(file)
    else:
        return {
            "model": "gemma2:latest",
            "base_url": "http://localhost:11434/v1",
            "api_key": "",
            "max_tokens": 4096,
            "temperature": 0.1,
            "llm_type": "ollama"
        }

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as file:
        json.dump(settings, file, indent=4)