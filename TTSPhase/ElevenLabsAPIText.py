import os
import requests
import json
import time
from dotenv import load_dotenv

# Load .env from project root to avoid CWD issues when run from subdirectories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=CURRENT_DIR, override=False)

api_key = os.getenv("ELEVENLABS_API_KEY")

# --- Load API key and config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
voice_id = config["voice_id"]
model_id = config.get("model_id", "eleven_monolingual_v1")
voice_settings = config.get("voice_settings", {"stability": 0.75, "similarity_boost": 0.75})

def genAudioText(text_to_speak, filename="output", directory="elevenAudio"):
    """
    Generate audio from text using ElevenLabs API, save as filenameEleven.mp3 in the specified directory.
    Returns the full path to the saved audio file.
    """
    os.makedirs(directory, exist_ok=True)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "text": text_to_speak,
        "model_id": model_id,
        "voice_settings": voice_settings
    }
    response = requests.post(url, headers=headers, json=data)
    output_filename = f"{filename}Eleven.mp3"
    output_path = os.path.join(directory, output_filename)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Audio file saved as {output_path}")
        return output_path
    else:
        print("❌ Failed to generate audio")
        print(response.status_code, response.text)
        return None

# Example usage
if __name__ == "__main__":
    text = "Hello! This is Shiv, from Clanker Customer Service Speaking, What do you want to know Miss Deleep?"
    genAudioText(text, filename="testQuery", directory="elevenAudio")