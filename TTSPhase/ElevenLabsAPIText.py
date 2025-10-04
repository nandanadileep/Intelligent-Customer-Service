import os
import requests
import json
import time
from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env

# --- Load API key from config file ---
api_key = os.getenv("ELEVENLABS_API_KEY")

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

voice_id = config["voice_id"]

# Your sample text
text_to_speak = "Hello! This is Shiv, from Clanker Customer Service Speaking, What do you want to know Miss Deleep ?"

# Endpoint URL
url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

# Headers
headers = {
    "xi-api-key": api_key,
    "Content-Type": "application/json"
}

# Data payload
data = {
    "text": text_to_speak,
    "model_id": "eleven_monolingual_v1",  # or use "eleven_multilingual_v1"
    "voice_settings": {
        "stability": 0.75,
        "similarity_boost": 0.75
    }
}

# Send the request
response = requests.post(url, headers=headers, json=data)
# Save the resulting audio to a file with a timestamp in the filename
if response.status_code == 200:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"output_audio_{timestamp}.mp3"
    with open(output_filename, "wb") as f:
        f.write(response.content)
    print(f"✅ Audio file saved as {output_filename}")
else:
    print("❌ Failed to generate audio")
    print(response.status_code, response.text)
