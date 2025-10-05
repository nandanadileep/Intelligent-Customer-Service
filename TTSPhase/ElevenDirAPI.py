import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env

# --- Load API key from config file ---
api_key = os.getenv("ELEVENLABS_API_KEY")
api_key = "sk_57ae3b6b247a6ca193723a5e3b0053957a7df45a0af19936"

CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

voice_id = config["voice_id"]

# --- Directories ---
TEXT_INPUT_DIR = "sampleTexts"
AUDIO_OUTPUT_DIR = "elevenAudio"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# --- ElevenLabs API settings ---
url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
headers = {
    "xi-api-key": api_key,
    "Content-Type": "application/json"
}
model_id = config.get("model_id", "eleven_monolingual_v1")
voice_settings = config.get("voice_settings", {"stability": 0.75, "similarity_boost": 0.75})

# --- Process each text file ---
for filename in os.listdir(TEXT_INPUT_DIR):
    if filename.endswith(".txt"):
        text_path = os.path.join(TEXT_INPUT_DIR, filename)
        with open(text_path, "r", encoding="utf-8") as f:
            text_to_speak = f.read().strip()
        if not text_to_speak:
            print(f"⚠️ Skipping empty file: {filename}")
            continue

        data = {
            "text": text_to_speak,
            "model_id": model_id,
            "voice_settings": voice_settings
        }

        print(f"🔊 Generating audio for: {filename}")
        response = requests.post(url, headers=headers, json=data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"{os.path.splitext(filename)[0]}_{timestamp}.mp3"
        audio_path = os.path.join(AUDIO_OUTPUT_DIR, audio_filename)

        if response.status_code == 200:
            with open(audio_path, "wb") as audio_file:
                audio_file.write(response.content)
            print(f"✅ Saved: {audio_path}")
        else:
            print(f"❌ Failed for {filename}: {response.status_code} {response.text}")