import sys
import os
import time

# Dynamically add project root to sys.path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from STTPhase.SimpleSTT import record_until_q
from STTPhase.wavWhisperSingleFile import processAudio
from RAGs.Implementation_with_GRPO import ask_query_with_grpo
from TTSPhase.ElevenLabsAPIText import genAudioText

def main():
    # Use absolute path for current working directory
    # abs_dir = os.path.abspath(os.getcwd())
    # Generate a unique filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_filename = f"query_{timestamp}.wav"

    # 1. Record audio
    audio_path = record_until_q(base_filename, current_dir)

    # 2. Transcribe audio
    transcript = processAudio(base_filename, current_dir)

    # 3. Query RAG with GRPO
    answer = ask_query_with_grpo(transcript)

    # 4. Generate audio from answer text
    genAudioText(answer, filename=f"response_{timestamp}", directory=current_dir)
    # print("Hello World")

if __name__ == "__main__":
    main()