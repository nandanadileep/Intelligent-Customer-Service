import os
import wave
import whisper

# Load the Whisper model
model = whisper.load_model("tiny")  # Change as needed

# --- CONFIGURATION CONSTANTS ---
WAV_FILE_DIR = "recordedWavs"
WAV_FILE = "rec_1759564478.wav"  # Your single WAV file
TRANSCRIPT_OUTPUT_DIR = "processed_text"
os.makedirs(TRANSCRIPT_OUTPUT_DIR, exist_ok=True)

def load_audio(filename):
    """Loads a WAV file and returns audio bytes and sample rate."""
    try:
        with wave.open(f"{WAV_FILE_DIR}/{filename}", "rb") as wf:
            audio_bytes = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()
        return audio_bytes, sample_rate
    except wave.Error as e:
        print(f"ERROR: Could not load WAV file: {filename}. {e}")
        return None, None

def call_stt_api(audio_content, sample_rate, audio_id):
    """Transcribes the audio file using Whisper."""
    print(f"  -> Sending {audio_id}.wav to STT API...")
    result = model.transcribe(f"{audio_id}.wav", language="English")
    print("Transcription:", result["text"])
    return f"[TRANSCRIPT]: Whisper output for {audio_id}. Sample rate: {sample_rate} Hz.\n{result['text']}"

if __name__ == "__main__":
    audio_id = WAV_FILE.replace('.wav', '')
    txt_output_path = os.path.join(TRANSCRIPT_OUTPUT_DIR, f"{audio_id}.txt")
    audio_content, sample_rate = load_audio(WAV_FILE)
    if audio_content is not None:
        stt_result = call_stt_api(audio_content, sample_rate, audio_id)
        try:
            with open(txt_output_path, 'w') as f_out:
                f_out.write(f"Audio ID: {audio_id}\n")
                f_out.write(f"Sample Rate: {sample_rate} Hz\n\n")
                f_out.write(stt_result)
            print(f"  -> Transcript saved to: {txt_output_path}")
        except Exception as e:
            print(f"  -> ERROR: Failed to write output file {txt_output_path}. Error: {e}")
    else:
        print(f"  -> Skipping {audio_id} due to loading error.")