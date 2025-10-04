import os
import wave
import whisper

# Load the Whisper model (choose "base", "small", etc.)
model = whisper.load_model("tiny")  # You can change to "tiny", "small", "medium", or "large"

# --- CONFIGURATION CONSTANTS ---
WAV_INPUT_DIR = "recordedWavs"  #change to your wav files
TRANSCRIPT_OUTPUT_DIR = "processed_text" # New directory for individual TXT files
FILE_LIMIT = 5 # Process only the first 5 files

# Ensure the output directory exists
os.makedirs(TRANSCRIPT_OUTPUT_DIR, exist_ok=True)

# --- UTILITY FUNCTIONS (Keep these) ---

def load_audio(filename):
    """Loads a WAV file and returns audio bytes and sample rate."""
    try:
        with wave.open(filename, "rb") as wf:
            audio_bytes = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()
        return audio_bytes, sample_rate
    except wave.Error as e:
        print(f"ERROR: Could not load WAV file: {filename}. {e}")
        return None, None

def call_stt_api(audio_content, sample_rate, audio_id):
    """Placeholder for actual API call, simulates a result."""
    # Simulate API call time    
    # time.sleep(0.5) 
    
    print(f"  -> Sending {audio_id}.wav to STT API...")
    
    # In a real scenario, this is where you'd send the audio_content
    # and receive the transcription string.
    
    # Transcribe the audio file
    result = model.transcribe(f"{WAV_INPUT_DIR}/{audio_id}.wav", language="English")

    # Print the recognized text
    print("Transcription:", result["text"])


    # Simulating a result:
    return f"[TRANSCRIPT]: This is the automated speech recognition output for audio file {audio_id}. The sample rate used was {sample_rate} Hz. \n {result['text']}"

# --------------------------------------------------------------------------
# 🎯 MAIN PROCESSING LOGIC
# --------------------------------------------------------------------------

def process_first_n_files(limit):
    """
    Reads the first N WAV files, calls the API, and saves each transcript 
    to a separate .txt file.
    """
    
    # 1. Get all WAV files and sort them
    all_wav_files = sorted([f for f in os.listdir(WAV_INPUT_DIR) if f.endswith(".wav")])
    
    if not all_wav_files:
        print(f"ERROR: No .wav files found in {WAV_INPUT_DIR}. Please check the directory.")
        return

    # 2. Select only the first 'limit' files
    files_to_process = all_wav_files[:limit]
    
    print(f"Found {len(all_wav_files)} WAV files. Processing the first {len(files_to_process)}.")

    for file_name in files_to_process:
        wav_path = os.path.join(WAV_INPUT_DIR, file_name)
        audio_id = file_name.replace('.wav', '')
        
        # 3. Define the output file path (e.g., processed_text/174-50561-0000.txt)
        txt_output_path = os.path.join(TRANSCRIPT_OUTPUT_DIR, f"{audio_id}.txt")

        print(f"\nProcessing file: {file_name}")

        # 4. Load the WAV file
        audio_content, sample_rate = load_audio(wav_path)
        
        if audio_content is not None:
            # 5. Call the STT API
            stt_result = call_stt_api(audio_content, sample_rate, audio_id)
            
            # 6. Save the transcription result to the individual TXT file
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
    
    print("\n✅ Limited batch processing complete.")

# --- EXECUTION ---
if __name__ == "__main__":
    process_first_n_files(limit=FILE_LIMIT)