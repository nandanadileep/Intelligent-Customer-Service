import os
import requests
import wave
from pydub import AudioSegment
import re
import tarfile

# --- CONFIGURATION CONSTANTS ---
DATASET_ROOT = "LibriSpeech/dev-clean"
WAV_OUTPUT_DIR = "processed_wavs"
LOG_FILE = "transcription_log.txt"
LIBRISPEECH_URL = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
ARCHIVE_NAME = "dev-clean.tar.gz"

# Ensure the output directory exists
os.makedirs(WAV_OUTPUT_DIR, exist_ok=True)

def download_and_extract_dataset():
    """Downloads and extracts the LibriSpeech dev-clean dataset if not present."""
    if not os.path.exists(DATASET_ROOT):
        print(f"Dataset not found at '{DATASET_ROOT}'. Downloading...")
        response = requests.get(LIBRISPEECH_URL, stream=True)
        with open(ARCHIVE_NAME, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete. Extracting...")
        with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
            tar.extractall(path=".")
        print("Extraction complete.")
        os.remove(ARCHIVE_NAME)
    else:
        print("Dataset already exists.")

# --- UTILITY FUNCTIONS (from your original script) ---

def load_audio(filename):
    """Loads a WAV file and returns audio bytes and sample rate."""
    # This is slightly modified to handle the expected WAV format from pydub
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
    print(f"  -> Calling STT API for {audio_id} at {sample_rate} Hz...")
    # In a real scenario, this is where you'd send a POST request to your API
    # with the audio_content and parameters.
    return f"[STT Result]: Simulated transcription for {audio_id}"

def convert_flac_to_wav(flac_path, wav_output_path):
    """Converts a FLAC file to a 16kHz WAV file using pydub."""
    try:
        # Load the FLAC file
        audio = AudioSegment.from_file(flac_path, format="flac")
        
        # Resample to 16kHz (Standard for most STT APIs) and save as WAV
        audio = audio.set_frame_rate(16000)
        audio.export(wav_output_path, format="wav")
        return True
    except Exception as e:
        print(f"  -> ERROR: Failed to convert {flac_path} to WAV. Ensure FFmpeg is installed and accessible.")
        print(f"  -> Error details: {e}")
        return False

# --- MAIN DATASET PROCESSING LOGIC ---

def process_dataset():
    """Iterates through the LibriSpeech dataset, converts FLAC to WAV, and calls the STT API."""
    
    total_processed = 0
    
    # 1. Find the master transcript file
    # LibriSpeech structure has a single .trans.txt file per chapter (e.g., in 121123/)
    for root, dirs, files in os.walk(DATASET_ROOT):
        for file in files:
            if file.endswith(".trans.txt"):
                trans_file_path = os.path.join(root, file)
                chapter_id = os.path.basename(root)
                print(f"Found transcript file for chapter {chapter_id}: {trans_file_path}")
                
                # 2. Process each line in the transcript file
                with open(trans_file_path, 'r') as f_trans, open(LOG_FILE, 'a') as f_log:
                    for line in f_trans:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Format: AUDIO_ID TRANSCRIPT_TEXT
                        parts = line.split(" ", 1)
                        audio_id = parts[0]
                        ground_truth = parts[1]
                        
                        # Construct file paths
                        # Example: 84-121123-0000 -> 84/121123/84-121123-0000.flac
                        speaker_id = audio_id.split('-')[0]
                        
                        flac_filename = f"{audio_id}.flac"
                        flac_path = os.path.join(DATASET_ROOT, speaker_id, chapter_id, flac_filename)
                        
                        wav_filename = f"{audio_id}.wav"
                        wav_output_path = os.path.join(WAV_OUTPUT_DIR, wav_filename)

                        print(f"\nProcessing {audio_id}...")
                        
                        # 3. Convert FLAC to WAV
                        if convert_flac_to_wav(flac_path, wav_output_path):
                            
                            # 4. Load the new WAV file
                            audio_content, sample_rate = load_audio(wav_output_path)
                            
                            if audio_content is not None:
                                # 5. Call the STT API
                                stt_result = call_stt_api(audio_content, sample_rate, audio_id)
                                
                                # 6. Log the results
                                log_entry = (
                                    f"ID: {audio_id}\n"
                                    f"GT: {ground_truth}\n"
                                    f"STT: {stt_result}\n"
                                    "------------------------------------------\n"
                                )
                                f_log.write(log_entry)
                                total_processed += 1
                            
                        # Optional: Remove the temporary WAV file to save space
                        # os.remove(wav_output_path)
                        
    print(f"\n--- Processing Complete ---")
    print(f"Total files processed: {total_processed}")
    print(f"Results logged to: {LOG_FILE}")

if __name__ == "__main__":
    download_and_extract_dataset()
    process_dataset()