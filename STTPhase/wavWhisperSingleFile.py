import os
import wave
import whisper

model = whisper.load_model("tiny")  # Change as needed

def processAudio(filename, directory="recordedWavs"):
    """
    Transcribes the given WAV file using Whisper, saves the transcript as .txt in the same directory,
    and returns the transcribed text.
    """
    wav_path = os.path.join(directory, filename)
    txt_path = os.path.join(directory, filename.replace('.wav', '.txt'))

    # Load audio
    try:
        with wave.open(wav_path, "rb") as wf:
            audio_bytes = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()
    except wave.Error as e:
        print(f"ERROR: Could not load WAV file: {wav_path}. {e}")
        return None

    # Transcribe using Whisper
    print(f"  -> Transcribing {wav_path} ...")
    result = model.transcribe(wav_path, language="English")
    transcript_text = result["text"]

    # Save transcript
    try:
        with open(txt_path, 'w', encoding='utf-8') as f_out:
            f_out.write(transcript_text)
        print(f"  -> Transcript saved to: {txt_path}")
    except Exception as e:
        print(f"  -> ERROR: Failed to write output file {txt_path}. Error: {e}")

    return transcript_text

# Example usage
if __name__ == "__main__":
    filename = "rec_1759564478.wav"
    directory = "recordedWavs"
    text = processAudio(filename, directory)
    print("Transcribed Text:", text)