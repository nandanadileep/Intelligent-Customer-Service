from pydub import AudioSegment
import whisper
import os

# Load your short audio clip (2 seconds)
input_path = "processed_wavs/84-121123-0007.wav"
audio = AudioSegment.from_wav(input_path)

# Pad with silence to make it 30 seconds (Whisper expects 30s chunks)
target_duration_ms = 30 * 1000  # 30 seconds in milliseconds
padding_needed = target_duration_ms - len(audio)

if padding_needed > 0:
    silence = AudioSegment.silent(duration=padding_needed)
    padded_audio = audio + silence
else:
    padded_audio = audio  # already long enough

# Save the padded audio to a temporary file
padded_path = "temp_padded.wav"
padded_audio.export(padded_path, format="wav")

# Load Whisper model
model = whisper.load_model("base")

# Transcribe the padded audio
result = model.transcribe(padded_path, language="English")

# Print the output
print("Transcription:", result["text"])

# Optional: remove the temporary padded file
os.remove(padded_path)
