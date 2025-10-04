import pyaudio
import wave
import keyboard
import time
import whisper

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
DIRECTORY_NAME = "recordedWavs"
FILENAME = "simple_recording.wav"

model = whisper.load_model("tiny")  # Change as needed

def call_stt_api(audio_content, sample_rate):
    """Simulates a call to a Speech-to-Text API."""
    if len(audio_content) > 5000:
        return f"*** [STT Result]: The user spoke {len(audio_content)} bytes of audio."
    else:
        return "No clear speech detected for transcription."

# ...existing code...

def record_until_q():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    # Generate filename with timestamp
    timestamp = int(time.time())
    filename = f"{DIRECTORY_NAME}/rec_{timestamp}.wav"

    print("Recording... Press 'q' to stop.")

    while True:
        if keyboard.is_pressed('q'):
            print("Stopping recording.")
            break
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save to WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    print(f"Audio saved to {filename}")

    # Send to API
    transcript = call_stt_api(b''.join(frames), RATE)
    # NOTE: Below not working in my stupid Windows PC
    # result = model.transcribe(f"{filename}", language="English")
    # print(f"TRANSCRIPTION RESULT:\n{result['text']}")
    print(f"TRANSCRIPTION RESULT:\n{transcript}")

# ...existing code...

if __name__ == "__main__":
    record_until_q()