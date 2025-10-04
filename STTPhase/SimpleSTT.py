import pyaudio
import wave
import keyboard
import time
import os

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

def record_until_q(filename=None, directory="recordedWavs"):
    """Record audio until 'q' is pressed and save to specified directory/filename."""
    os.makedirs(directory, exist_ok=True)
    full_path = f"{directory}/{filename}"

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    print(f"Recording... Press 'q' to stop. Saving to {full_path}")

    while True:
        if keyboard.is_pressed('q'):
            print("Stopping recording.")
            break
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(full_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    print(f"Audio saved to {full_path}")
    return filename

if __name__ == "__main__":
    import os
    timestamp = int(time.time())
    filename = f"rec_{timestamp}.wav"
    directory = "recordedWavs"
    record_until_q(filename, directory)