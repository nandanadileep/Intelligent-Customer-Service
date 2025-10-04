import pyaudio
import collections
import time
from queue import Queue
from threading import Thread
import struct
import numpy as np
import wave
import keyboard # <-- NEW IMPORT

# --- Global Control Flag ---
STOP_LISTENING_FLAG = False 

# --- Audio Parameters ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30
PADDING_DURATION_MS = 300
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
VOICE_ACTIVITY_TIMEOUT = 5000 
# --- VAD Parameters (RMS Method) ---
THRESHOLD = 500

# Queue for storing full speech segments
speech_queue = Queue()

def calculate_rms(chunk):
    """Calculate the Root Mean Square (RMS) energy of an audio chunk."""
    count = len(chunk) // 2
    format = "%dh" % count
    shorts = struct.unpack(format, chunk)
    return np.sqrt(np.mean(np.square(shorts)))

# --- NEW FUNCTION: Key Listener ---
def key_listener():
    """Monitors for a specific keypress to set the global stop flag."""
    global STOP_LISTENING_FLAG
    print("\nPress 'q' at any time to quit the listening system.")
    
    # Blocks until the 'q' key is pressed
    keyboard.wait('q')
    STOP_LISTENING_FLAG = True
    print("\n'q' pressed. Initiating graceful shutdown...")


def audio_producer():
    """Captures audio, detects speech via RMS, and puts full segments into a queue."""
    global STOP_LISTENING_FLAG
    audio = pyaudio.PyAudio()
    stream = None
    try:
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK_SIZE)

        print("--- LISTENING STARTED ---")
        print(f"--- RMS Threshold is set to: {THRESHOLD} ---")
        
        ring_buffer = collections.deque(maxlen=PADDING_CHUNKS) 
        triggered = False
        voiced_data = []
        silent_chunks = 0

        # Loop breaks if the keyboard thread sets the flag
        while not STOP_LISTENING_FLAG:

            
            # Check for immediate stop condition (to break before reading a chunk)
            if STOP_LISTENING_FLAG:
                break

            # print("--- LISTENING STARTED --- I'm Listening ", voiced_data)
                
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            rms_energy = calculate_rms(chunk)
            # is_speech = rms_energy > THRESHOLD
            is_speech = True #TODO: NOTE: NEED TO MAKE IT WORKKKKKKKK

            if not triggered:
                ring_buffer.append(chunk)
                if is_speech:
                    print(f"...Speech detected (RMS: {rms_energy:.0f}), starting recording...")
                    triggered = True
                    voiced_data.extend(list(ring_buffer))
                    ring_buffer.clear()
                    silent_chunks = 0
            else:
                voiced_data.append(chunk)
                if is_speech:
                    silent_chunks = 0
                else:
                    silent_chunks += 1
                
                if silent_chunks > PADDING_CHUNKS:
                    print("...End of speech detected, sending for transcription...")
                    
                    cutoff_index = len(voiced_data) - silent_chunks
                    full_audio_segment = b''.join(voiced_data[:cutoff_index])
                    
                    speech_queue.put(full_audio_segment)
                    
                    # Reset state
                    triggered = False
                    voiced_data = []
                    silent_chunks = 0
                    ring_buffer.extend(voiced_data[-PADDING_CHUNKS:]) 
        
        if triggered and voiced_data:
            print("...Saving last segment before quitting...")
            full_audio_segment = b''.join(voiced_data)
            speech_queue.put(full_audio_segment)

    except Exception as e:
        print(f"An error occurred in the audio producer: {e}")
        
    finally:
        
        print("--- Stopping audio stream. ---")
        if stream:
            stream.stop_stream()
            stream.close()
        audio.terminate()
        # Signal the consumer to stop only if the stop flag was set
        if STOP_LISTENING_FLAG:
             speech_queue.put(None)


def call_stt_api(audio_content, sample_rate):
    """Simulates a call to a Speech-to-Text API."""
    if len(audio_content) > 5000:
        time.sleep(1)
        return f"*** [STT Result]: The user spoke {len(audio_content)} bytes of audio. (RMS VAD used)"
    else:
        return "No clear speech detected for transcription."

def audio_consumer():
    """Consumes full speech segments, SAVES TO FILE, and calls the STT API."""
    print("--- Consumer started, waiting for speech segments. ---")
    file_counter = 1
    while True:
        # Use a timeout to occasionally check the STOP_LISTENING_FLAG
        try:
            audio_segment = speech_queue.get(timeout=0.1) 
        except:
            if STOP_LISTENING_FLAG:
                break
            continue
        
        if audio_segment is None or STOP_LISTENING_FLAG:
            break
        
        try:
            # 1. DEFINE FILENAME
            filename = f"speech_segment_{file_counter}.wav"
            
            # 2. SAVE THE AUDIO SEGMENT TO A .WAV FILE
            print(f"💾 Saving audio segment to {filename}...")
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(audio_segment)
            
            # 3. CALL THE API (or simulate the call)
            transcript = call_stt_api(audio_segment, RATE)
            print(f"\n✅ TRANSCRIPTION COMPLETE:\n{transcript}\n")
            
            file_counter += 1 
            
        except Exception as e:
            print(f"\n❌ ERROR during processing: {e}")
    
    print("--- Consumer stopping. ---")


# Start all threads
producer_thread = Thread(target=audio_producer)
consumer_thread = Thread(target=audio_consumer)
# --- NEW THREAD for Keyboard Listener ---
key_thread = Thread(target=key_listener)

key_thread.start()
producer_thread.start()
consumer_thread.start()

# Wait for threads to finish
key_thread.join()
producer_thread.join()
consumer_thread.join()

print("--- System shutdown complete. ---")