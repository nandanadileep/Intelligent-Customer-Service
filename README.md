# voiceLLM: Speech-to-Text Project

## Overview

**voiceLLM** is a Python-based toolkit for recording, processing, and transcribing speech audio using local models (OpenAI Whisper) and external APIs.  
It supports real-time microphone recording, dataset handling, and batch transcription.

---

## Directory Structure & Main Files

- **SimpleSTT.py**  
  Record audio from your microphone until you press `q`.  
  The recording is saved as a timestamped WAV file in the `recordedWavs` directory.  
  Optionally, transcribe the audio using Whisper.

- **wavWhisperSingleFile.py**  
  Transcribes a single WAV file (e.g., `rec_<timestamp>.wav`) using Whisper and saves the result to `processed_text`.

- **wavAPIDirectory.py**  
  Transcribes a Directory WAV file (e.g., `rec_<timestamp>.wav`) using Whisper and saves the result to `processed_text`.

- **DataSet.py**  
  Downloads and extracts the LibriSpeech `dev-clean` dataset if not present.  
  Converts FLAC files to WAV, loads transcripts, and prepares data for batch processing or API calls.

- **STTApi.py**  
  Downloads a sample audio file, loads it, and sends it to a placeholder speech-to-text API.  
  Useful for testing API integration.

- **recordedWavs/**  
  Directory for microphone recordings.

- **processed_wavs/**  
  Directory for WAV files converted from the LibriSpeech dataset.

- **processed_text/**  
  Directory for storing transcription results as text files.

---

## Typical Workflow

1. **Record Audio**  
   Run `SimpleSTT.py` to record speech.  
   Press `q` to stop and save the file.

2. **Transcribe Recording**  
   Use `wavWhisperSingleFile.py` to transcribe a specific WAV file using Whisper.

3. **Work With Datasets**  
   Run `DataSet.py` to download and prepare the LibriSpeech dataset.

4. **Test API Integration**  
   Use `STTApi.py` to send sample audio to your speech-to-text API.

---

## Requirements

- Python 3.8+
- `pyaudio` (for recording)
- `keyboard` (for keypress detection)
- `whisper` (for local transcription)
- `pydub` and `ffmpeg` (for audio conversion)
- `requests` (for downloading datasets)

---

## Getting Started

1. Install dependencies:
   ```
   pip install pyaudio keyboard openai-whisper pydub requests
   ```
2. Run any script as needed:
   ```
   python SimpleSTT.py
   python wavAPIDirectory.py
   python wavWhisperSingleFile.py
   python DataSet.py
   python STTApi.py
   ```

---

**Enjoy experimenting with speech-to-text!**