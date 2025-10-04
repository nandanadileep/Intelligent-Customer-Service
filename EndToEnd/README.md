# EndToEnd Voice-to-Response Pipeline (VOICELLM)

This project implements a complete **Voice-to-Response** pipeline, combining Speech-to-Text (STT), Retrieval-Augmented Generation (RAG), and Text-to-Speech (TTS) functionalities.

## 🚀 Overview

The `Pipeline.py` script orchestrates the following four main steps:

1.  **🎙️ Record Audio:** Captures user voice input and saves it to a unique WAV file (using `record_until_q`).
2.  **📝 Transcribe Audio (STT):** Converts the recorded audio file into text (a `transcript`) using a whisper-based process (`processAudio`).
3.  **🧠 Query RAG:** Uses the transcribed text to query a **Retrieval-Augmented Generation (RAG)** system with **GRPO** (using `ask_query_with_grpo`) to generate an informative `answer`.
4.  **🗣️ Generate Audio (TTS):** Converts the text `answer` back into an audio file using the ElevenLabs API (`genAudioText`), completing the voice-to-voice cycle.

## 🛠️ Dependencies

The pipeline relies on several internal modules:

* `STTPhase.SimpleSTT`
* `STTPhase.wavWhisperSingleFile`
* `RAGs.Implementation_with_GRPO`
* `TTSPhase.ElevenLabsAPIText`

Ensure all these modules and their respective dependencies (e.g., ElevenLabs API key, whisper models, RAG setup) are correctly configured in the project environment.

## 🏃 Usage

To run the full end-to-end pipeline:

```bash
python Pipeline.py