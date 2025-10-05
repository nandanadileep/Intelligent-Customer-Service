import os
import sys
import io
import time
import shutil
import wave
from typing import Optional
import subprocess

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is on path so we can import existing modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path=ENV_PATH, override=False)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import existing pipeline pieces
from STTPhase.wavWhisperSingleFile import processAudio
from RAGs.Implementation_with_GRPO import ask_query_with_grpo
from TTSPhase.ElevenLabsAPIText import genAudioText


app = FastAPI(title="voiceLLM Server")

# CORS - Allow all origins for now (restrict in production)
allowed_origins = os.getenv('ALLOWED_ORIGINS', '*')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins.split(',')] if allowed_origins != '*' else ['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static route to serve generated TTS audio directly
tts_audio_dir = os.path.join(PROJECT_ROOT, 'TTSPhase', 'elevenAudio')
os.makedirs(tts_audio_dir, exist_ok=True)
app.mount('/audio', StaticFiles(directory=tts_audio_dir), name='audio')

# Directory to store uploaded/normalized wavs
uploads_dir = os.path.join(CURRENT_DIR, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)


def convert_to_wav(input_path: str, output_path: str) -> str:
    """Convert input media to 16kHz mono WAV using ffmpeg. Returns output path.
    Requires ffmpeg to be installed and on PATH.
    """
    ffmpeg_bin = shutil.which('ffmpeg')
    if not ffmpeg_bin:
        raise RuntimeError('ffmpeg not found on PATH')

    cmd = [
        ffmpeg_bin,
        '-y',
        '-i', input_path,
        '-ac', '1',
        '-ar', '16000',
        '-f', 'wav',
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def ensure_wav(input_bytes: bytes, target_wav_path: str, original_filename: str) -> str:
    """Store upload as WAV if possible; if not, save original then convert to WAV via ffmpeg."""
    # If the upload file extension is already .wav, try to save directly
    _, ext = os.path.splitext(original_filename.lower())
    if ext == '.wav':
        try:
            with wave.open(io.BytesIO(input_bytes), 'rb'):
                pass
            with open(target_wav_path, 'wb') as f:
                f.write(input_bytes)
            return target_wav_path
        except wave.Error:
            # Fall through to conversion if header invalid
            pass

    # Save original bytes to temp file then convert
    tmp_input_path = target_wav_path.replace('.wav', ext if ext else '.webm')
    with open(tmp_input_path, 'wb') as f:
        f.write(input_bytes)

    try:
        return convert_to_wav(tmp_input_path, target_wav_path)
    finally:
        # Best-effort cleanup of temp source
        try:
            os.remove(tmp_input_path)
        except OSError:
            pass


@app.get('/')
def root():
    """Health check endpoint for Railway"""
    return {
        'status': 'healthy',
        'message': 'voiceLLM server running',
        'version': '1.0.0'
    }


@app.get('/health')
def health():
    """Additional health check endpoint"""
    return {'status': 'ok'}


@app.post('/process')
async def process(file: UploadFile = File(...)):
    """Main pipeline: STT -> RAG -> TTS"""
    try:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        base_name = f"query_{timestamp}.wav"
        save_path = os.path.join(uploads_dir, base_name)

        # Read upload into memory and store/convert to wav
        content = await file.read()
        stored_path = ensure_wav(content, save_path, file.filename or 'upload.webm')

        # Choose directory and filename for processAudio API
        directory = os.path.dirname(stored_path)
        filename = os.path.basename(stored_path)

        # Transcribe
        transcript: Optional[str] = processAudio(filename, directory=directory)
        if not transcript:
            return JSONResponse(status_code=400, content={'error': 'transcription_failed'})

        # RAG answer
        answer: str = ask_query_with_grpo(transcript)

        # TTS
        audio_filename = f"response_{timestamp}"
        audio_path = genAudioText(answer, filename=audio_filename, directory=tts_audio_dir)

        audio_url = None
        if audio_path and os.path.exists(audio_path):
            # Convert absolute path to public /audio URL
            audio_url = f"/audio/{os.path.basename(audio_path)}"

        return {
            'transcript': transcript,
            'answer': answer,
            'audio_url': audio_url,
        }
    
    except Exception as e:
        # Better error handling for production
        print(f"Error in /process endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': 'internal_server_error', 'details': str(e)}
        )


# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


    