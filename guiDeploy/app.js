// Configure your backend URL (FastAPI)
// For local dev: http://127.0.0.1:8000
// For deployed server: replace with your public backend URL
const API_BASE = 'http://127.0.0.1:8000';

const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const fileInput = document.getElementById('fileInput');
const sendFileBtn = document.getElementById('sendFileBtn');
const statusEl = document.getElementById('status');
const transcriptEl = document.getElementById('transcript');
const answerEl = document.getElementById('answer');
const player = document.getElementById('player');
const timerEl = document.getElementById('timer');

let mediaRecorder;
let recordedChunks = [];
let timerInterval;
let seconds = 0;

function setStatus(text) {
  statusEl.textContent = text;
}

function resetTimer() {
  clearInterval(timerInterval);
  seconds = 0;
  timerEl.textContent = '00:00';
}

function startTimer() {
  resetTimer();
  timerInterval = setInterval(() => {
    seconds += 1;
    const m = String(Math.floor(seconds / 60)).padStart(2, '0');
    const s = String(seconds % 60).padStart(2, '0');
    timerEl.textContent = `${m}:${s}`;
  }, 1000);
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = async () => {
      const blob = new Blob(recordedChunks, { type: 'audio/webm' });
      await sendAudioBlob(blob);
    };
    mediaRecorder.start();
    startTimer();
    setStatus('Recording...');
    recordBtn.disabled = true; stopBtn.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus('Microphone access denied or unavailable');
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  stopBtn.disabled = true; recordBtn.disabled = false; resetTimer();
  setStatus('Processing...');
}

async function sendAudioBlob(blob) {
  try {
    setStatus('Uploading audio...');
    const form = new FormData();
    // The backend expects a WAV filename; the server converts if needed
    form.append('file', blob, 'recording.webm');
    const res = await fetch(`${API_BASE}/process`, {
      method: 'POST',
      body: form
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    transcriptEl.textContent = data.transcript || '';
    answerEl.textContent = data.answer || '';
    if (data.audio_url) {
      player.src = `${API_BASE}${data.audio_url}`; // audio_url starts with '/audio/...'
      player.play().catch(() => {});
    }
    setStatus('Done');
  } catch (e) {
    console.error(e);
    setStatus('Failed to process audio');
  }
}

async function sendSelectedFile() {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  setStatus('Processing file...');
  await sendAudioBlob(file);
}

recordBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);
sendFileBtn.addEventListener('click', sendSelectedFile);


