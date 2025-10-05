# syntax=docker/dockerfile:1

FROM python:3.11-slim

# System deps: ffmpeg, git, and build essentials for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy minimal server and project files
COPY server/requirements.txt /app/server/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Copy entire repo (you can optimize later)
COPY . /app

# Expose port
EXPOSE 8000

# Environment for uvicorn
ENV HOST=0.0.0.0 PORT=8000

# Start FastAPI server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]


