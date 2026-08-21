FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py analyze_previews.py ./
RUN mkdir -p /data/audio-cache /data/jobs

ENV PORT=8000 AUDIO_CACHE_DIR=/data/audio-cache JOB_DIR=/data/jobs
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
