# Multi-arch base image (works natively on x86_64 and Oracle Ampere ARM64)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies:
# - ca-certificates & curl: secure HTTPS API communication
# - ffmpeg: audio and video transcoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application source code
COPY . .

# Ensure data directory structure exists
RUN mkdir -p /app/data/media

# Run the Telegram bot
CMD ["python", "bot.py"]
