FROM python:3.13-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libhdf5-dev \
    python3-dev \
    python3-pip \
    libfreetype6-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt dev-requirements.txt tox.ini ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r dev-requirements.txt

# Create necessary directories
RUN mkdir -p \
    data/raw/zipped \
    data/raw/extracted \
    data/inprogress \
    data/processed \
    data/external \
    reports/figures \
    models \
    logs \
    config

# Copy source code
COPY src/ src/
COPY setup.py .

# Install the project in editable mode
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV DVC_NO_ANALYTICS=1
