# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Install system dependencies including cmake and C++ compiler
RUN set -ex; \
  apt-get update && \
  apt-get install -y \
  cmake \
  build-essential \
  libopenblas-dev \
  liblapack-dev \
  libx11-dev \
  libgtk-3-dev \
  && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir numpy
RUN pip3 install --no-cache-dir dlib
RUN pip3 install --no-cache-dir face-recognition
RUN pip3 install --no-cache-dir opencv-python
RUN pip3 install --no-cache-dir imutils

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Run command
CMD ["python3", "index.py"] 