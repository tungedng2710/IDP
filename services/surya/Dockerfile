FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies for adding repositories
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# Install Python 3.10 and common tools
RUN apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3.10-dev \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Verify installation
RUN python3 --version && pip --version
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# Install Python dependencies for fast api
RUN pip install --no-cache-dir \
    PyMuPDF \
    fastapi \
    uvicorn \
    python-multipart \
    Pillow

RUN mkdir /app
WORKDIR /app