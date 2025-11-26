FROM vllm/vllm-openai:nightly

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    Pillow

# RUN pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
# # Copy FastAPI service file
# COPY api_service.py /app/

# EXPOSE 9666