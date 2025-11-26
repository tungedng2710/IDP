FROM vllm/vllm-openai:latest

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
    Pillow \
    "mineru-vl-utils[vllm]==0.1.14"

# Copy FastAPI service file
# COPY api_service.py /app/

# EXPOSE 9668