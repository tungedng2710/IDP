FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    PyMuPDF \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    openai>=1.5.0 \
    transformers \
    Pillow

# Clone dots.ocr repository for utils
# RUN git clone https://github.com/rednote-hilab/dots.ocr.git /app/dots.ocr

# Set Python path
ENV PYTHONPATH=/app/dots.ocr:$PYTHONPATH

# Copy FastAPI service file
# COPY api_service.py /app/

# EXPOSE 9667

# CMD ["python", "api_service.py"]