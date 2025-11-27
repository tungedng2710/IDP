FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        openai \
        pillow \
        python-multipart

COPY services/chandra_api/ /app/

EXPOSE 9670

CMD ["python", "app.py"]
