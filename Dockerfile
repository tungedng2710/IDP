FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.2.1-gpu-cuda12.6-cudnn9.5

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        python-multipart \
        pillow \
        opencv-python \
        dataclasses

COPY . .

EXPOSE 7860

CMD ["uvicorn", "submission_api:app", "--reload", "--host", "0.0.0.0", "--port", "7875"]
