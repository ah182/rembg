FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# إعداد فولدرات الكاش والـ model
RUN mkdir -p /app/.u2net /app/.cache/huggingface /tmp/numba_cache \
    && chmod -R 777 /app/.u2net /app/.cache /tmp/numba_cache

# نسخ ملفات المشروع
COPY . .

# تعيين متغيرات البيئة
ENV HF_HOME=/app/.cache/huggingface
ENV U2NET_HOME=/app/.u2net
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV NUMBA_DISABLE_JIT=1

# تشغيل التطبيق
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
