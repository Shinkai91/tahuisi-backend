FROM python:3.8-slim

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libglib2.0-dev \
    libpoppler-cpp-dev \
    tesseract-ocr \
    libtesseract-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --force-reinstall --no-deps -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]