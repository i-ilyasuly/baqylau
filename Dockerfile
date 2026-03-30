FROM python:3.11

# Жүйелік тәуелділіктер — dlib үшін міндетті
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Алдымен dlib жеке орнат (ең ұзақ орнатылатын)
RUN pip install --no-cache-dir dlib

# Қалған кітапханалар
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
