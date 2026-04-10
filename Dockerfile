FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/runpod-volume/huggingface_cache
ENV PYTHONPATH=$PYTHONPATH:/workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --ignore-installed \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt
RUN pip install --no-cache-dir runpod

COPY src /workspace/src
COPY handler.py .

CMD [ "python", "-u", "/workspace/handler.py" ]
