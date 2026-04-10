FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY models /workspace/models

COPY requirements.txt .
RUN pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt

COPY src /workspace/src
COPY handler.py .

CMD [ "python", "-u", "/workspace/handler.py" ]