FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get remove -y python3-blinker && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/runpod-volume/huggingface_cache
ENV PYTHONPATH=$PYTHONPATH:/workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod

COPY src /workspace/src
COPY handler.py .

CMD [ "python", "-u", "/workspace/handler.py" ]
