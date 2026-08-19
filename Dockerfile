# Sports tracker: RF-DETR detection plus supervision ByteTrack, over trickle.
#
# Base is python-slim, not nvidia/cuda:*-devel: nothing here compiles against
# CUDA, torch's cu128 wheels carry the runtime, and the driver arrives through
# the container runtime.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

# OpenCV's, not CUDA's, so slim needs them either way.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Pinned first so nothing downstream quietly swaps in a CPU build from PyPI.
RUN python -m pip install --no-cache-dir \
        torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# rfdetr: Apache-2.0 detector. supervision: MIT tracking and annotators.
RUN python -m pip install --no-cache-dir \
        rfdetr supervision av opencv-python-headless numpy aiohttp \
        "livepeer-gateway>=1.0.0"

# Bake the weights in so the container runs offline and the first session does
# not pay for the download.
ARG RFDETR_SIZE=medium
RUN python -c "import rfdetr; getattr(rfdetr, 'RFDETR' + '$RFDETR_SIZE'.capitalize())()"

# Fail the build, not the first session, if the CUDA stack cannot load.
RUN python -c "import torch, rfdetr, supervision; print('torch', torch.__version__, '/ cuda', torch.version.cuda)"

WORKDIR /app
COPY runner.py client.py ./

EXPOSE 8989

CMD ["python", "runner.py"]
