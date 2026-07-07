# Build for HPC (x86_64) even from Apple Silicon
# docker buildx build --platform linux/amd64 -t ghcr.io/stracquadaniolab/glycplm-nf:0.0.0 --push .

FROM mambaorg/micromamba:1.5.8 AS build

ARG MAMBA_DOCKERFILE_ACTIVATE=1

SHELL ["/bin/bash", "-c"]

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH

COPY environment.yml /tmp/environment.yml

RUN micromamba install -y -p $MAMBA_ROOT_PREFIX \
      -f /tmp/environment.yml \
      --override-channels && \
    micromamba clean --all -y

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
      huggingface_hub && \
    python -m pip install --no-cache-dir \
      git+https://github.com/Biohub/transformers.git@main && \
    python -m pip install --no-cache-dir \
      esm@git+https://github.com/Biohub/esm.git@main && \
    python -m pip install --no-cache-dir \
      torch==2.12.1 \
      --index-url https://download.pytorch.org/whl/cu126

RUN mkdir -p /opt/conda/models && \
    python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="biohub/ESMC-300M",
    local_dir="/opt/conda/models/ESMC-300M",
)
PY

FROM debian:bookworm-slim

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/lib

COPY --from=build /opt/conda /opt/conda
