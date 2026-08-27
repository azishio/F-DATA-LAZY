FROM python:3.12-slim AS base
ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

FROM base AS build
COPY pyproject.toml README.md /app/
COPY src /app/src
# CPU-only torch keeps the image a fraction of the CUDA build's size
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu "/app[onnx]"
# Bake both model backends into the image so runtime needs no network;
# the ONNX variant is the container default (~2-3x faster on CPU)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    SentenceTransformer('all-MiniLM-L6-v2', backend='onnx')"

FROM base
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin/fdata /usr/local/bin/fdata
COPY --from=build /opt/hf /opt/hf
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    FDATA_ENCODER_BACKEND=onnx
ENTRYPOINT ["fdata"]
CMD ["--help"]
