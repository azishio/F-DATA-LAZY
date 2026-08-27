FROM python:3.12-slim AS base
ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

FROM base AS build
COPY pyproject.toml README.md /app/
COPY src /app/src
# CPU-only torch keeps the image a fraction of the CUDA build's size
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu "/app[onnx]"
# Bake both model backends as LOCAL DIRECTORIES: loading by model ID queries
# the HF Hub (list_repo_files) to locate the backend file even when cached,
# so an HF-cache-only image is not actually offline-capable.
# The ONNX variant is the container default (~2-3x faster on CPU).
RUN python -c "from sentence_transformers import SentenceTransformer as ST; \
    ST('all-MiniLM-L6-v2').save('/opt/models/torch'); \
    ST('all-MiniLM-L6-v2', backend='onnx').save('/opt/models/onnx')"

FROM base
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin/fdata /usr/local/bin/fdata
COPY --from=build /opt/models /opt/models
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    FDATA_ENCODER_BACKEND=onnx \
    FDATA_MODEL_DIR=/opt/models
# Offline smoke check: under HF_HUB_OFFLINE any Hub call raises, so this
# fails the build if loading either backend still needs the network
RUN python -c "from fdata.embedding import load_model; \
    load_model('torch'); load_model('onnx'); print('offline load OK')"
ENTRYPOINT ["fdata"]
CMD ["--help"]
