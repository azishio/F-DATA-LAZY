FROM python:3.12-slim AS base
ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

FROM base AS build
COPY pyproject.toml README.md /app/
COPY src /app/src
# CPU-only torch keeps the image a fraction of the CUDA build's size
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu /app
# Bake the sentence-transformers model into the image so runtime needs no network
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

FROM base
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin/fdata /usr/local/bin/fdata
COPY --from=build /opt/hf /opt/hf
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["fdata"]
CMD ["--help"]
