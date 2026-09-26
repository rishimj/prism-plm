# PRISM-Bio live demo API (CPU). Build from the repository root:
#   docker build -f demo/backend/Dockerfile -t prism-bio-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false \
    PRISM_REPO_ROOT=/srv/prism \
    PRISM_ASSETS=/srv/prism/demo/backend/assets \
    MODEL_NAME=facebook/esm2_t12_35M_UR50D

WORKDIR /srv/prism

COPY demo/backend/requirements.txt demo/backend/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r demo/backend/requirements.txt

# Bake the model weights into the image so cold starts are fast and offline-safe
RUN python -c "from transformers import AutoModelForMaskedLM, AutoTokenizer as T; \
import os; n=os.environ['MODEL_NAME']; T.from_pretrained(n); AutoModelForMaskedLM.from_pretrained(n)"

# Only the parts of the research package the API imports
COPY src/__init__.py src/__init__.py
COPY src/utils src/utils
COPY src/steering src/steering
COPY demo/backend/app demo/backend/app
COPY demo/backend/assets demo/backend/assets

RUN useradd --system --uid 10001 prism && chown -R prism /opt/hf
USER prism

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1
CMD ["uvicorn", "demo.backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
