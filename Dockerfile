# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10.8-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements first (layer caching) ───────────────────────────────────
COPY requirements_app.txt .

# ── Install PyTorch CPU (HF Spaces free tier has no GPU) ─────────────────────
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    torchvision==0.20.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install MONAI and all other dependencies ──────────────────────────────────
RUN pip install --no-cache-dir \
    monai==1.5.2 \
    streamlit==1.45.1 \
    plotly==6.1.2 \
    huggingface_hub==0.33.0 \
    matplotlib==3.10.3 \
    numpy==1.26.4 \
    scipy==1.15.3 \
    scikit-image==0.25.2 \
    nibabel==5.3.2 \
    einops==0.8.1 \
    tqdm==4.67.1

# ── Copy application code ─────────────────────────────────────────────────────
COPY app.py .
COPY download_models.py .
COPY src/ ./src/

# ── Streamlit config ──────────────────────────────────────────────────────────
RUN mkdir -p /app/.streamlit
COPY .streamlit/config.toml .streamlit/config.toml

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Run ───────────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
