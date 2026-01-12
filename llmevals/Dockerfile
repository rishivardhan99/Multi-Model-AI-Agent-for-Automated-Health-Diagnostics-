# -------- Base image --------
FROM python:3.11-slim

# -------- Environment hygiene --------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# -------- Workdir --------
WORKDIR /app

# -------- System deps (minimal, safe TLS) --------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -------- Python deps --------
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# -------- Copy only what is needed --------
COPY llmevals_pkg/ llmevals_pkg/

# (Optional) copy default env; can be overridden at runtime
COPY .env .env

# -------- Default entrypoint --------
# You can override args (inputs/out/mode) at runtime
ENTRYPOINT ["python", "-m", "llmevals_pkg.run_eval"]
