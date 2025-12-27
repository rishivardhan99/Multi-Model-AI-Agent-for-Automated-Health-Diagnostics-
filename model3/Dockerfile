FROM python:3.10-slim

WORKDIR /app

# system deps for any potential libs (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "model3_runner.py"]
