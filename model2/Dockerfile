FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the *current folder* as a package
COPY . /app/model2

# Run as module so relative imports work
ENTRYPOINT ["python", "-m", "model2.model2_runner"]
