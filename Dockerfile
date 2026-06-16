FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY ml_training/ ./ml_training/
COPY data/processed/ ./data/processed/

# Copy trained models if they exist locally.
# For production Azure deployments, models should be loaded from
# Azure Blob Storage at startup rather than baked into the image.
# See infrastructure/azure_deploy.md for Blob Storage setup.
COPY data/models/ ./data/models/

# Do NOT copy data/raw/ — patient data never goes in the image
# Do NOT copy .env — secrets come from Key Vault at runtime

EXPOSE 8000

# Production: gunicorn with uvicorn workers
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--timeout", "120", "app.main:app"]
