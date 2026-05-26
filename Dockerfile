FROM python:3.11-slim

# Install curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer-cached separately from app code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (.env and models_store are excluded via .dockerignore)
COPY . .

# Create runtime directories — models and data are written here at runtime
RUN mkdir -p data models_store/isolation_forest models_store/lstm_autoencoder models_store/lstm_forecast models_store/cnn_pattern models_store/baselines

EXPOSE 8000

# Liveness probe — Railway and Docker both use this
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
