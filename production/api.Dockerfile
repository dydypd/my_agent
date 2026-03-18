# api.Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Builds the FastAPI gateway image.
# Context: my_agent/production/  (run docker-compose from that directory)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Install system deps for hiredis
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cache layer)
COPY production/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy production source
COPY production/ /app/production/

# Copy the existing graph source so graph/__init__.py can import it
COPY src/ /app/src/

# Ensure both directories are on PYTHONPATH
ENV PYTHONPATH="/app/production:/app/src"
WORKDIR /app/production

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
