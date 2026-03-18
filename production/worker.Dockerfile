# worker.Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Builds the Worker image.
# Context: my_agent/production/
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY production/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY production/ /app/production/
COPY src/ /app/src/

ENV PYTHONPATH="/app/production:/app/src"
WORKDIR /app/production

# Worker is a long-running process — no exposed port
CMD ["python", "-m", "worker.main"]
