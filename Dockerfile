FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    CHUNKS_PATH=/data/chunks/telco_v1.jsonl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY eval ./eval
COPY ingestion ./ingestion
COPY retrieval ./retrieval
COPY scripts ./scripts
COPY serving ./serving
COPY specs ./specs

EXPOSE 8080

RUN useradd --create-home --shell /usr/sbin/nologin appuser
USER appuser

CMD ["python", "-m", "serving.app"]
