FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/app backend/app
COPY models models
COPY data/cache data/cache
COPY data/raw data/raw
COPY data/street_images data/street_images
COPY data/overlays data/overlays

WORKDIR /app/backend

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
