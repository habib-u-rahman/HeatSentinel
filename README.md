# HeatSentinel

Hyperlocal urban-heat intelligence platform — FortyGuard Hackathon '26.

## Status

Task 1: repo scaffold + FortyGuard API client (submit-and-poll, on-disk cache, retries).
No ML models, agents, or frontend yet.

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # then fill in FORTYGUARD_API_KEY, GROQ_API_KEY, CITY_NAME, AOI_BBOX
```

## Run tests

```bash
cd backend
pytest
```

## Probe the FortyGuard API schema

The exact poll endpoint path and response field names are **unconfirmed** — see the
"VERIFY AGAINST DOCS" constants at the top of `app/fortyguard/client.py`. Run the probe
to submit a tiny AOI and print the raw submit + poll JSON, then update those constants:

```bash
cd backend
python -m app.fortyguard.client --probe
```

## Run the API

```bash
cd backend
uvicorn app.main:app --reload
```
