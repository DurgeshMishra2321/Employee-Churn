FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

ENV PYTHONPATH=/app/src

# Train at build time so the image is self-contained — no binary model
# artifact needs to be committed or uploaded separately. Deterministic
# (fixed random_state), so this reproduces the model verified in CI.
RUN python src/train.py

ENV MODEL_PATH=/app/models/model.joblib
ENV TRAINING_STATS_PATH=/app/models/training_feature_stats.csv
ENV PREDICTION_LOG_PATH=/app/reports/exports/prediction_log.csv

EXPOSE 8000

# $PORT is injected by most PaaS hosts (Render, Railway); falls back to 8000 locally.
CMD ["sh", "-c", "uvicorn serve:app --app-dir src --host 0.0.0.0 --port ${PORT:-8000}"]
