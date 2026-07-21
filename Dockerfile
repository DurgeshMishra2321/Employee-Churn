FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/model.joblib
ENV TRAINING_STATS_PATH=/app/models/training_feature_stats.csv
ENV PREDICTION_LOG_PATH=/app/reports/exports/prediction_log.csv

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
